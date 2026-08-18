"""Benchmark entrypoint: ResNet/CIFAR-10 time-to-accuracy and throughput.

Run one rank per device under mpiexec. On Aurora that is 12 ranks per node,
one per GPU tile:

    mpiexec -n 12 -ppn 12 python -m benchmark.train --nodes-from-pbs

Scaling modes:
  strong  fixed GLOBAL batch, split across ranks  -> accuracy stays comparable,
          so time-to-accuracy is meaningful as you add nodes
  weak    fixed LOCAL batch per rank              -> global batch grows with
          scale, so this measures throughput, NOT accuracy
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from . import data as data_mod
from . import models
from .metrics import (
    RunRecord,
    StepTimer,
    Stopwatch,
    anon_hostname,
    anonymize_environment,
    git_commit,
    load_peak_flops,
    pbs_environment,
)
from .platform import detect_platform, get_rank_info, init_distributed
from .power import PowerSampler, bound_device_indices

PRECISIONS = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def make_profiler(args, platform, is_main):
    """torch.profiler over a few steps on rank 0, or a no-op context.

    Rank 0 only: twelve ranks each writing a trace produces twelve files that
    say the same thing and a great deal of Lustre traffic. The wait/warmup
    phases exist because the first steps are JIT compilation -- profiling them
    measures the compiler, not the model.
    """
    if not args.profile or not is_main:
        return nullcontext()

    from torch.profiler import profile, schedule

    return profile(
        activities=platform.profiler_activities(),
        schedule=schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_steps,
            repeat=1,
        ),
        record_shapes=True,
    )


def write_profile(prof, platform, args, run_id, log):
    """Chrome trace for Perfetto, plus the top ops by device time.

    The trace is the useful artifact -- key_averages() totals per operator and
    so cannot show a gap, which is exactly what you look for when a step is
    slower than the work in it.
    """
    out_dir = Path(args.profile_dir) if args.profile_dir else Path(args.results_dir) / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{platform.name}_{run_id}_trace.json"
    prof.export_chrome_trace(str(trace_path))
    log(f"wrote {trace_path}  (open at https://ui.perfetto.dev)")

    try:
        log("top ops by device self time:")
        print(
            prof.key_averages().table(
                sort_by=platform.profiler_sort_key(), row_limit=15
            ),
            flush=True,
        )
    except Exception as exc:  # sort key varies by torch build; the trace is what matters
        log(f"profiler summary table unavailable ({exc})")


def usable_cores() -> int:
    """Cores this process may actually run on, not the node's core count.

    sched_getaffinity reports the affinity mask. A launcher that pins each rank
    to one core -- mpiexec without --depth, which is easy to do by hand -- leaves
    exactly one, on a node with dozens.
    """
    try:
        return len(os.sched_getaffinity(0))  # Linux only
    except AttributeError:
        return os.cpu_count() or 1


def default_workers() -> int:
    """Dataloader workers, capped by the cores actually available.

    Spawning four workers onto one core does not fail. It starves the device
    and yields a throughput number that looks entirely real -- which is worse
    than an error, because it gets written to results/ and compared against a
    machine that was bound correctly. Cap instead, so a run launched without
    the right flags is slow and honest rather than fast-looking and wrong.

    One core means zero workers: loading in the main process beats a worker
    fighting the training loop for the same core. Correct launcher flags
    (--depth 8 in both submit scripts) still give the previous default of 4.
    """
    usable = usable_cores()
    if usable <= 1:
        return 0
    return max(1, min(4, usable - 1))  # leave a core for the training process


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        "--platform",
        default="auto",
        # "cuda" is the unidentified-NVIDIA fallback; name the machine where you
        # know it, or results from Polaris and Sophia are indistinguishable.
        choices=["auto", "aurora", "polaris", "sophia", "cuda", "crux", "cpu"],
    )
    p.add_argument("--model", default="resnet20")
    p.add_argument("--workload", default=None, help="label for results; defaults to <model>_cifar10")

    # Scaling
    p.add_argument("--scaling", default="strong", choices=["strong", "weak"])
    # 1536 divides evenly by 12, 24, 48, 96 ranks (1-8 Aurora nodes at 12
    # ranks/node), so a full scaling sweep runs without changing the batch.
    p.add_argument("--global-batch-size", type=int, default=1536, help="strong scaling: total across all ranks")
    p.add_argument("--local-batch-size", type=int, default=128, help="weak scaling: per rank")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="stop early after N steps (0 = full epochs)")
    p.add_argument("--lr", type=float, default=0.1, help="base LR, defined at batch size 128")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup; needed at large global batch")
    p.add_argument("--precision", default="bf16", choices=list(PRECISIONS))
    p.add_argument("--seed", type=int, default=1234)

    # Measurement
    p.add_argument("--target-accuracy", type=float, default=0.90, help="top-1 target for time-to-accuracy")
    p.add_argument("--warmup-steps", type=int, default=20, help="steps excluded from steady-state stats")
    p.add_argument("--eval-every", type=int, default=1, help="epochs between validation passes")

    # Data / IO
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--synthetic-data", action="store_true", help="random tensors; removes all filesystem I/O")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="dataloader workers per rank (default: capped by CPU affinity, "
        "at most 4)",
    )
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--peak-flops-config", default="./configs/peak_flops.json")

    # Platform tuning
    p.add_argument("--no-ipex-optimize", action="store_true", help="skip vendor graph optimizer")
    p.add_argument("--note", action="append", default=[], help="free-text note recorded in the result")
    # --- profiling ---
    p.add_argument(
        "--profile",
        action="store_true",
        help="trace a few steps with torch.profiler on rank 0. Adds overhead -- "
        "never report throughput from a profiled run.",
    )
    p.add_argument("--profile-wait", type=int, default=5, help="steps to skip before profiling")
    p.add_argument("--profile-warmup", type=int, default=3, help="steps to trace but discard")
    p.add_argument("--profile-steps", type=int, default=5, help="steps to actually record")
    p.add_argument(
        "--profile-dir",
        default=None,
        help="where to write the Chrome trace (default: alongside --results-dir)",
    )
    # --- power sampling ---
    p.add_argument(
        "--power-interval",
        type=float,
        default=0.1,
        help="seconds between node-wide power samples; 0 disables. One rank per "
        "node reads every accelerator counter, idle ones included.",
    )
    p.add_argument(
        "--anonymize",
        action="store_true",
        help="hash hostnames and redact the job id, so results can be published. "
        "Does NOT scrub --note text -- that is yours to keep clean.",
    )
    args = p.parse_args()
    # Resolved here rather than left as None so the value the run actually used
    # is what lands in config.dataloader_workers -- a result that says "4" when
    # affinity forced 0 would misdescribe its own measurement.
    if args.workers is None:
        args.workers = default_workers()
    return args


def resolve_batch_sizes(args, world_size: int):
    """Return (local_batch, global_batch) for the chosen scaling mode."""
    if args.scaling == "strong":
        if args.global_batch_size % world_size != 0:
            raise SystemExit(
                f"--global-batch-size {args.global_batch_size} is not divisible by "
                f"world size {world_size}; every rank must get an equal batch."
            )
        local = args.global_batch_size // world_size
        return local, args.global_batch_size
    local = args.local_batch_size
    return local, local * world_size


def lr_at_epoch(args, epoch: int, global_batch: int) -> float:
    """Linear LR scaling with warmup (Goyal et al. 2017), then cosine decay.

    Without the linear scaling rule, changing node count changes the effective
    batch size and therefore the accuracy curve — you would be measuring
    hyperparameters, not machines.
    """
    scaled = args.lr * (global_batch / 128.0)
    if epoch < args.warmup_epochs:
        return scaled * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return scaled * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, platform, dtype, distributed: bool) -> float:
    """Global top-1 accuracy. Counts are allreduced so every rank agrees on the
    number that time-to-accuracy is judged against."""
    model.eval()
    correct = torch.zeros(1, device=platform.device)
    total = torch.zeros(1, device=platform.device)
    for images, labels in loader:
        images = images.to(platform.device, non_blocking=True)
        labels = labels.to(platform.device, non_blocking=True)
        with platform.autocast(dtype):
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum()
        total += labels.numel()
    if distributed:
        torch.distributed.all_reduce(correct)
        torch.distributed.all_reduce(total)
    model.train()
    return (correct / total).item()


def ensure_short_tmpdir(log) -> None:
    """Move TMPDIR somewhere short if it is long enough to break DataLoader.

    Workers talk to the parent over an AF_UNIX socket created under TMPDIR, and
    the kernel caps that path at 108 bytes. PBS sets TMPDIR to
    /var/spool/pbs/tmpdir/<jobid>.<long.fully.qualified.host>/, which spends
    most of the budget before multiprocessing appends its own
    pymp-XXXXXXXX/listener-XXXXXXXX.

    Every submit script exports TMPDIR=/tmp for this reason, but a run launched
    by hand inside an interactive job inherits the long one. The symptom is not
    an error: every worker dies at startup and the parent then waits on an empty
    queue until walltime, which reads as a hang.

    Fixed rather than only warned about, because the failure costs an entire
    allocation and the fix is what the scripts already do -- but it says so, so
    a changed TMPDIR is never a silent surprise.
    """
    current = os.environ.get("TMPDIR", "/tmp")
    # 108-byte cap, less roughly 50 for multiprocessing's own suffix.
    if len(current) <= 58:
        return
    fallback = "/tmp"
    if not os.path.isdir(fallback) or not os.access(fallback, os.W_OK):
        log(f"WARNING: TMPDIR is {len(current)} chars and {fallback} is not "
            "writable. DataLoader workers may fail with 'AF_UNIX path too "
            "long'; pass --workers 0 if they do.")
        return
    os.environ["TMPDIR"] = fallback
    log(f"TMPDIR was {len(current)} chars ({current}) -- too long for the "
        f"DataLoader worker socket, using {fallback} instead.")


def main():
    args = parse_args()
    workload = args.workload or f"{args.model}_cifar10"

    startup = Stopwatch()
    with startup:
        rank, world_size, local_rank = get_rank_info()
        platform = detect_platform(local_rank, force=args.platform)
        distributed = init_distributed(platform, rank, world_size)
        is_main = rank == 0

        torch.manual_seed(args.seed + rank)

        local_batch, global_batch = resolve_batch_sizes(args, world_size)
        dtype = PRECISIONS[args.precision]

    def log(msg):
        if is_main:
            print(f"[bench] {msg}", flush=True)

    log(f"platform={platform.name} device={platform.device_name()} world_size={world_size}")
    log(f"scaling={args.scaling} global_batch={global_batch} local_batch={local_batch} precision={args.precision}")

    # --- data -------------------------------------------------------------
    # Before any worker is spawned: the socket path is fixed at that point.
    ensure_short_tmpdir(log)
    data_setup = Stopwatch()
    with data_setup:
        train_loader, val_loader, train_sampler = data_mod.build_loaders(
            data_dir=args.data_dir,
            local_batch_size=local_batch,
            workers=args.workers,
            rank=rank,
            world_size=world_size,
            synthetic=args.synthetic_data,
            download=False,
            seed=args.seed,
        )

    # --- model ------------------------------------------------------------
    model = models.build_model(args.model).to(platform.device)
    flops_per_sample = models.count_flops_per_sample(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if not args.no_ipex_optimize:
        model, optimizer = platform.optimize(model, optimizer, dtype)

    if distributed:
        # Keyed off the device, not the platform name. DDP rejects device_ids on
        # a CPU module, and `name` is a machine label -- the moment a CPU machine
        # is called something other than "cpu" (crux), a name test routes it
        # down the accelerator branch and the run dies at construction.
        model = DistributedDataParallel(
            model,
            device_ids=None if platform.device.type == "cpu" else [platform.device],
        )

    criterion = nn.CrossEntropyLoss()
    platform.reset_peak_memory()

    # Read the scheduler context once — pbs_environment() re-reads $PBS_NODEFILE
    # off Lustre on every call.
    pbs_env = pbs_environment()
    node_count = pbs_env["pbs_node_count"] or 1
    env_blob = {**platform.environment(), **pbs_env, "git_commit": git_commit()}

    record = RunRecord(
        machine=platform.name,
        workload=workload,
        config={
            "model": args.model,
            "scaling": args.scaling,
            "nodes": node_count,
            "world_size": world_size,
            "ranks_per_node": world_size // node_count,
            "global_batch_size": global_batch,
            "local_batch_size": local_batch,
            "precision": args.precision,
            "epochs": args.epochs,
            "base_lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
            "synthetic_data": args.synthetic_data,
            "target_accuracy": args.target_accuracy,
            "dataloader_workers": args.workers,
            "ipex_optimize": not args.no_ipex_optimize,
        },
        environment=(
            anonymize_environment(env_blob) if args.anonymize else env_blob
        ),
        notes=list(args.note),
    )

    # Every rank generated its own run_id. The power sidecars are written by one
    # rank per node but have to be findable from the single result JSON rank 0
    # writes, so all ranks adopt rank 0's id.
    if distributed:
        ids = [record.run_id]
        try:
            # NCCL and XCCL move the pickled object through device memory, so
            # the device has to be named rather than inferred.
            torch.distributed.broadcast_object_list(
                ids, src=0, device=platform.device
            )
            record.run_id = ids[0]
        except Exception as exc:
            log(f"run_id broadcast failed ({exc}); power sidecars will not match")

    # --- train ------------------------------------------------------------
    timer = StepTimer()
    step = 0
    time_to_target = None
    epoch_to_target = None
    best_acc = 0.0

    platform.synchronize()
    energy_start = platform.energy_joules()
    energy_at_target = None
    train_start = time.perf_counter()

    # One rank per node samples the whole node. Every rank doing it would read
    # the same counters twelve times over for identical numbers, and the point
    # of reading node-wide is to catch devices that have no rank to read them.
    sampler = None
    if args.power_interval > 0 and local_rank == 0:
        sources = platform.node_energy_sources()
        if sources:
            sampler = PowerSampler(
                sources,
                interval_s=args.power_interval,
                t0=train_start,
                bound_devices=bound_device_indices(
                    world_size, node_count, platform.device_count()
                ),
                telemetry=platform.node_telemetry_sources(),
            ).start()
            log(
                f"power: sampling {len(sources)} counters every "
                f"{args.power_interval}s on each node"
            )
        else:
            log("power: no node energy counters on this platform, sampler off")

    with make_profiler(args, platform, is_main) as prof:
        for epoch in range(args.epochs):
            # Phase marks are what let the power series be read as "which part
            # of the run costs what" rather than just a mean.
            if sampler is not None:
                sampler.mark(f"epoch {epoch} train")
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)  # else every epoch reshuffles identically

            lr = lr_at_epoch(args, epoch, global_batch)
            for group in optimizer.param_groups:
                group["lr"] = lr

            epoch_loss = 0.0
            batches = 0
            data_start = time.perf_counter()

            for images, labels in train_loader:
                step_start = time.perf_counter()
                data_wait = step_start - data_start

                images = images.to(platform.device, non_blocking=True)
                labels = labels.to(platform.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with platform.autocast(dtype):
                    loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

                # Device work is async; without this sync the timer would measure
                # kernel launch, not kernel execution.
                platform.synchronize()
                now = time.perf_counter()
                timer.record(data_wait, now - step_start, now - step_start + data_wait)

                epoch_loss += loss.item()
                batches += 1
                step += 1
                data_start = time.perf_counter()

                # Drives the profiler's wait/warmup/active state machine. Must
                # be called every step, including the ones it is ignoring.
                if prof is not None:
                    prof.step()

                if args.max_steps and step >= args.max_steps:
                    break

            train_loss = epoch_loss / max(1, batches)

            acc = None
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                if sampler is not None:
                    sampler.mark(f"epoch {epoch} eval")
                acc = evaluate(model, val_loader, platform, dtype, distributed)
                best_acc = max(best_acc, acc)
                if time_to_target is None and acc >= args.target_accuracy:
                    time_to_target = time.perf_counter() - train_start
                    epoch_to_target = epoch + 1
                    # Read the counter at the crossing rather than scaling total
                    # energy by time_to_target/wall -- power is not constant
                    # across a run, so that estimate would be wrong by however
                    # much the early epochs differ from the late ones.
                    energy_at_target = platform.energy_joules()
                    if sampler is not None:
                        sampler.mark("target accuracy reached")

            elapsed = time.perf_counter() - train_start
            record.curve.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "elapsed_s": elapsed,
                    "lr": lr,
                    "train_loss": train_loss,
                    "val_top1": acc,
                }
            )
            log(
                f"epoch {epoch + 1}/{args.epochs} loss={train_loss:.4f} "
                f"top1={acc if acc is None else f'{acc:.4f}'} elapsed={elapsed:.1f}s"
            )

            if args.max_steps and step >= args.max_steps:
                break

    platform.synchronize()
    total_train_s = time.perf_counter() - train_start
    energy_end = platform.energy_joules()
    timeline = sampler.stop() if sampler is not None else None

    # --- power ------------------------------------------------------------
    # Each monitor rank owns its own node's series and writes it beside the
    # results; only the rollup is shared, because a 100 Hz series over 18
    # counters is megabytes and does not belong in the file every analysis
    # script parses.
    node_summary, timeline_name = None, None
    if timeline:
        node_summary = timeline.summary()
        # The filename carries the hostname, so --anonymize has to reach it too:
        # scrubbing the JSON while publishing a directory listing of real Aurora
        # node names would defeat the whole flag.
        hostname = socket.gethostname()
        try:
            path = timeline.write(
                Path(args.results_dir) / "power",
                platform.name,
                record.run_id,
                anon_hostname(hostname) if args.anonymize else hostname,
            )
            timeline_name = path.name
            print(f"[bench] wrote {path}", flush=True)
        except OSError as exc:
            # Must not propagate: the all_gather below is collective, and a rank
            # that bails on a filesystem error would hang every other one.
            print(f"[bench] power timeline write failed: {exc}", flush=True)

    node_summaries = [node_summary] if node_summary else []
    timeline_names = [timeline_name] if timeline_name else []
    if distributed:
        gathered = [None] * world_size
        try:
            torch.distributed.all_gather_object(
                gathered, (node_summary, timeline_name)
            )
            # Ordering follows rank, so node 0 is always first in the list.
            node_summaries = [s for s, _ in gathered if s]
            timeline_names = [n for _, n in gathered if n]
        except Exception as exc:
            # Object gather goes through the collective backend and is the one
            # thing here that can fail on an exotic build. Losing the other
            # nodes' rollups is not worth losing the run.
            log(f"power: summary gather failed ({exc}); reporting this node only")

    if prof is not None:
        write_profile(prof, platform, args, record.run_id, log)
        record.notes.append(
            "Profiled run: torch.profiler overhead is in these timings, so "
            "throughput and MFU here are not comparable to an unprofiled run."
        )

    # --- record -----------------------------------------------------------
    if is_main:
        record.timing = {
            "startup_s": startup.total(),
            "data_setup_s": data_setup.total(),
            "total_train_s": total_train_s,
            "warmup_steps_excluded": args.warmup_steps,
        }
        record.set_step_stats(timer, args.warmup_steps, samples_per_step=global_batch)
        # curve is appended once per epoch unconditionally -- not only on eval
        # epochs -- so its length is the epoch count without threading another
        # counter out of the loop. A final epoch cut short by --max-steps still
        # counts as one here, which is why the sample totals are derived from
        # `step` rather than from epochs: those stay exact either way.
        record.set_work(
            steps=step,
            epochs_completed=len(record.curve),
            epochs_requested=args.epochs,
            global_batch=global_batch,
            local_batch=local_batch,
        )
        record.accuracy = {
            "target": args.target_accuracy,
            "time_to_target_s": time_to_target,
            "epoch_to_target": epoch_to_target,
            "reached_target": time_to_target is not None,
            "final_top1": record.curve[-1]["val_top1"] if record.curve else None,
            "best_top1": best_acc,
        }
        record.memory = {"peak_bytes_per_device": platform.peak_memory_bytes()}
        peak_per_unit, peak_unit = load_peak_flops(
            args.peak_flops_config, platform.device_name(), args.precision
        )
        record.set_flops(
            flops_per_sample,
            record.throughput.get("samples_per_s", 0.0),
            peak_per_unit,
            peak_unit,
            world_size,
            node_count,
        )
        # This rank's device only, against this rank's samples -- so the ratios
        # are per device. Summing across ranks would double count on Aurora,
        # where power is reported per GPU card but there are two ranks per card.
        joules = (
            energy_end - energy_start
            if energy_end is not None and energy_start is not None
            else None
        )
        record.set_energy(
            joules=joules,
            joules_to_target=(
                energy_at_target - energy_start
                if energy_at_target is not None and energy_start is not None
                else None
            ),
            wall_s=total_train_s,
            samples_processed=step * local_batch,
            scope=platform.energy_scope(),
            devices_counted=1,
        )
        record.set_power(node_summaries, timeline_names)
        record.set_cost(record.config["nodes"], total_train_s)
        if args.synthetic_data:
            record.notes.append("Synthetic data: accuracy is meaningless, throughput only.")
        record.status = "complete"

        path = record.write(args.results_dir)
        log(f"wrote {path}")
        median_s = record.throughput["step_time"].get("median_s")
        log(
            f"median step {f'{median_s * 1e3:.2f} ms' if median_s else 'n/a'} | "
            f"{record.throughput.get('samples_per_s', 0):.0f} samples/s | "
            f"best top1 {best_acc:.4f} | TTA {time_to_target}"
        )
        if record.power.get("joules_total"):
            log(
                f"node accelerators {record.power['joules_total']:.0f} J across "
                f"{record.power['devices_total']} device(s) | "
                f"{record.power['devices_idle']} idle drew "
                f"{record.power.get('joules_idle') or 0:.0f} J "
                f"({(record.power.get('idle_fraction') or 0) * 100:.0f}%)"
            )

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
