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
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from . import data as data_mod
from . import models
from .metrics import (
    RunRecord,
    StepTimer,
    Stopwatch,
    anonymize_environment,
    git_commit,
    load_peak_flops,
    pbs_environment,
)
from .platform import detect_platform, get_rank_info, init_distributed

PRECISIONS = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--platform", default="auto", choices=["auto", "aurora", "cuda", "cpu"])
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--peak-flops-config", default="./configs/peak_flops.json")

    # Platform tuning
    p.add_argument("--no-ipex-optimize", action="store_true", help="skip vendor graph optimizer")
    p.add_argument("--note", action="append", default=[], help="free-text note recorded in the result")
    p.add_argument(
        "--anonymize",
        action="store_true",
        help="hash hostnames and redact the job id, so results can be published. "
        "Does NOT scrub --note text -- that is yours to keep clean.",
    )
    return p.parse_args()


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
        model = DistributedDataParallel(model, device_ids=None if platform.name == "cpu" else [platform.device])

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

    # --- train ------------------------------------------------------------
    timer = StepTimer()
    step = 0
    time_to_target = None
    epoch_to_target = None
    best_acc = 0.0

    platform.synchronize()
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
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

            if args.max_steps and step >= args.max_steps:
                break

        train_loss = epoch_loss / max(1, batches)

        acc = None
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            acc = evaluate(model, val_loader, platform, dtype, distributed)
            best_acc = max(best_acc, acc)
            if time_to_target is None and acc >= args.target_accuracy:
                time_to_target = time.perf_counter() - train_start
                epoch_to_target = epoch + 1

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

    # --- record -----------------------------------------------------------
    if is_main:
        record.timing = {
            "startup_s": startup.total(),
            "data_setup_s": data_setup.total(),
            "total_train_s": total_train_s,
            "warmup_steps_excluded": args.warmup_steps,
        }
        record.set_step_stats(timer, args.warmup_steps, samples_per_step=global_batch)
        record.accuracy = {
            "target": args.target_accuracy,
            "time_to_target_s": time_to_target,
            "epoch_to_target": epoch_to_target,
            "reached_target": time_to_target is not None,
            "final_top1": record.curve[-1]["val_top1"] if record.curve else None,
            "best_top1": best_acc,
        }
        record.memory = {"peak_bytes_per_device": platform.peak_memory_bytes()}
        record.set_flops(
            flops_per_sample,
            record.throughput.get("samples_per_s", 0.0),
            load_peak_flops(args.peak_flops_config, platform.device_name(), args.precision),
            world_size,
        )
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

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
