"""Timing instrumentation and the results record.

The JSON schema here is the contract every run writes into and every analysis
script reads from. Bump SCHEMA_VERSION on any breaking change — results from
different schema versions must not be silently compared.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3  # 3 adds the `work` block: how much work a run actually did


class Stopwatch:
    """Accumulating timer. Use as a context manager; total() is the sum of laps."""

    def __init__(self):
        self._total = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._total += time.perf_counter() - self._start
        self._start = None
        return False

    def total(self) -> float:
        return self._total


@dataclass
class StepTimer:
    """Per-step wall times, split into the phases we can honestly separate.

    A note on `compute` vs `comm`: DDP overlaps gradient allreduce with the
    backward pass by design, so backward time already contains some
    communication and the two cannot be cleanly separated from inside the
    training loop. Rather than invent a split, we fold both into `compute` and
    measure interconnect cost separately with the allreduce microbenchmark
    (see scripts/comm_bench.py). Treat `comm_estimate` as indicative only.
    """

    data: list = field(default_factory=list)
    compute: list = field(default_factory=list)
    step: list = field(default_factory=list)

    def record(self, data_s: float, compute_s: float, step_s: float) -> None:
        self.data.append(data_s)
        self.compute.append(compute_s)
        self.step.append(step_s)


def _summarize(values: list) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "p90_s": ordered[min(int(0.90 * len(ordered)), len(ordered) - 1)],
        "p99_s": ordered[min(int(0.99 * len(ordered)), len(ordered) - 1)],
        "stdev_s": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def pbs_environment() -> dict:
    """Scheduler context — job id, node list, queue. Without this you cannot
    tell later which run came from which allocation or node set."""
    nodefile = os.environ.get("PBS_NODEFILE", "")
    nodes = []
    if nodefile and Path(nodefile).exists():
        nodes = [n.strip() for n in Path(nodefile).read_text(encoding="utf-8").split() if n.strip()]
    return {
        "pbs_jobid": os.environ.get("PBS_JOBID", "interactive-or-local"),
        "pbs_queue": os.environ.get("PBS_QUEUE", "unset"),
        "pbs_nodes": sorted(set(nodes)),
        "pbs_node_count": len(set(nodes)),
    }


_SALT_FILE = Path.home() / ".alcf_bench_salt"


def _anon_salt() -> str:
    """Per-machine random salt, created once and reused.

    Salted rather than plain hashing because Aurora node names follow a known,
    enumerable scheme (x4001c0s0b0n0...) -- an unsalted hash could be reversed
    by brute force in seconds. Persisting the salt keeps hashes stable across
    runs, so you can still ask "did I land on the same node twice?" without
    publishing which node it was.
    """
    if _SALT_FILE.exists():
        return _SALT_FILE.read_text().strip()
    salt = secrets.token_hex(16)
    _SALT_FILE.write_text(salt)
    try:
        _SALT_FILE.chmod(0o600)
    except OSError:
        pass  # Windows and some filesystems don't support POSIX modes
    return salt


def _anon_id(value: str, salt: str) -> str:
    return "anon-" + hashlib.sha256((salt + value).encode()).hexdigest()[:10]


def anon_hostname(hostname: str) -> str:
    """The same salted hash anonymize_environment applies, for callers holding a
    bare hostname rather than an environment dict -- the power sidecar files
    carry one in their filename, which no amount of scrubbing inside the JSON
    would hide."""
    return _anon_id(hostname, _anon_salt())


def anonymize_environment(env: dict) -> dict:
    """Strip facility-identifying detail so results can be published.

    Removed: job id (traces to your allocation) and real hostnames.
    Kept: node COUNT, queue, and all software versions -- these are
    methodologically essential and not sensitive.

    Note this cannot scrub --note text, which is yours to write. Don't put a
    project name in it if you intend to publish.
    """
    salt = _anon_salt()
    out = dict(env)
    if out.get("hostname"):
        out["hostname"] = _anon_id(out["hostname"], salt)
    if out.get("pbs_nodes"):
        out["pbs_nodes"] = [_anon_id(n, salt) for n in out["pbs_nodes"]]
    if "pbs_jobid" in out:
        out["pbs_jobid"] = "redacted"
    out["anonymized"] = True
    return out


@dataclass
class RunRecord:
    """One benchmark run, start to finish. Serialized to results/<run_id>.json."""

    machine: str
    workload: str
    config: dict
    environment: dict
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    schema_version: int = SCHEMA_VERSION
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    timing: dict = field(default_factory=dict)
    work: dict = field(default_factory=dict)
    throughput: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    flops: dict = field(default_factory=dict)
    energy: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)
    status: str = "incomplete"
    notes: list = field(default_factory=list)

    # --- population helpers ------------------------------------------------
    def set_step_stats(self, timer: StepTimer, warmup_steps: int, samples_per_step: int):
        """Summarize steady-state steps only.

        The first steps include lazy kernel compilation, allocator growth and
        oneCCL bootstrap. Leaving them in would understate every machine, and
        would penalize whichever machine has the slowest JIT rather than the
        slowest hardware — so they are excluded here and reported separately as
        timing.warmup_s.

        A short run (--max-steps below --warmup-steps) would otherwise discard
        every step and report nothing, so warmup is clamped to half the recorded
        steps. The clamped value is what gets recorded, so the result says how
        many steps were actually excluded rather than how many were requested.
        """
        recorded = len(timer.step)
        warmup_steps = min(warmup_steps, recorded // 2)

        steady = slice(warmup_steps, None)
        self.throughput["step_time"] = _summarize(timer.step[steady])
        self.throughput["data_wait"] = _summarize(timer.data[steady])
        self.throughput["compute"] = _summarize(timer.compute[steady])

        median = self.throughput["step_time"].get("median_s")
        if median:
            self.throughput["samples_per_s"] = samples_per_step / median
        self.throughput["warmup_steps_excluded"] = warmup_steps

    def set_work(
        self,
        steps: int,
        epochs_completed: int,
        epochs_requested: int,
        global_batch: int,
        local_batch: int,
    ):
        """How much work the run did, as distinct from how fast it did it.

        Rates alone cannot say whether two runs are comparable on energy, and
        the run itself decides how much work it does: --target-accuracy stops
        early once the target is hit, --max-steps truncates, and a wall-clock
        limit can cut an epoch short. Any of those produces a joules column that
        looks comparable and is not. Recording the volume is what lets
        summarize.py check that instead of assuming it.

        The two sample counts are deliberately both kept. `samples_global` is
        what the job processed; `samples_per_rank` is what THIS rank processed
        and is the numerator of energy.samples_per_joule, whose denominator is
        this rank's device only. Reporting a global count against a per-device
        energy would overstate efficiency by the rank count.
        """
        self.work = {
            "steps": steps,
            "epochs_completed": epochs_completed,
            "epochs_requested": epochs_requested,
            "samples_global": steps * global_batch,
            "samples_per_rank": steps * local_batch,
            # Not the same question as "did it reach the target": a run can stop
            # early having hit target accuracy (success) or having run out of
            # steps (truncation). This flags only that the epoch budget was not
            # spent, and accuracy.reached_target says which case it was.
            "stopped_early": epochs_completed < epochs_requested,
        }

    def set_flops(
        self,
        flops_per_sample_fwd: int,
        samples_per_s: float,
        peak_per_device: float | None,
        world_size: int,
    ):
        """Achieved FLOP/s and MFU.

        Training cost is taken as 3x forward: one forward plus a backward that
        is roughly two forwards' work (input grads + weight grads). This is the
        standard convention, so numbers stay comparable to published MFU.

        samples_per_s is aggregate over every rank, so the peak it is divided by
        has to be the aggregate peak too. peak_flops.json stores ONE DEVICE, so
        scale it by world_size -- dividing by the per-device figure would report
        12x the true MFU on a full Aurora node.
        """
        train_flops_per_sample = 3 * flops_per_sample_fwd
        achieved = train_flops_per_sample * samples_per_s
        peak_total = peak_per_device * world_size if peak_per_device else None
        self.flops = {
            "fwd_flops_per_sample": flops_per_sample_fwd,
            "train_flops_per_sample": train_flops_per_sample,
            "achieved_flops_per_s": achieved,
            "peak_flops_per_s_per_device": peak_per_device,
            "peak_flops_per_s": peak_total,
            "mfu": (achieved / peak_total) if peak_total else None,
        }
        if peak_total is None:
            self.notes.append(
                "MFU not computed: no peak FLOP/s configured for this device "
                "in configs/peak_flops.json"
            )

    def set_energy(
        self,
        joules: float | None,
        joules_to_target: float | None,
        wall_s: float,
        samples_processed: int,
        scope: str,
        devices_counted: int,
    ):
        """Energy for the run, and the two ratios worth comparing machines on.

        samples_per_joule is the energy twin of samples/s, and joules_to_target
        the twin of time-to-accuracy -- the pair that answers "high throughput
        at low power" rather than just "fast".

        scope and devices_counted are recorded because the number is meaningless
        without them: an accelerator-only figure excludes CPU, memory, NICs and
        cooling, and a node draws well more than the sum of its accelerators.
        """
        if not joules:
            self.energy = {"joules": None, "scope": scope}
            self.notes.append(f"Energy not recorded: {scope}")
            return

        self.energy = {
            "joules": joules,
            "avg_watts": joules / wall_s if wall_s else None,
            # The numerator, kept beside the ratio it produced. Without it a
            # samples_per_joule column cannot be checked, re-derived, or
            # compared between runs that did different amounts of work.
            "samples_processed": samples_processed,
            "samples_per_joule": samples_processed / joules if joules else None,
            "joules_to_target_accuracy": joules_to_target,
            "scope": scope,
            "devices_counted": devices_counted,
            "wall_s": wall_s,
        }

    def set_power(self, node_summaries: list, timeline_files: list):
        """Per-device energy across every monitored node, from the sampler.

        Deliberately a separate block rather than folded into set_energy:
        energy.joules stays "this rank's device", so results written before the
        sampler existed remain comparable to ones written after. This block is
        the node-complete view, and the only one that can see a device the job
        never used -- energy.joules is read by a rank, and an unused device has
        no rank to read it.
        """
        if not node_summaries:
            return

        bound = sum(s.get("joules_bound") or 0 for s in node_summaries)
        idle = sum(s.get("joules_idle") or 0 for s in node_summaries)
        total = bound + idle
        idle_devices = sum(s.get("devices_idle") or 0 for s in node_summaries)

        self.power = {
            "nodes_monitored": len(node_summaries),
            "devices_total": sum(s.get("device_count") or 0 for s in node_summaries),
            "devices_idle": idle_devices,
            "joules_total": total or None,
            "joules_bound": bound or None,
            "joules_idle": idle or None,
            "idle_fraction": (idle / total) if total else None,
            "sample_interval_s": node_summaries[0].get("sample_interval_s"),
            "timeline_files": timeline_files,
            "per_node": node_summaries,
        }

        if idle and total:
            self.notes.append(
                f"{idle / total * 100:.0f}% of accelerator energy ({idle:.0f} J) "
                f"went to {idle_devices} device(s) this job never used."
            )

    def set_cost(self, node_count: int, wall_s: float):
        node_seconds = node_count * wall_s
        self.cost = {
            "node_count": node_count,
            "wall_s": wall_s,
            "node_seconds": node_seconds,
            "node_hours": node_seconds / 3600.0,
        }

    def write(self, results_dir: str) -> Path:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.machine}_{self.workload}_{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False), encoding="utf-8")
        return path


def load_peak_flops(config_path: str, device_name: str, precision: str):
    """Look up vendor peak FLOP/s for MFU.

    Matched by substring against the device name reported by the runtime, so
    the config keys can stay short. Returns None when unknown — MFU is then
    omitted rather than guessed.
    """
    path = Path(config_path)
    if not path.exists():
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    for key, entry in table.items():
        if key.lower() in device_name.lower():
            value = entry.get(precision)
            return float(value) if value else None
    return None
