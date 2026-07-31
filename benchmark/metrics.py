"""Timing instrumentation and the results record.

The JSON schema here is the contract every run writes into and every analysis
script reads from. Bump SCHEMA_VERSION on any breaking change — results from
different schema versions must not be silently compared.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


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
        nodes = [n.strip() for n in Path(nodefile).read_text().split() if n.strip()]
    return {
        "pbs_jobid": os.environ.get("PBS_JOBID", "interactive-or-local"),
        "pbs_queue": os.environ.get("PBS_QUEUE", "unset"),
        "pbs_nodes": sorted(set(nodes)),
        "pbs_node_count": len(set(nodes)),
    }


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
    throughput: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    flops: dict = field(default_factory=dict)
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
        """
        steady = slice(warmup_steps, None)
        self.throughput["step_time"] = _summarize(timer.step[steady])
        self.throughput["data_wait"] = _summarize(timer.data[steady])
        self.throughput["compute"] = _summarize(timer.compute[steady])

        median = self.throughput["step_time"].get("median_s")
        if median:
            self.throughput["samples_per_s"] = samples_per_step / median
        self.throughput["warmup_steps_excluded"] = warmup_steps

    def set_flops(self, flops_per_sample_fwd: int, samples_per_s: float, peak: float | None):
        """Achieved FLOP/s and MFU.

        Training cost is taken as 3x forward: one forward plus a backward that
        is roughly two forwards' work (input grads + weight grads). This is the
        standard convention, so numbers stay comparable to published MFU.
        """
        train_flops_per_sample = 3 * flops_per_sample_fwd
        achieved = train_flops_per_sample * samples_per_s
        self.flops = {
            "fwd_flops_per_sample": flops_per_sample_fwd,
            "train_flops_per_sample": train_flops_per_sample,
            "achieved_flops_per_s": achieved,
            "peak_flops_per_s": peak,
            "mfu": (achieved / peak) if peak else None,
        }
        if peak is None:
            self.notes.append(
                "MFU not computed: no peak FLOP/s configured for this device "
                "in configs/peak_flops.json"
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
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False))
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
    table = json.loads(path.read_text())
    for key, entry in table.items():
        if key.lower() in device_name.lower():
            value = entry.get(precision)
            return float(value) if value else None
    return None
