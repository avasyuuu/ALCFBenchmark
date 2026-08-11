"""Node-wide power sampling: every accelerator on the node, including idle ones.

Two counter reads around the training loop answer "what did this rank's device
cost". Two questions that matter more need more than two samples:

  - How power moves *during* a run -- startup, steady state, eval, collectives.
    That is a shape, and two points cannot describe one.
  - What the devices this job did NOT use were drawing the whole time. No
    reading taken by a rank that owns a device can ever see that, because a
    device with no rank on it has nobody to read it.

So one rank per node samples every counter on the node on a timer thread and
keeps the series. Everything machine-specific stays in platform.py; this file
only knows how to read a list of counters on a clock.

Energy is accumulated as a sum of positive deltas rather than last-minus-first.
For a monotonic counter the two are identical, but hwmon and NVML counters both
reset when the driver reloads, and a reset partway through a run would otherwise
produce a large negative number that looks like a plausible reading.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EnergySource:
    """One readable cumulative-energy counter, in joules.

    device_index is the accelerator this counter maps to in torch's numbering,
    or None when torch cannot address it at all -- a GPU masked out by
    CUDA_VISIBLE_DEVICES is exactly the unused device we are here to catch, so
    it still counts toward the node total.

    aggregate marks a counter that covers silicon another source already
    reports, and so must be excluded from totals: Aurora's whole-card counter
    spans both of that card's tiles.
    """

    key: str
    scope: str
    read: Callable[[], float | None]
    device_index: int | None = None
    aggregate: bool = False


class PowerSampler:
    """Samples every source on a fixed cadence in a background thread.

    Timestamps come from time.perf_counter() and are stored relative to t0, so
    they line up with the elapsed values already recorded in RunRecord.curve --
    a power series you cannot align against the training timeline only tells you
    the average, which the counters already did.
    """

    def __init__(
        self,
        sources: list[EnergySource],
        interval_s: float = 0.1,
        t0: float | None = None,
        bound_devices: frozenset = frozenset(),
    ):
        self.sources = list(sources)
        self.interval_s = interval_s
        self.t0 = time.perf_counter() if t0 is None else t0
        self.bound_devices = frozenset(bound_devices)
        self._samples: list[tuple[float, list]] = []
        self._marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> "PowerSampler":
        if not self.sources:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="power-sampler", daemon=True
        )
        self._thread.start()
        return self

    def mark(self, label: str) -> None:
        """Annotate the timeline. Epoch and eval boundaries are what turn a
        power trace into an answer about which *phase* costs what."""
        if self._thread is not None:
            self._marks.append((time.perf_counter() - self.t0, label))

    def stop(self) -> "PowerTimeline":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return PowerTimeline(
            sources=self.sources,
            samples=self._samples,
            marks=self._marks,
            interval_s=self.interval_s,
            bound_devices=self.bound_devices,
        )

    # --- sampling ----------------------------------------------------------
    def _loop(self) -> None:
        # Absolute deadlines rather than sleep(interval): sleeping for the
        # interval accumulates the read time into every tick, so a 100 ms
        # cadence quietly drifts to 105 ms and the integrated energy is off by
        # the drift. If a read runs long enough to miss a deadline, resync to
        # now instead of firing a burst of catch-up samples.
        deadline = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            self._samples.append(
                (now - self.t0, [source.read() for source in self.sources])
            )
            deadline += self.interval_s
            delay = deadline - time.perf_counter()
            if delay <= 0:
                deadline = time.perf_counter()
                continue
            self._stop.wait(delay)


class PowerTimeline:
    """The recorded series, plus the per-device rollup worth putting in a result."""

    def __init__(self, sources, samples, marks, interval_s, bound_devices):
        self.sources = sources
        self.samples = samples
        self.marks = marks
        self.interval_s = interval_s
        self.bound_devices = bound_devices

    def __bool__(self) -> bool:
        return len(self.samples) > 1

    def watts(self, index: int) -> list[tuple[float, float]]:
        """(elapsed_s, watts) for one source, from consecutive counter deltas.

        Gaps -- an unreadable counter, or a reset -- are dropped rather than
        interpolated. A hole in the series is honest; a straight line drawn
        across one is not.
        """
        out = []
        for (t0, prev), (t1, curr) in zip(self.samples, self.samples[1:]):
            a, b, dt = prev[index], curr[index], t1 - t0
            if a is None or b is None or dt <= 0 or b < a:
                continue
            out.append((t1, (b - a) / dt))
        return out

    def _joules(self, index: int) -> tuple[float, bool]:
        """(joules, saw_reset) as the sum of positive deltas."""
        total, saw_reset = 0.0, False
        for prev, curr in zip(self.samples, self.samples[1:]):
            a, b = prev[1][index], curr[1][index]
            if a is None or b is None:
                continue
            if b < a:
                saw_reset = True
                continue
            total += b - a
        return total, saw_reset

    def span_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1][0] - self.samples[0][0]

    def per_device(self) -> list[dict]:
        """One row per counter: what it drew, and whether this job used it.

        `bound` is computed from the rank-to-device binding rule, not inferred
        from the power level. A device can idle at a third of its peak, so a
        power threshold would misclassify a lightly loaded device as unused --
        and "was anything running on it" is a fact we already know exactly.
        """
        span = self.span_s()
        rows = []
        for i, source in enumerate(self.sources):
            joules, saw_reset = self._joules(i)
            watts = [w for _, w in self.watts(i)]
            rows.append(
                {
                    "key": source.key,
                    "scope": source.scope,
                    "device_index": source.device_index,
                    "bound": None
                    if source.aggregate
                    else (source.device_index in self.bound_devices),
                    "aggregate": source.aggregate,
                    "joules": joules or None,
                    "mean_watts": (joules / span) if span and joules else None,
                    # The floor of the distribution is the idle draw: a device
                    # doing nothing still burns this, and it is the number that
                    # makes unused-device waste concrete.
                    "floor_watts": _quantile(watts, 0.05),
                    "median_watts": statistics.median(watts) if watts else None,
                    "max_watts": max(watts) if watts else None,
                    "samples": len(watts),
                    "counter_reset": saw_reset or None,
                }
            )
        return rows

    def summary(self) -> dict:
        """Node rollup. Totals cover per-device counters only -- summing those
        with an aggregate card counter would count the same silicon twice."""
        rows = self.per_device()
        per_device = [r for r in rows if not r["aggregate"] and r["joules"]]
        used = sum(r["joules"] for r in per_device if r["bound"])
        idle = sum(r["joules"] for r in per_device if r["bound"] is False)
        total = used + idle
        # "Nothing was readable" and "measured, and it came to zero" are
        # different answers and must not share a null. `idle or None` gave both
        # the same one, so a fully subscribed node printed a dash under Idle J
        # beside an Idle % of 0.0 -- one fact rendered two ways, reading as if
        # the columns disagreed. Zero idle joules on a node where every device
        # had a rank is a measurement, and the honest report of it is 0.
        measured = bool(per_device)
        return {
            "devices": rows,
            "device_count": len(per_device),
            "devices_bound": sum(1 for r in per_device if r["bound"]),
            "devices_idle": sum(1 for r in per_device if r["bound"] is False),
            "joules_total": total if measured else None,
            "joules_bound": used if measured else None,
            "joules_idle": idle if measured else None,
            # The headline: share of the node's accelerator energy spent on
            # devices this job never used.
            "idle_fraction": (idle / total) if total else None,
            "sample_interval_s": self.interval_s,
            "sample_count": len(self.samples),
            "span_s": self.span_s(),
        }

    def write(self, out_dir, machine: str, run_id: str, hostname: str) -> Path:
        """Full series to its own file. Thousands of samples per source do not
        belong in the result JSON that every analysis script parses."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{machine}_{run_id}_{hostname}_power.json"
        path.write_text(
            json.dumps(
                {
                    "kind": "power_timeline",
                    "machine": machine,
                    "run_id": run_id,
                    "hostname": hostname,
                    "sample_interval_s": self.interval_s,
                    "sources": [
                        {
                            "key": s.key,
                            "scope": s.scope,
                            "device_index": s.device_index,
                            "aggregate": s.aggregate,
                        }
                        for s in self.sources
                    ],
                    "marks": [
                        {"t_s": round(t, 4), "label": label} for t, label in self.marks
                    ],
                    # Columnar: elapsed seconds, then cumulative joules per
                    # source in `sources` order. Long and boring to read by eye,
                    # but it loads straight into a dataframe.
                    #
                    # Rounded to microseconds and millijoules, both far finer
                    # than either clock or counter resolves. Full float repr
                    # doubled the file for digits that were never measurements.
                    "t_s": [round(t, 6) for t, _ in self.samples],
                    "joules": [
                        [None if row[i] is None else round(row[i], 3)
                         for _, row in self.samples]
                        for i in range(len(self.sources))
                    ],
                },
                separators=(",", ":"),
            )
        )
        return path


def _quantile(values: list, q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def bound_device_indices(world_size: int, node_count: int, device_count: int) -> frozenset:
    """Which local device indices this job actually put a rank on.

    Mirrors the binding rule the Platform constructors use (local_rank modulo
    device count). Assumes ranks are spread evenly over nodes, which is what
    mpiexec -ppn guarantees and what the submit scripts always pass.
    """
    if device_count <= 0:
        return frozenset()
    ranks_per_node = max(1, world_size // max(1, node_count))
    return frozenset((lr % device_count) for lr in range(ranks_per_node))
