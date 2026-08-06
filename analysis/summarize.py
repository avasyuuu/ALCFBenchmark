"""Turn results/*.json into a comparison table.

    python analysis/summarize.py                    # markdown table
    python analysis/summarize.py --csv out.csv      # also write CSV
    python analysis/summarize.py --scaling          # add parallel efficiency

Stdlib only, so it runs on a login node without installing anything.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

COLUMNS = [
    ("when", "When"),
    ("machine", "Machine"),
    ("nodes", "Nodes"),
    ("world_size", "Ranks"),
    ("precision", "Prec"),
    ("global_batch", "Global BS"),
    ("samples_per_s", "Samples/s"),
    ("median_step_ms", "Step (ms)"),
    ("p90_step_ms", "p90 (ms)"),
    ("jitter_pct", "Jitter %"),
    ("tail_ratio", "Max/med"),
    ("best_top1", "Best top-1"),
    ("tta_s", "TTA (s)"),
    ("mfu_pct", "MFU %"),
    ("node_hours", "Node-hrs"),
]

# Printed as its own table, and only when some run actually has energy data --
# on machines with no counter it would be a block of dashes.
ENERGY_COLUMNS = [
    ("when", "When"),
    ("machine", "Machine"),
    ("nodes", "Nodes"),
    ("ranks", "Ranks"),
    ("avg_watts", "Avg W"),
    ("joules", "Joules"),
    ("samples_per_joule", "Samples/J"),
    ("joules_to_target", "J to acc"),
    ("energy_scope", "Scope"),
]

NOTE_WIDTH = 28


def short_when(timestamp: str | None) -> str | None:
    """UTC timestamp -> 'MM-DD HH:MM:SS'. Seconds are worth the width: back to
    back runs of the same config land in the same minute. Full value, year and
    timezone included, stays in the JSON."""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp).strftime("%m-%d %H:%M:%S")
    except ValueError:
        return None


def load_runs(results_dir: str):
    runs = []
    for path in sorted(Path(results_dir).glob("*.json")):
        blob = json.loads(path.read_text())
        if blob.get("kind") == "allreduce_microbenchmark":
            continue
        if blob.get("status") != "complete":
            print(f"  skipping incomplete run: {path.name}")
            continue
        runs.append(flatten(blob))
    # Chronological, not filename order -- run ids are random hex, so sorting
    # by path scrambled repeated runs of the same config.
    runs.sort(key=lambda r: r.get("timestamp_utc") or "")
    return runs


def flatten(blob: dict) -> dict:
    cfg = blob.get("config", {})
    thr = blob.get("throughput", {})
    step = thr.get("step_time", {})
    acc = blob.get("accuracy", {})
    flops = blob.get("flops", {})
    cost = blob.get("cost", {})
    energy = blob.get("energy") or {}

    median = step.get("median_s")
    p90 = step.get("p90_s")
    slowest = step.get("max_s")

    # --note values come first in the list; anything the run appended for
    # itself (synthetic-data or missing-MFU warnings) follows. Showing the
    # first one means a labelled run reads back as its label.
    notes = blob.get("notes") or []
    note = notes[0] if notes else None
    if note and len(note) > NOTE_WIDTH:
        note = note[: NOTE_WIDTH - 1] + "…"

    return {
        "run_id": blob.get("run_id"),
        "timestamp_utc": blob.get("timestamp_utc"),
        "when": short_when(blob.get("timestamp_utc")),
        "note": note,
        "workload": blob.get("workload"),
        "machine": blob.get("machine"),
        "nodes": cfg.get("nodes"),
        "world_size": cfg.get("world_size"),
        "precision": cfg.get("precision"),
        "scaling": cfg.get("scaling"),
        "global_batch": cfg.get("global_batch_size"),
        "synthetic": cfg.get("synthetic_data"),
        "samples_per_s": thr.get("samples_per_s"),
        "median_step_ms": median * 1e3 if median else None,
        "p90_step_ms": p90 * 1e3 if p90 else None,
        # Spread of the normal case, as a share of median. Was stdev/median,
        # which two identical 12-rank runs scored 43% and 93% on -- stdev is
        # dominated by a handful of very slow steps (epoch boundaries respawn
        # the dataloader workers), so it measured how bad the worst step
        # happened to be, not how steady the machine is. p90 vs median is
        # bounded by construction and reproduces.
        "jitter_pct": ((p90 - median) / median * 100) if median and p90 else None,
        # Stragglers get their own column instead of being smeared into the
        # one above: every rank waits on the slowest at each all-reduce, so a
        # rare terrible step is worth seeing, not averaging away.
        "tail_ratio": (slowest / median) if median and slowest else None,
        "best_top1": acc.get("best_top1"),
        "tta_s": acc.get("time_to_target_s"),
        "mfu_pct": (flops.get("mfu") * 100) if flops.get("mfu") else None,
        "node_hours": cost.get("node_hours"),
        # "ranks" duplicates world_size under the label the energy table uses,
        # so both tables can share one flattened row.
        "ranks": cfg.get("world_size"),
        "joules": energy.get("joules"),
        "avg_watts": energy.get("avg_watts"),
        "samples_per_joule": energy.get("samples_per_joule"),
        "joules_to_target": energy.get("joules_to_target_accuracy"),
        "energy_scope": energy.get("scope"),
    }


def add_scaling_efficiency(runs):
    """Parallel efficiency against the smallest run in each (machine, workload,
    scaling, precision) group. Efficiency = throughput_N / (N x throughput_base)
    normalized by rank count."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["machine"], r["workload"], r["scaling"], r["precision"])].append(r)

    for group in groups.values():
        group.sort(key=lambda r: r["world_size"] or 0)
        base = group[0]
        if not base["samples_per_s"] or not base["world_size"]:
            continue
        per_rank_base = base["samples_per_s"] / base["world_size"]
        for r in group:
            if r["samples_per_s"] and r["world_size"]:
                per_rank = r["samples_per_s"] / r["world_size"]
                r["efficiency_pct"] = per_rank / per_rank_base * 100
    return runs


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value >= 1000:
            return f"{value:,.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def print_table(runs, columns):
    headers = [label for _, label in columns]
    rows = [[fmt(r.get(key)) for key, _ in columns] for r in runs]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print()
    print(line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--csv", help="also write a CSV here")
    p.add_argument("--scaling", action="store_true", help="add parallel efficiency column")
    p.add_argument("--include-synthetic", action="store_true")
    args = p.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no complete runs found in {args.results_dir}")

    if not args.include_synthetic:
        runs = [r for r in runs if not r["synthetic"]]

    # Group for comparison first; oldest to newest within a group.
    runs.sort(
        key=lambda r: (
            r["machine"] or "",
            r["workload"] or "",
            r["world_size"] or 0,
            r["timestamp_utc"] or "",
        )
    )

    columns = list(COLUMNS)
    if args.scaling:
        runs = add_scaling_efficiency(runs)
        columns.append(("efficiency_pct", "Efficiency %"))

    print(f"\n{len(runs)} run(s) from {args.results_dir}")
    print_table(runs, columns)

    metered = [r for r in runs if r.get("joules")]
    if metered:
        print(f"energy — {len(metered)} of {len(runs)} run(s) metered")
        print_table(metered, ENERGY_COLUMNS)

    if args.csv:
        keys = list(runs[0].keys())
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(runs)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
