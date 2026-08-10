"""Turn AIPerf artifact directories into power, token and efficiency tables.

    python analysis/summarize_aiperf.py results/aiperf/polaris-*
    python analysis/summarize_aiperf.py results/aiperf/* --idle results/aiperf/idle.json
    python analysis/summarize_aiperf.py results/aiperf/* --json summary.json

Each positional argument is one AIPerf `--output-artifact-dir`, i.e. a directory
holding `profile_export_aiperf.json`. Rows are ordered by concurrency, which is
read from the export's own input_config rather than the directory name.

Stdlib only, so it runs on a login node without installing anything.

Two checks are printed loudly because both silently invalidated an earlier run of
this benchmark on a laptop:

  * unequal request counts across rows -- absolute joules and durations are then
    not comparable between concurrency levels, only rates and ratios are;
  * achieved OSL below the requested OSL -- a shorter generation amortises the
    fixed prefill over fewer decode tokens, which moves tokens/joule on its own
    and confounds it with whatever the concurrency change did. Serve with
    `ignore_eos` so the two stay separable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPORT = "profile_export_aiperf.json"


def val(node, stat: str = "avg"):
    if node is None:
        return None
    return node.get(stat) if isinstance(node, dict) else node


def fmt(x, places: int = 2) -> str:
    return f"{x:.{places}f}" if isinstance(x, (int, float)) else "n/a"


def table(title: str, headers: list[str], rows: list[list]) -> None:
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
        for i, h in enumerate(headers)
    ]
    line = " ".join(h.rjust(w) for h, w in zip(headers, widths))
    print(f"\n{title}")
    print(line)
    print("-" * len(line))
    for row in cells:
        print(" ".join(c.rjust(w) for c, w in zip(row, widths)))


def load(directory: Path, idle_w: float | None) -> dict | None:
    export = directory / EXPORT
    if not export.exists():
        return None
    # utf-8-sig: a shell redirect on Windows can leave a BOM that plain utf-8
    # chokes on, and these files get copied between machines.
    d = json.loads(export.read_text(encoding="utf-8-sig"))

    cfg = d.get("input_config") or {}
    profiling = _profiling_phase(cfg)
    dur = val(d.get("benchmark_duration"))
    avg_w = val(d.get("nvidia_average_gpu_power"))
    osl = d.get("output_sequence_length") or {}
    out_tok = val(d.get("total_output_tokens"))
    in_tok = val(d.get("total_isl"))

    dyn_w = avg_w - idle_w if (avg_w is not None and idle_w is not None) else None
    dyn_j = dyn_w * dur if (dyn_w is not None and dur is not None) else None

    return {
        "name": directory.name,
        "concurrency": profiling.get("concurrency"),
        "requested_osl": _requested(cfg, "osl"),
        "requested_isl": _requested(cfg, "isl"),
        "duration_s": dur,
        "requests": val(d.get("request_count")),
        "req_per_s": val(d.get("request_throughput")),
        "out_tok_per_s": val(d.get("output_token_throughput")),
        "all_tok_per_s": val(d.get("total_token_throughput")),
        "ttft_ms": val(d.get("time_to_first_token")),
        "itl_ms": val(d.get("inter_token_latency")),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": (in_tok + out_tok) if None not in (in_tok, out_tok) else None,
        "osl_avg": val(osl, "avg"),
        "osl_p50": val(osl, "p50"),
        "osl_min": val(osl, "min"),
        "osl_max": val(osl, "max"),
        "osl_std": val(osl, "std"),
        "avg_gpu_w": avg_w,
        "dynamic_w": dyn_w,
        "total_energy_j": val(d.get("nvidia_total_gpu_energy")),
        "dynamic_energy_j": dyn_j,
        "tok_per_joule": val(d.get("nvidia_output_tokens_per_joule")),
        "tok_per_joule_dynamic": (out_tok / dyn_j) if (dyn_j and out_tok) else None,
        "mj_per_output_token": val(d.get("nvidia_energy_per_output_token")),
        "mj_per_total_token": val(d.get("nvidia_energy_per_total_token")),
        "energy_per_req_j": val(d.get("nvidia_energy_per_request")),
    }


def _dig(node, *keys):
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _profiling_phase(cfg: dict) -> dict:
    """The measured phase's own settings.

    input_config.phases is a list -- warmup first, then profiling -- and only the
    profiling entry describes the run whose numbers get reported. Reading
    concurrency from the config rather than the directory name means a
    hand-renamed or re-run directory still lands in the right row.
    """
    for phase in cfg.get("phases") or []:
        if isinstance(phase, dict) and phase.get("kind") == "profiling":
            return phase
    return {}


def _requested(cfg: dict, which: str):
    """Requested ISL/OSL mean from the first synthetic dataset, if any.

    datasets is a list; a mixed workload built with --sequence-distribution has
    no single requested length, so this stays None there rather than reporting
    the first shape as if it were the whole run.
    """
    datasets = cfg.get("datasets") or []
    if len(datasets) != 1 or not isinstance(datasets[0], dict):
        return None
    return _dig(datasets[0], "prompts", which, "mean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="AIPerf artifact directories")
    ap.add_argument("--idle", help="JSON from analysis/nvml_idle.py")
    ap.add_argument("--idle-watts", type=float, help="override the idle floor directly")
    ap.add_argument("--json", help="write the collected rows here")
    args = ap.parse_args()

    idle_w = args.idle_watts
    if idle_w is None and args.idle:
        blob = json.loads(Path(args.idle).read_text(encoding="utf-8-sig"))
        # node_idle_w counts every device, matching AIPerf's pynvml collector,
        # which also reports every device NVML can see rather than just the busy
        # one. Mixing a per-device floor with node-wide draw understates nothing
        # quietly -- it just makes "dynamic" watts too large.
        idle_w = blob.get("node_idle_w") or blob.get("power_w_avg")

    rows = [r for d in args.dirs if (r := load(Path(d), idle_w)) is not None]
    if not rows:
        raise SystemExit("no profile_export_aiperf.json found in the given directories")
    rows.sort(key=lambda r: (r["concurrency"] is None, r["concurrency"], r["name"]))

    print(
        f"idle floor: {fmt(idle_w, 2) if idle_w is not None else 'not supplied'} W"
        f"   |   {len(rows)} run(s)"
    )
    if idle_w is None:
        print("  (no --idle: 'dyn' columns are omitted; absolute watts include the floor)")

    table(
        "Load and power",
        ["conc", "dur_s", "req/s", "outTok/s", "allTok/s", "TTFT_ms", "ITL_ms",
         "avg_W", "dyn_W", "energy_J"],
        [
            [r["concurrency"], fmt(r["duration_s"]), fmt(r["req_per_s"]),
             fmt(r["out_tok_per_s"], 1), fmt(r["all_tok_per_s"], 1), fmt(r["ttft_ms"], 1),
             fmt(r["itl_ms"], 2), fmt(r["avg_gpu_w"], 1), fmt(r["dynamic_w"], 1),
             fmt(r["total_energy_j"], 1)]
            for r in rows
        ],
    )

    table(
        "Tokens produced",
        ["conc", "reqs", "in_tok", "out_tok", "all_tok", "OSLavg", "OSLp50",
         "OSLmin", "OSLmax", "OSLstd", "tgt"],
        [
            [r["concurrency"], fmt(r["requests"], 0), fmt(r["input_tokens"], 0),
             fmt(r["output_tokens"], 0), fmt(r["total_tokens"], 0), fmt(r["osl_avg"], 1),
             fmt(r["osl_p50"], 0), fmt(r["osl_min"], 0), fmt(r["osl_max"], 0),
             fmt(r["osl_std"], 1), fmt(r["requested_osl"], 0)]
            for r in rows
        ],
    )

    table(
        "Token energy efficiency",
        ["conc", "tok/J", "tok/J_dyn", "mJ/outTok", "mJ/allTok", "J/req"],
        [
            [r["concurrency"], fmt(r["tok_per_joule"]), fmt(r["tok_per_joule_dynamic"]),
             fmt(r["mj_per_output_token"], 1), fmt(r["mj_per_total_token"], 1),
             fmt(r["energy_per_req_j"])]
            for r in rows
        ],
    )

    _warn(rows)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"idle_w": idle_w, "runs": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")


def _warn(rows: list[dict]) -> None:
    notes = []

    counts = {r["requests"] for r in rows if r["requests"] is not None}
    if len(counts) > 1:
        notes.append(
            f"request counts differ across rows ({sorted(counts)}). Absolute energy_J "
            "and dur_s are NOT comparable between rows; use rates and ratios only."
        )

    for r in rows:
        target, got = r["requested_osl"], r["osl_avg"]
        if target and got and abs(got - target) / target > 0.05:
            notes.append(
                f"conc={r['concurrency']}: achieved OSL {got:.1f} vs requested {target:.0f} "
                f"({(got - target) / target * 100:+.1f}%). Serve with ignore_eos, or "
                "tokens/joule is confounded with generation length."
            )

    if notes:
        print("\nWARNINGS")
        for n in notes:
            print(f"  * {n}")


if __name__ == "__main__":
    main()
