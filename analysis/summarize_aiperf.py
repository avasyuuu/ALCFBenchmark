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
import sys
from datetime import datetime
from pathlib import Path

# analysis/ is a directory of scripts, not a package; the energy integration
# lives with the sampler that wrote the series so both use one delta rule.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.power import joules_from_series  # noqa: E402

EXPORT = "profile_export_aiperf.json"

# Written into the artifact directory by analysis/power_sidecar.py. AIPerf's own
# power collectors are DCGM, pynvml and amdsmi, so on Intel silicon every
# nvidia_* field in the export is absent -- the sidecar samples the i915 hwmon
# counters from beside the run instead, and this is where its joules come in.
SIDECAR = "sidecar_power.json"


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

    # Energy comes from AIPerf where AIPerf can measure it, and from the sidecar
    # where it cannot. Never from both: the two cover different silicon -- AIPerf
    # reports the GPUs NVML enumerates, the sidecar reports every accelerator
    # counter on the node -- so adding or averaging them would invent a scope
    # that no counter has. `energy_src` says which one a row used, because
    # tokens/joule from the two is not the same ratio.
    energy_src = "aiperf nvml" if avg_w is not None else None
    idle_devices = idle_share = wall_j = window_src = xcheck = None
    windowed = False
    total_j = val(d.get("nvidia_total_gpu_energy"))
    side = _sidecar(directory, profiling_s=dur)

    if avg_w is not None and side is not None and side.get("watts"):
        # Both instruments ran. AIPerf's figure stays primary -- it is the
        # vendor tool and the one the NVIDIA rows have always used -- but the
        # sidecar measured the same GPUs by a different path, so the gap between
        # them is worth a column. Never averaged: they cannot both be right, and
        # a mean would hide which.
        xcheck = (side["watts"] - avg_w) / avg_w * 100
        idle_devices, idle_share = side.get("devices_idle"), side.get("idle_fraction")

    if avg_w is None and side is not None:
        total_j = side.get("joules")
        # Narrowed to the profiling window by _sidecar() when the series is on
        # disk, so these divide the same stretch of time AIPerf's tokens came
        # from. Re-integrated from the samples in that window rather than
        # rescaled from the whole-run average, which would assume a flat draw
        # across a phase the series can see is not flat.
        avg_w = side.get("watts")
        dur = side.get("wall_s") or dur
        energy_src = f"sidecar {side.get('machine') or '?'}"
        idle_devices, idle_share = side.get("devices_idle"), side.get("idle_fraction")
        wall_j, windowed = side.get("wall_joules"), side.get("windowed")
        window_src = side.get("window_source")

    dyn_w = avg_w - idle_w if (avg_w is not None and idle_w is not None) else None
    dyn_j = dyn_w * dur if (dyn_w is not None and dur is not None) else None

    # AIPerf computes these itself from its own collectors; with sidecar joules
    # they have to be divided here, from the token counts AIPerf did report.
    tok_per_j = val(d.get("nvidia_output_tokens_per_joule"))
    if tok_per_j is None and total_j and out_tok:
        tok_per_j = out_tok / total_j
    mj_out = val(d.get("nvidia_energy_per_output_token"))
    if mj_out is None and total_j and out_tok:
        mj_out = total_j / out_tok * 1000
    mj_all = val(d.get("nvidia_energy_per_total_token"))
    if mj_all is None and total_j and in_tok and out_tok:
        mj_all = total_j / (in_tok + out_tok) * 1000
    j_per_req = val(d.get("nvidia_energy_per_request"))
    reqs = val(d.get("request_count"))
    if j_per_req is None and total_j and reqs:
        j_per_req = total_j / reqs

    return {
        "name": directory.name,
        "energy_src": energy_src,
        "idle_devices": idle_devices,
        "idle_share": idle_share,
        "wall_energy_j": wall_j,
        "sidecar_delta_pct": xcheck,
        "windowed": windowed,
        "window_source": window_src,
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
        "total_energy_j": total_j,
        "dynamic_energy_j": dyn_j,
        "tok_per_joule": tok_per_j,
        "tok_per_joule_dynamic": (out_tok / dyn_j) if (dyn_j and out_tok) else None,
        "mj_per_output_token": mj_out,
        "mj_per_total_token": mj_all,
        "energy_per_req_j": j_per_req,
    }


def _sidecar(directory: Path, profiling_s: float | None = None) -> dict | None:
    """Joules, mean watts and wall time from a sidecar run in this directory.

    Returns None rather than raising when the file is absent, which is the
    normal case on NVIDIA -- there AIPerf measures its own power and the sidecar
    is not run at all.

    When profiling_s is given and the full series is on disk, the energy is
    re-integrated over just that window. The sidecar brackets the whole aiperf
    process; AIPerf's tokens describe only its profiling phase. Dividing one by
    the other is the difference between 0.07 and 0.15 tokens/joule on the first
    Aurora run, so the window is narrowed rather than the mismatch noted.
    """
    path = directory / SIDECAR
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    power = blob.get("power") or {}
    joules, span = power.get("joules_total"), power.get("span_s")
    row = {
        "joules": joules,
        # From the sampler's own span, not from wall_s: the counters bound what
        # was actually measured, and if sampling stopped early the honest watts
        # come from the window that has samples in it.
        "watts": (joules / span) if joules and span else None,
        "wall_s": blob.get("wall_s"),
        "machine": blob.get("machine"),
        "devices": power.get("device_count"),
        "devices_idle": power.get("devices_idle"),
        "idle_fraction": power.get("idle_fraction"),
        # Kept alongside the windowed figures. The whole-process cost is a real
        # number and the one an allocation is billed for -- it is the wrong
        # denominator for tokens/joule, not a wrong measurement.
        "wall_joules": joules,
        "windowed": False,
        "window_source": None,
    }

    if not profiling_s:
        return row

    # One entry per node since the sidecar learned to span them; sweeps recorded
    # before that have no list and describe the single node they ran on. Each
    # node is re-integrated against its OWN start time -- the samplers are
    # separate processes with separate clocks, and one shared offset would slide
    # every other node's window off the phase it is supposed to bracket.
    nodes = blob.get("nodes") or [{
        "started_utc": blob.get("started_utc"),
        "bound_devices": blob.get("bound_devices"),
        "series_file": blob.get("series_file"),
    }]

    joules = idle_j = 0.0
    spans, sources, windowed_any = [], None, False
    for node in nodes:
        name = node.get("series_file")
        if not name or not (directory / name).exists():
            return row       # a node we cannot read is not a node we can subtract
        try:
            series = json.loads((directory / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return row
        window = _profiling_window(directory, node.get("started_utc"))
        part = joules_from_series(
            series,
            window=window,
            tail_s=None if window else profiling_s,
            bound_devices=frozenset(node.get("bound_devices") or []),
        )
        if not part or not part.get("joules_total"):
            return row
        joules += part["joules_total"]
        idle_j += part.get("joules_idle") or 0.0
        spans.append(part["span_s"])
        windowed_any = windowed_any or part["windowed"]
        sources = "manifest" if window else "tail"

    # Concurrent samplers, so the job's span is the longest node's, never the
    # sum -- watts is joules over that, and adding the spans would halve it.
    span = max(spans)
    row["window_source"] = sources
    row.update(
        joules=joules,
        watts=joules / span if span else None,
        wall_s=span,
        idle_fraction=(idle_j / joules) if joules else None,
        windowed=windowed_any,
    )
    return row


def _profiling_window(directory: Path, started_utc: str | None) -> tuple | None:
    """The profiling phase's bounds, in the power series' own seconds.

    AIPerf 0.12 writes phase_manifest.json with start_ns/end_ns per phase in
    Unix epoch nanoseconds. The sidecar records started_utc immediately before
    constructing the sampler, and the sampler's t0 is set microseconds later in
    its constructor, so `started_utc + t_s` is the wall-clock time of a sample to
    well under a millisecond -- fine against a window measured in minutes.

    That turns the tail-slice approximation into an exact answer. On the first
    Aurora sweep the two differed by 3-5 s of teardown counted in place of the
    same amount of early profiling, worth 0.1-0.6% of the energy. Small, but
    there is no reason to carry an assumption when the file states the fact.

    Takes the start time rather than the whole sidecar record because each node
    of a multi-node run has its own: the phase bounds are wall-clock and shared,
    but the offset from them into a series is per-sampler.

    Returns None whenever anything is missing or unparseable -- older artifacts
    have no manifest, and the caller falls back to the tail slice.
    """
    started = started_utc
    if not started:
        return None
    path = directory / "phase_manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        t0 = datetime.fromisoformat(started).timestamp()
        phase = next(
            p for p in manifest.get("phases") or []
            if p.get("phase_kind") == "profiling"
        )
        start, end = phase["start_ns"] / 1e9 - t0, phase["end_ns"] / 1e9 - t0
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return None
    return (start, end) if end > start else None


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
        if idle_w is None and blob.get("kind") == "power_sidecar":
            # An idle floor taken by the sidecar itself, which is how Aurora
            # gets one -- nvml_idle.py needs NVML and there is none. Same
            # node-wide scope as the sidecar's measured rows, so the two
            # subtract cleanly.
            power = blob.get("power") or {}
            joules, span = power.get("joules_total"), power.get("span_s")
            idle_w = (joules / span) if joules and span else None

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
         "avg_W", "dyn_W", "energy_J", "idleDev", "idle%", "xcheck%", "src"],
        [
            [r["concurrency"], fmt(r["duration_s"]), fmt(r["req_per_s"]),
             fmt(r["out_tok_per_s"], 1), fmt(r["all_tok_per_s"], 1), fmt(r["ttft_ms"], 1),
             fmt(r["itl_ms"], 2), fmt(r["avg_gpu_w"], 1), fmt(r["dynamic_w"], 1),
             fmt(r["total_energy_j"], 1), fmt(r["idle_devices"], 0),
             fmt(r["idle_share"] * 100, 1) if r["idle_share"] is not None else "n/a",
             (f'{r["sidecar_delta_pct"]:+.1f}'
              if r["sidecar_delta_pct"] is not None else "n/a"),
             r["energy_src"] or "none"]
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

    # Mixing scopes is worse than missing one. AIPerf's figure covers the GPUs
    # NVML enumerates; the sidecar's covers every accelerator counter on the
    # node. A table with both would put two different denominators under one
    # tokens/joule column and read as a machine comparison.
    # A sidecar row whose series could not be narrowed is divided by the whole
    # process's joules -- startup and warmup included -- while its tokens cover
    # only the profiling phase. On the first Aurora run that was the difference
    # between 0.07 and 0.15 tokens/joule, so it is stated rather than left to be
    # noticed.
    unwindowed = [
        r for r in rows
        if (r["energy_src"] or "").startswith("sidecar") and not r.get("windowed")
    ]
    if unwindowed:
        notes.append(
            f"{len(unwindowed)} sidecar row(s) not narrowed to the profiling window "
            "(no series file, or sampling was shorter than benchmark_duration). Their "
            "energy covers AIPerf startup and warmup too, so tok/J is understated. "
            "tok/J_dyn is unaffected -- the excess is idle and subtracts out."
        )

    # Narrowed, but by the approximation rather than by the stated bounds. Worth
    # a line because the two differ most exactly where the rows are shortest --
    # 6.8% of dynamic watts at concurrency 32 on the first Aurora sweep, where a
    # 52 s window had 3 s of teardown in it.
    approx = [r for r in rows if r.get("window_source") == "tail"]
    if approx:
        notes.append(
            f"{len(approx)} row(s) used the tail-slice window (no phase_manifest.json). "
            "That counts teardown in place of early profiling; the error grows as the "
            "row gets shorter. AIPerf 0.12+ writes the manifest and is exact."
        )

    # The cross-check is only worth running if a bad answer is noticed. Two
    # instruments on the same GPUs should agree closely; 5% is generous for
    # counters reading the same silicon, and anything past it means one of them
    # is wrong -- which matters well beyond this row, since Aurora's numbers
    # come from the sidecar alone and have nothing to be checked against.
    off = [r for r in rows
           if r.get("sidecar_delta_pct") is not None
           and abs(r["sidecar_delta_pct"]) > 5]
    if off:
        worst = max(off, key=lambda r: abs(r["sidecar_delta_pct"]))
        notes.append(
            f"{len(off)} row(s) where AIPerf and the sidecar disagree by more than "
            f"5% (worst {worst['sidecar_delta_pct']:+.1f}% at concurrency "
            f"{worst['concurrency']}). They measure the same GPUs, so one is wrong. "
            "Check the sidecar's window and that --bound-devices matches what the "
            "server was given."
        )

    sources = {r["energy_src"] for r in rows if r["energy_src"]}
    if len(sources) > 1:
        notes.append(
            f"rows mix energy sources ({', '.join(sorted(sources))}). These cover "
            "different silicon -- compare tokens/joule only within one source."
        )

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
