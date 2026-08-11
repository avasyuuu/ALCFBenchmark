"""Inline SVG charts for the site. Stdlib only, like everything else here.

No plotting library and no client-side script: the page promises to be one
self-contained file that renders offline and inside restricted networks, which
rules out matplotlib (a build dependency the login nodes do not have) and
Chart.js (a CDN fetch). Hand-built SVG costs more code and keeps both promises.

Colour comes from CSS custom properties rather than baked hex, so one palette
definition serves the light and dark themes and the viewer's toggle keeps
working. Series slots are validated against the page's own surfaces -- see
SERIES_SLOT.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

# Fixed machine -> palette slot. Assigned by entity, never by rank or by
# position in the current result set: a machine keeps its colour when another
# is added or filtered out, so a reader who learned "aurora is blue" stays
# right. Slots 1-4 of the reference categorical palette, validated on this
# page's surfaces (#fbfbfa / #16161a) for the adjacent pairlist that line and
# bar charts use -- worst CVD dE 9.1 light, 8.4 dark against a floor of 8.
SERIES_SLOT = {"aurora": 1, "polaris": 2, "crux": 3, "sophia": 4}

TARGET_LABEL = "target 0.90"


def canonical_runs(results_dir: str) -> dict:
    """The one full-length run per machine, keyed by machine.

    A chart of every run would draw ten near-identical aurora curves and say
    nothing. Restricted to runs that spent their whole epoch budget, because a
    truncated curve stops mid-climb and would read as a machine that never got
    there. Most recent wins where several qualify.
    """
    best: dict = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        work = blob.get("work") or {}
        machine = blob.get("machine")
        curve = blob.get("curve") or []
        if blob.get("status") != "complete" or not machine or len(curve) < 2:
            continue
        if (blob.get("config") or {}).get("synthetic_data"):
            continue
        if work.get("epochs_completed") != work.get("epochs_requested"):
            continue
        stamp = blob.get("timestamp_utc") or ""
        if machine not in best or stamp > (best[machine].get("timestamp_utc") or ""):
            best[machine] = blob
    return dict(sorted(best.items()))


def _slot(machine: str) -> int:
    return SERIES_SLOT.get(machine, 8)


def _fmt_s(seconds: float) -> str:
    return f"{seconds:,.0f} s" if seconds >= 100 else f"{seconds:.1f} s"


def _esc(text) -> str:
    return html.escape(str(text))


def accuracy_chart(runs: dict) -> str:
    """Validation accuracy against wall-clock, one line per machine.

    The log x-axis is the point rather than a flourish: the three machines
    finish 48 s, 86 s and 932 s apart, and on a linear axis the two accelerators
    collapse into a vertical wall in the leftmost tenth. Log spreads them while
    keeping the distance between them honest -- and the axis says so, because a
    log scale read as linear understates a 16x gap.
    """
    if len(runs) < 2:
        return ""

    W, H = 720, 330
    L, R, T, B = 46, 20, 14, 46          # margins; B holds the x-axis band
    pw, ph = W - L - R, H - T - B

    xs = [p["elapsed_s"] for b in runs.values() for p in b["curve"] if p["elapsed_s"] > 0]
    x_lo, x_hi = 1.0, 10 ** math.ceil(math.log10(max(xs)))

    def px(seconds: float) -> float:
        f = (math.log10(max(seconds, x_lo)) - math.log10(x_lo)) / (
            math.log10(x_hi) - math.log10(x_lo)
        )
        return L + f * pw

    def py(acc: float) -> float:
        return T + (1.0 - acc) * ph

    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Validation accuracy against wall-clock time, one line per machine">'
    ]

    # --- grid: hairline, solid, one step off the surface ------------------
    for acc in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(acc)
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{acc:.2f}</text>'
        )

    decade = int(math.log10(x_lo))
    while 10 ** decade <= x_hi:
        for mult in (1, 3):
            t = mult * 10 ** decade
            if not x_lo <= t <= x_hi:
                continue
            x = px(t)
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
            if mult == 1:  # label decades only; the 3s are unlabelled hairlines
                parts.append(
                    f'<text class="tick" x="{x:.1f}" y="{T+ph+18}" '
                    f'text-anchor="middle">{t:,g}s</text>'
                )
        decade += 1

    # --- the accuracy target ---------------------------------------------
    ty = py(0.90)
    parts.append(f'<line class="target" x1="{L}" y1="{ty:.1f}" x2="{L+pw}" y2="{ty:.1f}"/>')
    parts.append(
        f'<text class="tick" x="{L+4}" y="{ty-6:.1f}" text-anchor="start">{TARGET_LABEL}</text>'
    )

    # --- one line per machine --------------------------------------------
    for machine, blob in runs.items():
        pts = " ".join(
            f"{px(p['elapsed_s']):.1f},{py(p['val_top1']):.1f}"
            for p in blob["curve"]
            if p["elapsed_s"] > 0
        )
        tta = (blob.get("accuracy") or {}).get("time_to_target_s")
        parts.append(
            f'<polyline class="ln s{_slot(machine)}" points="{pts}">'
            f"<title>{_esc(machine)}</title></polyline>"
        )
        # One marker per series, at the moment it crosses the target -- the
        # value the whole page is about. Ringed in the surface colour so it
        # stays legible where a line passes under it.
        if tta:
            parts.append(
                f'<circle class="dot s{_slot(machine)}" cx="{px(tta):.1f}" '
                f'cy="{ty:.1f}" r="4.5">'
                f"<title>{_esc(machine)}: {_fmt_s(tta)} to 0.90</title></circle>"
            )

    parts.append(
        f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">'
        "wall-clock seconds (log scale)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def tail_chart(runs: dict) -> str:
    """Slowest step as a multiple of the median -- one bar per machine.

    A single series, so no legend: the caption names what is plotted. The
    jitter and tail columns in the runs table are easy to scroll past, and this
    is the one place the CPU machine wins outright, which is worth a form that
    cannot be missed.
    """
    if len(runs) < 2:
        return ""

    rows = []
    for machine, blob in runs.items():
        step = ((blob.get("throughput") or {}).get("step_time") or {})
        med, slowest = step.get("median_s"), step.get("max_s")
        if med and slowest:
            rows.append((machine, slowest / med))
    if len(rows) < 2:
        return ""
    rows.sort(key=lambda r: -r[1])

    BAR, GAP, L, R, T = 22, 18, 66, 54, 8
    W = 720
    H = T + len(rows) * (BAR + GAP)
    pw = W - L - R
    hi = max(v for _, v in rows) * 1.08

    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Slowest training step as a multiple of the median, per machine">'
    ]
    for i, (machine, ratio) in enumerate(rows):
        y = T + i * (BAR + GAP)
        w = max(ratio / hi * pw, 1.0)
        r = min(4.0, w)  # 4px rounded data-end, square at the baseline
        parts.append(
            f'<path class="bar s{_slot(machine)}" d="M{L},{y} H{L+w-r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} V{y+BAR-r:.1f} a{r},{r} 0 0 1 -{r},{r} H{L} Z">'
            f"<title>{_esc(machine)}: slowest step {ratio:.1f}x the median</title></path>"
        )
        parts.append(
            f'<text class="tick" x="{L-8}" y="{y+BAR/2+4:.1f}" text-anchor="end">'
            f"{_esc(machine)}</text>"
        )
        parts.append(
            f'<text class="val" x="{L+w+8:.1f}" y="{y+BAR/2+4:.1f}">{ratio:.1f}×</text>'
        )
    parts.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{H}"/>')
    parts.append("</svg>")
    return "".join(parts)


def series_legend(runs: dict) -> str:
    """Identity plus the headline number, outside the plot.

    All three lines converge on the same accuracy, so end-labels would collide
    at the right edge -- the legend carries identity and the time-to-target
    instead of nudging labels apart, which detaches them from their lines.
    """
    items = ""
    for machine, blob in runs.items():
        tta = (blob.get("accuracy") or {}).get("time_to_target_s")
        value = f" — {_fmt_s(tta)} to 0.90" if tta else ""
        items += (
            f'<span class="key"><span class="sw s{_slot(machine)}"></span>'
            f"{_esc(machine)}{value}</span>"
        )
    return f'<div class="legend-row">{items}</div>'


def efficiency_chart(rows: list) -> str:
    """Node-wide samples per joule, one bar per run.

    The only energy ratio on this page that compares across machines: whole-job
    samples over whole-node joules, both sides covering a node. The per-rank
    figure cannot be charted this way -- an Aurora tile is half a card and a
    Polaris reading covers a whole board, so bars drawn side by side would
    invite a comparison the measurement does not support.

    Every run that reached the accuracy target is shown rather than one per
    machine, because the repeats are the argument: two Aurora runs at 64.7 and
    66.9 and two Polaris at 91.4 and 93.9 establish the spread, which is what
    makes the single-rank 6.1 read as a result rather than as noise.
    """
    if len(rows) < 2:
        return ""
    rows = sorted(rows, key=lambda r: -r["eff"])

    BAR, GAP, L, R, T, BOT = 18, 12, 118, 52, 8, 28
    W = 720
    H = T + len(rows) * (BAR + GAP) + BOT
    pw = W - L - R
    hi = max(r["eff"] for r in rows) * 1.1

    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label='
        f'"Node-wide samples per joule, one bar per run">'
    ]
    for i, row in enumerate(rows):
        y = T + i * (BAR + GAP)
        w = max(row["eff"] / hi * pw, 1.0)
        r = min(4.0, w)
        label = f'{row["machine"]} · {row["ranks"]} rank' + ("" if row["ranks"] == 1 else "s")
        parts.append(
            f'<path class="bar s{_slot(row["machine"])}" d="M{L},{y} H{L+w-r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} V{y+BAR-r:.1f} a{r},{r} 0 0 1 -{r},{r} H{L} Z">'
            f'<title>{_esc(label)}: {row["eff"]:.1f} samples per joule, '
            f'{row["samples"]:,} samples over {row["joules"]:,.0f} node J</title></path>'
        )
        parts.append(
            f'<text class="tick" x="{L-8}" y="{y+BAR/2+4:.1f}" text-anchor="end">'
            f"{_esc(label)}</text>"
        )
        parts.append(
            f'<text class="val" x="{L+w+8:.1f}" y="{y+BAR/2+4:.1f}">{row["eff"]:.1f}</text>'
        )
    parts.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{H-BOT}"/>')
    parts.append(
        f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-8}" text-anchor="middle">'
        "samples per joule, whole node</text>"
    )
    parts.append("</svg>")
    return "".join(parts)
