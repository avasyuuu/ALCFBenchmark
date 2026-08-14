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


def machine_tag(machine) -> str:
    """A machine name in its own series colour, matching every chart here.

    Colour is the machine everywhere on this site -- lines, bars, swatches -- so
    a name rendered plain is the one place it is not, and a reader who learned
    "aurora is blue" from a chart has to translate it back to a word.

    Uses --ct rather than --c: the chart colours separate marks from each other
    and fail a text contrast floor on the light surface, so the stylesheet keeps
    a darkened twin for text. An unknown machine falls through to currentColor
    and stays plain rather than invisible.
    """
    slot = SERIES_SLOT.get(machine)
    cls = f"m s{slot}" if slot else "m"
    return f'<span class="{cls}">{_esc(machine or "—")}</span>'



def canonical_runs(results_dir: str) -> dict:
    """The one full-length run per machine, keyed by machine.

    A chart of every run would draw ten near-identical aurora curves and say
    nothing. Restricted to runs that spent their whole epoch budget, because a
    truncated curve stops mid-climb and would read as a machine that never got
    there.

    Fewest nodes wins, then most recent. Recency alone was the rule until Crux
    was run at two node counts on the same day, and it picked between them by
    which job happened to finish last -- so a 2-node curve could have landed
    beside 1-node curves from every other machine with nothing on the chart
    saying so. Node count is the one thing these curves must agree on, since the
    horizontal gap between them is the entire comparison. Fewest rather than a
    hardcoded 1 so a machine only ever run multi-node still appears, and
    node_count() below puts the basis in the legend either way.
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
        # Missing node count sorts last rather than first: an unlabelled run
        # should not outrank a run that says it used one node. Written out
        # rather than packed into a sort key because the two fields break ties
        # in opposite directions -- fewest nodes, then latest timestamp -- and
        # a tuple cannot express that over a string.
        nodes = node_count(blob) or math.inf
        stamp = blob.get("timestamp_utc") or ""
        if machine not in best:
            best[machine] = (nodes, stamp, blob)
            continue
        prior_nodes, prior_stamp, _ = best[machine]
        if nodes < prior_nodes or (nodes == prior_nodes and stamp > prior_stamp):
            best[machine] = (nodes, stamp, blob)
    return {m: blob for m, (_, _, blob) in sorted(best.items())}


def node_count(blob: dict):
    return (blob.get("config") or {}).get("nodes")


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
        # Node count is stated, always, even at the 1 that every machine
        # currently sits on. canonical_runs() picks the fewest-node run and a
        # machine could arrive whose fewest is not 1; a label that appears only
        # in that case is a label nobody has learned to look for, and its
        # absence would read as agreement rather than as nothing to report.
        nodes = node_count(blob)
        basis = f" ({nodes:,} node{'s' if nodes != 1 else ''})" if nodes else ""
        items += (
            f'<span class="key"><span class="sw s{_slot(machine)}"></span>'
            f"{_esc(machine)}{basis}{value}</span>"
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


def inference_chart(rows: list) -> str:
    """Throughput against dynamic power as concurrency rises, both normalised.

    Two series on one axis because the finding is the gap between them, and the
    gap only reads as a gap when they share a scale. Absolute units cannot do
    that -- 631 tokens/s and 114 watts on one axis makes the watts a flat line
    at the bottom whatever they do, which would look like the same picture if
    power had tripled.

    Normalised to concurrency 1, so both start at 1.0 and the y-axis is "times
    the single-request value". Throughput ends at 12.6x and dynamic power at
    0.94x, and that divergence is the whole result: the accelerator draws what
    it draws, and serving more at once is nearly free in watts.
    """
    rows = [r for r in rows if r.get("concurrency") and r.get("out_tok_per_s")]
    if len(rows) < 3:
        return ""
    rows = sorted(rows, key=lambda r: r["concurrency"])
    base_tok = rows[0]["out_tok_per_s"]
    base_w = rows[0].get("dynamic_w")
    if not base_tok or not base_w:
        return ""

    W, H = 720, 330
    L, R, T, B = 46, 20, 14, 46
    pw, ph = W - L - R, H - T - B

    concs = [r["concurrency"] for r in rows]
    ratios = [r["out_tok_per_s"] / base_tok for r in rows]
    ratios += [(r.get("dynamic_w") or 0) / base_w for r in rows]
    y_hi = math.ceil(max(ratios) / 2) * 2

    def px(c: float) -> float:
        # Log x: the sweep doubles each step, so linear would crowd every low
        # level into the first eighth and spend half the width between 16 and 32.
        f = (math.log2(c) - math.log2(concs[0])) / (math.log2(concs[-1]) - math.log2(concs[0]))
        return L + f * pw

    def py(v: float) -> float:
        return T + (1 - v / y_hi) * ph

    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Output '
        f'throughput and dynamic power against concurrency, both relative to '
        f'concurrency 1">'
    ]
    for v in range(0, y_hi + 1, 2):
        y = py(v)
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{v}x</text>'
        )
    for c in concs:
        x = px(c)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{c}</text>'
        )
    # The 1.0 line: what concurrency 1 did. Power sits on it the whole way and
    # throughput leaves it, which is the sentence this chart is making.
    parts.append(f'<line class="target" x1="{L}" y1="{py(1):.1f}" x2="{L+pw}" y2="{py(1):.1f}"/>')

    for slot, key, label, unit in (
        (1, "out_tok_per_s", "output tokens/s", "tok/s"),
        (2, "dynamic_w", "dynamic power", "W"),
    ):
        base = base_tok if key == "out_tok_per_s" else base_w
        pts, dots = [], []
        for r in rows:
            v = r.get(key)
            if not v:
                continue
            x, y = px(r["concurrency"]), py(v / base)
            pts.append(f"{x:.1f},{y:.1f}")
            dots.append(
                f'<circle class="dot s{slot}" cx="{x:.1f}" cy="{y:.1f}" r="4">'
                f'<title>concurrency {r["concurrency"]}: {v:,.1f} {unit} '
                f'({v / base:.2f}x)</title></circle>'
            )
        parts.append(f'<polyline class="ln s{slot}" points="{" ".join(pts)}">'
                     f"<title>{label}</title></polyline>")
        parts.extend(dots)

    parts.append(
        f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">'
        "concurrent requests (log scale)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def inference_legend() -> str:
    return (
        '<div class="legend-row">'
        '<span class="key"><span class="sw s1"></span>output tokens/s</span>'
        '<span class="key"><span class="sw s2"></span>dynamic power</span>'
        "</div>"
    )


def efficiency_compare_chart(sweeps: list) -> str:
    """Tokens per joule against concurrency, every machine, both denominators.

    Log-log on purpose. Both metrics are close to linear in concurrency, so a
    log-log plot draws them as near-straight lines and the vertical distance
    between two lines is their ratio -- which means "Polaris is 4x better
    throughout" reads as a constant gap rather than as a fan that the eye has to
    integrate. The values span 0.04 to 5.5, two decades, which a linear axis
    would crush into the bottom eighth.

    Colour is the machine, always, per SERIES_SLOT. The second series for the
    same machine is dashed instead: total and dynamic tokens/joule are two
    measures of one machine, and giving the second one its own colour would
    say they were two machines.

    One model per chart, therefore. Colour has no room left to encode a model
    too, and two sweeps of different models on one machine would draw two
    identical-looking lines. The caller groups by model and calls this once per
    group; comparing machines is the job, and that only means anything with the
    model held fixed anyway.
    """
    usable = [s for s in sweeps if len(s.get("rows") or []) >= 3]
    if not usable:
        return ""

    series = []
    for sweep in usable:
        rows = sorted(sweep["rows"], key=lambda r: r["concurrency"])
        for key, dashed, label in (
            ("tok_per_joule", False, "tokens/J"),
            ("tok_per_joule_dynamic", True, "tokens/J dynamic"),
        ):
            pts = [(r["concurrency"], r[key]) for r in rows if r.get(key)]
            if len(pts) >= 3:
                series.append((sweep["machine"], dashed, label, pts))
    if not series:
        return ""

    W, H = 720, 360
    L, R, T, B = 52, 20, 14, 46
    pw, ph = W - L - R, H - T - B

    xs = [c for _, _, _, pts in series for c, _ in pts]
    ys = [v for _, _, _, pts in series for _, v in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo = 10 ** math.floor(math.log10(min(ys)))
    y_hi = 10 ** math.ceil(math.log10(max(ys)))

    def px(c):
        return L + (math.log10(c) - math.log10(x_lo)) / (
            math.log10(x_hi) - math.log10(x_lo)) * pw

    def py(v):
        return T + (1 - (math.log10(v) - math.log10(y_lo)) / (
            math.log10(y_hi) - math.log10(y_lo))) * ph

    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Tokens '
        f'per joule against concurrency for each machine, total and dynamic">'
    ]
    decade = math.log10(y_lo)
    while decade <= math.log10(y_hi) + 1e-9:
        v = 10 ** decade
        y = py(v)
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        label = f"{v:g}" if v >= 1 else f"{v:.2f}".rstrip("0")
        parts.append(
            f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{label}</text>')
        decade += 1
    for c in sorted(set(xs)):
        x = px(c)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{c}</text>')

    for machine, dashed, label, pts in series:
        cls = f"ln s{_slot(machine)}" + (" dash" if dashed else "")
        line = " ".join(f"{px(c):.1f},{py(v):.1f}" for c, v in pts)
        parts.append(f'<polyline class="{cls}" points="{line}">'
                     f"<title>{_esc(machine)} {_esc(label)}</title></polyline>")
        for c, v in pts:
            parts.append(
                f'<circle class="dot s{_slot(machine)}" cx="{px(c):.1f}" '
                f'cy="{py(v):.1f}" r="3.5"><title>{_esc(machine)} {_esc(label)} '
                f'at concurrency {c}: {v:,.2f}</title></circle>')

    parts.append(
        f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">'
        "concurrent requests (both axes logarithmic)</text>")
    parts.append("</svg>")
    return "".join(parts)


def efficiency_compare_legend(sweeps: list) -> str:
    """A key for the lines actually drawn, and no others.

    Filtered on the same rule the chart uses. Iterating every sweep listed ten
    entries against four lines, with the same colour and the same words twice --
    which is what a legend is for and exactly what it stopped doing.
    """
    keys = ""
    for sweep in [s for s in sweeps if len(s.get("rows") or []) >= 3]:
        slot = _slot(sweep["machine"])
        keys += (f'<span class="key"><span class="sw s{slot}"></span>'
                 f"{_esc(sweep['machine'])} tokens/J</span>"
                 f'<span class="key"><span class="sw dash s{slot}"></span>'
                 f"{_esc(sweep['machine'])} tokens/J dynamic</span>")
    return f'<div class="legend-row">{keys}</div>'


# Dash patterns for the dashboard, indexed by model. Colour stays the machine
# everywhere on this site, so a second dimension needs a second channel -- and
# the dashboard is the one page that shows several models at once.
MODEL_DASHES = ("", "6 3", "2 3", "9 3 2 3")


def dashboard_chart(sweeps: list, metric: str, label: str, log_y: bool = True) -> str:
    """One metric against concurrency, every configuration, each tagged.

    Every series is emitted, always, inside a <g> carrying its machine, model
    and tensor parallelism. The page's script hides groups rather than redrawing
    anything, which is why there is no charting code in JavaScript: this
    function stays the only thing that knows how a chart is built, and a browser
    with scripting off shows the complete picture instead of an empty box.

    A configuration with one concurrency level gets a marker and no line. That
    is honest -- a single point has no slope -- and it keeps the 70B and the
    first gemma runs visible rather than dropping them for being short.
    """
    series = []
    for sweep in sweeps:
        pts = [(r["concurrency"], r.get(metric)) for r in sweep["rows"]]
        pts = [(c, v) for c, v in pts if c and isinstance(v, (int, float)) and v > 0]
        if pts:
            series.append((sweep, sorted(pts)))
    if not series:
        return ""

    models = sorted({s["model"] for s, _ in series})
    W, H = 760, 380
    L, R, T, B = 58, 16, 14, 46
    pw, ph = W - L - R, H - T - B
    xs = [c for _, pts in series for c, _ in pts]
    ys = [v for _, pts in series for _, v in pts]
    x_lo, x_hi = min(xs), max(xs)
    if log_y:
        y_lo = 10 ** math.floor(math.log10(min(ys)))
        y_hi = 10 ** math.ceil(math.log10(max(ys)))
    else:
        y_lo, y_hi = 0, max(ys) * 1.08

    def px(c):
        if x_hi == x_lo:
            return L + pw / 2
        return L + (math.log2(c) - math.log2(x_lo)) / (math.log2(x_hi) - math.log2(x_lo)) * pw

    def py(v):
        if log_y:
            f = (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        else:
            f = (v - y_lo) / (y_hi - y_lo)
        return T + (1 - f) * ph

    parts = [f'<svg class="chart" data-metric="{_esc(metric)}" viewBox="0 0 {W} {H}" '
             f'role="img" aria-label="{_esc(label)} against concurrency">']

    if log_y:
        d = math.log10(y_lo)
        while d <= math.log10(y_hi) + 1e-9:
            v = 10 ** d
            y = py(v)
            lab = f"{v:g}" if v >= 1 else f"{v:.3f}".rstrip("0")
            parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{lab}</text>')
            d += 1
    else:
        for i in range(5):
            v = y_hi * i / 4
            y = py(v)
            parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{v:,.0f}</text>')
    for c in sorted(set(xs)):
        x = px(c)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{c}</text>')

    for sweep, pts in series:
        machine = sweep["machine"]
        model = (sweep["model"] or "?").split("/")[-1]
        dash = MODEL_DASHES[models.index(sweep["model"]) % len(MODEL_DASHES)]
        style = f' style="stroke-dasharray:{dash}"' if dash else ""
        tp = sweep.get("tp") or ""
        parts.append(
            f'<g class="series s{_slot(machine)}" data-machine="{_esc(machine)}" '
            f'data-model="{_esc(model)}" data-tp="{_esc(tp)}">'
        )
        if len(pts) > 1:
            line = " ".join(f"{px(c):.1f},{py(v):.1f}" for c, v in pts)
            parts.append(f'<polyline class="ln" points="{line}"{style}/>')
        for c, v in pts:
            parts.append(
                f'<circle class="dot" cx="{px(c):.1f}" cy="{py(v):.1f}" r="4">'
                f'<title>{_esc(machine)} {_esc(model)} TP={_esc(tp)} — '
                f'concurrency {c}: {v:,.3g} {_esc(label)}</title></circle>'
            )
        parts.append("</g>")

    parts.append(f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" '
                 f'text-anchor="middle">concurrent requests (log scale)</text>')
    parts.append("</svg>")
    return "".join(parts)


def dashboard_legend(sweeps: list) -> str:
    """One key per configuration, tagged so the script can dim what is hidden."""
    models = sorted({s["model"] for s in sweeps})
    keys = ""
    for sweep in sorted(sweeps, key=lambda s: (s["machine"], s["model"] or "")):
        machine = sweep["machine"]
        model = (sweep["model"] or "?").split("/")[-1]
        dash = MODEL_DASHES[models.index(sweep["model"]) % len(MODEL_DASHES)]
        bg = (f"repeating-linear-gradient(90deg,var(--c) 0 6px,transparent 6px 9px)"
              if dash else "var(--c)")
        keys += (f'<span class="key s{_slot(machine)}" data-machine="{_esc(machine)}" '
                 f'data-model="{_esc(model)}">'
                 f'<span class="sw" style="background:{bg}"></span>'
                 f'{machine_tag(machine)} · {_esc(model)}'
                 + (f' · TP={_esc(sweep["tp"])}' if sweep.get("tp") else "") + "</span>")
    return f'<div class="legend-row">{keys}</div>'
