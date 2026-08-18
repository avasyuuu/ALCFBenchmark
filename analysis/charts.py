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

# Marker shape is tensor parallelism -- the third thing a line has to say, after
# the machine it ran on (colour) and the model it served (dash). Assigned by
# position among the TPs present rather than by value, matching MODEL_DASHES, so
# adding a TP=2 sweep does not repaint the ones already on the page.
#
# Shape leads rather than colour: dash is taken by the model, and hue has to
# stay the machine's. Shape survives being four pixels across and survives the
# colour-vision check the palette was picked against.
#
# Fill varies with it, in the stylesheet, for the case shape alone does not
# cover -- two markers on the same point. See _marker.
MARKER_SHAPES = ("circle", "square", "triangle", "diamond")


def _tp_slot(tp, tps: list) -> int:
    """Which marker treatment one sweep's tensor parallelism gets."""
    try:
        return tps.index(tp) % len(MARKER_SHAPES)
    except ValueError:
        return 0


def _marker(cx: float, cy: float, r: float, slot: int, title: str = "") -> str:
    """One data point: a shape from the slot, and a fill from it too.

    Shape alone was not enough where two sweeps cross. A solid marker hides the
    one under it completely, so a TP=4 point landing on a TP=8 point read as a
    single point of unknown identity -- which is the ambiguity the shapes were
    added to remove, returned in a smaller form.

    So alternating slots are hollow, in the stylesheet: adjacent slots always
    differ in fill as well as outline, and a hollow marker shows what it lands
    on rather than erasing it. Fill and not tint -- a tinted marker desaturates
    into a colour that reads as another machine, and hue is the machine
    everywhere on this site.

    Redundant with shape on purpose: either channel alone identifies the series,
    so neither has to survive being small on its own.
    """
    shape = MARKER_SHAPES[slot % len(MARKER_SHAPES)]
    if shape == "square":
        tag, attrs = "rect", (f'x="{cx - r:.1f}" y="{cy - r:.1f}" '
                              f'width="{2 * r:.1f}" height="{2 * r:.1f}"')
    elif shape == "triangle":
        k = r * 1.3
        tag, attrs = "path", (f'd="M{cx:.1f},{cy - k:.1f} L{cx + k:.1f},{cy + k * 0.72:.1f} '
                              f'L{cx - k:.1f},{cy + k * 0.72:.1f} Z"')
    elif shape == "diamond":
        k = r * 1.35
        tag, attrs = "path", (f'd="M{cx:.1f},{cy - k:.1f} L{cx + k:.1f},{cy:.1f} '
                              f'L{cx:.1f},{cy + k:.1f} L{cx - k:.1f},{cy:.1f} Z"')
    else:
        tag, attrs = "circle", f'cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
    inner = f"<title>{title}</title>" if title else ""
    return f'<{tag} class="dot k{slot % len(MARKER_SHAPES)}" {attrs}>{inner}</{tag}>'


def swept_over_concurrency(rows) -> bool:
    """Whether these levels actually differ in concurrency.

    Three rows used to be enough to mean "a curve", because concurrency was the
    only thing a sweep ever varied. A shape sweep holds it fixed and varies ISL
    and OSL instead, so its levels are several points at one x: a vertical stack
    on anything drawn against concurrency, and a divide-by-zero on the charts
    that normalise by the axis span.

    Every caller that plots against concurrency asks this first. A shape sweep
    is not a broken concurrency sweep, it is a different sweep, and it belongs
    on an axis this file does not draw yet.
    """
    return len({r.get("concurrency") for r in rows or []} - {None}) >= 2


def _log_ticks(lo: float, hi: float) -> list:
    """Gridline values for a log axis, at a density the range can carry.

    Decades are the right stops for an axis covering several of them and the
    wrong ones for an axis covering half: a ratio chart running 0.75 to 2.1 has
    exactly one decade inside it, and a single labelled line is not an axis.
    """
    span = math.log10(hi / lo)
    mults = (1,) if span >= 2 else (1, 2, 5) if span >= 0.7 else (1, 1.5, 2, 3, 5, 7)
    out, d = [], math.floor(math.log10(lo))
    while 10 ** d <= hi * (1 + 1e-9):
        out += [m * 10 ** d for m in mults if lo <= m * 10 ** d <= hi]
        d += 1
    return out


def _marker_swatch(slot: int) -> str:
    """The same marker at legend size, so the key and the chart agree."""
    return (f'<svg class="mk" viewBox="0 0 12 12" aria-hidden="true">'
            f'{_marker(6, 6, 3.4, slot)}</svg>')

# Left margin on the charts that name their y axis. Wide enough for a rotated
# title at x=14 plus the tick labels it must not touch.
Y_TITLE_L = 62


def _y_title(text: str, T: float, ph: float, x: float = 14) -> str:
    """The y axis said in words, rotated up the left margin.

    Every chart here named its x axis and left the y to the caption beside it --
    and on the dashboard, where the reader changes the metric, to a dropdown
    outside the frame entirely. A chart that only means something while the
    control above it is in view is a chart that cannot be screenshotted into a
    slide or a report, which is most of what these get used for.
    """
    y = T + ph / 2
    return (f'<text class="axis-title" x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
            f'transform="rotate(-90 {x:.0f} {y:.0f})">{_esc(text)}</text>')


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
    L, R, T, B = Y_TITLE_L, 20, 14, 46   # margins; B holds the x-axis band
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
    parts.append(_y_title("validation accuracy", T, ph))
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
    if len(rows) < 3 or not swept_over_concurrency(rows):
        return ""
    rows = sorted(rows, key=lambda r: r["concurrency"])
    base_tok = rows[0]["out_tok_per_s"]
    base_w = rows[0].get("dynamic_w")
    if not base_tok or not base_w:
        return ""

    W, H = 720, 330
    L, R, T, B = Y_TITLE_L, 20, 14, 46
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
    parts.append(_y_title("× the concurrency-1 value", T, ph))
    parts.append("</svg>")
    return "".join(parts)


def inference_legend() -> str:
    return (
        '<div class="legend-row">'
        '<span class="key"><span class="sw s1"></span>output tokens/s</span>'
        '<span class="key"><span class="sw s2"></span>dynamic power</span>'
        "</div>"
    )


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
        if not swept_over_concurrency(sweep["rows"]):
            continue
        pts = [(r["concurrency"], r.get(metric)) for r in sweep["rows"]]
        pts = [(c, v) for c, v in pts if c and isinstance(v, (int, float)) and v > 0]
        if pts:
            series.append((sweep, sorted(pts)))
    if not series:
        return ""

    # Dash and shape come from every sweep on the page, not from the ones this
    # chart happens to draw. Derived from `series`, a metric that only some
    # sweeps have -- the TP=1 ratio, where the baselines are absent by
    # construction -- renumbered the slots, and TP=8 was a triangle on one chart
    # and a hollow square on the next. The encoding has to survive a reader
    # changing the metric, exactly as colour survives filtering a machine out.
    models = sorted({s["model"] for s in sweeps})
    tps = sorted({s.get("tp") for s in sweeps}, key=lambda v: (v is None, v))
    W, H = 760, 380
    L, R, T, B = 58, 16, 14, 46
    pw, ph = W - L - R, H - T - B
    xs = [c for _, pts in series for c, _ in pts]
    ys = [v for _, pts in series for _, v in pts]
    x_lo, x_hi = min(xs), max(xs)
    if log_y:
        # Whole decades cost whole decades: tokens/joule bottoms out at 0.0091
        # and the floor snapped to 0.001, spending a third of the axis below any
        # data. Same fix, and the same reason, as the power chart's.
        y_lo, y_hi = _nice_log_bounds(ys)
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
        for v in _log_ticks(y_lo, y_hi):
            y = py(v)
            lab = f"{v:g}" if v >= 1 else f"{v:.3f}".rstrip("0").rstrip(".")
            parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{lab}</text>')
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
        slot = _tp_slot(sweep.get("tp"), tps)
        for c, v in pts:
            parts.append(_marker(
                px(c), py(v), 4, slot,
                f'{_esc(machine)} {_esc(model)} TP={_esc(tp)} — '
                f'concurrency {c}: {v:,.3g} {_esc(label)}'))
        parts.append("</g>")

    parts.append(f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" '
                 f'text-anchor="middle">concurrent requests (log scale)</text>')
    # The metric is the reader's choice here, so the axis has to say which one
    # it ended up on -- the dropdown that made the choice scrolls away, and a
    # saved image of this chart carries no dropdown at all.
    parts.append(_y_title(label, T, ph))
    parts.append("</svg>")
    return "".join(parts)


def dashboard_legend(sweeps: list) -> str:
    """One key per configuration, tagged so the script can dim what is hidden."""
    models = sorted({s["model"] for s in sweeps})
    tps = sorted({s.get("tp") for s in sweeps}, key=lambda v: (v is None, v))
    keys = ""
    # Same set the charts draw. A key for a sweep no chart on this page can
    # plot sends the reader looking for a line that is not there.
    for sweep in sorted([s for s in sweeps if swept_over_concurrency(s["rows"])],
                        key=lambda s: (s["machine"], s["model"] or "",
                                       s.get("tp") or 0)):
        machine = sweep["machine"]
        model = (sweep["model"] or "?").split("/")[-1]
        dash = MODEL_DASHES[models.index(sweep["model"]) % len(MODEL_DASHES)]
        bg = (f"repeating-linear-gradient(90deg,var(--c) 0 6px,transparent 6px 9px)"
              if dash else "var(--c)")
        # Line and marker both, because the line carries two of the three
        # encodings and the marker carries the third. A key showing only the
        # line cannot tell two TPs of the same model apart -- which is the exact
        # ambiguity this legend existed to resolve.
        keys += (f'<span class="key s{_slot(machine)}" data-machine="{_esc(machine)}" '
                 f'data-model="{_esc(model)}" data-tp="{_esc(sweep.get("tp") or "")}">'
                 f'<span class="sw" style="background:{bg}"></span>'
                 f'{_marker_swatch(_tp_slot(sweep.get("tp"), tps))}'
                 f'{machine_tag(machine)} · {_esc(model)}'
                 + (f' · TP={_esc(sweep["tp"])}' if sweep.get("tp") else "") + "</span>")
    return f'<div class="legend-row">{keys}</div>'


def _nice_log_bounds(values: list, pad: float = 0.03) -> tuple:
    """Round a range out to the next 1-2-5 step rather than the next decade.

    Snapping to whole decades is the obvious rule on a log axis, and the cost is
    paid in whole decades: a sweep peaking at 1088 tokens/s pushed the ceiling
    to 10,000 and spent a third of the plot on empty space above the data.

    That is not only wasted room. Pixels per decade is what sets the slope of
    the iso-efficiency diagonals, so an axis padded out to a decade it does not
    use flattens them toward horizontal, and a line that reads as a gridline
    cannot be read as an efficiency.

    pad is then a few percent of a decade beyond the snapped bound, because the
    snap can land exactly on the extreme value -- 102 W under a bound of 100 put
    a marker on top of the axis. It is margin, not range: the decade gridlines
    stay where they were and sit just inside the frame.
    """
    # A 1-2-5 grid is too coarse for data that spans less than a decade: a ratio
    # running 0.96 to 1.52 snaps out to 0.5..2 and spends most of the axis on
    # emptiness. Finer stops where the range is already tight.
    span = math.log10(max(values) / min(values)) if min(values) > 0 else 1.0
    steps = (1, 2, 5, 10) if span >= 0.7 else (1, 1.5, 2, 3, 4, 5, 6, 8, 9, 10)

    def snap(v, up):
        exp = math.floor(math.log10(v))
        mantissa = v / 10 ** exp
        if up:
            mantissa = next(s for s in steps if s >= mantissa - 1e-9)
        else:
            mantissa = next(s for s in reversed(steps) if s <= mantissa + 1e-9)
        return mantissa * 10 ** exp

    return (snap(min(values), False) / 10 ** pad,
            snap(max(values), True) * 10 ** pad)


def power_throughput_chart(sweeps: list, power_key: str, label: str) -> str:
    """Throughput against power, one trajectory per configuration.

    Log-log on purpose, and not for range: tokens/joule is tokens/s over watts,
    so a line of constant efficiency is y = kx, and on log axes that is a
    straight line of slope 1. The faint diagonals are those lines. Efficiency
    stops being a column to cross-reference and becomes a position on the grid --
    a point sitting above a diagonal beats that efficiency, and the vertical gap
    to it is by how much.

    Slope 1 holds in log space; on screen it is whatever the pixels per decade
    make it, which is why this chart computes its own height rather than taking
    a fixed box. See _nice_log_bounds and the height below -- between them they
    are the difference between a diagonal and a second set of gridlines.

    Each configuration is drawn in concurrency order with the marker growing
    along the way, so direction is visible without an arrowhead: small dot is
    one concurrent request, large is the most. Up and to the left is better.

    power_key picks the x axis. Dynamic power compares silicon; node power
    compares what an allocation is billed, and switching between them slides
    each machine right by its own idle floor -- which is the whole argument
    about these two machines, as one movement.
    """
    series = []
    for sweep in sweeps:
        # Marker size runs with concurrency here, so a sweep that held it fixed
        # would draw a trajectory whose growing dots mean nothing.
        if not swept_over_concurrency(sweep["rows"]):
            continue
        pts = [(r.get(power_key), r.get("out_tok_per_s"), r["concurrency"])
               for r in sweep["rows"] if r.get("concurrency")]
        pts = [(x, y, c) for x, y, c in pts
               if isinstance(x, (int, float)) and isinstance(y, (int, float))
               and x > 0 and y > 0]
        if pts:
            series.append((sweep, sorted(pts, key=lambda p: p[2])))
    if not series:
        return ""

    # Dash and shape come from every sweep on the page, not from the ones this
    # chart happens to draw. Derived from `series`, a metric that only some
    # sweeps have -- the TP=1 ratio, where the baselines are absent by
    # construction -- renumbered the slots, and TP=8 was a triangle on one chart
    # and a hollow square on the next. The encoding has to survive a reader
    # changing the metric, exactly as colour survives filtering a machine out.
    models = sorted({s["model"] for s in sweeps})
    tps = sorted({s.get("tp") for s in sweeps}, key=lambda v: (v is None, v))
    W = 760
    L, R, T, B = 58, 16, 14, 46
    pw = W - L - R
    xs = [x for _, pts in series for x, _, _ in pts]
    ys = [y for _, pts in series for _, y, _ in pts]
    x_lo, x_hi = _nice_log_bounds(xs)
    y_lo, y_hi = _nice_log_bounds(ys)
    x_dec = math.log10(x_hi) - math.log10(x_lo)
    y_dec = math.log10(y_hi) - math.log10(y_lo)

    # Height comes from the data's own shape, not from a fixed box. Equal pixels
    # per decade on both axes is what puts the diagonals at 45 degrees, which is
    # the entire reason for drawing them -- but these sweeps span two decades of
    # throughput inside one of power, and honouring that literally gives a plot
    # three times taller than it is wide. Half a decade of y per decade of x is
    # the compromise: steep enough to read as a diagonal rather than as a second
    # set of gridlines, capped so the chart still fits on a screen.
    ph = max(300.0, min(640.0, pw * 0.5 * y_dec / x_dec))
    H = ph + T + B

    def px(v):
        return L + (math.log10(v) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * pw

    def py(v):
        return T + (1 - (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))) * ph

    parts = [f'<svg class="chart xy" viewBox="0 0 {W} {H:.0f}" role="img" aria-label="'
             f'Output throughput against {_esc(label)}, one line per configuration">']

    # Gridlines stay on the decades even though the bounds no longer have to be
    # decades, so every labelled line is still a round number. Starting the walk
    # at the bound itself would put them at 2, 20, 200 the moment a bound landed
    # on a 2 or a 5.
    d = math.ceil(math.log10(x_lo) - 1e-9)
    while d <= math.log10(x_hi) + 1e-9:
        v = 10 ** d
        parts.append(f'<line class="grid" x1="{px(v):.1f}" y1="{T}" x2="{px(v):.1f}" y2="{T+ph:.1f}"/>')
        parts.append(f'<text class="tick" x="{px(v):.1f}" y="{T+ph+18:.1f}" '
                     f'text-anchor="middle">{v:,g}</text>')
        d += 1
    d = math.ceil(math.log10(y_lo) - 1e-9)
    while d <= math.log10(y_hi) + 1e-9:
        v = 10 ** d
        parts.append(f'<line class="grid" x1="{L}" y1="{py(v):.1f}" x2="{L+pw}" y2="{py(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{L-8}" y="{py(v)+3.5:.1f}" text-anchor="end">{v:,g}</text>')
        d += 1

    # Iso-efficiency diagonals: y = kx, clipped to the plot box. Drawn under the
    # data and labelled where they leave the frame. A decade whose line only
    # clips a corner is dropped rather than drawn: a 40 px stub carrying a full
    # "0.01 tok/J" label is all label and no line, and the reader can find that
    # efficiency by counting decades off the neighbour it was crowding.
    k = 10 ** math.floor(math.log10(min(ys) / max(xs)))
    while k <= max(ys) / min(xs):
        a = (max(x_lo, y_lo / k), min(x_hi, y_hi / k))
        if a[0] < a[1] and math.log10(a[1] / a[0]) >= 0.12 * x_dec:
            parts.append(f'<line class="target" x1="{px(a[0]):.1f}" y1="{py(k*a[0]):.1f}" '
                         f'x2="{px(a[1]):.1f}" y2="{py(k*a[1]):.1f}" opacity=".45"/>')
            lbl = f"{k:g}" if k >= 1 else f"{k:.3f}".rstrip("0")
            parts.append(f'<text class="tick" x="{px(a[1])-4:.1f}" y="{py(k*a[1])-5:.1f}" '
                         f'text-anchor="end" opacity=".7">{lbl} tok/J</text>')
        k *= 10

    for sweep, pts in series:
        machine = sweep["machine"]
        model = (sweep["model"] or "?").split("/")[-1]
        dash = MODEL_DASHES[models.index(sweep["model"]) % len(MODEL_DASHES)]
        style = f' style="stroke-dasharray:{dash}"' if dash else ""
        parts.append(f'<g class="series s{_slot(machine)}" data-machine="{_esc(machine)}" '
                     f'data-model="{_esc(model)}" data-tp="{_esc(sweep.get("tp") or "")}">')
        if len(pts) > 1:
            line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in pts)
            parts.append(f'<polyline class="ln" points="{line}"{style}/>')
        # Size still runs with concurrency and shape now runs with TP. The two
        # do not collide: size is read along a line, shape between lines.
        slot = _tp_slot(sweep.get("tp"), tps)
        for i, (x, y, c) in enumerate(pts):
            r = 2.6 + 2.6 * (i / max(1, len(pts) - 1))
            parts.append(_marker(
                px(x), py(y), r, slot,
                f'{_esc(machine)} {_esc(model)} TP={_esc(sweep.get("tp") or "?")} — '
                f'concurrency {c}: {y:,.1f} tok/s at {x:,.0f} W = {y/x:.3f} tok/J'))
        parts.append("</g>")

    parts.append(f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6:.0f}" text-anchor="middle">'
                 f'{_esc(label)} — both axes logarithmic, diagonals are constant tokens/joule</text>')
    parts.append(_y_title("output tokens/s", T, ph))
    parts.append("</svg>")
    return "".join(parts)


def timeline_watts(timeline: dict, bins: int = 240) -> dict:
    """Cumulative joule counters -> binned watts, per device and total.

    Watts are consecutive counter deltas over consecutive timestamps -- the
    same arithmetic power.py's summary does, kept out of the chart so the
    stats under it and the lines in it cannot disagree.

    Aggregate sources (Aurora's whole-card counters) are excluded from the
    total for the same reason they are excluded everywhere: they cover silicon
    the per-tile counters already report, and summing both counts it twice.
    """
    t = timeline["t_s"]
    n = len(t)
    if n < 3:
        return {}
    keep = [i for i, s in enumerate(timeline["sources"]) if not s.get("aggregate")]
    idxs = sorted({round(i * (n - 1) / bins) for i in range(bins + 1)})
    mid, per_dev = [], []
    for a, b in zip(idxs, idxs[1:]):
        if t[b] <= t[a]:
            continue
        mid.append((t[a] + t[b]) / 2)
    for si in keep:
        j = timeline["joules"][si]
        w, k = [], 0
        for a, b in zip(idxs, idxs[1:]):
            if t[b] <= t[a]:
                continue
            w.append((j[b] - j[a]) / (t[b] - t[a]))
        per_dev.append(w)
    total = [sum(col) for col in zip(*per_dev)]
    return {
        "t": mid, "devices": per_dev, "total": total,
        "keys": [timeline["sources"][i]["key"] for i in keep],
        "avg_w": sum(total) / len(total),
        "peak_w": max(total),
        "span_s": t[-1] - t[0],
    }


def power_timeline_chart(timeline: dict, slot: int, aria: str) -> str:
    """Watts against wall-clock: one thin line per device, one bold total.

    The thin lines are the reason this chart exists. A single busy Aurora tile
    over eleven flat ones, or all twelve rising together, is the difference
    between an inference node and a training node -- and it is exactly the
    thing every average in every table on this site has to flatten. The bold
    total is what the node's accelerators drew; its average and peak are in
    the caption, where they can be compared against the tables.

    Marks: only 'reached' (time-to-accuracy) earns a line. Training timelines
    carry two marks per epoch and a hundred epochs of them is a fence, not an
    annotation.
    """
    d = timeline_watts(timeline)
    if not d:
        return ""
    W, H = 720, 300
    L, R, T, B = Y_TITLE_L, 16, 12, 44
    pw, ph = W - L - R, H - T - B
    t0, t1 = d["t"][0], d["t"][-1]
    raw = max(1.0, d["peak_w"] * 1.06)
    # The ceiling is chosen so the quarter-gridlines land on round numbers --
    # a 1,700 W axis reads 425 at its first tick, which no one wants to hold.
    quarter = raw / 4
    mag = 10 ** math.floor(math.log10(quarter))
    for m in (1, 2, 2.5, 4, 5, 10):
        if m * mag >= quarter:
            quarter = m * mag
            break
    y_hi = 4 * quarter

    def px(x: float) -> float:
        return L + (x - t0) / (t1 - t0) * pw

    def py(v: float) -> float:
        return T + (1 - v / y_hi) * ph

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(aria)}">']
    for i in range(5):
        v = y_hi * i / 4
        y = py(v)
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{L-8}" y="{y+3.5:.1f}" text-anchor="end">{v:,.0f}</text>')
    for i in range(6):
        x = L + pw * i / 5
        sec = t0 + (t1 - t0) * i / 5
        parts.append(f'<text class="tick" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{_fmt_s(sec)}</text>')

    for w in d["devices"]:
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(d["t"], w))
        parts.append(f'<polyline class="ln dev s{slot}" points="{pts}"/>')
    pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(d["t"], d["total"]))
    parts.append(
        f'<polyline class="ln s{slot}" points="{pts}">'
        f'<title>all {len(d["devices"])} devices: avg {d["avg_w"]:,.0f} W, '
        f'peak {d["peak_w"]:,.0f} W</title></polyline>')

    for m in timeline.get("marks", []):
        if "reached" in m["label"]:
            x = px(m["t_s"])
            parts.append(f'<line class="target" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}"/>')
            parts.append(f'<text class="tick" x="{x+4:.1f}" y="{T+12}">{_esc(m["label"])}</text>')

    parts.append(f'<text class="axis-title" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">wall-clock</text>')
    parts.append(_y_title("accelerator watts", T, ph))
    parts.append("</svg>")
    return "".join(parts)
