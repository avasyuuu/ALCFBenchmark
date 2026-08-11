"""Render results/*.json into docs/index.html for GitHub Pages.

    python analysis/build_site.py                      # writes docs/index.html
    python analysis/build_site.py --results-dir ./results --out docs/index.html
    python analysis/build_site.py --include-synthetic

Serve it by setting Pages to "main branch /docs" in the repo settings. The page
is self-contained -- no CDN, no fonts, no scripts -- so it renders offline and
inside restricted networks.

Numbers are never written by hand here. Everything comes from the result JSON,
so the page is only as current as the last `git pull`, and a stale page is
always a missing commit rather than a forgotten edit.

Stdlib only, so it runs on a login node.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

# `legend` here is the column glossary further down; the chart key is
# imported under its own name so the two never shadow each other.
from charts import (accuracy_chart, canonical_runs, idle_power_run, power_chart,
                    series_legend, tail_chart)
from summarize import load_runs

AUTHOR = "Avasyu Chukkapalli"
REPO_URL = "https://github.com/avasyuuu/ALCFBenchmark"

# Dropped next to the output file, and inlined into it at build time. A logo
# left as <img src="argonne-logo.png"> would break the promise in this module's
# docstring -- one self-contained file that renders offline and inside
# restricted networks -- so the bytes go into the page rather than beside it.
# Absent is fine: the byline just renders without it.
LOGO_STEM = "argonne-logo"

# Sniffed from the bytes rather than taken from the file extension, because a
# logo saved from a browser routinely arrives as a JPEG named .png. Declaring
# the wrong type in a data: URI leaves the image at the mercy of each browser's
# sniffing, which is not a thing to rely on for the one image on the page.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def logo_data_uri(out_dir: Path) -> str | None:
    """Read a logo from out_dir and return it as a data: URI, or None.

    SVG is preferred explicitly when several exist -- plain alphabetical order
    would pick .jpg -- because it is usually smaller than a bitmap and stays
    sharp on the high-DPI displays where a 26px raster goes soft.
    """
    prefer_svg = lambda p: (p.suffix.lower() != ".svg", p.name)
    for path in sorted(out_dir.glob(f"{LOGO_STEM}.*"), key=prefer_svg):
        blob = path.read_bytes()
        mime = next((m for sig, m in _MAGIC if blob.startswith(sig)), None)
        if mime is None and blob.lstrip()[:1] == b"<":
            mime = "image/svg+xml"  # <svg> or an XML declaration ahead of it
        if mime is None:
            continue  # not an image we can name; a wrong label is worse
        return f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"
    return None


# Scope strings name a measurement boundary, not a machine. An Aurora tile is
# half a card; a Polaris reading covers the whole board including HBM. Ratios
# built from them are not comparable, which is why the per-rank Samples/J is
# kept out of the headline and the node table carries the comparison instead.
SCOPE_WARNING = (
    "Per-rank energy scopes differ by vendor — an Aurora tile is half a card, "
    "a Polaris reading covers the whole board including HBM. Compare machines "
    "on the node table, never on per-rank Samples/J."
)


def node_efficiency(run: dict):
    """Node-wide samples per joule: the only cross-machine energy ratio here.

    Uses samples_global (the whole job) against joules_total (every accelerator
    on the node, including ones no rank bound to). Both sides then cover the
    same thing -- a node -- which is also the unit an allocation is billed in.
    Returns None on schema < 3, where samples_global was never recorded.
    """
    samples = run.get("samples_global")
    joules = run.get("power_joules_total")
    if not samples or not joules:
        return None
    return samples / joules


def headline(runs: list) -> list:
    """Best throughput and best node efficiency per machine, for the cards."""
    by_machine: dict = {}
    for r in runs:
        m = r.get("machine")
        if not m:
            continue
        slot = by_machine.setdefault(m, {"machine": m, "sps": None, "eff": None,
                                         "nodes": None, "ranks": None})
        sps = r.get("samples_per_s")
        if sps and (slot["sps"] is None or sps > slot["sps"]):
            slot["sps"] = sps
            slot["nodes"] = r.get("nodes")
            slot["ranks"] = r.get("ranks")
        eff = node_efficiency(r)
        if eff and (slot["eff"] is None or eff > slot["eff"]):
            slot["eff"] = eff
    return sorted(by_machine.values(), key=lambda s: -(s["sps"] or 0))


def takeaway(cards: list) -> str:
    """The throughput-vs-energy sentence, derived rather than written.

    Whichever machine leads on samples/s and whichever leads on node-wide
    samples/J are found from the data, so the claim re-derives on every build
    and cannot drift from the tables above it. Renders nothing until at least
    two machines have both numbers -- with one machine there is no tradeoff to
    state, and saying so would be inventing a comparison.
    """
    usable = [c for c in cards if c["sps"] and c["eff"]]
    if len(usable) < 2:
        return ""

    fast = max(usable, key=lambda c: c["sps"])
    green = max(usable, key=lambda c: c["eff"])

    if fast["machine"] == green["machine"]:
        others = [c for c in usable if c is not fast]
        best_other = max(others, key=lambda c: c["sps"])
        body = (
            f"<strong>{html.escape(str(fast['machine']))}</strong> leads on both axes — "
            f"{fast['sps'] / best_other['sps']:.2f}× the throughput of "
            f"{html.escape(str(best_other['machine']))} and "
            f"{fast['eff'] / best_other['eff']:.2f}× the node-wide energy efficiency. "
            "No tradeoff to make on this workload."
        )
    else:
        body = (
            f"<strong>{html.escape(str(fast['machine']))}</strong> is "
            f"{fast['sps'] / green['sps']:.2f}× faster, but "
            f"<strong>{html.escape(str(green['machine']))}</strong> is "
            f"{green['eff'] / fast['eff']:.2f}× more energy-efficient per node. "
            "Fastest and most efficient are not the same machine — which one "
            "wins depends on whether the budget is wall-clock or joules."
        )

    return f"""
<h2>The tradeoff</h2>
<p class="takeaway">{body}</p>
<p class="fineprint">Throughput is each machine's peak samples/s; efficiency is
its best node-wide samples/joule. Both are ratios, so runs of different lengths
compare cleanly — but this is a handful of runs per machine on a workload at
0.5–1.4% of peak, so read it as a direction, not a verdict.</p>"""


# Every column on the page, in the words of the code that produces it. Kept
# here rather than in the table captions because a caption explains what a
# table is for, and this answers the narrower question of what one number is --
# and because a reader who needs "what is MFU" needs it once, not three times.
LEGEND = [
    ("Shared by every table", [
        ("When", "when the run started, UTC — hover for the exact timestamp"),
        ("Machine", "the ALCF system the run executed on"),
        ("Nodes", "compute nodes in the job"),
        ("Ranks", "worker processes, one bound to each accelerator"),
    ]),
    ("Runs", [
        ("Prec", "numeric precision the model trained in"),
        ("Global BS", "samples per optimizer step, summed across all ranks"),
        ("Steps", "optimizer steps completed"),
        ("Epochs", "completed of requested — fewer means the run stopped early"),
        ("Samples/s", "global batch ÷ median step time, aggregate over all ranks"),
        ("Step ms", "median time for one training step, warmup excluded"),
        ("Best top-1", "highest validation accuracy reached, 0–1"),
        ("TTA s", "seconds of training to first cross the accuracy target; "
                  "blank means it never did"),
        ("MFU %", "achieved FLOP/s as a share of the vendor's dense peak"),
    ]),
    ("Energy — per rank", [
        ("Avg W", "this rank's device energy ÷ the seconds it was measured over"),
        ("Joules", "energy drawn by this rank's own accelerator"),
        ("Samples", "samples this rank processed — the numerator of Samples/J"),
        ("Samples/J", "samples per joule, one rank on one device"),
        ("J to acc", "energy spent getting to the accuracy target"),
        ("Scope", "what the counter physically covered — vendor-specific, which "
                  "is why these rows do not compare across machines"),
    ]),
    ("Energy — per node", [
        ("Devices", "accelerators on the node, used or not"),
        ("Idle", "accelerators no rank bound to"),
        ("Node J", "energy across every accelerator on the node"),
        ("Idle J", "the share of Node J spent on accelerators nobody used"),
        ("Idle %", "Idle J as a percentage of Node J"),
        ("Samples/J", "whole-job samples ÷ whole-node joules — the one energy "
                      "ratio that compares across machines"),
    ]),
]


def identity(logo_uri: str | None) -> str:
    """Byline block for the top corner: logo, name, repo link."""
    logo = (
        f'<img class="logo" src="{logo_uri}" '
        f'alt="Argonne National Laboratory">'
        if logo_uri
        else ""
    )
    return f"""<aside class="ident">{logo}
<div class="who">{html.escape(AUTHOR)}</div>
<a href="{html.escape(REPO_URL)}">{html.escape(REPO_URL.split("//")[-1])}</a>
</aside>"""


def legend() -> str:
    blocks = ""
    for heading, terms in LEGEND:
        items = "".join(
            f"<dt>{html.escape(term)}</dt><dd>{html.escape(meaning)}</dd>"
            for term, meaning in terms
        )
        blocks += (
            f"<section><h3>{html.escape(heading)}</h3><dl>{items}</dl></section>"
        )
    return f'<div class="legend">{blocks}</div>'


SPEC_COLUMNS = [
    ("nodes", "Nodes"),
    ("accelerator", "Accelerator"),
    ("accelerator_memory", "Accel. memory"),
    ("cpu", "CPU"),
    ("memory", "Node memory"),
    ("interconnect", "Interconnect"),
]


def spec_rows(specs: dict, measured: set) -> list:
    """One row per configured machine, in the config's own order.

    Machines with no runs yet are listed and tagged rather than hidden: the
    table answers "what is being compared", and a system that is targeted but
    unmeasured is part of that answer.
    """
    rows = []
    for machine, spec in specs.items():
        if machine.startswith("_"):
            continue  # _comment and friends
        tag = "" if machine in measured else ' <span class="tag">no runs yet</span>'
        rows.append(
            [f'<span class="m">{html.escape(machine)}</span>{tag}']
            + [html.escape(str(spec.get(key, "—"))) for key, _ in SPEC_COLUMNS]
        )
    return rows


def when_cell(run: dict) -> str:
    """'Aug 10', with the full UTC timestamp on hover.

    summarize.py's `when` carries seconds down to "08-10 17:12:04", because two
    runs of the same config land in the same minute and a terminal table has no
    other way to tell them apart. A page does: the rows are already in order,
    and the exact stamp rides along in a title attribute. So the visible column
    drops to the part anyone actually reads, and nothing is lost.

    Day is zero-padded to keep the column aligned under tabular-nums.
    """
    stamp = run.get("timestamp_utc")
    label = run.get("when") or "—"
    if stamp:
        try:
            label = datetime.fromisoformat(stamp).strftime("%b %d")
        except ValueError:
            pass  # keep summarize.py's label rather than invent a date
    if not stamp:
        return html.escape(label)
    return f'<span title="{html.escape(stamp)}">{html.escape(label)}</span>'


def num(value, places=0, dash="—"):
    if value is None:
        return dash
    if isinstance(value, float) and places == 0:
        return f"{value:,.0f}"
    if isinstance(value, float):
        return f"{value:,.{places}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def table(headers: list, rows: list, caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure><div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{cap}</figure>"
    )


def build(runs: list, logo_uri: str | None = None, specs: dict | None = None,
          curves: dict | None = None, idle_run: dict | None = None) -> str:
    machine_stats = headline(runs)
    curves = curves or {}
    tta_svg = accuracy_chart(curves)
    tta_section = (
        "<h2>Time to accuracy</h2>" + tta_svg + series_legend(curves)
        + '<p class="fineprint">Validation top-1 against wall-clock, for the '
          'full 100-epoch run on each machine. Every machine crosses 0.90 '
          'within three epochs of the others — same global batch, same '
          'schedule — so the horizontal distance between the curves is the '
          'whole comparison. The x-axis is logarithmic: read the gaps as '
          'multiples, not lengths.</p>'
        if tta_svg else ""
    )
    power_svg = power_chart(idle_run)
    ranks = ((idle_run or {}).get("config") or {}).get("world_size")
    devices = ((idle_run or {}).get("power") or {}).get("devices_total")
    power_section = (
        "<h2>What an idle accelerator costs</h2>" + power_svg
        + '<p class="fineprint">Mean watts per accelerator during a single '
          f'{ranks}-rank run on a {devices}-device node. The device doing all '
          'of the work draws barely more than the ones doing none — at this '
          "workload's 0.5% of peak, almost all of the node's energy is floor "
          'rather than computation. That is why the node table exists: a '
          'per-rank figure cannot see the eleven devices nobody asked for.</p>'
        if power_svg else ""
    )
    tail_svg = tail_chart(curves)
    tail_section = (
        "<h2>Step-time tail</h2>" + tail_svg
        + '<p class="fineprint">Slowest training step as a multiple of the '
          'median, from the same runs. Every rank waits on the slowest at each '
          'all-reduce, so the tail is what a rare bad step actually costs. The '
          'CPU machine is the steadiest thing on this page.</p>'
        if tail_svg else ""
    )
    measured = {r.get("machine") for r in runs if r.get("machine")}
    spec_table = (
        table(
            ["Machine"] + [label for _, label in SPEC_COLUMNS],
            spec_rows(specs, measured),
            "Per node, which is both the unit this benchmark scales in and the "
            "unit an allocation is billed in. Where each figure came from is "
            "recorded in the _source fields of configs/machines.json.",
        )
        if specs
        else ""
    )
    cards = ""
    for h in machine_stats:
        eff = f"{h['eff']:,.1f}" if h["eff"] else "—"
        cards += f"""
        <div class="card">
          <h3>{html.escape(str(h['machine']))}</h3>
          <div class="stat"><span class="v">{num(h['sps'])}</span><span class="u">samples/s</span></div>
          <div class="sub">peak, {h['nodes']} node · {h['ranks']} ranks</div>
          <div class="stat"><span class="v">{eff}</span><span class="u">samples/J</span></div>
          <div class="sub">node-wide, best run</div>
        </div>"""

    run_rows = []
    for r in runs:
        flag = ""
        if r.get("stopped_early"):
            flag = ' <span class="tag">early</span>'
        run_rows.append([
            when_cell(r),
            f'<span class="m">{html.escape(str(r.get("machine") or "—"))}</span>',
            num(r.get("nodes")), num(r.get("ranks")),
            html.escape(str(r.get("precision") or "—")),
            num(r.get("global_batch")),
            num(r.get("steps")) + flag,
            html.escape(str(r.get("epochs") or "—")),
            num(r.get("samples_per_s")),
            num(r.get("median_step_ms"), 1),
            num(r.get("best_top1"), 3),
            num(r.get("tta_s"), 1),
            num(r.get("mfu_pct"), 2),
        ])

    energy_rows = []
    for r in runs:
        if not r.get("joules"):
            continue
        energy_rows.append([
            when_cell(r),
            f'<span class="m">{html.escape(str(r.get("machine") or "—"))}</span>',
            num(r.get("ranks")),
            num(r.get("avg_watts"), 1),
            num(r.get("joules")),
            num(r.get("samples_processed")),
            num(r.get("samples_per_joule"), 1),
            num(r.get("joules_to_target")),
            f'<span class="scope">{html.escape(str(r.get("energy_scope") or "—"))}</span>',
        ])

    node_rows = []
    for r in runs:
        if not r.get("power_joules_total"):
            continue
        eff = node_efficiency(r)
        node_rows.append([
            when_cell(r),
            f'<span class="m">{html.escape(str(r.get("machine") or "—"))}</span>',
            num(r.get("power_devices")),
            num(r.get("power_devices_idle")),
            num(r.get("power_joules_total")),
            num(r.get("power_joules_idle")),
            num(r.get("power_idle_pct"), 1),
            f'<strong>{eff:,.1f}</strong>' if eff else "—",
        ])

    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Power and Performance Across ALCF Machines</title>
<style>
:root {{
  --bg:#fbfbfa; --fg:#1a1a18; --dim:#6b6b66; --line:#e2e2dd;
  --card:#ffffff; --accent:#8a5a2b; --tag:#f0e6d8; --stripe:#f4f4f1;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16161a; --fg:#e8e8e4; --dim:#9a9a94; --line:#2c2c32;
          --card:#1e1e23; --accent:#d9a066; --tag:#33291d; --stripe:#1c1c21; }}
}}
:root[data-theme="dark"] {{ --bg:#16161a; --fg:#e8e8e4; --dim:#9a9a94;
  --line:#2c2c32; --card:#1e1e23; --accent:#d9a066; --tag:#33291d;
  --stripe:#1c1c21; }}
:root[data-theme="light"] {{ --bg:#fbfbfa; --fg:#1a1a18; --dim:#6b6b66;
  --line:#e2e2dd; --card:#ffffff; --accent:#8a5a2b; --tag:#f0e6d8;
  --stripe:#f4f4f1; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:3rem 1.25rem 5rem; }}
h1 {{ font-size:1.9rem; margin:0 0 .4rem; letter-spacing:-.02em; }}
h2 {{ font-size:1.15rem; margin:2.75rem 0 .75rem; letter-spacing:-.01em; }}
h2::before {{ content:""; display:inline-block; width:3px; height:.95em;
  background:var(--accent); margin-right:.55rem; vertical-align:-.08em; }}
.lede {{ color:var(--dim); margin:0 0 2rem; max-width:60ch; }}
/* Byline sits opposite the title and wraps under it on a phone, where a
   right-aligned column beside a headline would leave both too narrow. */
.head {{ display:flex; gap:2rem; align-items:flex-start;
  justify-content:space-between; flex-wrap:wrap; }}
.ident {{ margin-left:auto; text-align:right; flex-shrink:0; padding-top:.4rem;
  font-size:.72rem; line-height:1.55; color:var(--dim); }}
/* No plate or rounding: the mark is a transparent PNG, so the page background
   shows through its hollow centre on either theme. */
.ident .logo {{ height:28px; width:auto; display:block; margin:0 0 .45rem auto; }}
.ident .who {{ color:var(--fg); opacity:.8; font-weight:600; }}
.ident a {{ color:var(--dim); text-decoration:none;
  border-bottom:1px solid var(--line); }}
.ident a:hover {{ color:var(--accent); border-bottom-color:var(--accent); }}
.cards {{ display:grid; gap:.9rem;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }}
.card {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:1.1rem 1.2rem; }}
.card h3 {{ margin:0 0 .7rem; font-size:.78rem; text-transform:uppercase;
  letter-spacing:.09em; color:var(--accent); }}
.stat {{ display:flex; align-items:baseline; gap:.4rem; }}
.stat .v {{ font-size:1.6rem; font-weight:650;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
.stat .u {{ font-size:.8rem; color:var(--dim); }}
.sub {{ font-size:.76rem; color:var(--dim); margin:.1rem 0 .8rem; }}
.card .sub:last-child {{ margin-bottom:0; }}
.scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
table {{ border-collapse:collapse; width:100%; font-size:.83rem;
  font-variant-numeric:tabular-nums; }}
th {{ text-align:right; font-weight:600; color:var(--dim); padding:.5rem .6rem;
  border-bottom:1px solid var(--line); white-space:nowrap;
  font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }}
td {{ text-align:right; padding:.45rem .6rem;
  border-bottom:1px solid var(--line); white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:none; }}
/* These tables are 9-13 columns wide and scroll sideways on anything narrower
   than a laptop. Pinning the date column means a row stays identifiable while
   the numbers slide past it -- otherwise the far-right columns are unlabelled.
   The pinned cell needs an opaque background of its own, or the scrolled
   content shows through it. */
th:first-child, td:first-child {{ text-align:left;
  position:sticky; left:0; background:var(--bg); }}
th:first-child {{ z-index:1; }}
/* After the sticky rule, so even rows repaint the pinned cell too. */
tbody tr:nth-child(even) td {{ background:var(--stripe); }}
tbody tr:hover td {{ background:var(--tag); }}
.m {{ font-weight:600; }}
.scope {{ font-size:.74rem; color:var(--dim); white-space:normal; }}
.tag {{ background:var(--tag); color:var(--accent); font-size:.66rem;
  padding:.1rem .35rem; border-radius:4px; letter-spacing:.03em; }}
figure {{ margin:0; }}
figcaption {{ font-size:.78rem; color:var(--dim); margin-top:.6rem;
  max-width:70ch; }}
.note {{ border-left:2px solid var(--accent); padding:.15rem 0 .15rem .9rem;
  margin:1.1rem 0; color:var(--dim); font-size:.87rem; max-width:70ch; }}
.takeaway {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:1.1rem 1.25rem; margin:.4rem 0 .8rem;
  font-size:1.02rem; line-height:1.65; max-width:72ch; }}
.takeaway strong {{ color:var(--accent); }}
.fineprint {{ font-size:.78rem; color:var(--dim); margin:0; max-width:70ch; }}
footer {{ margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
  color:var(--dim); font-size:.78rem; }}
code {{ font-size:.85em; background:var(--tag); padding:.1rem .3rem;
  border-radius:3px; }}
/* Reference material, so it sits quieter than the tables it explains. The
   column-width form collapses to one column on a phone instead of squeezing
   two unreadable ones. */
.legend {{ font-size:.74rem; color:var(--dim); line-height:1.5;
  columns:19rem 2; column-gap:2.5rem; margin-top:.4rem; }}
.legend section {{ break-inside:avoid; margin:0 0 1rem; }}
.legend h3 {{ font-size:.67rem; text-transform:uppercase; letter-spacing:.08em;
  margin:0 0 .4rem; font-weight:600; color:var(--accent); opacity:.75; }}
.legend dl {{ margin:0; }}
.legend dt {{ color:var(--fg); opacity:.7; font-weight:600; }}
.legend dd {{ margin:0 0 .35rem; }}

/* Categorical series slots. Assigned to machines by entity in charts.py, so a
   colour never moves when a machine is added. Both modes are selected steps of
   the same hues, validated against this page's own surfaces rather than
   flipped automatically. */
:root {{ --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --series-4:#eda100; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --series-4:#c98500; }}
}}
:root[data-theme="dark"] {{ --series-1:#3987e5; --series-2:#d95926;
  --series-3:#199e70; --series-4:#c98500; }}
:root[data-theme="light"] {{ --series-1:#2a78d6; --series-2:#eb6834;
  --series-3:#1baf7a; --series-4:#eda100; }}
.s1 {{ --c:var(--series-1); }} .s2 {{ --c:var(--series-2); }}
.s3 {{ --c:var(--series-3); }} .s4 {{ --c:var(--series-4); }}
.chart {{ display:block; width:100%; height:auto; overflow:visible; }}
/* Grid and axes: hairline, solid, one step off the surface. Never dashed --
   dashing reads as "projection" when it is only a grid. */
.grid {{ stroke:var(--line); stroke-width:1; }}
.axis {{ stroke:var(--line); stroke-width:1; }}
.target {{ stroke:var(--dim); stroke-width:1; opacity:.7; }}
.ln {{ fill:none; stroke:var(--c); stroke-width:2;
  stroke-linejoin:round; stroke-linecap:round; }}
/* 2px ring in the surface colour, so a marker stays legible where a line
   passes under it -- and so the hit target beats the 8px mark. */
.dot {{ fill:var(--c); stroke:var(--bg); stroke-width:2; }}
.bar {{ fill:var(--c); }}
/* Context marks in emphasis charts: the muted text token at reduced
   weight, so eleven bars recede and one reads as the subject. */
.bar-mute {{ fill:var(--dim); opacity:.32; }}
.sw-mute {{ background:var(--dim); opacity:.32; }}
/* Text never wears the series colour; identity comes from the mark beside it. */
.tick, .axis-title, .val {{ fill:var(--dim); font-size:11px;
  font-variant-numeric:tabular-nums; }}
.val {{ fill:var(--fg); opacity:.75; font-weight:600; }}
.legend-row {{ display:flex; flex-wrap:wrap; gap:.35rem 1.4rem;
  margin:.7rem 0 .2rem; font-size:.78rem; color:var(--dim); }}
.key {{ display:inline-flex; align-items:center; gap:.45rem; }}
.sw {{ width:14px; height:3px; border-radius:2px; background:var(--c);
  flex:none; }}
</style></head><body><div class="wrap">

<header class="head">
<div>
<h1>Power and Performance Across ALCF Machines</h1>
<p class="lede">A portable benchmark — ResNet-20 on CIFAR-10 — run identically on
every ALCF system and compared on throughput, time-to-accuracy and energy, down
to the accelerators nobody was using. One harness, one result schema, one
table.</p>
</div>
{identity(logo_uri)}
</header>

<h2>Machines</h2>
<div class="cards">{cards}</div>
{spec_table}

{tta_section}

<h2>Runs</h2>
{table(
    ["When","Machine","Nodes","Ranks","Prec","Global BS","Steps","Epochs",
     "Samples/s","Step ms","Best top-1","TTA s","MFU %"],
    run_rows,
    "Every complete run on disk, oldest first. MFU is reported for diagnosis, "
    "not as an efficiency claim — this workload runs at 0.5–1.4% of peak, so it "
    "largely measures kernel-launch overhead rather than the accelerator.",
)}

{tail_section}

{power_section}

<h2>Energy — per rank</h2>
{table(
    ["When","Machine","Ranks","Avg W","Joules","Samples","Samples/J","J to acc","Scope"],
    energy_rows,
    SCOPE_WARNING,
)}

<h2>Energy — per node</h2>
{table(
    ["When","Machine","Devices","Idle","Node J","Idle J","Idle %","Samples/J"],
    node_rows,
    "Node-wide sampling covers every accelerator, including devices no rank "
    "bound to. Samples/J here is the cross-machine comparison: whole-job "
    "samples over whole-node joules, the unit an allocation is billed in.",
)}

<div class="note">A device left idle still draws power. A single-rank run on a
12-tile Aurora node spent over 90% of node energy on tiles nobody used — the
per-rank column cannot see that, which is the reason both tables exist.</div>

{takeaway(machine_stats)}

<h2>Reading these numbers</h2>
<div class="note">Runs marked <span class="tag">early</span> stopped before
their epoch budget, so their raw joules cover less training. Compare those on
Samples/J or joules-to-accuracy, never on the Joules column.</div>
<div class="note">Energy scope is vendor-specific and is printed with every row.
An accelerator-only figure excludes CPU, memory, NICs and cooling, so a node
draws well more than the sum of its accelerators.</div>

<h2>Legend</h2>
{legend()}

<footer>
Generated {generated} from {len(runs)} run(s) · machines: {html.escape(", ".join(machines)) or "none"}<br>
Rebuild with <code>python analysis/build_site.py</code> after
<code>git pull</code>. Numbers come from <code>results/*.json</code>; the page
is never edited by hand.
</footer>
</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--machines", default="./configs/machines.json",
                    help="node specs for the machines table; skipped if absent")
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not args.include_synthetic:
        runs = [r for r in runs if not r.get("synthetic")]
    if not runs:
        raise SystemExit(f"no complete runs found in {args.results_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logo = logo_data_uri(out.parent)
    # Optional: a checkout without the config still builds, just without the
    # specs table, rather than failing on a file that carries no measurements.
    specs_path = Path(args.machines)
    specs = json.loads(specs_path.read_text(encoding="utf-8")) if specs_path.exists() else {}
    curves = canonical_runs(args.results_dir)
    idle_run = idle_power_run(args.results_dir)
    out.write_text(build(runs, logo, specs, curves, idle_run), encoding="utf-8")
    # Said out loud because an inlined image is the one thing here that can bloat
    # the page, and a silently-missing logo otherwise looks like a CSS bug.
    print(f"logo: {'inlined, ' + str(len(logo) // 1024) + ' KB' if logo else 'none found'}")
    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    print(f"wrote {out} — {len(runs)} run(s), machines: {', '.join(machines)}")


if __name__ == "__main__":
    main()
