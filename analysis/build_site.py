"""Render results/*.json into the docs/ site for GitHub Pages.

    python analysis/build_site.py                      # writes the whole site
    python analysis/build_site.py --results-dir ./results --out-dir docs
    python analysis/build_site.py --include-synthetic

Two pages, both written by this script:

    index.html   the comparison -- specs, runs, energy tables, charts
    power.html   per-machine power profiles, one section per system

Serve them by setting Pages to "main branch /docs" in the repo settings; Pages
serves whatever static files are in that folder, so a second page needs no
configuration beyond existing. Each page is self-contained -- no CDN, no fonts,
no scripts -- so it renders offline and inside restricted networks.

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# `legend` here is the column glossary further down; the chart key is
# imported under its own name so the two never shadow each other.
from charts import (SERIES_SLOT, accuracy_chart, canonical_runs,
                    dashboard_chart, dashboard_legend, machine_tag,
                    power_throughput_chart,
                    efficiency_chart, efficiency_compare_chart,
                    efficiency_compare_legend, inference_chart, inference_legend,
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


# Slug -> display name. The results carry "resnet20_cifar10"; a reader wants
# "ResNet-20 on CIFAR-10". Anything unrecognised falls back to its own slug
# rather than to a guessed prettification.
WORKLOAD_NAMES = {"resnet20_cifar10": "<strong>ResNet-20</strong> on <strong>CIFAR-10</strong>"}


def workload_strip(runs: list) -> str:
    """Name the experiment, once, above everything it applies to.

    What is being trained is the first question a reader has and the page only
    answered it in passing, halfway through a sentence of prose. It is stated
    here instead -- and only while every run agrees on it, so the line can
    never claim a uniformity the results have stopped having.
    """
    workloads = {r.get("workload") for r in runs if r.get("workload")}
    batches = {r.get("global_batch") for r in runs if r.get("global_batch")}
    targets = {r.get("target_top1") for r in runs if r.get("target_top1")}
    if len(workloads) != 1 or len(batches) != 1:
        return ""

    slug = workloads.pop()
    name = WORKLOAD_NAMES.get(slug, html.escape(slug))
    bits = [name, f"global batch {batches.pop():,}"]
    if len(targets) == 1:
        bits.append(f"target top-1 {targets.pop():.2f}")
    bits.append("identical on every machine")
    return '<p class="workload">' + " · ".join(bits) + "</p>"


def equal_work_note(runs: list) -> str:
    """State the machines that did byte-for-byte identical work, if any.

    Raw joules are only comparable when the runs behind them did the same
    amount of training, which is exactly what summarize.py warns about and what
    every ratio on this page exists to work around. Under strong scaling with a
    fixed global batch and a fixed epoch budget, several machines land on the
    same sample count exactly -- and for those the joules columns can be read
    straight across with no normalisation at all.

    Derived rather than written, so it cannot drift from the results: the
    sentence disappears on its own if no two machines share a sample count.
    """
    groups: dict = defaultdict(dict)
    for r in runs:
        samples, joules = r.get("samples_global"), r.get("power_joules_total")
        if not samples or not joules:
            continue
        machine = r.get("machine")
        # One run per machine -- the most recent, matching the cards.
        prior = groups[samples].get(machine)
        if not prior or (r.get("timestamp_utc") or "") > (prior.get("timestamp_utc") or ""):
            groups[samples][machine] = r
    best = max(groups.items(), key=lambda kv: len(kv[1]), default=(None, {}))
    samples, members = best
    if len(members) < 2:
        return ""

    ordered = sorted(members.values(), key=lambda r: r["power_joules_total"])
    named = ", ".join(
        f"{html.escape(str(r['machine']))} {r['power_joules_total']:,.0f} J" for r in ordered
    )
    lo, hi = ordered[0], ordered[-1]
    ratio = hi["power_joules_total"] / lo["power_joules_total"]
    return (
        f'<p class="fineprint">Every machine here runs a fixed global batch for a '
        f"fixed epoch budget, so several land on <strong>exactly the same "
        f"{samples:,} samples</strong> — identical work, not merely comparable. For "
        f"those the node-energy column reads straight across with no normalising: "
        f"{named}. {html.escape(str(hi['machine']))} spent {ratio:.2f}× the energy of "
        f"{html.escape(str(lo['machine']))} to train the same model to the same "
        f"accuracy.</p>"
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
#
# Rendered under the heading "Glossary". It defines terms, which is what a
# glossary does; a legend keys the marks on a chart, and the chart keys on this
# page are the .legend-row strips beside each figure.
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
        ("MFU", "achieved FLOP/s as a share of the vendor's dense peak, as a "
                "percentage — every run here is under 2%, which is the workload "
                "being far too small to fill the machine rather than the machine "
                "being slow"),
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


# Node specs, rendered per machine on the power page. Kept on one page only:
# the index compares machines and the power page describes them, and a spec
# printed twice is a spec that can be corrected in one place and not the other.
SPEC_COLUMNS = [
    ("nodes", "Nodes"),
    ("accelerator", "Accelerator"),
    ("accelerator_memory", "Accel. memory"),
    ("cpu", "CPU"),
    ("memory", "Node memory"),
    ("interconnect", "Interconnect"),
]


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


# ---------------------------------------------------------------------------
# Page shell, shared by every page in docs/.
#
# The stylesheet stays inlined in each page rather than becoming a docs/style.css
# that both link, because this module's docstring promises a page rendering with
# no network and no companion files -- the same reason the logo is a data: URI.
# Ten kilobytes duplicated across two files is a smaller cost than a page that
# loses its styling the moment it is opened from anywhere but the site.
#
# Braces below are doubled: this is an f-string body, and the CSS moved into it
# verbatim. Substituted values are not re-scanned, so callers need not escape.

PAGES = [
    ("index.html", "Data & analysis"),
    ("power.html", "Power profiles"),
    ("dashboard.html", "Dashboard"),
]


def nav(current: str) -> str:
    """Links to every page, with the current one marked and not a link.

    A second page nobody can reach from the first is a second page nobody
    reads, so this renders on both rather than only on the one being added.
    """
    links = ""
    for href, label in PAGES:
        if href == current:
            links += f'<span class="here">{html.escape(label)}</span>'
        else:
            links += f'<a href="{href}">{html.escape(label)}</a>'
    return f'<nav class="nav">{links}</nav>'


def shell(*, title: str, heading: str, lede: str, body: str, footer: str,
          logo_uri: str | None = None, here: str = "index.html",
          strip: str = "") -> str:
    """One complete HTML document. Every page on the site is built from this."""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
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
.lede {{ color:var(--dim); margin:0 0 .55rem; max-width:60ch; }}
/* Names the experiment above everything it applies to. Sits between the lede
   and the cards, so "what is being trained" is answered before any number. */
.workload {{ margin:0 0 1.9rem; font-size:.83rem; color:var(--dim);
  border-left:2px solid var(--accent); padding:.1rem 0 .1rem .7rem; }}
.workload strong {{ color:var(--fg); font-weight:650; }}
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
.m {{ font-weight:600; color:var(--ct, currentColor); }}
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
  --series-4:#eda100;
  /* The same hues, dark enough to read as body text. The chart colours were
     chosen to separate marks from each other, which is a different problem: on
     the light surface they run 2.09-4.26 against a 4.5 contrast floor, so
     setting them as text colour would make a name harder to read, not easier.
     Dark mode needs no such variant -- there they clear 4.65 already. */
  --series-1-text:#2772cb; --series-2-text:#bc5329;
  --series-3-text:#13815a; --series-4-text:#9a6800; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --series-4:#c98500;
    --series-1-text:#3987e5; --series-2-text:#d95926;
    --series-3-text:#199e70; --series-4-text:#c98500; }}
}}
:root[data-theme="dark"] {{ --series-1:#3987e5; --series-2:#d95926;
  --series-3:#199e70; --series-4:#c98500;
  --series-1-text:#3987e5; --series-2-text:#d95926;
  --series-3-text:#199e70; --series-4-text:#c98500; }}
:root[data-theme="light"] {{ --series-1:#2a78d6; --series-2:#eb6834;
  --series-3:#1baf7a; --series-4:#eda100;
  --series-1-text:#2772cb; --series-2-text:#bc5329;
  --series-3-text:#13815a; --series-4-text:#9a6800; }}
.s1 {{ --c:var(--series-1); --ct:var(--series-1-text); }}
.s2 {{ --c:var(--series-2); --ct:var(--series-2-text); }}
.s3 {{ --c:var(--series-3); --ct:var(--series-3-text); }}
.s4 {{ --c:var(--series-4); --ct:var(--series-4-text); }}
.chart {{ display:block; width:100%; height:auto; overflow:visible; }}
/* The throughput-against-power chart sizes its own height so that a decade of
   one axis keeps a fixed relationship to a decade of the other -- that ratio is
   what the iso-efficiency diagonals are read against. Stretching it to a wide
   viewport would scale both axes and keep the ratio, but at 1.5x it is a chart
   taller than the window, so it stops at its natural width and centres. */
.chart.xy {{ max-width:760px; margin-inline:auto; }}
/* Legend marker: the same shape the chart draws, at the size a key needs.
   Sits on the text baseline so a row of keys does not go ragged. */
.mk {{ width:12px; height:12px; vertical-align:-2px; margin-right:4px;
  overflow:visible; }}
/* Grid and axes: hairline, solid, one step off the surface. Never dashed --
   dashing reads as "projection" when it is only a grid. */
.grid {{ stroke:var(--line); stroke-width:1; }}
.axis {{ stroke:var(--line); stroke-width:1; }}
.target {{ stroke:var(--dim); stroke-width:1; opacity:.7; }}
.ln {{ fill:none; stroke:var(--c); stroke-width:2;
  stroke-linejoin:round; stroke-linecap:round; }}
/* Dashed encodes a second measure of the SAME machine -- total against dynamic
   tokens/joule. Colour is the machine and has to stay the machine, so the
   second series needs a different channel. This is the one place a dash is not
   the grid's "provisional" meaning, which is why the grid never dashes. */
.ln.dash {{ stroke-dasharray:5 3; }}
.sw.dash {{ background:repeating-linear-gradient(90deg,
  var(--c) 0 5px, transparent 5px 8px); }}
/* 2px ring in the surface colour, so a marker stays legible where a line
   passes under it -- and so the hit target beats the 8px mark. */
.dot {{ fill:var(--c); stroke:var(--bg); stroke-width:2; }}
.bar {{ fill:var(--c); }}
/* Text never wears the series colour; identity comes from the mark beside it. */
.tick, .axis-title, .val {{ fill:var(--dim); font-size:11px;
  font-variant-numeric:tabular-nums; }}
.val {{ fill:var(--fg); opacity:.75; font-weight:600; }}
.legend-row {{ display:flex; flex-wrap:wrap; gap:.35rem 1.4rem;
  margin:.7rem 0 .2rem; font-size:.78rem; color:var(--dim); }}
.key {{ display:inline-flex; align-items:center; gap:.45rem; }}
.sw {{ width:14px; height:3px; border-radius:2px; background:var(--c);
  flex:none; }}

/* Sits under the header on every page. The current page is a <span>, not a
   dead link to itself -- the only reliable cue for which page you are on. */
.nav {{ display:flex; gap:.45rem; flex-wrap:wrap; margin:1.6rem 0 0; }}
.nav a, .nav .here {{ font-size:.76rem; padding:.32rem .8rem; border-radius:999px;
  border:1px solid var(--line); text-decoration:none; }}
.nav a {{ color:var(--dim); }}
.nav a:hover {{ color:var(--accent); border-color:var(--accent); }}
.nav .here {{ color:var(--accent); border-color:var(--accent);
  background:var(--tag); font-weight:600; }}
/* Body links. Underlined on the baseline rather than through the descenders,
   and never the browser's blue, which belongs to neither theme. The .ident and
   .nav rules are more specific, so they keep their own treatment. */
a {{ color:var(--accent); text-decoration:none;
  border-bottom:1px solid var(--line); }}
a:hover {{ border-bottom-color:var(--accent); }}

/* Per-machine specs on the power page: the same fields as the index's specs
   table, turned on their side because one machine is a record, not a row. */
.specs {{ display:grid; grid-template-columns:auto 1fr; gap:.28rem 1.2rem;
  margin:.2rem 0 1.1rem; font-size:.83rem; }}
.specs dt {{ color:var(--dim); font-size:.7rem; text-transform:uppercase;
  letter-spacing:.05em; padding-top:.22rem; white-space:nowrap; }}
.specs dd {{ margin:0; }}
/* Content that is measured but not yet drawn. Dashed on purpose: this is the
   one place on the site where "provisional" is the correct reading of a
   border, everywhere else it would be lying about finished numbers. */
.todo {{ border:1px dashed var(--line); border-radius:10px;
  padding:1rem 1.2rem; color:var(--dim); font-size:.83rem; max-width:70ch;
  line-height:1.6; }}
.todo strong {{ color:var(--fg); font-weight:600; }}
.mhead {{ display:flex; align-items:baseline; gap:.55rem; flex-wrap:wrap;
  margin:2.75rem 0 .2rem; }}
.mhead h2 {{ margin:0; }}

/* Dashboard controls. Chips rather than a multi-select because a checkbox
   shows its state without being opened, and the whole point of this page is
   seeing what is currently included. */
.controls {{ display:flex; flex-wrap:wrap; gap:1.4rem 2rem; margin:1.4rem 0 1rem;
  padding:1rem 1.2rem; background:var(--card); border:1px solid var(--line);
  border-radius:10px; }}
.ctl {{ display:flex; flex-direction:column; gap:.45rem; font-size:.72rem;
  text-transform:uppercase; letter-spacing:.07em; color:var(--dim); }}
.ctl select {{ font:inherit; font-size:.85rem; text-transform:none;
  letter-spacing:0; color:var(--fg); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:.35rem .5rem; }}
.chips {{ display:flex; flex-wrap:wrap; gap:.4rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; cursor:pointer;
  font-size:.78rem; text-transform:none; letter-spacing:0; color:var(--fg);
  border:1px solid var(--line); border-radius:999px; padding:.22rem .7rem; }}
.chip:hover {{ border-color:var(--accent); }}
.chip input {{ margin:0; accent-color:var(--accent); }}
/* Hidden series keep their key visible but dimmed: which machines exist is
   part of the answer, and removing the key would hide that one was excluded. */
.legend-row .key {{ transition:opacity .12s; }}
/* Collapsible reference tables. Closed by default: these are what a claim can
   be checked against, not what a reader came for. <details> rather than a
   scripted toggle so they still open with JavaScript off, like everything here.
   
   The disclosure triangle alone was too quiet -- a small glyph beside a heading
   reads as decoration. The word show/hide on the right says what it is, and the
   whole bar takes a border and a hover so it looks like a control rather than a
   title that happens to move. */
.fold {{ margin-top:2.2rem; border:1px solid var(--line); border-radius:10px;
  background:var(--card); }}
.fold summary {{ cursor:pointer; font-size:1.05rem; letter-spacing:-.01em;
  list-style:none; display:flex; align-items:center; gap:.55rem;
  padding:.75rem 1.1rem; user-select:none; }}
.fold summary::-webkit-details-marker {{ display:none; }}
.fold summary::before {{ content:"▶"; color:var(--accent); font-size:.7em;
  transition:transform .15s; flex:none; }}
.fold[open] summary::before {{ transform:rotate(90deg); }}
.fold summary::after {{ content:"show"; margin-left:auto; font-size:.7rem;
  text-transform:uppercase; letter-spacing:.09em; color:var(--accent);
  border:1px solid var(--line); border-radius:999px; padding:.15rem .6rem; }}
.fold[open] summary::after {{ content:"hide"; }}
.fold summary:hover {{ color:var(--accent); }}
.fold summary:hover::after {{ border-color:var(--accent); background:var(--tag); }}
.fold summary .count {{ color:var(--dim); font-size:.78rem; font-weight:400; }}
.fold[open] summary {{ border-bottom:1px solid var(--line); }}
.fold figure {{ margin:.9rem 1.1rem 1.1rem; }}
.fold > p {{ margin:.9rem 1.1rem 1.1rem; }}
</style></head><body><div class="wrap">

<header class="head">
<div>
<h1>{heading}</h1>
<p class="lede">{lede}</p>
{strip}
</div>
{identity(logo_uri)}
</header>
{nav(here)}
{body}

<footer>
{footer}
</footer>
</div></body></html>
"""


def index_body(runs: list, specs: dict | None = None,
               curves: dict | None = None) -> str:
    """The comparison page: specs, runs, energy tables and charts."""
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
    # Runs that actually trained the model to target. A three-epoch run has a
    # node-wide efficiency too, but almost all of its energy is startup
    # amortised over no training, so it would sit on the chart looking like a
    # slow machine rather than a short run.
    # One bar per configuration -- best run of each (machine, rank count) --
    # rather than one per run. Repeats added nothing a reader could act on: two
    # Aurora 12-rank bars 3% apart, two Polaris ones the same.
    #
    # Deliberately NOT one bar per machine. Aurora's 1-rank run at 6.1
    # samples/J against its 12-rank at 66.9 is the point this chart makes, and
    # the caption below and the note under the energy tables both say so; a
    # best-per-machine rule would delete the finding and leave the prose
    # claiming it. Rank count is a configuration, not a repeat.
    best: dict = {}
    for r in runs:
        eff = node_efficiency(r)
        if not eff or not r.get("tta_s"):
            continue
        key = (r["machine"], r["world_size"])
        if key not in best or eff > best[key]["eff"]:
            best[key] = {
                "machine": r["machine"],
                "ranks": r["world_size"],
                "eff": eff,
                "samples": r["samples_global"],
                "joules": r["power_joules_total"],
            }
    eff_rows = list(best.values())
    eff_svg = efficiency_chart(eff_rows)
    eff_section = (
        "<h2>Energy per sample</h2>" + eff_svg
        + '<p class="fineprint">Whole-job samples over whole-node joules — the '
          'one energy figure on this page that compares across machines, since '
          'both sides cover a node. Filling the node matters far more than '
          'which node it is: the same Aurora hardware is an order of magnitude '
          'less efficient at one rank than at twelve, while the gap between '
          'two fully subscribed machines is well under 2×. Runs that never '
          'reached the accuracy target are left out — their energy is mostly '
          'startup amortised over no training.</p>'
        if eff_svg else ""
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
    # The cards cover machines with runs. Systems that are targeted but not yet
    # measured have no card and no row here, so the pointer is what keeps them
    # reachable -- "what is being compared" is still part of the answer, it just
    # lives on the page that describes the hardware now.
    spec_link = (
        '<p class="fineprint">Node specifications for every targeted system, '
        'including ones with no runs yet, are on the '
        '<a href="power.html">power profiles</a> page.</p>'
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
            machine_tag(r.get("machine")),
            num(r.get("nodes")), num(r.get("ranks")),
            html.escape(str(r.get("precision") or "—")),
            num(r.get("global_batch")),
            num(r.get("steps")) + flag,
            html.escape(str(r.get("epochs") or "—")),
            num(r.get("samples_per_s")),
            num(r.get("median_step_ms"), 1),
            num(r.get("best_top1"), 3),
            num(r.get("tta_s"), 1),
            # The unit rides in the cell, not just the header. The column to
            # its left is Best top-1, a fraction -- 0.917 meaning 91.7% -- and a
            # reader carrying that convention two columns right turns 1.83 into
            # 183% and reports a machine running past its own peak. That
            # happened. A three-character suffix ends it.
            (f'{r["mfu_pct"]:.2f}%' if r.get("mfu_pct") is not None else "—"),
        ])

    energy_rows = []
    for r in runs:
        if not r.get("joules"):
            continue
        energy_rows.append([
            when_cell(r),
            machine_tag(r.get("machine")),
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
            machine_tag(r.get("machine")),
            num(r.get("power_devices")),
            num(r.get("power_devices_idle")),
            num(r.get("power_joules_total")),
            num(r.get("power_joules_idle")),
            num(r.get("power_idle_pct"), 1),
            f'<strong>{eff:,.1f}</strong>' if eff else "—",
        ])

    return f"""
<h2>Machines</h2>
<div class="cards">{cards}</div>
{spec_link}

{tta_section}

<details class="fold">
<summary>Runs <span class="count">({len(run_rows)} run{"" if len(run_rows) == 1 else "s"} on disk)</span></summary>
{table(
    ["When","Machine","Nodes","Ranks","Prec","Global BS","Steps","Epochs",
     "Samples/s","Step ms","Best top-1","TTA s","MFU"],
    run_rows,
    "Every complete run on disk, oldest first. MFU is reported for diagnosis, "
    "not as an efficiency claim — this workload runs at 0.5–1.4% of peak, so it "
    "largely measures kernel-launch overhead rather than the accelerator.",
)}
</details>

{tail_section}

{eff_section}

<details class="fold">
<summary>Energy — per rank <span class="count">({len(energy_rows)} metered run{"" if len(energy_rows) == 1 else "s"})</span></summary>
{table(
    ["When","Machine","Ranks","Avg W","Joules","Samples","Samples/J","J to acc","Scope"],
    energy_rows,
    SCOPE_WARNING,
)}
</details>

<details class="fold">
<summary>Energy — per node <span class="count">({len(node_rows)} metered run{"" if len(node_rows) == 1 else "s"})</span></summary>
{table(
    ["When","Machine","Devices","Idle","Node J","Idle J","Idle %","Samples/J"],
    node_rows,
    "Node-wide sampling covers every accelerator, including devices no rank "
    "bound to. Samples/J here is the cross-machine comparison: whole-job "
    "samples over whole-node joules, the unit an allocation is billed in.",
)}
{equal_work_note(runs)}
</details>

<div class="note">A device left idle still draws power. A single-rank run on a
12-tile Aurora node spent over 90% of node energy on tiles nobody used — the
per-rank column cannot see that, which is the reason both tables exist.</div>

{takeaway(machine_stats)}

<h2>Glossary</h2>
{legend()}
"""


def load_aiperf(results_dir: str) -> list:
    """Every AIPerf sweep under results/aiperf/, newest first.

    A sweep is a directory of concurrency levels plus the summary.json that
    summarize_aiperf.py wrote. Machine comes from the directory prefix, matching
    timeline_counts(); the served model and sequence lengths come from one
    level's own export, because a sweep that changed model midway is not a sweep
    and the page should not average over one.
    """
    seen: set = set()
    sweeps = []
    # Newest first, one per (machine, model, tensor parallelism). Directory names
    # are timestamped, so reverse order puts the current run ahead of the ones it
    # supersedes -- a rerun of the same configuration is a rerun, and two of them
    # would draw one machine twice and read as two.
    #
    # Keyed on the configuration rather than the machine, which is what it was
    # until a second model arrived. Machine alone meant a gemma run silently
    # replaced the 8B sweep instead of joining it, deleting the comparison the
    # page was built on. Superseded runs stay in results/ either way.
    #
    # Single-level runs are kept, unlike before. Six-level sweeps carry the
    # concurrency charts; a one-level run cannot, but it still says what a model
    # costs on a machine, and dropping it silently was how the first 70B and
    # gemma results would have vanished from a page asked to show them.
    for path in sorted(Path(results_dir).glob("aiperf/*/summary.json"), reverse=True):
        try:
            blob = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        rows = [r for r in (blob.get("runs") or []) if r.get("concurrency")]
        if not rows:
            continue
        cfg = _aiperf_config(path.parent)
        machine = path.parent.name.split("-")[0]
        key = (machine, cfg.get("model"), cfg.get("tp"))
        if key in seen:
            continue
        seen.add(key)
        sweeps.append({
            "name": path.parent.name,
            "machine": path.parent.name.split("-")[0],
            "idle_w": blob.get("idle_w"),
            "rows": [_with_latency_split(r)
                     for r in sorted(rows, key=lambda r: r["concurrency"])],
            **_aiperf_config(path.parent),
        })
    return sweeps


def _with_latency_split(row: dict) -> dict:
    """Add the prefill/decode split of a request, derived not measured.

    TTFT + ITL x (tokens - 1) is the whole of a request's latency: the wait for
    the first token, then one inter-token gap for each one after it. It holds
    exactly -- checked against AIPerf's own request_latency across every level
    of every sweep here, and it reconstructs it to 0.00%.

    Derived at read time rather than written by summarize_aiperf because it adds
    no measurement: both terms are already in the row, and a stored copy would
    be a third number that can disagree with the two it came from.
    """
    ttft, itl = row.get("ttft_ms"), row.get("itl_ms")
    osl = row.get("osl_avg") or row.get("requested_osl")
    if None in (ttft, itl, osl) or osl < 2:
        return row
    latency = ttft + itl * (osl - 1)
    return {**row,
            "request_latency_ms": latency,
            "prefill_share_pct": (ttft / latency * 100) if latency else None}


def _aiperf_config(sweep_dir: Path) -> dict:
    """Model, ISL and OSL as the run itself recorded them.

    Read from an export rather than from the submit script's defaults, so a
    sweep launched with MODEL= overridden describes what it actually served.
    """
    for export in sorted(sweep_dir.glob("c*/profile_export_aiperf.json")):
        try:
            cfg = (json.loads(export.read_text(encoding="utf-8-sig"))
                   .get("input_config") or {})
        except (OSError, ValueError):
            continue
        datasets = cfg.get("datasets") or [{}]
        prompts = (datasets[0].get("prompts") or {}) if datasets else {}
        return {
            "model": (cfg.get("tokenizer") or {}).get("name"),
            "isl": (prompts.get("isl") or {}).get("mean"),
            "osl": (prompts.get("osl") or {}).get("mean"),
            "streaming": (cfg.get("endpoint") or {}).get("streaming"),
            "tp": _tensor_parallel(sweep_dir),
        }
    return {"model": None, "isl": None, "osl": None, "streaming": None,
            "tp": _tensor_parallel(sweep_dir)}


def _tensor_parallel(sweep_dir: Path):
    """Devices the server was given, from the run's own provenance.

    Not derivable from the AIPerf export -- AIPerf is a client and never learns
    how the server was sharded. run_meta.json records it, which is why the
    sweeps started writing one. None for runs that predate it.
    """
    path = sweep_dir / "run_meta.json"
    if path.exists():
        try:
            tp = json.loads(path.read_text(encoding="utf-8-sig")).get("tensor_parallel")
            if tp:
                return tp
        except (OSError, ValueError):
            pass
    # Runs predating run_meta.json still recorded the devices the server was
    # given: the sidecar was told them as --bound-devices, and wrote them down.
    # Recovered rather than assumed -- the alternative is a dash in the table for
    # the sweep the whole page was originally built on.
    for side in sorted(sweep_dir.glob("c*/sidecar_power.json")):
        try:
            bound = json.loads(side.read_text(encoding="utf-8-sig")).get("bound_devices")
        except (OSError, ValueError):
            continue
        if bound:
            return len(bound)
    return None


def inference_section(sweeps: list) -> str:
    """The inference sweeps, kept apart from the training numbers on purpose.

    Same node, same counters, same sampler as the training profiles above --
    and a different benchmark. Tokens per joule and samples per joule answer
    different questions and share no denominator, so they share a page and
    nothing else.
    """
    if not sweeps:
        return ""

    blocks = ""
    # Only multi-level runs get a section: the chart and the takeaway both
    # describe a curve, and a single point has none. Those runs are in the model
    # table above instead, which is the view they can actually support.
    for sweep in [s for s in sweeps if len(s["rows"]) >= 3]:
        rows = sweep["rows"]
        first, last = rows[0], rows[-1]
        chart = inference_chart(rows)
        served = " · ".join(
            bit for bit in (
                f"<strong>{html.escape(str(sweep['model']))}</strong>" if sweep["model"] else None,
                f"ISL {sweep['isl']:,.0f}" if sweep["isl"] else None,
                f"OSL {sweep['osl']:,.0f}" if sweep["osl"] else None,
                f"idle floor {sweep['idle_w']:,.0f} W" if sweep["idle_w"] else None,
            ) if bit
        )
        blocks += f"""
<div class="mhead"><h2>{html.escape(sweep["machine"])} — {html.escape((sweep["model"] or "inference").split("/")[-1])}</h2>
<span class="key s{SERIES_SLOT.get(sweep["machine"], 8)}"><span class="sw"></span></span></div>
<p class="workload">{served}</p>
{chart}
{inference_legend()}
{_inference_takeaway(first, last)}"""
    # One chart per model, because colour encodes the machine and has nothing
    # left for a model. Only models at least two machines swept: a single line
    # is not a comparison, and the heading would promise one.
    by_model: dict = defaultdict(list)
    for s in sweeps:
        if len(s["rows"]) >= 3:
            by_model[s["model"]].append(s)
    compare = ""
    for model, group in sorted(by_model.items()):
        if len({s["machine"] for s in group}) < 2:
            continue
        name = html.escape((model or "?").split("/")[-1])
        compare += f"""
<h2>Which machine serves more per joule — {name}</h2>
{efficiency_compare_chart(group)}
{efficiency_compare_legend(group)}
{_compare_takeaway(group)}"""

    return f"""
<h2>Serving, not training</h2>
<p class="fineprint">The same node, the same counters and the same sampler as the
profiles above, measuring a different thing: vLLM answering requests instead of a
model being trained. Tokens per joule and samples per joule share no denominator
and never belong in one table, which is why this sits beside the training
profiles rather than among them. AIPerf collects power through DCGM, pynvml and
amdsmi, so on NVIDIA it measures its own; on Intel it can read nothing, and the
energy comes instead from <code>analysis/power_sidecar.py</code> sampling the
same hwmon counters the training runs use, from beside the run. The Src column
on each table says which.</p>
<p class="fineprint">These sections carry the charts and what they mean. Every
concurrency level of every sweep, in numbers and filterable by machine and
model, is on the <a href="dashboard.html">dashboard</a> — it was printed under
each section too, and one table maintained in two places is one table that
disagrees with itself.</p>
{_xcheck_note(sweeps)}
{model_table(sweeps)}
{compare}
{blocks}"""


def _inference_takeaway(first: dict, last: dict) -> str:
    """The throughput-versus-power sentence, derived from the two end rows."""
    tok = first.get("out_tok_per_s"), last.get("out_tok_per_s")
    watts = first.get("dynamic_w"), last.get("dynamic_w")
    joules = first.get("total_energy_j"), last.get("total_energy_j")
    if not all(tok) or not all(watts) or not all(joules):
        return ""
    return (
        f'<p class="takeaway">From {first["concurrency"]} concurrent request to '
        f'{last["concurrency"]}, output throughput rises '
        f'<strong>{tok[1] / tok[0]:.1f}×</strong> while dynamic power goes '
        f'{watts[0]:,.0f} W to {watts[1]:,.0f} W — '
        f'<strong>{watts[1] / watts[0]:.2f}×</strong>. The same work costs '
        f'{joules[0] / joules[1]:.1f}× less energy, almost entirely by finishing '
        f'sooner rather than by drawing less.</p>'
        f'<p class="fineprint">Paid for in latency: time to first token goes '
        f'{first["ttft_ms"]:,.0f} ms to {last["ttft_ms"]:,.0f} ms and inter-token '
        f'latency {first["itl_ms"]:,.1f} ms to {last["itl_ms"]:,.1f} ms. Tok/J is '
        f'the figure to budget with, since an allocation bills for the node either '
        f'way; Tok/J dyn is what the silicon did. Dynamic power is a small '
        f'difference between two large numbers, so it holds only while the floor '
        f'comes from this node in this job — which is why one is sampled per run '
        f'rather than carried over. Given that, it is the steadier of the two: two '
        f'Aurora runs on different nodes differed 4.5% in node draw and 0.07% in '
        f'dynamic, because a node that idles high runs high under load too and the '
        f'offset cancels.</p>'
    )


def model_table(sweeps: list) -> str:
    """Every configuration at one concurrency, so models compare directly.

    The concurrency charts need a sweep; most of these runs are a single level,
    because a 70B takes two minutes per level and the queue allows an hour. They
    still answer the question the sweeps cannot -- what a model costs on a
    machine -- and one shared concurrency is enough to ask it.

    W/dev is the column worth reading. Dynamic watts alone conflate "the model
    is bigger" with "more devices are working"; divided by the devices actually
    serving, it says how hard each one was driven, which is the only way to see
    that Aurora's tiles are busier on an 8B at TP=1 than on a 27B at TP=4.
    """
    levels = [{r["concurrency"] for r in s["rows"]} for s in sweeps]
    shared = set.intersection(*levels) if levels else set()
    if not shared:
        return ""
    conc = 4 if 4 in shared else max(shared)

    rows = []
    for s in sorted(sweeps, key=lambda s: (s["machine"], s["model"] or "")):
        r = next((x for x in s["rows"] if x["concurrency"] == conc), None)
        if not r:
            continue
        tp = s.get("tp")
        idle = r.get("idle_devices")
        total = (tp + idle) if (tp is not None and idle is not None) else None
        per_dev = (r["dynamic_w"] / tp) if (r.get("dynamic_w") and tp) else None
        rows.append([
            f'<span class="m">{html.escape((s["model"] or "?").split("/")[-1])}</span>',
            machine_tag(s["machine"]),
            num(tp) if tp else "—",
            f"{idle} of {total}" if total is not None else "—",
            num(r.get("out_tok_per_s"), 1),
            num(r.get("itl_ms"), 1),
            num(r.get("dynamic_w")),
            num(per_dev),
            f'<strong>{r["tok_per_joule"]:.3f}</strong>' if r.get("tok_per_joule") else "—",
            num(r.get("tok_per_joule_dynamic"), 2),
        ])
    if len(rows) < 2:
        return ""
    return f"""
<h2>What the model costs</h2>
{table(
    ["Model", "Machine", "TP", "Idle dev", "Out tok/s", "ITL ms", "Dyn W",
     "W/dev", "Tok/J", "Tok/J dyn"],
    rows,
    f"Every row at concurrency {conc}, the one level all of these runs share. "
    "TP is how many accelerators served the model; Idle dev is how many on the "
    "node did not. W/dev divides dynamic power by the serving devices, which "
    "separates a bigger model from a wider one — the two move together in every "
    "other column here.",
)}
{_model_takeaway(sweeps, conc)}"""


def _model_takeaway(sweeps: list, conc: int) -> str:
    """The like-for-like pair, where one exists.

    Two machines running the same model at the same TP is the only comparison
    here that holds anything constant, so it is the only one stated as a result.
    """
    by_model: dict = defaultdict(list)
    for s in sweeps:
        r = next((x for x in s["rows"] if x["concurrency"] == conc), None)
        if r and r.get("tok_per_joule") and s.get("tp"):
            by_model[(s["model"], s["tp"])].append((s["machine"], r))
    pairs = [(k, v) for k, v in by_model.items() if len(v) == 2]
    if not pairs:
        return ""
    (model, tp), members = pairs[0]
    members.sort(key=lambda kv: -kv[1]["tok_per_joule"])
    (fast, fr), (slow, sr) = members
    name = html.escape((model or "?").split("/")[-1])
    return (
        f'<p class="takeaway"><strong>{name}</strong> at TP={tp} is the one pair '
        f'here holding model, sharding and concurrency constant. '
        f'<strong>{html.escape(fast)}</strong> delivers '
        f'{fr["tok_per_joule"] / sr["tok_per_joule"]:.1f}× the tokens per joule '
        f'of {html.escape(slow)} — and drives each device '
        f'{(fr["dynamic_w"] / tp) / (sr["dynamic_w"] / tp):.1f}× harder '
        f'({fr["dynamic_w"] / tp:,.0f} W against {sr["dynamic_w"] / tp:,.0f} W), '
        f'while finishing a token in {fr["itl_ms"]:.0f} ms against '
        f'{sr["itl_ms"]:.0f} ms.</p>'
        f'<p class="fineprint">These are single-level runs, not sweeps — one '
        f'concurrency each, so they carry no curve and no repeat. Read them as '
        f'the cost of a model on a machine, not as a measurement of either '
        f'machine\'s best. The vLLM versions still differ by machine, and every '
        f'run is eager-mode.</p>'
    )


def _xcheck_note(sweeps: list) -> str:
    """How closely the two instruments agreed, where both ran.

    Worth stating on the page rather than only in a table column, because it is
    the only evidence the Aurora rows have. AIPerf cannot read Intel counters,
    so those joules come from the sidecar with nothing to check them against;
    the check has to happen on NVIDIA, where both instruments work, and then be
    carried across as an argument about the method rather than the machine.
    """
    deltas = [
        abs(r["sidecar_delta_pct"]) for s in sweeps for r in s["rows"]
        if r.get("sidecar_delta_pct") is not None
    ]
    if not deltas:
        return ""
    machines = sorted({
        s["machine"] for s in sweeps
        if any(r.get("sidecar_delta_pct") is not None for r in s["rows"])
    })
    return (
        f'<p class="fineprint">On {html.escape(", ".join(machines))} both '
        f'instruments ran at once and agreed to within '
        f'<strong>{max(deltas):.1f}%</strong> across {len(deltas)} levels — '
        f"AIPerf's own NVML collector against the sidecar sampling the same "
        f"GPUs. That is the only check the method gets: the Intel rows have no "
        f"second instrument available, so their joules rest on this agreement "
        f"holding somewhere it could be tested.</p>"
    )


def _compare_takeaway(sweeps: list) -> str:
    """Which machine wins, on each denominator, at the top of the sweep.

    Derived because the two answers disagree, and the disagreement is the
    result. Written by hand it would need rewriting every time a sweep is added,
    and a stale sentence naming the wrong winner is worse than no sentence.
    """
    tops = []
    for sweep in sweeps:
        top = sorted(sweep["rows"], key=lambda r: r["concurrency"])[-1]
        if top.get("tok_per_joule") and top.get("tok_per_joule_dynamic"):
            tops.append((sweep["machine"], top))
    if len(tops) < 2:
        return ""

    levels = {t["concurrency"] for _, t in tops}
    at = (f"At concurrency {tops[0][1]['concurrency']}"
          if len(levels) == 1 else "At the top of each sweep")
    by_total = sorted(tops, key=lambda kv: -kv[1]["tok_per_joule"])
    by_dyn = sorted(tops, key=lambda kv: -kv[1]["tok_per_joule_dynamic"])
    tw, tl = by_total[0], by_total[-1]
    dw, dl = by_dyn[0], by_dyn[-1]
    total_ratio = tw[1]["tok_per_joule"] / tl[1]["tok_per_joule"]
    dyn_ratio = dw[1]["tok_per_joule_dynamic"] / dl[1]["tok_per_joule_dynamic"]

    if tw[0] == dw[0]:
        body = (
            f"{at}, <strong>{html.escape(tw[0])}</strong> leads on both measures — "
            f"{total_ratio:.1f}× the tokens per joule of {html.escape(tl[0])}, and "
            f"{dyn_ratio:.2f}× on dynamic energy. No tradeoff to make."
        )
    else:
        body = (
            f"The two measures disagree, and that is the result. {at}, "
            f"<strong>{html.escape(tw[0])}</strong> delivers {total_ratio:.1f}× the "
            f"tokens per joule of {html.escape(tl[0])} — but per joule of "
            f"<em>work</em>, <strong>{html.escape(dw[0])}</strong> is "
            f"{dyn_ratio:.2f}× ahead. The more efficient accelerator loses, because "
            f"its node spends so much more simply being switched on."
        )
    return (
        f'<p class="takeaway">{body}</p>'
        '<p class="fineprint">Both axes are logarithmic, so the vertical gap between '
        'two lines is their ratio — parallel lines mean a constant advantage across '
        'the whole range rather than a growing one. Solid is tokens per joule of '
        'node energy, what an allocation bills for. Dashed is per joule of dynamic '
        'energy, node draw above the idle floor measured on that node before its '
        'server started. The machines did not run the same vLLM: versions are '
        'recorded in each sweep\'s run_meta.json, and both ran with '
        '<code>--enforce-eager</code>.</p>'
    )



# Metrics the dashboard offers, each pre-rendered as its own complete chart.
# Switching metric shows a different SVG rather than redrawing one, because the
# axes differ by orders of magnitude between them -- tokens/joule sits near 0.1
# and dynamic watts near 400, and one axis cannot serve both.
DASHBOARD_METRICS = [
    ("tok_per_joule", "tokens per joule", True),
    ("tok_per_joule_dynamic", "tokens per joule (dynamic)", True),
    ("out_tok_per_s", "output tokens per second", True),
    ("itl_ms", "inter-token latency (ms)", False),
    ("ttft_ms", "time to first token (ms)", False),
    # The two halves of a request, and how the split moves under load. Drawn as
    # shares rather than stacked absolutes on purpose: prefill is 1-16% of a
    # request here, and a stack of 1% on 99% shows a line of colour and hides
    # the thing worth seeing. The share is the finding; request latency beside
    # it is what the share is a share of.
    ("request_latency_ms", "request latency (ms)", False),
    ("prefill_share_pct", "prefill share of request latency (%)", False),
    ("dynamic_w", "dynamic power (W)", False),
]

# Throughput against power, which is a different chart shape: not a metric over
# concurrency but a trajectory through the two. Two x-axis choices, because
# which power you divide by is the entire disagreement between these machines --
# dynamic compares silicon, node compares what an allocation is billed.
DASHBOARD_XY = [
    ("xy_dynamic_w", "dynamic power (W)", "dynamic_w"),
    ("xy_avg_gpu_w", "node power (W)", "avg_gpu_w"),
]


def dashboard_body(sweeps: list) -> str:
    """Every inference configuration, filterable in the browser.

    The filtering is presentation only: every series and every row is in the
    page already, tagged with its machine and model, and the script sets
    display. Nothing is fetched, nothing is computed client-side, and with
    scripting off the page shows the complete set -- which is the same thing
    the other two pages show.
    """
    if not sweeps:
        return "<p class=\"fineprint\">No inference sweeps in results/aiperf yet.</p>"

    machines = sorted({s["machine"] for s in sweeps})
    models = sorted({(s["model"] or "?").split("/")[-1] for s in sweeps})
    # Sorted numerically, not as the strings they become in the DOM: TP 10 has
    # to sit after TP 8 rather than between 1 and 4.
    tps = [str(t) for t in sorted({s.get("tp") for s in sweeps if s.get("tp")})]

    def boxes(kind, values):
        # Machines are labelled in their own colour, models plainly -- a model
        # has no colour anywhere on this site, and inventing one here would
        # imply a mapping the charts do not share.
        label = machine_tag if kind == "machine" else html.escape
        if kind == "tp":
            label = lambda v: f"TP={html.escape(v)}"   # noqa: E731
        return "".join(
            f'<label class="chip"><input type="checkbox" data-filter="{kind}" '
            f'value="{html.escape(v)}" checked> {label(v)}</label>'
            for v in values
        )

    options = (
        '<optgroup label="against concurrency">'
        + "".join(
            f'<option value="{key}"{" selected" if i == 0 else ""}>{html.escape(label)}</option>'
            for i, (key, label, _) in enumerate(DASHBOARD_METRICS)
        )
        + '</optgroup><optgroup label="throughput against power">'
        + "".join(
            f'<option value="{key}">{html.escape(label)}</option>'
            for key, label, _ in DASHBOARD_XY
        )
        + "</optgroup>"
    )
    charts = "".join(
        f'<div class="chartwrap" data-metric="{key}"{"" if i == 0 else " hidden"}>'
        f"{dashboard_chart(sweeps, key, label, log_y)}</div>"
        for i, (key, label, log_y) in enumerate(DASHBOARD_METRICS)
    ) + "".join(
        f'<div class="chartwrap" data-metric="{key}" hidden>'
        f"{power_throughput_chart(sweeps, col, label)}</div>"
        for key, label, col in DASHBOARD_XY
    )

    rows = []
    for sweep in sorted(sweeps, key=lambda s: (s["machine"], s["model"] or "",
                                               s.get("tp") or 0)):
        model = (sweep["model"] or "?").split("/")[-1]
        for r in sorted(sweep["rows"], key=lambda r: r["concurrency"]):
            rows.append((sweep["machine"], model, str(sweep.get("tp") or ""), [
                machine_tag(sweep["machine"]),
                html.escape(model),
                num(sweep.get("tp")) if sweep.get("tp") else "—",
                num(r.get("concurrency")),
                num(r.get("out_tok_per_s"), 1),
                num(r.get("ttft_ms")),
                num(r.get("itl_ms"), 1),
                num(r.get("dynamic_w")),
                num(r.get("total_energy_j")),
                f'<strong>{r["tok_per_joule"]:.3f}</strong>' if r.get("tok_per_joule") else "—",
                num(r.get("tok_per_joule_dynamic"), 2),
            ]))
    head = "".join(f"<th>{h}</th>" for h in (
        "Machine", "Model", "TP", "Conc", "Out tok/s", "TTFT ms", "ITL ms",
        "Dyn W", "Joules", "Tok/J", "Tok/J dyn"))
    body = "".join(
        f'<tr data-machine="{html.escape(m)}" data-model="{html.escape(mo)}" '
        f'data-tp="{html.escape(tp)}">'
        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        for m, mo, tp, cells in rows
    )

    return f"""
<div class="controls">
  <label class="ctl">Metric
    <select id="metric">{options}</select>
  </label>
  <div class="ctl"><span>Machines</span><div class="chips">{boxes("machine", machines)}</div></div>
  <div class="ctl"><span>Models</span><div class="chips">{boxes("model", models)}</div></div>
  <div class="ctl"><span>Tensor parallel</span><div class="chips">{boxes("tp", tps)}</div></div>
</div>
{charts}
{dashboard_legend(sweeps)}
<p class="fineprint" id="empty" hidden>Nothing selected — tick a machine and a model.</p>

<details class="fold">
<summary>Every row <span class="count" id="rowcount"></span></summary>
<figure><div class="scroll"><table id="rows"><thead><tr>{head}</tr></thead>
<tbody>{body}</tbody></table></div>
<figcaption>Every concurrency level of every sweep, filtered by the same
controls. Joules are comparable only between rows that ran the same request
count — see each sweep's run_meta.json.</figcaption></figure>
</details>

<script>
(function () {{
  var metric = document.getElementById('metric');
  function selected(kind) {{
    var out = {{}};
    document.querySelectorAll('[data-filter="' + kind + '"]').forEach(function (b) {{
      if (b.checked) out[b.value] = true;
    }});
    return out;
  }}
  function apply() {{
    var m = selected('machine'), mo = selected('model'), tp = selected('tp'), shown = 0;
    // A sweep with no recorded TP is never hidden by the TP filter -- it has
    // no chip to tick, and filtering it out would make it unreachable.
    function keep(el) {{
      var t = el.getAttribute('data-tp');
      return m[el.getAttribute('data-machine')] && mo[el.getAttribute('data-model')]
             && (!t || tp[t]);
    }}
    document.querySelectorAll('.chartwrap').forEach(function (w) {{
      w.hidden = w.getAttribute('data-metric') !== metric.value;
    }});
    document.querySelectorAll('.series').forEach(function (g) {{
      var on = keep(g);
      g.style.display = on ? '' : 'none';
      if (on) shown++;
    }});
    var rows = document.querySelectorAll('#rows tbody tr'), visible = 0;
    rows.forEach(function (tr) {{
      var on = keep(tr);
      tr.style.display = on ? '' : 'none';
      if (on) visible++;
    }});
    // Shown on the summary because that is the only part visible when the
    // section is shut, and "how much is in there" is the question a collapsed
    // block raises.
    document.getElementById('rowcount').textContent =
      visible === rows.length ? '(' + rows.length + ')'
                              : '(' + visible + ' of ' + rows.length + ')';
    document.querySelectorAll('.legend-row .key').forEach(function (k) {{
      k.style.opacity = keep(k) ? '' : '0.3';
    }});
    document.getElementById('empty').hidden = shown > 0;
  }}
  metric.addEventListener('change', apply);
  document.querySelectorAll('[data-filter]').forEach(function (b) {{
    b.addEventListener('change', apply);
  }});
  apply();
}})();
</script>"""

def timeline_counts(results_dir: str) -> dict:
    """How many node power timelines are on disk, per machine.

    Counted from filenames -- power.py writes machine_runid_host_power.json and
    no machine name contains an underscore -- rather than by parsing the files,
    because one Aurora timeline is 1.5 MB and the only question here is how many
    there are.
    """
    counts: dict = defaultdict(int)
    for path in Path(results_dir).glob("power/*_power.json"):
        counts[path.name.split("_")[0]] += 1
    return dict(counts)


def power_state(machine: str, runs: list, timelines: int) -> tuple:
    """(tag, sentence) describing what power data exists for this machine.

    Derived from what is on disk, never asserted. "No timeline" has more than
    one cause and they are not interchangeable: a machine that reports energy
    counters but was run without the sampler is waiting on a flag, while Crux
    exposes no accelerator counter at all and always will. Writing one
    explanation for both would put a guess on a public page.
    """
    mine = [r for r in runs if r.get("machine") == machine]
    if timelines:
        return "", (
            f"<strong>{timelines} node timeline{'s' if timelines != 1 else ''} "
            f"recorded</strong>, in <code>results/power/</code>. Per-device watts "
            f"against wall-clock, with epoch and eval boundaries marked, will be "
            f"drawn here."
        )
    if not mine:
        return ' <span class="tag">no runs yet</span>', (
            "<strong>Not yet run.</strong> Targeted by the benchmark, with no "
            "results on disk — the specs above are what it will be measured on."
        )
    if any(r.get("joules") or r.get("power_joules_total") for r in mine):
        return ' <span class="tag">no timeline</span>', (
            f"<strong>{len(mine)} run(s) with energy counters, no node "
            f"timeline.</strong> The counters bracket training regardless of the "
            f"sampler, so these ran with <code>--power-interval 0</code> or "
            f"predate node sampling. Their totals are on the index page."
        )
    return ' <span class="tag">no counter</span>', (
        f"<strong>{len(mine)} run(s), no energy measured.</strong> This machine "
        f"exposes no accelerator energy counter to read, so its rows carry timing "
        f"only and are absent from the energy tables rather than sitting in them "
        f"full of zeros."
    )


def machine_section(machine: str, spec: dict, tag: str, state: str) -> str:
    """One machine on the power page: its specs, then what is coming.

    This is the only place the specs are rendered. They read as a record per
    machine rather than a row per machine, which is the shape a reader wants
    beside one system's power trace -- and it is the shape the index's table
    could never be, since a table compares and this describes.
    """
    fields = "".join(
        f"<dt>{html.escape(label)}</dt>"
        f"<dd>{html.escape(str(spec.get(key, '—')))}</dd>"
        for key, label in SPEC_COLUMNS
    )
    slot = SERIES_SLOT.get(machine, 8)
    return f"""
<div class="mhead">
<h2>{html.escape(machine)}</h2>
<span class="key s{slot}"><span class="sw"></span></span>{tag}
</div>
<dl class="specs">{fields}</dl>
<div class="todo">{state}</div>"""


def power_body(specs: dict, runs: list, timelines: dict, sweeps: list) -> str:
    """The power page: one section per targeted machine, in config order."""
    sections = ""
    for machine, spec in specs.items():
        if machine.startswith("_"):
            continue
        tag, state = power_state(machine, runs, timelines.get(machine, 0))
        sections += machine_section(machine, spec, tag, state)
    return f"""
<div class="note">Placeholders. The specs and the timeline counts are read from
the repo on every build; the profiles themselves are not drawn yet. Nothing on
this page is a measurement except the counts.</div>
<p class="fineprint">Specifications are per node — both the unit this benchmark
scales in and the unit an allocation is billed in. Where each figure came from
is recorded in the <code>_source</code> fields of
<code>configs/machines.json</code>; the measurements are on the
<a href="index.html">data and analysis</a> page.</p>
{sections}

{inference_section(sweeps)}

<h2>What a training profile will show</h2>
<p class="fineprint">One line per accelerator on the node, sampled by
<code>benchmark/power.py</code> at <code>--power-interval</code> seconds, plotted
as watts from consecutive energy-counter deltas. Devices no rank bound to get a
line too — on a single-rank Aurora node those flat lines are over 90% of the
node's energy, which is the fact the index page can only state as a percentage.
Aurora's whole-card counters are excluded, as they are from every total, because
they cover silicon the per-tile counters already report.</p>"""


def footer_text(runs: list, generated: str) -> str:
    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    return (
        f'Generated {generated} from {len(runs)} run(s) · machines: '
        f'{html.escape(", ".join(machines)) or "none"}<br>\n'
        "Rebuild both pages with <code>python analysis/build_site.py</code> after\n"
        "<code>git pull</code>. Numbers come from <code>results/*.json</code>; the\n"
        "pages are never edited by hand."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--out-dir", default="docs",
                    help="directory the pages are written into")
    ap.add_argument("--machines", default="./configs/machines.json",
                    help="node specs for the machines table; skipped if absent")
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not args.include_synthetic:
        runs = [r for r in runs if not r.get("synthetic")]
    if not runs:
        raise SystemExit(f"no complete runs found in {args.results_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logo = logo_data_uri(out_dir)
    # Optional: a checkout without the config still builds, just without the
    # specs table, rather than failing on a file that carries no measurements.
    specs_path = Path(args.machines)
    specs = json.loads(specs_path.read_text(encoding="utf-8")) if specs_path.exists() else {}
    curves = canonical_runs(args.results_dir)
    measured = {r.get("machine") for r in runs if r.get("machine")}
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = footer_text(runs, generated)
    sweeps = load_aiperf(args.results_dir)

    pages = {
        "index.html": shell(
            title="Power and Performance Across ALCF Machines",
            heading="Power and Performance Across ALCF Machines",
            lede="One portable benchmark, run identically on every ALCF system "
                 "and compared on throughput, time-to-accuracy and energy — down "
                 "to the accelerators nobody was using. One harness, one result "
                 "schema, one table.",
            strip=workload_strip(runs),
            body=index_body(runs, specs, curves),
            footer=footer,
            logo_uri=logo,
            here="index.html",
        ),
        "power.html": shell(
            title="Power Profiles — ALCF Machines",
            heading="Power Profiles",
            lede="What each machine actually draws while it trains, per node and "
                 "per accelerator. One section per system the benchmark targets.",
            body=power_body(specs, runs, timeline_counts(args.results_dir), sweeps),
            footer=footer,
            logo_uri=logo,
            here="power.html",
        ),
    }
    pages["dashboard.html"] = shell(
        title="ALCF Benchmark Dashboard",
        heading="Dashboard",
        lede="Every inference configuration measured so far, filtered in the "
             "browser. Pick a metric, then choose which machines and models to "
             "show. Nothing is fetched — the whole dataset is in this page.",
        body=dashboard_body(sweeps),
        footer=footer,
        logo_uri=logo,
        here="dashboard.html",
    )
    for name, text in pages.items():
        (out_dir / name).write_text(text, encoding="utf-8")

    # Pages runs Jekyll over the publishing folder unless told not to. Nothing
    # here starts with an underscore today, but the cost of being wrong later is
    # a page that silently stops updating, and the cost of the file is nothing.
    (out_dir / ".nojekyll").touch()

    # Said out loud because an inlined image is the one thing here that can bloat
    # the page, and a silently-missing logo otherwise looks like a CSS bug.
    print(f"logo: {'inlined, ' + str(len(logo) // 1024) + ' KB' if logo else 'none found'}")
    for name in pages:
        print(f"wrote {out_dir / name} — {(out_dir / name).stat().st_size // 1024} KB")
    print(f"{len(runs)} run(s), machines: {', '.join(sorted(measured))}")


if __name__ == "__main__":
    main()
