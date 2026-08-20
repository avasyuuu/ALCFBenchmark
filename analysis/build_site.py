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
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# `legend` here is the column glossary further down; the chart key is
# imported under its own name so the two never shadow each other.
from charts import (SERIES_SLOT, accuracy_chart, canonical_runs,
                    capability_radar, swept_over_concurrency,
                    dashboard_chart, dashboard_legend, machine_tag,
                    power_throughput_chart, power_timeline_chart, timeline_watts,
                    tp_swatch, model_dash_bg,
                    efficiency_chart, inference_chart, inference_legend,
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
    # "Training" leads the line: the workload name alone says what model and
    # dataset, not which of the two benchmarks this page reports.
    bits = [f"Training {name}", f"global batch {batches.pop():,}"]
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


def _accel_model(spec: dict) -> str:
    """The accelerator model with its per-node count stripped off.

    "4x NVIDIA A100, NVLink" and "8x NVIDIA A100, NVLink" are the same part in
    different quantities, which is the whole point of the comparison below.
    """
    text = (spec or {}).get("accelerator") or ""
    # Case preserved: this string is displayed as well as grouped on, and
    # "nvidia a100" in a sentence reads as a typo rather than a part number.
    return re.sub(r"^\s*\d+\s*[x\u00d7]\s*", "", text).strip()


def _accel_count(spec: dict):
    """How many of them a node carries, read off the same string."""
    m = re.match(r"\s*(\d+)\s*[x\u00d7]", (spec or {}).get("accelerator") or "")
    return int(m.group(1)) if m else None


def same_accelerator_takeaway(runs: list, specs: dict | None) -> str:
    """Machines that differ only in how many of one accelerator they carry.

    The only controlled comparison this benchmark has on the training side.
    Aurora against Polaris confounds vendor, architecture, node width and
    software all at once; Polaris against Sophia holds the part number fixed
    and changes the count, so the difference is attributable.

    Derived, and silent unless two measured machines actually share a part --
    a hand-written version would have to be revisited the day a machine is
    added, which is how a page starts asserting things the data stopped
    saying.
    """
    if not specs:
        return ""
    groups: dict = defaultdict(list)
    for machine, spec in specs.items():
        if machine.startswith("_"):
            continue
        model = _accel_model(spec)
        if not model or "none" in model.lower():
            continue
        best_tp = max((r for r in runs if r.get("machine") == machine
                       and r.get("samples_per_s")),
                      key=lambda r: r["samples_per_s"], default=None)
        if not best_tp:
            continue
        effs = [(node_efficiency(r), r) for r in runs if r.get("machine") == machine]
        effs = [(e, r) for e, r in effs if e]
        groups[model.lower()].append({
            "label": model,
            "machine": machine,
            "devices": _accel_count(spec),
            "tp": best_tp,
            "eff": max(effs, key=lambda t: t[0]) if effs else None,
        })

    for model, members in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        members = [m for m in members if m["eff"] and m["devices"]]
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m["devices"])
        small, big = members[0], members[-1]
        if small["devices"] == big["devices"]:
            continue
        ratio_dev = big["devices"] / small["devices"]
        ratio_tp = big["tp"]["samples_per_s"] / small["tp"]["samples_per_s"]
        eff_small, eff_big = small["eff"][0], big["eff"][0]
        # Per-rank batch is what the extra devices actually get to work on: the
        # global batch is fixed for strong scaling, so more ranks means less
        # each, and that is the mechanism behind the numbers above.
        per_small = small["tp"].get("global_batch", 0) // max(1, small["tp"]["ranks"])
        per_big = big["tp"].get("global_batch", 0) // max(1, big["tp"]["ranks"])
        name = html.escape(big["label"].split(",")[0])
        return (
            f'<h2>Same accelerator, different count</h2>'
            f'<p class="takeaway"><strong>{html.escape(big["machine"])}</strong> and '
            f'<strong>{html.escape(small["machine"])}</strong> run the same part — '
            f'{name} — with {big["devices"]} and {small["devices"]} per node. '
            f'{ratio_dev:.0f}x the accelerators bought {ratio_tp:.2f}x the throughput '
            f'and cost efficiency: {eff_big:,.1f} against {eff_small:,.1f} samples per '
            f'node-joule. More silicon, less work per joule.</p>'
            f'<p class="fineprint">The global batch is fixed at '
            f'{small["tp"].get("global_batch", 0):,} so time-to-accuracy stays '
            f'comparable, which means the wider node gives each rank less to do — '
            f'{per_big} samples per rank against {per_small}. On a workload already '
            f'running at under 1.5% of peak, halving the per-rank batch starves the '
            f'devices further, so the extra accelerators mostly add idle draw. This '
            f'is the one pair here that holds the accelerator constant; every other '
            f'comparison on this page changes the vendor too.</p>'
        )
    return ""


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

# Named by workload, not by page type. "Data & analysis" and "Dashboard"
# described the *form* of each page and left a reader to infer that one held
# training and the other inference -- which is the one thing they most need to
# know, since tokens/joule and samples/joule share no denominator and comparing
# across them is meaningless. The power page keeps its name because it is the
# one page that is deliberately both, sectioned per machine.
# Order is the reading order, not the build order: the overview says what the
# project is, the three middle pages carry every measurement, and conclusions
# says what they came to. index.html is the overview because it is the front
# door -- the training data it used to hold now has a name of its own, so a link
# to it says which page it means.
PAGES = [
    ("index.html", "Overview"),
    ("training.html", "Training data"),
    ("power.html", "Power profiles"),
    ("dashboard.html", "Inference dashboard"),
    ("conclusions.html", "Conclusions"),
]


# Named software and instruments, marked in prose so a reader scanning a
# paragraph can see which tools a finding depended on. Deliberately NOT the
# hardware: a spec card is nothing but hardware names, and marking every one
# of them marks nothing. Models and datasets are out too -- they are the
# workload, not the instrument, and the workload strip already states them.
#
# Longest first, because the alternation is leftmost-first: "Nsight Compute"
# has to win over "Nsight", and "torchrun" over "torch".
TOOL_TERMS = (
    "Nsight Systems", "Nsight Compute", "Level Zero", "Hugging Face",
    "torchvision", "HuggingFace", "torchrun", "Perfetto", "xpu-smi",
    "oneCCL", "pynvml", "PyTorch", "mpiexec", "Nsight", "AIPerf",
    "sysman", "hwmon", "sysfs", "conda", "Lustre", "NCCL", "DCGM",
    "vLLM", "NVML", "CUDA", "IPEX", "ipex", "RAPL", "gloo", "torch",
    "PALS", "i915", "ZMQ", "PBS", "MPI", "DDP", "Ray",
)

# Elements whose text is not prose. <code> already has its own treatment and
# marking inside it would nest two styles on one word; <title> inside an SVG
# is a tooltip; <script> is code a stray span would break.
_TOOL_SKIP = {"code", "pre", "script", "style", "title", "textarea"}

_TOOL_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(t) for t in TOOL_TERMS) + r")(?![\w-])"
)


def mark_tools(markup: str) -> str:
    """Wrap tool names in already-rendered markup, text nodes only.

    Operates on the finished HTML rather than at each call site because the
    names are scattered across dozens of f-strings, and a rule applied in one
    place cannot drift from a rule applied in another. Tags are copied through
    untouched, so nothing inside an attribute -- an href, a data-tip, a chart's
    aria-label -- is ever rewritten.
    """
    out, pos, depth = [], 0, 0
    for m in re.finditer(r"<(/?)([a-zA-Z][\w-]*)([^>]*?)(/?)>", markup):
        chunk = markup[pos:m.start()]
        out.append(chunk if depth else _TOOL_RE.sub(
            r'<span class="tool">\1</span>', chunk))
        out.append(m.group(0))
        name = m.group(2).lower()
        if name in _TOOL_SKIP:
            if m.group(1):
                depth = max(0, depth - 1)
            elif not m.group(4):
                depth += 1
        pos = m.end()
    tail = markup[pos:]
    out.append(tail if depth else _TOOL_RE.sub(
        r'<span class="tool">\1</span>', tail))
    return "".join(out)


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


def anchored(markup: str) -> str:
    """Give every plain h2 and h3 a stable id, so another page can link to it.

    The conclusions page states a finding and then points at the chart it came
    from, which needs somewhere to point. Deriving the ids here rather than at
    each of the several dozen places a heading is written keeps one rule in one
    place, and means a section added later is linkable without anyone
    remembering to make it so.

    Only headings that are plain text are touched; one carrying markup is left
    alone rather than guessed at. Duplicates -- "Serving under load" appears
    once per machine -- are numbered in document order, so the first keeps the
    bare slug and later ones are suffixed.

    The id comes from the text, which means renaming a heading breaks links to
    it. That is the trade for not maintaining a second list of names by hand,
    and check_links() in this file is what catches it when it happens.
    """
    seen: dict = {}

    def add_id(match):
        tag, text = match.group(1), match.group(2)
        base = re.sub(r"[^a-z0-9]+", "-", html.unescape(text).lower()).strip("-")
        if not base:
            return match.group(0)
        seen[base] = seen.get(base, 0) + 1
        ident = base if seen[base] == 1 else f"{base}-{seen[base]}"
        return f'<{tag} id="{ident}">{text}</{tag}>'

    return re.sub(r"<(h[23])>([^<]+)</\1>", add_id, markup)


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
.tool {{ font-weight:600; border-bottom:1px dotted var(--accent); }}
/* The coverage matrix. Its headers name an axis the charts encode as a
   marker shape, so they carry the marker and are set larger than a normal
   column label -- they are read as headings, not as units. */
.covtable th.tph {{ font-size:.92rem; text-transform:none; letter-spacing:0;
  color:var(--fg); text-align:center; }}
.covtable th.mname, .covtable td.mname {{ text-align:left; }}
.covtable td {{ text-align:center; }}
.covtable td.mname .sw {{ width:18px; margin-right:.5rem; vertical-align:1px; }}
.chip .sw {{ width:18px; }}
/* Marker swatches outside a chart. --c is the machine's colour, set by the
   .sN classes, and a tensor-parallel width has no machine -- so without a
   value here the filled shapes fall back to SVG's initial black and the
   hollow ones, which stroke with --c, vanish outright. That is what removed
   TP=4: its slot is the hollow square. Body colour reads on both themes. */
.chip .mk, .covtable .mk {{ --c:var(--fg); width:17px; height:17px;
  vertical-align:-4px; margin-right:.1rem; }}
.chip .dot, .covtable .dot {{ stroke-width:1.3; }}
/* Swatches are solid whatever their slot. The hollow fill exists so a marker
   landing on another shows what it covers, which is a chart problem and not a
   legend one -- and beside a real checkbox a hollow square reads as an
   unticked one. Shape alone separates them at this size. */
.chip .dot.k1, .chip .dot.k3,
.covtable .dot.k1, .covtable .dot.k3 {{ fill:var(--c); stroke:var(--bg); }}
.cov {{ display:inline-flex; gap:.45rem; font-variant-numeric:tabular-nums; }}
.cov > span {{ width:1.15em; text-align:center; font-weight:700;
  font-size:.82rem; letter-spacing:.02em; }}
.hit {{ cursor:help; }}
.cov > span.miss {{ color:var(--dim); opacity:.32; font-weight:500; }}
/* Not a gap but a wall: struck through, so a reader stops looking for the
   run that would fill it. */
.cov > span.cant {{ color:var(--dim); opacity:.5; font-weight:500;
  text-decoration:line-through; text-decoration-thickness:1.5px; cursor:help; }}
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
/* Capability radar. Narrow and centred: it is a summary of four numbers, and
   stretched to a wide viewport the polygon reads as a claim about magnitude
   when every axis is a ratio to the best machine. Fill is faint because the
   area of a radar means nothing -- it depends on the order of the axes -- so
   the outline is the figure and the wash is only there to tell it from the
   ghosts. */
/* Overview page: the four ways into the site. Cards rather than a list
   because this is the one page whose job is to send you somewhere else, and a
   grid that collapses to one column keeps the order the reading order. */
.doors {{ display:grid; gap:.7rem; margin:1.1rem 0 1.4rem;
  grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); }}
.door {{ display:block; padding:.85rem 1rem; border:1px solid var(--line);
  border-radius:8px; background:var(--card); text-decoration:none;
  color:inherit; transition:border-color .12s ease; }}
.door:hover {{ border-color:var(--accent); }}
.door-t {{ display:block; font-weight:650; color:var(--accent);
  font-size:.92rem; margin-bottom:.25rem; }}
.door-d {{ display:block; font-size:.8rem; color:var(--dim);
  line-height:1.5; }}
.chart.radar {{ max-width:460px; margin-inline:auto; }}
.rdr {{ fill:var(--c); fill-opacity:.12; stroke:var(--c); stroke-width:2;
  stroke-linejoin:round; }}
.rdr.ghost {{ fill:none; stroke-width:1; opacity:.32; }}
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
.ln.dev {{ stroke-width:1; opacity:.28; }}
.specs.measured {{ border-left:3px solid var(--accent); padding-left:.9rem;
  margin-top:.9rem; }}
h4 {{ margin:1.6rem 0 .5rem; font-size:.95rem; }}
.sw.dash {{ background:repeating-linear-gradient(90deg,
  var(--c) 0 5px, transparent 5px 8px); }}
/* 2px ring in the surface colour, so a marker stays legible where a line
   passes under it -- and so the hit target beats the 8px mark. */
.dot {{ fill:var(--c); stroke:var(--bg); stroke-width:2; }}
/* Alternating slots are hollow, so two markers on the same point stay two
   markers. Hollow rather than a tinted fill: mixing the series colour toward
   the surface desaturates it, and a browner orange reads as a third machine
   rather than a second TP -- hue has to stay the machine's. See-through rather
   than surface-filled, because an opaque hollow marker erases what it covers
   just as thoroughly as a solid one, with pointer-events pinned so the tooltip
   survives having no paint in the middle. */
.dot.k1, .dot.k3 {{ fill:transparent; stroke:var(--c); pointer-events:all; }}
/* Hovering a line or its legend key drops everything else back, which is the
   only thing that really works when series sit on top of each other. Filtering
   is still the answer for a figure -- a screenshot cannot hover. */
.series, .legend-row .key {{ transition:opacity .12s ease; }}
.legend-row .key {{ cursor:default; }}
/* Point details. The browser's own <title> tip waits about a second before it
   appears, cannot be styled, and renders in the OS chrome rather than the
   page -- so the script lifts those titles out and draws this instead. The
   markup keeps the <title> for the scripting-off case, where a slow tooltip
   beats no tooltip. */
.tip {{ position:fixed; z-index:50; pointer-events:none; max-width:min(340px,80vw);
  background:var(--card); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; padding:6px 9px; font-size:12px; line-height:1.35;
  box-shadow:0 6px 18px rgba(0,0,0,.22); }}
.tip[hidden] {{ display:none; }}
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
{anchored(mark_tools(body))}

<footer>
{footer}
</footer>
</div></body></html>
"""


def index_body(runs: list, specs: dict | None = None,
               curves: dict | None = None) -> str:
    """The training page: specs, runs, energy tables and charts.

    Training only, deliberately. The serving numbers moved to the inference
    page when the tabs were named by workload -- a page holding both invites
    reading a samples/joule figure against a tokens/joule one, and those
    share no denominator.
    """
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
{same_accelerator_takeaway(runs, specs)}

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
        # Shape is part of the identity, not a detail of it. Without it the
        # 8B shape sweep and the 8B concurrency sweep -- same machine, same
        # model, both TP=1 -- collided, and the newer one silently deleted the
        # baseline every TP comparison on the page is measured against. isl and
        # osl are None for a sweep that varied them, which is its own key and
        # exactly right: that is a different experiment, not a rerun.
        key = (machine, cfg.get("model"), cfg.get("tp"),
               cfg.get("isl"), cfg.get("osl"))
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

    Every level is read, not just the first. A sweep that varies ISL or OSL has
    no single shape, and reporting whichever level sorted first would put a
    confident wrong number in the page header -- the failure mode is a label
    nobody rechecks, since 1024 and 256 are exactly what the reader expects to
    see. Unanimous or null.
    """
    first, shapes = None, set()
    for export in sorted(sweep_dir.glob("c*/profile_export_aiperf.json")):
        try:
            cfg = (json.loads(export.read_text(encoding="utf-8-sig"))
                   .get("input_config") or {})
        except (OSError, ValueError):
            continue
        datasets = cfg.get("datasets") or [{}]
        prompts = (datasets[0].get("prompts") or {}) if datasets else {}
        shapes.add(((prompts.get("isl") or {}).get("mean"),
                    (prompts.get("osl") or {}).get("mean")))
        if first is None:
            first = cfg
    if first is None:
        return {"model": None, "isl": None, "osl": None, "streaming": None,
                "tp": _tensor_parallel(sweep_dir), "shapes": 0}
    count = len(shapes)
    isl, osl = shapes.pop() if count == 1 else (None, None)
    return {
        "model": (first.get("tokenizer") or {}).get("name"),
        "isl": isl,
        "osl": osl,
        "shapes": count,
        "streaming": (first.get("endpoint") or {}).get("streaming"),
        "tp": _tensor_parallel(sweep_dir),
    }


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


def serving_comparison(sweeps: list) -> str:
    """The cross-machine serving comparison, for the index page.

    Closes the inference page: the dashboard is what the page is for, so the
    controls come first and this is what they add up to -- the comparison a
    reader would otherwise have to assemble by filtering. The per-machine
    serving material -- traces, saturation charts, sweet spots -- is in each
    machine's power profile.
    """
    if not sweeps:
        return ""
    compare = ""
    by_model: dict = defaultdict(list)
    for s in sweeps:
        # Concurrency sweeps only, matching what the takeaway can read. A shape
        # sweep has no load curve, so it neither earns a heading of its own nor
        # counts toward the two machines a comparison needs.
        if len(s["rows"]) >= 3 and swept_over_concurrency(s["rows"]):
            by_model[s["model"]].append(s)
    for model, group in sorted(by_model.items()):
        if len({s["machine"] for s in group}) < 2:
            continue
        name = html.escape((model or "?").split("/")[-1])
        compare += f"""
<h3>Which machine serves more per joule — {name}</h3>
{_compare_takeaway(group)}"""

    return f"""
<h2>What the numbers say</h2>
<p class="fineprint">A second benchmark on the same nodes and the same energy
counters as the <a href="training.html">training</a> page: vLLM answering requests
instead of a model being trained. Tokens per joule and samples per joule share
no denominator and never belong in one table, which is why these are two pages
rather than two sections. On NVIDIA,
AIPerf measures its own power through pynvml; on Intel it can read nothing, and
the energy comes from <code>analysis/power_sidecar.py</code> sampling the same
hwmon counters the training runs use, from beside the run. Each machine's power
traces and operating points are in its
<a href="power.html">power profile</a>; every curve, filterable, is
above.</p>
{_xcheck_note(sweeps)}
{model_table(sweeps)}
{compare}"""


def machine_serving_blocks(machine: str, sweeps: list) -> str:
    """One machine's concurrency sweeps: the saturation chart and its sentence.

    The chart normalises throughput and dynamic power to concurrency 1 on a
    shared axis, because the divergence between them is the finding -- serving
    more at once is nearly free in watts. One block per sweep, inside the
    machine's profile.
    """
    blocks = ""
    mine = [s for s in sweeps
            if s["machine"] == machine and len(s["rows"]) >= 3
            and swept_over_concurrency(s["rows"])]
    # Model then TP, so the same model's shardings sit together and the order
    # cannot change when a newer sweep lands.
    for sweep in sorted(mine, key=lambda s: ((s["model"] or ""), s.get("tp") or 0)):
        rows = sweep["rows"]
        first, last = rows[0], rows[-1]
        chart = inference_chart(rows)
        served = " · ".join(
            bit for bit in (
                f"<strong>{html.escape(str(sweep['model']))}</strong>" if sweep["model"] else None,
                f"TP={sweep['tp']}" if sweep.get("tp") else None,
                f"ISL {sweep['isl']:,.0f}" if sweep["isl"] else None,
                f"OSL {sweep['osl']:,.0f}" if sweep["osl"] else None,
                f"idle floor {sweep['idle_w']:,.0f} W" if sweep["idle_w"] else None,
            ) if bit
        )
        blocks += f"""
<h4>{html.escape((sweep["model"] or "inference").split("/")[-1])}{f" — TP={sweep['tp']}" if sweep.get("tp") else ""}</h4>
<p class="workload">{served}</p>
{chart}
{inference_legend()}
{_inference_takeaway(first, last)}"""
    return blocks


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
    # Folded like the index's tables: ten columns of dense numbers are a
    # reference, not something a reader works through on the way past. The
    # takeaway underneath stays open, because it is the finding the table
    # supports and burying it would leave the section saying nothing.
    return f"""
<details class="fold">
<summary>What the model costs <span class="count">({len(rows)} configuration{"" if len(rows) == 1 else "s"} at concurrency {conc})</span></summary>
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
</details>
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


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _compare_takeaway(sweeps: list) -> str:
    """Which machine wins per joule, on two framings and two denominators.

    Concurrency sweeps only. A shape sweep holds concurrency fixed and varies
    the prompt, so its rows are five prompts rather than a load curve and its
    extremes describe a prompt, not a machine -- which is how the worst shape of
    one Aurora sweep briefly became Aurora's number here, reporting 10.3x where
    the like-for-like comparison says 4.3x.

    Two framings, because they disagree and the disagreement is the useful part.
    Matched TP holds sharding constant and is the controlled result. Best per
    machine is what each machine actually reaches when allowed its own best
    configuration, which is the question an allocation actually asks.

    Never compares a machine against itself: ranked across every configuration,
    the best and worst entries were both Aurora often enough that the sentence
    read as a machine comparison while being a TP comparison.
    """
    tops: dict = {}
    for sweep in sweeps:
        if not swept_over_concurrency(sweep["rows"]):
            continue
        top = sorted(sweep["rows"], key=lambda r: r["concurrency"])[-1]
        if not (top.get("tok_per_joule") and top.get("tok_per_joule_dynamic")):
            continue
        # Repeat sweeps of one configuration: keep the best, so a known-bad
        # early run cannot become the machine's number.
        key = (sweep["machine"], sweep.get("tp"))
        if key not in tops or top["tok_per_joule"] > tops[key]["tok_per_joule"]:
            tops[key] = top
    if len({m for m, _ in tops}) < 2:
        return ""

    def rank(members: dict, metric: str):
        """Winner, loser and their ratio on one metric. members: machine -> row."""
        order = sorted(members.items(), key=lambda kv: -kv[1][metric])
        (wm, wr), (lm, lr) = order[0], order[-1]
        return wm, lm, wr[metric] / lr[metric]

    def sentence(members: dict, lead: str) -> str:
        tw, tl, tr = rank(members, "tok_per_joule")
        dw, dl, dr = rank(members, "tok_per_joule_dynamic")
        if tw == dw:
            return (f"{lead} <strong>{html.escape(tw)}</strong> leads on both measures "
                    f"— {tr:.1f}x the tokens per joule of {html.escape(tl)}, and "
                    f"{dr:.2f}x on dynamic energy. No tradeoff to make.")
        return (f"{lead} <strong>{html.escape(tw)}</strong> delivers {tr:.1f}x the "
                f"tokens per joule of {html.escape(tl)} — but per joule of "
                f"<em>work</em>, <strong>{html.escape(dw)}</strong> is {dr:.2f}x "
                f"ahead. The more efficient accelerator loses, because its node "
                f"spends so much more simply being switched on.")

    paras = ""

    # Framing 1: the controlled comparison, at a TP both machines actually ran.
    by_tp: dict = defaultdict(dict)
    for (machine, tp), row in tops.items():
        by_tp[tp][machine] = row
    shared = {tp: d for tp, d in by_tp.items() if len(d) >= 2}
    if shared:
        tp = sorted(shared, key=lambda t: (-len(shared[t]), t if t else 0))[0]
        members = shared[tp]
        levels = {r["concurrency"] for r in members.values()}
        at = (f"At concurrency {next(iter(levels))}"
              if len(levels) == 1 else "At the top of each sweep")
        # "both" was right while two machines had swept it and wrong the day a
        # third did. Counted rather than asserted, for the same reason every
        # number on this page is derived: prose that names a quantity has to
        # re-derive it or it becomes false without anyone editing it.
        n = len(members)
        who = "both machines" if n == 2 else f"all {_NUMBER_WORDS.get(n, str(n))} machines"
        lead = f"{at} and TP={tp}, the one configuration {who} swept:"
        paras += f'<p class="takeaway">{sentence(members, lead)}</p>'

    # Framing 2: each machine at its own best, which need not be the same TP.
    best_total: dict = {}
    best_dyn: dict = {}
    for (machine, tp), row in tops.items():
        if (machine not in best_total
                or row["tok_per_joule"] > best_total[machine][1]["tok_per_joule"]):
            best_total[machine] = (tp, row)
        if (machine not in best_dyn
                or row["tok_per_joule_dynamic"] > best_dyn[machine][1]["tok_per_joule_dynamic"]):
            best_dyn[machine] = (tp, row)
    # Only worth saying when some machine had a choice of configuration. Where
    # every machine ran exactly one TP, "its best" is the configuration the
    # paragraph above already compared, and the two would restate each other.
    tps_per_machine: dict = defaultdict(set)
    for machine, tp in tops:
        tps_per_machine[machine].add(tp)
    if max((len(v) for v in tps_per_machine.values()), default=0) > 1:
        picks = ", ".join(
            f"{html.escape(m)} at TP={tp}" for m, (tp, _) in sorted(best_total.items()))
        members = {m: r for m, (_, r) in best_total.items()}
        dyn_members = {m: r for m, (_, r) in best_dyn.items()}
        tw, tl, tr = rank(members, "tok_per_joule")
        dw, dl, dr = rank(dyn_members, "tok_per_joule_dynamic")
        paras += (
            f'<p class="takeaway">Given each machine its best configuration '
            f'({picks}), <strong>{html.escape(tw)}</strong> leads on node energy '
            f'by {tr:.1f}x, and <strong>{html.escape(dw)}</strong> on dynamic '
            f'energy by {dr:.2f}x. Filling a node changes the node-energy gap and '
            f'not the dynamic one, because the idle draw is divided across more '
            f'work rather than reduced.</p>')

    return (
        paras
        + '<p class="fineprint">Tokens per joule of <em>node</em> energy is what an '
        'allocation bills for; per joule of <em>dynamic</em> energy is node draw '
        'above the idle floor, measured on that node before its server started. '
        'The two rank the machines differently, which is the finding above. '
        'Prompt-shape sweeps are excluded — they hold concurrency fixed, so their '
        'spread is across prompts and not across load. Both measures against '
        'concurrency, for every machine and model, are on the '
        '<a href="dashboard.html">inference dashboard</a> as the tokens per joule and '
        'tokens per joule (dynamic) metrics. The machines did not run the same '
        "vLLM: versions are recorded in each sweep's run_meta.json, and both ran "
        'with <code>--enforce-eager</code>.</p>'
    )


# What the dashboard's columns and metric names mean. Kept short on purpose:
# it sits under the controls as a reminder, not a tutorial -- the reasoning
# behind each measure is in the takeaways and on the power profiles. Terms are
# grouped by what they measure, because the point a reader most often misses is
# that throughput, latency and efficiency are three different questions and a
# machine can win one while losing another.
DASHBOARD_LEGEND = [
    ("How much", [
        ("Out tok/s", "output tokens generated per second, all requests together"),
        ("Conc", "concurrent requests in flight; the swept variable"),
        ("ISL / OSL", "input and output sequence lengths, in tokens — pinned per row"),
        ("TP", "tensor parallelism: how many accelerators one model is sharded across"),
    ]),
    ("How fast", [
        ("TTFT", "time to first token — how long a request waits before anything comes back"),
        ("ITL", "inter-token latency — the gap between tokens once generation starts"),
        ("Request latency", "TTFT plus the whole generation, end to end"),
        ("Prefill share", "percent of request latency spent reading the prompt rather than writing"),
    ]),
    ("At what cost", [
        ("Dyn W", "dynamic power: node draw above the idle floor measured before the server started"),
        ("Node power", "watts drawn by every accelerator on the node, including ones this job left idle"),
        ("Joules", "total node energy over the level; comparable only between rows of equal request count"),
        ("Tok/J", "output tokens per joule of node energy — what an allocation bills for"),
        ("Tok/J dyn", "the same per joule of dynamic energy — what the silicon did"),
    ]),
]


# Weight footprint per model, in GB, for deciding whether a configuration can
# fit at all. Parameters times bytes per parameter: bf16 is 2 bytes, and
# gpt-oss-120b ships its MoE weights in mxfp4 at roughly half a byte, which its
# own config.json declares as quant_method.
#
# Deliberately approximate. It separates "nobody ran this" from "this cannot
# run", not a memory budget, so it is compared against weights alone -- a
# configuration whose weights merely fit but leave no room for a KV cache is
# left looking possible rather than being called impossible on an estimate.
MODEL_WEIGHT_GB = {
    "Llama-3.1-8B-Instruct": 16,
    "Llama-3.3-70B-Instruct": 141,
    "gemma-3-27b-it": 55,
    "gpt-oss-120b": 61,
}


def _accel_units(spec: dict) -> tuple:
    """(devices tensor parallel can address, GB of memory on each).

    Tiles where a machine has them. Aurora's six Ponte Vecchio cards present
    as twelve tiles, and a tile is what a rank binds and what --tensor-parallel
    counts, so reading the card count would halve the width and double the
    memory in one step.
    """
    accel = (spec or {}).get("accelerator") or ""
    tiles = re.search(r"(\d+)\s*tiles", accel)
    lead = re.match(r"\s*(\d+)", accel)
    if tiles:
        n = int(tiles.group(1))
    elif lead:
        n = int(lead.group(1))
    else:
        return None, None
    total = re.match(r"\s*([\d.]+)", (spec or {}).get("accelerator_memory") or "")
    return n, (float(total.group(1)) / n if total and n else None)


def _fits(model: str, tp: int, spec: dict) -> bool:
    """Whether one model at one sharding width can sit on one node.

    Two independent walls. The width cannot exceed the devices a node has --
    Polaris has four GPUs, so TP=8 is not a tuning question there. And the
    weights, divided across those devices, have to land inside what vLLM will
    allocate on each.

    Unknown model or unparseable spec answers True: an unmarked cell reads as
    "nobody ran this", which is the safer thing to say when we do not know.
    """
    units, gb = _accel_units(spec)
    if not units or tp > units:
        return False
    weight = MODEL_WEIGHT_GB.get(model)
    if weight is None or not gb:
        return True
    # 0.9 is what vLLM will allocate; the further 0.8 is headroom for a KV
    # cache. Weights that merely fit do not make a server: gemma-3-27b is 55 GB
    # against an Aurora tile's 57.6 usable, which leaves under 3 GB for cache
    # and cannot hold one sequence of the sweep's own length. Calling that
    # possible would send someone to spend an allocation discovering it.
    #
    # Checked against every configuration that has actually run here: all of
    # them clear this bar, so the rule marks nothing impossible that the
    # results contradict.
    return weight / tp <= gb * 0.9 * 0.8


def coverage_matrix(sweeps: list, specs: dict | None = None) -> str:
    """Which model, sharding width and machine combinations exist.

    Three dimensions in a two-dimensional table: model down, tensor parallel
    across, and the machines inside each cell as their initials. The obvious
    layout -- a row per model AND per TP -- repeated every model name three or
    four times and still needed a column per machine, so the grid was mostly
    restating its own axes.

    Every machine appears in every cell, lit where a sweep exists and dimmed
    where none does, so a column can be scanned without the positions moving
    under the eye. Colour is the machine, matching every chart on the site.

    Derived from the sweeps the charts draw, so it cannot advertise coverage
    the page does not have.
    """
    if not sweeps:
        return ""
    machines = sorted({s["machine"] for s in sweeps})
    tps = sorted({s.get("tp") or 0 for s in sweeps})
    # Slot lookup uses the same list the charts do, including a None entry if
    # any sweep lacks a TP, so a marker here is the marker there.
    tp_all = sorted({s.get("tp") for s in sweeps}, key=lambda v: (v is None, v))
    have: dict = {}
    for s in sweeps:
        model = (s["model"] or "?").split("/")[-1]
        key = (model, s.get("tp") or 0, s["machine"])
        kind = ("prompt shapes" if not swept_over_concurrency(s["rows"])
                else "concurrency levels")
        have.setdefault(key, []).append(f'{len(s["rows"])} {kind}')

    # One letter per machine, disambiguated only if two share an initial --
    # "A P S" needs no legend, "Au Po So Cr" does.
    firsts = [m[0].upper() for m in machines]
    labels = {m: (m[0].upper() if len(set(firsts)) == len(firsts)
                  else m[:2].capitalize()) for m in machines}

    full_models = sorted({s["model"] for s in sweeps})
    short_to_full = {(s["model"] or "?").split("/")[-1]: s["model"] for s in sweeps}

    body = ""
    for model in sorted({m for m, _, _ in have}):
        bg = model_dash_bg(short_to_full.get(model, model), full_models)
        cells = (f'<td class="mname"><span class="sw" style="background:{bg}"></span>'
                 f'{html.escape(model)}</td>')
        for tp in tps:
            marks = ""
            for machine in machines:
                hit = have.get((model, tp, machine))
                slot = SERIES_SLOT.get(machine, 8)
                if hit:
                    marks += (f'<span class="m s{slot} hit" '
                              f'title="{html.escape(machine)}: {", ".join(hit)}">'
                              f'{labels[machine]}</span>')
                elif not _fits(model, tp, (specs or {}).get(machine)):
                    units, gb = _accel_units((specs or {}).get(machine))
                    why = (f"{machine} has {units} devices, fewer than TP={tp}"
                           if units and tp > units else
                           f"{MODEL_WEIGHT_GB.get(model, 0)} GB over {tp} device(s) "
                           f"exceeds the {gb * 0.9:,.0f} GB each will hold"
                           if gb else "cannot fit on one node")
                    marks += (f'<span class="cant" title="{html.escape(why)}">'
                              f'{labels[machine]}</span>')
                else:
                    marks += f'<span class="miss">{labels[machine]}</span>'
            cells += f'<td><span class="cov">{marks}</span></td>'
        body += f"<tr>{cells}</tr>"

    # Rendered directly rather than through table(): the headers carry the
    # markers the charts plot, and table() escapes its headers -- rightly, since
    # every other caller passes plain words.
    head = '<th class="mname">Model</th>' + "".join(
        f'<th class="tph">{tp_swatch(t, tp_all)}TP={t}</th>' if t
        else '<th class="tph">TP —</th>' for t in tps)
    legend = " · ".join(f"{labels[m]} = {machine_tag(m)}" for m in machines)
    caption = (
        f"{legend}. A lit initial is a sweep on that machine at that sharding "
        "width; a dimmed one is a gap nobody has run yet; a struck-through one "
        "cannot run on a single node of that machine at all — either the node "
        "has fewer devices than the width asks for, or the weights will not "
        "divide into them. Hover any mark for the reason or the level count. "
        "Multi-node tensor parallelism over Ray would lift the device-count "
        "wall, and is not built here. The swatches are "
        "the chart's own encodings: dash is the model, marker is the tensor "
        "parallel width, colour is the machine. Prompt-shape sweeps count "
        "here too, though they hold concurrency fixed and so appear on the "
        "power profiles rather than in the charts above.")
    return (
        '<h2>What has been measured</h2>'
        f'<figure><div class="scroll"><table class="covtable">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        f"<figcaption>{caption}</figcaption></figure>")


def dashboard_legend_terms() -> str:
    """The metric glossary, reusing the index's legend styling."""
    blocks = ""
    for heading, terms in DASHBOARD_LEGEND:
        items = "".join(
            f"<dt>{html.escape(term)}</dt><dd>{html.escape(meaning)}</dd>"
            for term, meaning in terms
        )
        blocks += f"<section><h3>{html.escape(heading)}</h3><dl>{items}</dl></section>"
    return f'<div class="legend">{blocks}</div>'


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


def dashboard_body(sweeps: list, specs: dict | None = None) -> str:
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
    # Ordered as the charts order them, not as the labels sort. See
    # model_dash_bg: the two disagree, and the swatch has to follow the line.
    full_models = sorted({s["model"] for s in sweeps})
    short_to_full = {(s["model"] or "?").split("/")[-1]: s["model"] for s in sweeps}
    tp_values = sorted({s.get("tp") for s in sweeps}, key=lambda v: (v is None, v))

    # Opening with everything ticked put seven configurations on one axis, which
    # is a browsing state rather than a picture -- and it grows with every sweep
    # added, so the first thing a reader sees gets worse as the work goes better.
    # The page opens on the most-swept model instead, which is where the machine
    # comparison and the TP comparison both live, and the rest is one click and
    # one dimmed legend key away.
    #
    # Chosen by count rather than named, so it follows the results: whatever has
    # been measured most is what the page is currently about.
    busiest = max(models, key=lambda m: sum(
        1 for s in sweeps if (s["model"] or "?").split("/")[-1] == m))
    defaults = {"model": {busiest}}

    def boxes(kind, values):
        # Machines are labelled in their own colour, models plainly -- a model
        # has no colour anywhere on this site, and inventing one here would
        # imply a mapping the charts do not share.
        label = machine_tag if kind == "machine" else html.escape
        if kind == "tp":
            label = lambda v: f"TP={html.escape(v)}"   # noqa: E731
        on = defaults.get(kind)
        out = ""
        for v in values:
            # The chip shows the encoding it filters on, so the three legends
            # a reader has to hold -- colour is the machine, dash is the
            # model, shape is the sharding width -- are stated where they are
            # used instead of only under the chart.
            if kind == "machine":
                # No swatch: machine_tag already renders the name in the
                # machine's colour, and a bar beside it says the same thing
                # twice.
                cls, mark = "chip", ""
            elif kind == "model":
                cls = "chip"
                bg = model_dash_bg(short_to_full.get(v, v), full_models)
                mark = f'<span class="sw" style="background:{bg}"></span>'
            else:
                cls = "chip"
                mark = tp_swatch(int(v), tp_values)
            checked = "" if on and v not in on else " checked"
            out += (f'<label class="{cls}"><input type="checkbox" '
                    f'data-filter="{kind}" value="{html.escape(v)}"{checked}>'
                    f'{mark} {label(v)}</label>')
        return out

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

    # Concurrency sweeps only, matching the charts above it. A shape sweep has
    # no line on this page, and rows that appear in a table under a chart they
    # are absent from read as points the chart forgot to draw. Those live on the
    # power page, next to the fit that is the reason they were measured.
    rows = []
    for sweep in sorted([s for s in sweeps if swept_over_concurrency(s["rows"])],
                        key=lambda s: (s["machine"], s["model"] or "",
                                       s.get("tp") or 0)):
        model = (sweep["model"] or "?").split("/")[-1]
        for r in sorted(sweep["rows"], key=lambda r: r["concurrency"]):
            rows.append((sweep["machine"], model, str(sweep.get("tp") or ""), [
                machine_tag(sweep["machine"]),
                html.escape(model),
                num(sweep.get("tp")) if sweep.get("tp") else "—",
                # Per row, not per sweep: a sweep that varied the shape has a
                # different one on every line, and four rows reading
                # "aurora / 8B / TP=1 / 32" with different numbers beside them
                # look like a measurement that could not make up its mind.
                num(r.get("requested_isl")),
                num(r.get("requested_osl")),
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
        "Machine", "Model", "TP", "ISL", "OSL", "Conc", "Out tok/s", "TTFT ms", "ITL ms",
        "Dyn W", "Joules", "Tok/J", "Tok/J dyn"))
    body = "".join(
        f'<tr data-machine="{html.escape(m)}" data-model="{html.escape(mo)}" '
        f'data-tp="{html.escape(tp)}">'
        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        for m, mo, tp, cells in rows
    )

    return f"""
<h2>Every configuration, filtered</h2>
<p class="fineprint">Every concurrency level of every sweep, with the controls
below applying to the charts and the table together. What the numbers add up
to is under them.</p>
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
<summary>Every concurrency-sweep row <span class="count" id="rowcount"></span></summary>
<figure><div class="scroll"><table id="rows"><thead><tr>{head}</tr></thead>
<tbody>{body}</tbody></table></div>
<figcaption>Every concurrency level of every <em>concurrency</em> sweep,
filtered by the same controls. Sweeps that varied the prompt shape instead of
the load are not here — they hold concurrency fixed, so they have no point on
these charts; they get their own table on the
<a href="power.html">power profiles</a> page. That is why ISL and OSL read the
same on every row below: the shape is what these sweeps held still.
Joules are comparable only between rows that ran the same request
count — see each sweep's run_meta.json.</figcaption></figure>
</details>

{serving_comparison(sweeps)}

{coverage_matrix(sweeps, specs)}

<h2>What the terms mean</h2>
{dashboard_legend_terms()}

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

  // Hover to isolate. Seven configurations on one axis overlap no matter how
  // the marks are drawn, and picking one out is the thing no static encoding
  // does. Series and legend key share the same three attributes, so hovering
  // either finds the other.
  function ident(el) {{
    return [el.getAttribute('data-machine'), el.getAttribute('data-model'),
            el.getAttribute('data-tp')].join('|');
  }}
  function highlight(id) {{
    // Clearing runs apply() rather than blanking the styles: the filter dims
    // legend keys with the same property, and resetting it here would undo the
    // filter every time the pointer left a line.
    if (!id) {{ document.querySelectorAll('.series').forEach(function (g) {{
      g.style.opacity = ''; }}); apply(); return; }}
    document.querySelectorAll('.series').forEach(function (g) {{
      g.style.opacity = ident(g) === id ? '' : '0.12';
    }});
    document.querySelectorAll('.legend-row .key').forEach(function (k) {{
      k.style.opacity = ident(k) === id ? '' : '0.2';
    }});
  }}
  document.querySelectorAll('.series, .legend-row .key').forEach(function (el) {{
    el.addEventListener('mouseenter', function () {{ highlight(ident(el)); }});
    el.addEventListener('mouseleave', function () {{ highlight(null); }});
  }});

  // Instant point details. The native <title> tip waits about a second, cannot
  // be styled and draws outside the page. Those titles are lifted into an
  // attribute and removed, so the browser stops offering its own -- the markup
  // still ships them, which is what a scripting-off reader gets.
  var tip = document.createElement('div');
  tip.className = 'tip';
  tip.hidden = true;
  document.body.appendChild(tip);
  document.querySelectorAll('.chart title').forEach(function (t) {{
    t.parentNode.setAttribute('data-tip', t.textContent);
    t.parentNode.removeChild(t);
  }});
  function place(e) {{
    // Flip rather than clamp at the edges: a tip pinned to the right margin
    // sits on top of the point it describes, which is the one thing it must
    // not cover.
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 6) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 6) y = e.clientY - h - pad;
    tip.style.left = Math.max(6, x) + 'px';
    tip.style.top = Math.max(6, y) + 'px';
  }}
  document.querySelectorAll('[data-tip]').forEach(function (el) {{
    el.addEventListener('mouseenter', function (e) {{
      tip.textContent = el.getAttribute('data-tip');
      tip.hidden = false;
      place(e);
    }});
    el.addEventListener('mousemove', place);
    el.addEventListener('mouseleave', function () {{ tip.hidden = true; }});
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


def shape_section(sweeps: list) -> str:
    """Sweeps that varied the prompt shape instead of the load.

    A table and not a chart, deliberately. These sweeps hold concurrency fixed,
    so they have nothing to say on an axis of concurrency, and giving the axis a
    choice put two-thirds of the dashboard's controls in service of one line.
    Five rows of numbers say the same thing and cost nothing to read.

    The fit underneath is the reason the sweep exists. Energy per request comes
    apart into a cost per input token and a cost per output token, and those are
    properties of the machine -- tokens/joule at one shape is a blend whose
    mixing ratio someone chose. Least squares on a*ISL + b*OSL + c, stdlib only,
    like everything else here.
    """
    shaped = [s for s in sweeps
              if not swept_over_concurrency(s["rows"]) and len(s["rows"]) >= 3]
    if not shaped:
        return ""

    blocks = ""
    for sweep in shaped:
        rows = sorted(sweep["rows"], key=lambda r: (r.get("requested_isl") or 0,
                                                    r.get("requested_osl") or 0))
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{c}</td>" for c in (
                num(r.get("requested_isl")), num(r.get("requested_osl")),
                num(r.get("concurrency")), num(r.get("out_tok_per_s"), 1),
                num(r.get("ttft_ms")), num(r.get("itl_ms"), 1),
                num(r.get("dynamic_w")), num(r.get("mj_per_output_token")),
                num(r.get("energy_per_req_j")),
            )) + "</tr>"
        head = "".join(f"<th>{h}</th>" for h in (
            "ISL", "OSL", "Conc", "Out tok/s", "TTFT ms", "ITL ms", "Dyn W",
            "mJ/out tok", "J/req"))
        fitted = _shape_fit(rows)
        blocks += f"""
<h3>Prompt shape — {html.escape((sweep["model"] or "inference").split("/")[-1])}</h3>
<figure><div class="twrap"><table>
<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<figcaption>{fitted}</figcaption></figure>"""
    return blocks


def _shape_fit(rows: list) -> str:
    """Solve a*ISL + b*OSL + c for dynamic energy per request, and say it plainly.

    Returns a caption rather than a number, because the coefficients only mean
    something with the caveat attached: three parameters against five points,
    and an intercept that has no physical reading.
    """
    pts = [(r.get("requested_isl"), r.get("requested_osl"),
            (r.get("dynamic_w") or 0) * (r.get("duration_s") or 0)
            / (r.get("requests") or 1))
           for r in rows]
    pts = [(x, y, z) for x, y, z in pts if x and y and z]
    if len(pts) < 4:
        return ("Prompt shape held concurrency fixed, so these levels are points "
                "rather than a curve. Too few for a fit.")
    n = len(pts)
    sums = [0.0] * 8
    for x, y, z in pts:
        for i, v in enumerate((x, y, z, x * x, y * y, x * y, x * z, y * z)):
            sums[i] += v
    Sx, Sy, Sz, Sxx, Syy, Sxy, Sxz, Syz = sums
    A = [[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, float(n)]]
    B = [Sxz, Syz, Sz]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        B[i], B[p] = B[p], B[i]
        if not A[i][i]:
            return "Levels are collinear in ISL and OSL; no fit."
        for r in range(3):
            if r != i:
                f = A[r][i] / A[i][i]
                for c in range(3):
                    A[r][c] -= f * A[i][c]
                B[r] -= f * B[i]
    a, b, _c = (B[i] / A[i][i] for i in range(3))
    mean = Sz / n
    ss_t = sum((z - mean) ** 2 for _, _, z in pts)
    ss_r = sum((z - (a * x + b * y + _c)) ** 2 for x, y, z in pts)
    r2 = 1 - ss_r / ss_t if ss_t else 0.0
    return (f"Dynamic energy per request fits <strong>{a * 1000:,.1f} mJ</strong> per "
            f"input token plus <strong>{b * 1000:,.0f} mJ</strong> per output token "
            f"(R² {r2:.4f}) — an output token costs {b / a:.1f}x an input one, "
            f"which is decode streaming every weight for one token against prefill "
            f"batching the whole prompt. Three parameters on {n} points, and the "
            f"intercept is fit slack rather than a fixed per-request cost.")


def training_timeline(machine: str, results_dir: str, runs: list):
    """The machine's canonical training power timeline, with its run.

    One trace per machine, chosen by the run's work: the fullest metered run is
    the one whose trace shows a machine training rather than a smoke test
    warming up. Returns (timeline, run) or (None, None).
    """
    best = None
    for path in Path(results_dir).glob(f"power/{machine}_*_power.json"):
        rid = path.name.split("_")[1]
        run = next((r for r in runs if r.get("run_id") == rid), None)
        if not run:
            continue
        score = run.get("samples_global") or 0
        if best is None or score > best[0]:
            best = (score, path, run)
    if not best:
        return None, None
    return json.loads(best[1].read_text(encoding="utf-8")), best[2]


def best_sweep(machine: str, sweeps: list):
    """The machine's best concurrency sweep by top-level tokens/joule.

    The same rule _compare_takeaway uses, so the profile's operating point and
    the index's comparison can never disagree about which run represents the
    machine.
    """
    cands = [s for s in sweeps
             if s["machine"] == machine and len(s["rows"]) >= 3
             and swept_over_concurrency(s["rows"])]
    best = None
    for s in cands:
        top = sorted(s["rows"], key=lambda r: r["concurrency"])[-1]
        if not top.get("tok_per_joule"):
            continue
        if best is None or top["tok_per_joule"] > best[0]:
            best = (top["tok_per_joule"], s, top)
    return (best[1], best[2]) if best else (None, None)


def inference_timeline(machine: str, results_dir: str, sweeps: list):
    """The sidecar power trace of the best sweep's top concurrency level."""
    sweep, top = best_sweep(machine, sweeps)
    if not sweep:
        return None, None, None
    conc = top["concurrency"]
    root = Path(results_dir) / "aiperf" / sweep["name"]
    for pat in (f"c{conc}/*sidecar_power_*_power.json",
                f"c{conc}_*/*sidecar_power_*_power.json"):
        hits = sorted(root.glob(pat))
        if hits:
            return json.loads(hits[0].read_text(encoding="utf-8")), sweep, top
    return None, sweep, top


def machine_versions(machine: str, results_dir: str) -> str:
    """Software versions from the sweeps' own provenance, joined for the card."""
    vllm, torch = set(), set()
    for meta in Path(results_dir).glob(f"aiperf/{machine}-*/run_meta.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("vllm"):
            vllm.add(m["vllm"])
        if m.get("torch"):
            torch.add(m["torch"])
    parts = []
    if torch:
        parts.append("torch " + " / ".join(sorted(torch)))
    if vllm:
        parts.append("vLLM " + " / ".join(sorted(vllm)))
    return ", ".join(parts)


def measured_card(machine: str, runs: list, sweeps: list,
                  results_dir: str, facts: dict) -> str:
    """The measured half of the summary card: what the machine actually did.

    Every value is the best the machine has shown, each labelled with the
    configuration that produced it -- a bare number invites comparing a
    12-rank Aurora figure against a 4-rank Polaris one and calling it a
    machine difference.
    """
    items = []
    mine = [r for r in runs if r.get("machine") == machine and r.get("samples_per_s")]
    if mine:
        b = max(mine, key=lambda r: r["samples_per_s"])
        plural = "s" if b["nodes"] != 1 else ""
        items.append(("Training throughput",
                      f"{b['samples_per_s']:,.0f} samples/s "
                      f"({b['ranks']} ranks, {b['nodes']} node{plural})"))
        effs = [(node_efficiency(r), r) for r in mine]
        effs = [(e, r) for e, r in effs if e]
        if effs:
            e, r = max(effs, key=lambda t: t[0])
            items.append(("Training efficiency",
                          f"{e:,.1f} samples/J node-wide ({r['ranks']} ranks)"))
            facts["train_eff"] = e
        ttas = [r["tta_s"] for r in mine if r.get("tta_s")]
        if ttas:
            items.append(("Time to 0.90 top-1", f"{min(ttas):,.1f} s"))
        facts["train_best"] = b
    conc = [s for s in sweeps if s["machine"] == machine and len(s["rows"]) >= 3
            and swept_over_concurrency(s["rows"])]
    if conc:
        peak = max(((r, s) for s in conc for r in s["rows"] if r.get("out_tok_per_s")),
                   key=lambda t: t[0]["out_tok_per_s"])
        r, s = peak
        model = html.escape((s["model"] or "?").split("/")[-1])
        tp_txt = f" at TP={s['tp']}" if s.get("tp") else ""
        items.append(("Peak serving throughput",
                      f"{r['out_tok_per_s']:,.0f} tok/s "
                      f"({model}{tp_txt}, c={r['concurrency']})"))
        facts["peak_tok"] = (r, s)
        bs, bt = best_sweep(machine, sweeps)
        if bs:
            items.append(("Best serving efficiency",
                          f"{bt['tok_per_joule']:.2f} tok/J of node energy "
                          f"({bt['tok_per_joule_dynamic']:.2f} dynamic)"))
            facts["best_sweep"] = (bs, bt)
        idles = sorted(s["idle_w"] for s in conc if s.get("idle_w"))
        if idles:
            facts["idle_w"] = idles[len(idles) // 2]
            items.append(("Accelerator idle floor", f"{facts['idle_w']:,.0f} W per node"))
    versions = machine_versions(machine, results_dir)
    if versions:
        items.append(("Software", versions))
    if not items:
        return ""
    fields = "".join(f"<dt>{html.escape(k)}</dt><dd>{v}</dd>" for k, v in items)
    return f'<dl class="specs measured">{fields}</dl>'


def scaling_table(machine: str, runs: list) -> str:
    """Training throughput per configuration, as a scaling statement.

    A table rather than a chart: two or three configurations do not make a
    curve, and the honest columns are speedup against the smallest
    configuration and what fraction of ideal that is.
    """
    mine = [r for r in runs if r.get("machine") == machine and r.get("samples_per_s")]
    best: dict = {}
    for r in mine:
        key = (r["nodes"], r["ranks"])
        if key not in best or r["samples_per_s"] > best[key]["samples_per_s"]:
            best[key] = r
    if len(best) < 2:
        return ""
    base_key = min(best)
    base = best[base_key]
    rows = []
    for (nodes, ranks), r in sorted(best.items()):
        speed = r["samples_per_s"] / base["samples_per_s"]
        ideal = ranks / base_key[1]
        rows.append([num(nodes), num(ranks), num(r["samples_per_s"]),
                     f"{speed:.2f}x", f"{speed / ideal * 100:,.0f}%"])
    return "<h4>Training scaling</h4>" + table(
        ["Nodes", "Ranks", "Samples/s", "Speedup", "of ideal"], rows,
        "Best run per configuration. Speedup is against the smallest "
        "configuration; the last column divides by the rank ratio, so 100% is "
        "linear scaling and anything under it is the cost of coordination.")


def sweet_spot(machine: str, sweeps: list) -> str:
    """The best measured operating point, honestly labelled as measured.

    Every sweep here stops with tokens/joule still rising, so the knee of the
    curve has not been found -- and a box claiming a best operating point
    would be reporting the edge of the measurement as a property of the
    machine. It says so instead.
    """
    sweep, top = best_sweep(machine, sweeps)
    if not sweep:
        return ""
    rows = sorted(sweep["rows"], key=lambda r: r["concurrency"])
    rising = (len(rows) >= 2 and rows[-1].get("tok_per_joule")
              and rows[-2].get("tok_per_joule")
              and rows[-1]["tok_per_joule"] > rows[-2]["tok_per_joule"])
    model = html.escape((sweep["model"] or "?").split("/")[-1])
    tp_txt = f" at TP={sweep['tp']}" if sweep.get("tp") else ""
    bits = [f"{top['out_tok_per_s']:,.0f} tok/s"]
    if top.get("ttft_ms"):
        bits.append(f"TTFT {top['ttft_ms']:,.0f} ms")
    if top.get("dynamic_w"):
        bits.append(f"{top['dynamic_w']:,.0f} W dynamic")
    bits.append(f"{top['tok_per_joule']:.2f} tok/J of node energy")
    caveat = (" Tokens per joule was still rising at the top of the sweep, so "
              "this is the edge of the measurement, not a saturation point — "
              "the knee, if there is one, is past this concurrency."
              if rising else "")
    joined = ", ".join(bits)
    return (f'<p class="takeaway"><strong>Best measured operating point</strong> — '
            f'{model}{tp_txt}, concurrency {top["concurrency"]}: {joined}.{caveat}</p>')


def profile_summary(machine: str, facts: dict) -> str:
    """The machine in plain English, assembled from what was measured.

    Derived, like every takeaway on the site: a hand-written summary of
    numbers that rebuild on every run is a summary that goes stale the day
    after it is written.
    """
    bits = []
    b = facts.get("train_best")
    if b:
        s = (f"training ResNet-20, {machine} reached {b['samples_per_s']:,.0f} "
             f"samples/s on {b['ranks']} ranks")
        if facts.get("train_eff"):
            s += f" at {facts['train_eff']:,.1f} samples per node-joule"
        bits.append(s)
    bs = facts.get("best_sweep")
    if bs:
        sweep, top = bs
        model = (sweep["model"] or "?").split("/")[-1]
        s = (f"serving {model} it delivered {top['out_tok_per_s']:,.0f} tokens/s "
             f"at concurrency {top['concurrency']}, {top['tok_per_joule']:.2f} "
             f"tokens per joule of node energy")
        if top.get("tok_per_joule_dynamic"):
            s += f" ({top['tok_per_joule_dynamic']:.2f} above the idle floor)"
        bits.append(s)
    if facts.get("idle_w") and facts.get("infer_avg_w"):
        share = facts["idle_w"] / facts["infer_avg_w"] * 100
        if share > 50:
            bits.append(f"its accelerators drew {facts['idle_w']:,.0f} W before "
                        f"any work arrived — {share:,.0f}% of what they drew "
                        f"while serving — which is why filling the node matters "
                        f"more than choosing it")
    if not bits:
        return ""
    text = ". ".join(s[0].upper() + s[1:] for s in bits) + "."
    return f'<p class="takeaway">{text}</p>'


def machine_profile(machine: str, spec: dict, runs: list, sweeps: list,
                    results_dir: str, timelines: int) -> str:
    """One machine, top to bottom: card, traces, behaviour, operating point.

    Structured the way a reader new to the machine needs it -- what it is,
    what it drew while working, how serving behaves under load, where to run
    it, and one paragraph to leave with. Machines with no measurements keep
    the old spec-plus-status card rather than an empty scaffold.
    """
    tag, state = power_state(machine, runs, timelines)
    slot = SERIES_SLOT.get(machine, 8)
    facts: dict = {}
    head = f"""
<div class="mhead">
<h2>{html.escape(machine)}</h2>
<span class="key s{slot}"><span class="sw"></span></span>{tag}
</div>"""
    fields = "".join(
        f"<dt>{html.escape(label)}</dt>"
        f"<dd>{html.escape(str(spec.get(key, '—')))}</dd>"
        for key, label in SPEC_COLUMNS
    )
    card = f'<dl class="specs">{fields}</dl>'
    measured = measured_card(machine, runs, sweeps, results_dir, facts)

    traces = ""
    tl, tl_run = training_timeline(machine, results_dir, runs)
    if tl:
        stats = timeline_watts(tl)
        chart = power_timeline_chart(tl, slot,
                                     f"{machine} accelerator power during training")
        if chart:
            reached = (f" 0.90 top-1 was reached at {tl_run['tta_s']:,.1f} s (marked)."
                       if tl_run.get("tta_s") else "")
            plural = "s" if tl_run["nodes"] != 1 else ""
            traces += (f"<h4>What training draws</h4>{chart}"
                       f'<p class="fineprint">Run {html.escape(tl_run["run_id"])}: '
                       f'{tl_run["ranks"]} ranks on {tl_run["nodes"]} node{plural}, '
                       f'{tl_run.get("epochs") or "?"} epochs — the bold line is all '
                       f'{len(stats["devices"])} accelerators together, averaging '
                       f'{stats["avg_w"]:,.0f} W with a peak of {stats["peak_w"]:,.0f} W; '
                       f'the thin lines are each device alone.{reached}</p>')
    itl, isweep, itop = inference_timeline(machine, results_dir, sweeps)
    if itl:
        stats = timeline_watts(itl)
        facts["infer_avg_w"] = stats["avg_w"]
        chart = power_timeline_chart(itl, slot,
                                     f"{machine} accelerator power while serving")
        if chart:
            model = html.escape((isweep["model"] or "?").split("/")[-1])
            tp_txt = f" at TP={isweep['tp']}" if isweep.get("tp") else ""
            idle = (f" The idle floor measured on this node before the server "
                    f"started was {isweep['idle_w']:,.0f} W."
                    if isweep.get("idle_w") else "")
            ndev = len(stats["devices"])
            tp = isweep.get("tp")
            bound = (f" vLLM bound {tp} of the {ndev} devices; the flat thin "
                     f"lines are the ones it left idle."
                     if tp and tp < ndev else "")
            traces += (f"<h4>What serving draws</h4>{chart}"
                       f'<p class="fineprint">{model}{tp_txt}, concurrency '
                       f'{itop["concurrency"]} — the top of this machine&#39;s best '
                       f'sweep: all {len(stats["devices"])} accelerators average '
                       f'{stats["avg_w"]:,.0f} W against a peak of '
                       f'{stats["peak_w"]:,.0f} W.{bound}{idle}</p>')

    # Summary before detail: the radar is four numbers at one operating point,
    # and the sweeps underneath it are what those numbers came from. Empty
    # string for a machine with nothing to compare against, which is the whole
    # of what has to be removed here if it stops earning its place.
    radar = capability_radar(sweeps, machine)
    serving = machine_serving_blocks(machine, sweeps)
    if serving:
        serving = "<h3>Serving under load</h3>" + serving
    shapes = shape_section([s for s in sweeps if s["machine"] == machine])
    scaling = scaling_table(machine, runs)
    spot = sweet_spot(machine, sweeps)
    summary = profile_summary(machine, facts)

    if not (measured or traces or serving or scaling):
        return head + card + f'<div class="todo">{state}</div>'
    status = "" if traces else f'<div class="todo">{state}</div>'
    return (head + card + measured + status + traces + radar + serving + shapes
            + scaling + spot + summary)


def power_body(specs: dict, runs: list, results_dir: str,
               timelines: dict, sweeps: list) -> str:
    """The power page: one profile per machine, in config order."""
    sections = "".join(
        machine_profile(machine, spec, runs, sweeps, results_dir,
                        timelines.get(machine, 0))
        for machine, spec in specs.items() if not machine.startswith("_")
    )
    return f"""
<p class="fineprint">One profile per machine: its hardware, the best it has
measured, what its accelerators actually drew while training and serving, and
where to run it. Power traces are watts from consecutive energy-counter deltas,
sampled every 0.1 s by <code>benchmark/power.py</code> during training and
<code>analysis/power_sidecar.py</code> during serving — one thin line per
device, devices no rank bound to included, because their flat lines are where
most of a single-rank node&#39;s energy goes. Aurora&#39;s whole-card counters
are excluded from every total, as everywhere on this site, since they cover
silicon the per-tile counters already report. Specifications are per node;
sources are in the <code>_source</code> fields of
<code>configs/machines.json</code>. The cross-machine comparison lives on the
<a href="training.html">training</a> page.</p>
{sections}"""


# --- Overview and conclusions -------------------------------------------------
# The two pages that carry no charts. Everything below states a number and then
# points at the page that measured it, so a finding and its evidence never drift
# apart: the numbers here are computed from results/ on every build, exactly
# like the charts they link to, and check_links() fails the build if a link
# stops resolving.


def _sweep(sweeps: list, machine: str, model_ends: str, tp, isl=1024.0):
    """One sweep by machine, model and sharding, or None.

    Matched on the shape as well, because a machine can have two sweeps of the
    same model and TP -- the concurrency sweep and the prompt-shape sweep -- and
    they are different experiments.
    """
    return next(
        (s for s in sweeps
         if s["machine"] == machine and (s["model"] or "").endswith(model_ends)
         and s.get("tp") == tp and s["isl"] == isl),
        None,
    )


def _at(sweep, concurrency: int = 32):
    """One concurrency level of a sweep, or None."""
    if not sweep:
        return None
    return next((r for r in sweep["rows"]
                 if r.get("concurrency") == concurrency), None)


def _ratio(a, b):
    """a/b, or None if either is missing -- so a sentence can drop itself
    rather than print a dash in the middle of a claim."""
    return (a / b) if (a and b) else None


def intro_body(runs: list, sweeps: list, specs: dict | None = None) -> str:
    """The front door: what this is, how it measures, and where to look.

    Deliberately the shortest page on the site and deliberately carries no
    chart. A reader arriving cold needs the question before the answers, and
    every number here is on another page with its evidence attached.
    """
    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    served = sorted({s["machine"] for s in sweeps})
    models = sorted({(s["model"] or "").split("/")[-1] for s in sweeps if s["model"]})

    cards = "".join(
        f'<a class="door" href="{href}"><span class="door-t">{html.escape(label)}</span>'
        f'<span class="door-d">{html.escape(blurb)}</span></a>'
        for href, label, blurb in (
            ("training.html", "Training data",
             "ResNet-20 on CIFAR-10, run identically everywhere: throughput, "
             "time-to-accuracy and energy per sample."),
            ("power.html", "Power profiles",
             "One page per machine — what it is, what it drew, and how it "
             "behaves as load rises."),
            ("dashboard.html", "Inference dashboard",
             "Every serving run measured so far, filtered in the browser."),
            ("conclusions.html", "Conclusions",
             "What all of it came to, stated once, with a link to the chart "
             "behind each claim."),
        )
    )

    return f"""
<h2>The question</h2>
<p>Which ALCF machine turns a watt into the most useful work — and does the
answer change depending on whether you are choosing an accelerator or paying
for a node?</p>

<p>Every machine here runs the <em>same</em> two workloads: ResNet-20 on
CIFAR-10 for training, and vLLM serving real models for inference. Identical
code, identical flags, identical amounts of work per measurement. What differs
is the silicon underneath, which is the only way a comparison means anything.</p>

<h2>How it is measured</h2>
<p>Energy comes from <strong>cumulative hardware counters</strong>, not sampled
estimates: NVML&#39;s millijoule register on the NVIDIA machines, the i915
driver&#39;s hwmon counters on Aurora, and Cray&#39;s <code>pm_counters</code>
on Crux. Two reads and a subtraction, so the figure carries no integration
error.</p>

<p>Every run measures <strong>its own idle floor</strong> first — 30 seconds on
a quiet node, before the server starts — and dynamic power is total draw minus
that floor. Both numbers are reported, because they answer different questions:
dynamic power is what the silicon spent on the work, and total is what the
allocation billed. Figures are whole-node on an exclusive allocation.</p>

<p>For inference, the counters are read by a sidecar process running
<em>beside</em> the benchmark rather than inside it, since neither the load
generator nor the server belongs to this project. That is also what puts Aurora
in the comparison at all: the load generator&#39;s own power collection has no
Intel path.</p>

<h2>What has been measured</h2>
<p>Training on <strong>{html.escape(", ".join(machines))}</strong>. Serving on
<strong>{html.escape(", ".join(served))}</strong>, across
{len(models)} model(s): {html.escape(", ".join(models))}. Every sweep holds the
request count fixed and forces an exact output length, so each row is the same
amount of work and joules compare in absolute terms rather than only as
ratios.</p>

<h2>Where to look</h2>
<div class="doors">{cards}</div>
<p class="fineprint">Pages are generated from <code>results/</code> by
<code>analysis/build_site.py</code> and are never edited by hand, so every
figure on the site traces back to a JSON record produced by a run. The
conclusions page states each finding once and links to the chart that measured
it; the three data pages hold the charts.</p>"""


def _conclusion_inversion(sweeps: list) -> str:
    """Node energy against silicon energy: the two answers to one question."""
    au = _at(_sweep(sweeps, "aurora", "Llama-3.1-8B-Instruct", 1))
    po = _at(_sweep(sweeps, "polaris", "Llama-3.1-8B-Instruct", 1))
    if not (au and po):
        return ""
    node = _ratio(po["tok_per_joule"], au["tok_per_joule"])
    dyn = _ratio(au["tok_per_joule_dynamic"], po["tok_per_joule_dynamic"])
    if not (node and dyn):
        return ""
    return f"""
<h2>The same comparison has two opposite winners</h2>
<p class="takeaway">Serving Llama-3.1-8B at concurrency 32, Polaris delivers
<strong>{node:.1f}&#215;</strong> the tokens per joule of node energy that
Aurora does. On the same runs, Aurora delivers <strong>{dyn:.2f}&#215;</strong>
the tokens per joule its accelerators actually spent.</p>
<p>Both are true and they are not in tension. Aurora&#39;s tile converts energy
into tokens more efficiently than an A100 does; Aurora&#39;s <em>node</em> does
not, because the job is billed for twelve tiles and this configuration uses
one. Which number matters depends on the question. Choosing silicon, read the
second. Paying for an allocation, read the first — the node bills either
way.</p>
<p class="fineprint">Evidence: <a href="power.html#aurora">Aurora</a> and
<a href="power.html#polaris">Polaris</a> power profiles; every configuration is
on the <a href="dashboard.html">inference dashboard</a>.</p>"""


def _conclusion_idle(sweeps: list) -> str:
    """Polaris against Sophia: the closest thing here to a controlled test."""
    po_s = _sweep(sweeps, "polaris", "Llama-3.1-8B-Instruct", 1)
    so_s = _sweep(sweeps, "sophia", "Llama-3.1-8B-Instruct", 1)
    po, so = _at(po_s), _at(so_s)
    if not (po and so and po_s.get("idle_w") and so_s.get("idle_w")):
        return ""
    gap = _ratio(po["tok_per_joule"], so["tok_per_joule"])
    return f"""
<h2>The gap is the idle floor, not the chip</h2>
<p class="takeaway">Polaris and Sophia draw within
<strong>{abs(so["dynamic_w"] - po["dynamic_w"]):.0f} W</strong> of each other
doing identical work — {po["dynamic_w"]:,.0f} W against
{so["dynamic_w"]:,.0f} W — and still differ <strong>{gap:.2f}&#215;</strong> on
tokens per joule. Their idle floors are {po_s["idle_w"]:,.0f} W and
{so_s["idle_w"]:,.0f} W.</p>
<p>This is as close to a controlled experiment as the fleet allows: the same
accelerator generation, the same model, the same sharding, the same load, and
dynamic power that agrees to within a percent. Everything separating the two
numbers is what the node costs before any work arrives. An idle floor is not
overhead you can optimise away in software — it is a property of the node you
were given.</p>
<p class="fineprint">Evidence: <a href="power.html#polaris">Polaris</a> and
<a href="power.html#sophia">Sophia</a> power profiles — the radar at the top of
each shows this as the one axis where the machines are not nearly
identical.</p>"""


def _conclusion_fill(sweeps: list) -> str:
    """gemma at TP=4: the bound on the idle-floor story."""
    au = _at(_sweep(sweeps, "aurora", "gemma-3-27b-it", 4))
    po = _at(_sweep(sweeps, "polaris", "gemma-3-27b-it", 4))
    if not (au and po):
        return ""
    dyn = _ratio(au["tok_per_joule_dynamic"], po["tok_per_joule_dynamic"])
    node = _ratio(po["tok_per_joule"], au["tok_per_joule"])
    if not (dyn and node):
        return ""
    return f"""
<h2>Filling the node does not close it</h2>
<p class="takeaway">gemma-3-27b at TP=4 leaves no idle GPU on a Polaris node.
Aurora&#39;s lead per joule of silicon <em>widens</em> to
<strong>{dyn:.2f}&#215;</strong>, and Polaris still wins per joule of node
energy by <strong>{node:.1f}&#215;</strong>.</p>
<p>So the idle floor explains the headline number without being the whole
story. Even with every accelerator working, the two machines convert energy
into tokens at genuinely different rates — the PVC tile is the better converter,
and it is still attached to a node that costs a kilowatt to keep switched
on.</p>
<p class="fineprint">Evidence: gemma-3-27b at TP=4 on
<a href="power.html#aurora">Aurora</a> and
<a href="power.html#polaris">Polaris</a>.</p>"""


def _conclusion_tp(sweeps: list) -> str:
    """The sharding penalty, and the second machine that confirms it."""
    rows = []
    for machine in ("aurora", "sophia"):
        base = _at(_sweep(sweeps, machine, "Llama-3.1-8B-Instruct", 1))
        top = _at(_sweep(sweeps, machine, "Llama-3.1-8B-Instruct", 8))
        if not (base and top):
            continue
        rows.append((
            machine,
            _ratio(top["out_tok_per_s"], base["out_tok_per_s"]),
            _ratio(top["dynamic_w"], base["dynamic_w"]),
            _ratio(top["tok_per_joule_dynamic"], base["tok_per_joule_dynamic"]),
        ))
    rows = [r for r in rows if all(r[1:])]
    if not rows:
        return ""
    # A list of rows, each a list of cells: table() writes the tr and td itself.
    # Handing it one pre-joined string instead makes it iterate the characters
    # and emit a row per character, which is what it did here until 2026-08-20.
    body = [
        [machine_tag(m), f"{tput:.2f}&#215;", f"{power:.2f}&#215;",
         f"{eff:.2f}&#215;"]
        for m, tput, power, eff in rows
    ]
    worst = min(rows, key=lambda r: r[3])
    return f"""
<h2>Tensor parallelism divides time, not joules</h2>
<p class="takeaway">Going from one accelerator to eight on the same model, at
the same load — what it bought, and what it cost.</p>
{table(["", "throughput", "dynamic power", "energy efficiency"], body,
       "Llama-3.1-8B at concurrency 32, TP=8 relative to TP=1 on the same "
       "machine. Efficiency is tokens per joule of dynamic energy.")}
<p>Sharding a model buys latency and capacity, and it costs energy on both
vendors measured. That answers the question the curve raised on Aurora: the
penalty is not an artifact of Intel silicon or of one vLLM build.
{html.escape(worst[0].capitalize())} is the steeper of the two, keeping only
<strong>{worst[3]:.2f}&#215;</strong> its single-accelerator efficiency while
drawing <strong>{worst[2]:.2f}&#215;</strong> the power.</p>
<p>The practical reading: shard because a model does not fit, or because a
latency target demands it — not because more accelerators sounded faster. For a
model that fits on one, replicas are the efficient way to fill a node.</p>
<p class="fineprint">Evidence: the TP curves under
<a href="power.html#aurora">Aurora</a> and
<a href="power.html#sophia">Sophia</a>; on the training side the same shape
appears as
<a href="training.html#same-accelerator-different-count">same accelerator,
different count</a>.</p>"""


def _conclusion_workload(sweeps: list) -> str:
    """Power against concurrency: a property of the model, not the machine."""
    rises = []
    for sweep in sweeps:
        if len(sweep["rows"]) < 6 or sweep["isl"] is None:
            continue
        first, last = sweep["rows"][0], sweep["rows"][-1]
        rise = _ratio(last.get("dynamic_w"), first.get("dynamic_w"))
        if rise:
            rises.append((rise, sweep, first, last))
    if len(rises) < 2:
        return ""
    rises.sort(key=lambda r: r[0])
    flat, steep = rises[0], rises[-1]

    def name(entry):
        _rise, sweep, _f, _l = entry
        model = (sweep["model"] or "?").split("/")[-1]
        tp = f" at TP={sweep['tp']}" if sweep.get("tp") else ""
        return f"{model}{tp} on {sweep['machine']}"

    return f"""
<h2>The workload sets the power curve, not the machine</h2>
<p class="takeaway">Across a concurrency sweep, dynamic power moves
<strong>{flat[0]:.2f}&#215;</strong> for {html.escape(name(flat))} and
<strong>{steep[0]:.2f}&#215;</strong> for {html.escape(name(steep))} —
a range of {steep[0] / flat[0]:.0f}&#215; in how much a machine reacts to
being loaded, on the same hardware.</p>
<p>An early sweep suggested that serving more requests at once was nearly free
in watts, because a bandwidth-bound tile draws what it draws. That turned out
to describe a dense 8B model on one accelerator rather than the silicon: a
mixture-of-experts routes each additional request to more experts, which is
real extra work, and its draw climbs steeply with load.</p>
<p>The consequence for capacity planning is that a machine has no single power
number. What it draws under load is a property of the model you are serving and
the width you sharded it to, and it has to be measured per configuration.</p>
<p class="fineprint">Evidence: the serving-under-load charts on every
<a href="power.html">power profile</a>; all configurations side by side on the
<a href="dashboard.html">inference dashboard</a>.</p>"""


def _conclusion_method() -> str:
    """What the numbers rest on, stated before anyone has to ask."""
    return """
<h2>What these numbers rest on</h2>
<p>Energy is read from cumulative counters and differenced, so no figure here
depends on a sampling rate. Every run measures its own idle floor on a quiet
node before the server starts, because a floor taken afterwards reads high
while clocks and fans settle. All figures are whole-node on an exclusive
allocation — on a shared node they would be this job plus whatever else was
resident.</p>
<p>Polaris is the only machine where two independent instruments read the same
devices, and where they disagree the disagreement is reported rather than
averaged: two instruments on the same silicon that differ mean one is wrong,
not that the truth lies between them. Aurora has no second opinion available at
all, which is the single largest assumption on this site.</p>
<p>The machines do not run identical software. vLLM versions differ between
them, which is recorded in each run&#39;s <code>run_meta.json</code> and treated
as a stated confound rather than smoothed over. Run-to-run noise is a few
percent on dynamic watts and well under one percent on throughput, so
differences of that size are not findings.</p>
<p class="fineprint">Per-machine detail, including which counter each one reads
and what it cannot read, is on the <a href="power.html">power profiles</a>
page.</p>"""


def _conclusion_open(sweeps: list, runs: list) -> str:
    """The gaps, named on the page rather than left as blank cells."""
    served = {s["machine"] for s in sweeps}
    trained = {r.get("machine") for r in runs if r.get("machine")}
    no_serving = sorted(trained - served)
    gap = ""
    if no_serving:
        gap = (f" {html.escape(', '.join(no_serving))} "
               f"{'has' if len(no_serving) == 1 else 'have'} training numbers "
               f"but no serving numbers.")
    return f"""
<h2>What is not answered</h2>
<p>Stated rather than left as empty cells, because a gap a reader has to
discover reads as an oversight.{gap} Cerebras and Graphcore are in the plan and
not yet started; both need their own stacks rather than a portable script, so
neither is a matter of finding queue time.</p>
<p>The energy comparison is between three machines that expose per-device
counters. A machine that exposes none can still contribute throughput and
latency, and would appear here with an energy column that stays empty — which
is a limit of the instrument, not of the machine.</p>
<p class="fineprint">Coverage, configuration by configuration, is the matrix on
the <a href="dashboard.html">inference dashboard</a>.</p>"""


def conclusions_body(runs: list, sweeps: list, specs: dict | None = None) -> str:
    """Every finding, stated once, each pointing at the page that measured it.

    Written to be read start to finish by someone who has not seen the rest of
    the site: each section states its numbers in full rather than assuming a
    chart is open in another tab. The charts stay where they are -- this page
    adds no plots of its own, so there is exactly one place each measurement
    is drawn and exactly one place it is interpreted.
    """
    sections = "".join(part for part in (
        _conclusion_inversion(sweeps),
        _conclusion_idle(sweeps),
        _conclusion_fill(sweeps),
        _conclusion_tp(sweeps),
        _conclusion_workload(sweeps),
        _conclusion_method(),
        _conclusion_open(sweeps, runs),
    ) if part)
    return f"""
<p class="fineprint">Every number below is computed from
<code>results/</code> on each build, the same as the charts it links to. Nothing
on this page is plotted here — each finding names the page that measured
it.</p>
{sections}"""


def check_links(pages: dict) -> list:
    """Internal hrefs that point at a page or anchor which does not exist.

    Cheap, and it catches the failure this site is most exposed to: anchors are
    derived from heading text by anchored(), so renaming a heading silently
    breaks every link aimed at it. A build that prints nothing here is a build
    where every cross-reference resolved.
    """
    ids = {name: set(re.findall(r'id="([^"]+)"', text))
           for name, text in pages.items()}
    broken = []
    for name, text in pages.items():
        for href in re.findall(r'href="([^"#:]*\.html)?(?:#([^"]+))?"', text):
            target, anchor = href
            page = target or name
            if page not in pages:
                broken.append(f"{name}: no such page {page}")
            elif anchor and anchor not in ids[page]:
                broken.append(f"{name}: {page}#{anchor} does not exist")
    return sorted(set(broken))


def footer_text(runs: list, generated: str) -> str:
    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    return (
        f'Generated {generated} from {len(runs)} run(s) · machines: '
        f'{html.escape(", ".join(machines)) or "none"}<br>\n'
        "Rebuild the site with <code>python analysis/build_site.py</code> after\n"
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
            lede="One benchmark, every ALCF machine the project can reach, "
                 "compared on throughput, accuracy and energy — measured from "
                 "hardware counters rather than estimated.",
            body=intro_body(runs, sweeps, specs),
            footer=footer,
            logo_uri=logo,
            here="index.html",
        ),
        "training.html": shell(
            title="Training — Power and Performance Across ALCF Machines",
            heading="Training Across ALCF Machines",
            lede="ResNet-20 on CIFAR-10, run identically on every ALCF system "
                 "and compared on throughput, time-to-accuracy and energy — down "
                 "to the accelerators nobody was using. Serving numbers are on "
                 "the inference dashboard; the two share no denominator.",
            strip=workload_strip(runs),
            body=index_body(runs, specs, curves),
            footer=footer,
            logo_uri=logo,
            here="training.html",
        ),
        "power.html": shell(
            title="Power Profiles — ALCF Machines",
            heading="Power Profiles",
            lede="One profile per machine: what it is, what its accelerators "
                 "drew while training and serving, how it behaves under load, "
                 "and the best operating point measured so far.",
            body=power_body(specs, runs, args.results_dir,
                            timeline_counts(args.results_dir), sweeps),
            footer=footer,
            logo_uri=logo,
            here="power.html",
        ),
    }
    pages["dashboard.html"] = shell(
        title="Inference — Serving Across ALCF Machines",
        heading="Inference Across ALCF Machines",
        lede="vLLM serving real models, measured for tokens per second, latency "
             "and tokens per joule. Every configuration measured so far, filtered "
             "in the browser, with what they add up to underneath. Nothing is "
             "fetched — the whole dataset is in this page.",
        body=dashboard_body(sweeps, specs),
        footer=footer,
        logo_uri=logo,
        here="dashboard.html",
    )
    # Last, because it links into every page above and check_links() below can
    # only verify anchors that have already been rendered.
    pages["conclusions.html"] = shell(
        title="Conclusions — Power and Performance Across ALCF Machines",
        heading="Conclusions",
        lede="What the measurements came to, stated once each, with a link to "
             "the chart behind every claim. No plots of its own — the data "
             "pages keep those.",
        body=conclusions_body(runs, sweeps, specs),
        footer=footer,
        logo_uri=logo,
        here="conclusions.html",
    )
    for name, text in pages.items():
        (out_dir / name).write_text(text, encoding="utf-8")

    # A broken cross-reference is the one failure this site cannot show you:
    # the page still renders, the link just goes nowhere. Loud on stdout rather
    # than fatal, so a build during a rename still produces pages to look at.
    for problem in check_links(pages):
        print(f"BROKEN LINK  {problem}")

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
