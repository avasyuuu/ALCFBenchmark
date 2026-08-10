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
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from summarize import load_runs

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


def build(runs: list) -> str:
    cards = ""
    for h in headline(runs):
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
            html.escape(r.get("when") or "—"),
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
            html.escape(r.get("when") or "—"),
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
            html.escape(r.get("when") or "—"),
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
<title>ALCF Machine Benchmark</title>
<style>
:root {{
  --bg:#fbfbfa; --fg:#1a1a18; --dim:#6b6b66; --line:#e2e2dd;
  --card:#ffffff; --accent:#8a5a2b; --tag:#f0e6d8;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16161a; --fg:#e8e8e4; --dim:#9a9a94; --line:#2c2c32;
          --card:#1e1e23; --accent:#d9a066; --tag:#33291d; }}
}}
:root[data-theme="dark"] {{ --bg:#16161a; --fg:#e8e8e4; --dim:#9a9a94;
  --line:#2c2c32; --card:#1e1e23; --accent:#d9a066; --tag:#33291d; }}
:root[data-theme="light"] {{ --bg:#fbfbfa; --fg:#1a1a18; --dim:#6b6b66;
  --line:#e2e2dd; --card:#ffffff; --accent:#8a5a2b; --tag:#f0e6d8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:3rem 1.25rem 5rem; }}
h1 {{ font-size:1.9rem; margin:0 0 .4rem; letter-spacing:-.02em; }}
h2 {{ font-size:1.15rem; margin:2.75rem 0 .75rem; letter-spacing:-.01em; }}
h2::before {{ content:""; display:inline-block; width:3px; height:.95em;
  background:var(--accent); margin-right:.55rem; vertical-align:-.08em; }}
.lede {{ color:var(--dim); margin:0 0 2rem; max-width:60ch; }}
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
th:first-child, td:first-child {{ text-align:left; }}
td {{ text-align:right; padding:.45rem .6rem;
  border-bottom:1px solid var(--line); white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:none; }}
.m {{ font-weight:600; }}
.scope {{ font-size:.74rem; color:var(--dim); white-space:normal; }}
.tag {{ background:var(--tag); color:var(--accent); font-size:.66rem;
  padding:.1rem .35rem; border-radius:4px; letter-spacing:.03em; }}
figure {{ margin:0; }}
figcaption {{ font-size:.78rem; color:var(--dim); margin-top:.6rem;
  max-width:70ch; }}
.note {{ border-left:2px solid var(--accent); padding:.15rem 0 .15rem .9rem;
  margin:1.1rem 0; color:var(--dim); font-size:.87rem; max-width:70ch; }}
footer {{ margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
  color:var(--dim); font-size:.78rem; }}
code {{ font-size:.85em; background:var(--tag); padding:.1rem .3rem;
  border-radius:3px; }}
</style></head><body><div class="wrap">

<h1>ALCF Machine Benchmark</h1>
<p class="lede">ResNet-20 on CIFAR-10, run identically across ALCF systems and
compared on throughput, time-to-accuracy and energy. One portable harness, one
result schema, one table.</p>

<h2>Machines</h2>
<div class="cards">{cards}</div>

<h2>Runs</h2>
{table(
    ["When","Machine","Nodes","Ranks","Prec","Global BS","Steps","Epochs",
     "Samples/s","Step ms","Best top-1","TTA s","MFU %"],
    run_rows,
    "Every complete run on disk, oldest first. MFU is reported for diagnosis, "
    "not as an efficiency claim — this workload runs at 0.5–1.4% of peak, so it "
    "largely measures kernel-launch overhead rather than the accelerator.",
)}

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

<h2>Reading these numbers</h2>
<div class="note">Runs marked <span class="tag">early</span> stopped before
their epoch budget, so their raw joules cover less training. Compare those on
Samples/J or joules-to-accuracy, never on the Joules column.</div>
<div class="note">Energy scope is vendor-specific and is printed with every row.
An accelerator-only figure excludes CPU, memory, NICs and cooling, so a node
draws well more than the sum of its accelerators.</div>

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
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not args.include_synthetic:
        runs = [r for r in runs if not r.get("synthetic")]
    if not runs:
        raise SystemExit(f"no complete runs found in {args.results_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(runs), encoding="utf-8")
    machines = sorted({r.get("machine") for r in runs if r.get("machine")})
    print(f"wrote {out} — {len(runs)} run(s), machines: {', '.join(machines)}")


if __name__ == "__main__":
    main()
