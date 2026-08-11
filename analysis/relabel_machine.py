"""Rename the machine a recorded run says it came from.

    python analysis/relabel_machine.py --from cuda --to polaris          # preview
    python analysis/relabel_machine.py --from cuda --to polaris --apply

Needed once, because CudaPlatform.name was "cuda" -- a backend, not a machine --
so early Polaris runs recorded machine="cuda". Left alone they would group as a
different system from later runs, and summarize.py's parallel-efficiency
comparison groups by machine.

This rewrites measurements, so it prints what it would do and changes nothing
until --apply. It only touches the label: no timing, energy or power value is
altered, and a run whose machine does not match --from is left alone.

Stdlib only, so it runs on a login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rename_to(path: Path, old: str, new: str) -> Path:
    """Swap the leading machine name in a filename, leaving the rest intact.

    Result files are <machine>_<workload>_<runid>.json and power sidecars are
    <machine>_<runid>_<host>_power.json -- both lead with the machine, and only
    the first field may change. A blind replace would also corrupt a workload or
    hostname that happened to contain the old name.
    """
    if not path.name.startswith(f"{old}_"):
        return path
    return path.with_name(f"{new}_{path.name[len(old) + 1:]}")


def plan(results_dir: Path, old: str, new: str):
    """(result path, new path, [(sidecar, new sidecar)]) for each matching run."""
    jobs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if blob.get("machine") != old:
            continue

        sidecars = []
        for name in (blob.get("power") or {}).get("timeline_files") or []:
            src = results_dir / "power" / name
            if src.exists():
                sidecars.append((src, rename_to(src, old, new)))
        jobs.append((path, rename_to(path, old, new), blob, sidecars))
    return jobs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--from", dest="old", required=True, help="machine name to replace")
    p.add_argument("--to", dest="new", required=True, help="machine name to write")
    p.add_argument(
        "--apply", action="store_true", help="actually write; otherwise preview only"
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    jobs = plan(results_dir, args.old, args.new)
    if not jobs:
        raise SystemExit(f"no runs with machine={args.old!r} in {results_dir}")

    for path, new_path, blob, sidecars in jobs:
        when = blob.get("timestamp_utc", "?")
        print(f"{path.name}  ({when})")
        print(f"  machine: {args.old} -> {args.new}")
        if path != new_path:
            print(f"  rename:  {path.name} -> {new_path.name}")
        for src, dst in sidecars:
            print(f"  sidecar: {src.name} -> {dst.name}")

        if not args.apply:
            continue

        for src, dst in sidecars:
            side = json.loads(src.read_text(encoding="utf-8"))
            side["machine"] = args.new
            dst.write_text(
                json.dumps(side, separators=(",", ":")),
                encoding="utf-8",
                newline="\n",
            )
            if dst != src:
                src.unlink()

        blob["machine"] = args.new
        if blob.get("power", {}).get("timeline_files"):
            blob["power"]["timeline_files"] = [
                rename_to(results_dir / "power" / n, args.old, args.new).name
                for n in blob["power"]["timeline_files"]
            ]
        # newline="\n" so a rewrite on Windows does not flip every result to
        # CRLF for .gitattributes to normalise straight back.
        new_path.write_text(
            json.dumps(blob, indent=2, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        if new_path != path:
            path.unlink()

    print()
    print(
        f"{len(jobs)} run(s) relabelled."
        if args.apply
        else f"{len(jobs)} run(s) would change. Re-run with --apply to write."
    )


if __name__ == "__main__":
    main()
