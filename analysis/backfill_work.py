"""Reconstruct the `work` block on results that predate it.

    python analysis/backfill_work.py            # preview
    python analysis/backfill_work.py --apply    # write

set_work() was added part-way through, so earlier results record how fast a run
went and not how much it did. That leaves the site's Epochs column blank, and
keeps node-wide Samples/J -- the one energy ratio that compares across machines
-- uncomputable for every run that predates it.

Nothing here is guessed. Each field is read back out of the run's own curve,
which records one point per epoch with a cumulative step count:

    epochs_completed  max(epoch) over the curve
    epochs_requested  config.epochs
    steps             the last curve point's step
    samples_global    steps x config.global_batch_size
    samples_per_rank  steps x config.local_batch_size
    stopped_early     epochs_completed < epochs_requested

The two are cross-checked before anything is written: a curve must have exactly
one point per epoch and its final step count must equal epochs x steps-per-epoch
at the recorded batch size. A file that fails either check is skipped rather
than filled in, because a curve with gaps cannot say how much work happened
between them.

The block is marked `derived_from_curve` so nobody later reads it as something
the harness measured. This rewrites measurements, so like relabel_machine.py it
prints what it would do and changes nothing without --apply.

Stdlib only, so it runs on a login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def derive(blob: dict):
    """(work block, note) for one result, or (None, reason-it-was-skipped)."""
    if blob.get("work"):
        return None, "already has work"
    curve = blob.get("curve") or []
    cfg = blob.get("config") or {}
    if len(curve) < 1:
        return None, "no curve"

    epochs_req = cfg.get("epochs")
    gbs, lbs = cfg.get("global_batch_size"), cfg.get("local_batch_size")
    if not epochs_req or not gbs or not lbs:
        return None, "config missing epochs or batch size"

    epochs = [p.get("epoch") for p in curve]
    steps = curve[-1].get("step")
    if None in epochs or steps is None:
        return None, "curve missing epoch or step"

    # One point per epoch, numbered 1..N with no gaps. A curve that skipped
    # epochs would still have a final step count, and multiplying it out would
    # silently invent work that was never recorded.
    completed = max(epochs)
    if sorted(epochs) != list(range(1, completed + 1)):
        return None, f"curve is not one point per epoch (n={len(curve)}, max={completed})"

    # The step count has to agree with the epoch count at this batch size, or
    # the two halves of the record disagree and neither is trustworthy.
    per_epoch, remainder = divmod(steps, completed)
    if remainder or per_epoch < 1:
        return None, f"{steps} steps does not divide into {completed} epochs"

    return {
        "steps": steps,
        "epochs_completed": completed,
        "epochs_requested": epochs_req,
        "samples_global": steps * gbs,
        "samples_per_rank": steps * lbs,
        "stopped_early": completed < epochs_req,
        # Provenance, not decoration: these numbers were reconstructed from the
        # curve, not measured by the run that wrote the file.
        "derived_from_curve": True,
    }, f"{completed}/{epochs_req} epochs, {steps} steps ({per_epoch}/epoch)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--apply", action="store_true", help="write; otherwise preview only")
    args = ap.parse_args()

    filled = skipped = 0
    for path in sorted(Path(args.results_dir).glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if blob.get("kind") in ("allreduce_microbenchmark", "power_timeline"):
            continue

        work, why = derive(blob)
        if work is None:
            if why != "already has work":
                print(f"  skip  {path.name}: {why}")
                skipped += 1
            continue

        print(f"  fill  {path.name}: {why}")
        filled += 1
        if args.apply:
            blob["work"] = work
            # newline="\n" so a run on Windows does not rewrite every result
            # with CRLF for .gitattributes to normalise straight back.
            path.write_text(
                json.dumps(blob, indent=2, sort_keys=False),
                encoding="utf-8",
                newline="\n",
            )

    print()
    print(
        f"{filled} result(s) backfilled, {skipped} skipped."
        if args.apply
        else f"{filled} result(s) would change, {skipped} skipped. "
        "Re-run with --apply to write."
    )


if __name__ == "__main__":
    main()
