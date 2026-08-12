"""Intel GPU energy counters, read straight from sysfs.

Split out of platform.py so a reader that needs nothing else can have it.
platform.py imports torch at module scope, which is correct for a file about
binding ranks to devices and wrong for anything that only wants to know what a
node is drawing -- analysis/power_sidecar.py runs inside the AIPerf virtualenv,
where torch is absent and installing it to read a sysfs file would be absurd.

Stdlib only, and no import of anything in this package except the EnergySource
record itself, so it stays importable from either environment.
"""

from __future__ import annotations

import re
from pathlib import Path

from .power import EnergySource

_DRM_CARD = re.compile(r"^card(\d+)$")


def intel_energy_counters():
    """Yield (card_index, hwmon_name, energy_path) for every readable i915
    energy counter on this node, cards in numeric order.

    Single place that knows the sysfs layout, so the per-rank counter and the
    node-wide enumeration cannot drift apart. /sys/class/drm also holds
    connector entries (card0-DP-1) and render nodes, hence the strict match on
    card<N>; and cards are sorted numerically because card10 sorts before card2
    as a string.
    """
    try:
        entries = list(Path("/sys/class/drm").iterdir())
    except OSError:
        return

    cards = [
        (int(m.group(1)), entry)
        for entry in entries
        if (m := _DRM_CARD.match(entry.name))
    ]
    for card, entry in sorted(cards):
        try:
            hwmons = sorted((entry / "device" / "hwmon").glob("hwmon*"))
        except OSError:
            continue
        for hwmon in hwmons:
            try:
                name = (hwmon / "name").read_text().strip()
            except OSError:
                continue
            counter = hwmon / "energy1_input"
            try:
                int(counter.read_text())
            except (OSError, ValueError):
                continue
            yield card, name, counter


def microjoule_reader(path: Path):
    """Closure over one counter path. A factory rather than a lambda in a loop,
    which would capture the loop variable and make every source read the last
    path."""

    def read():
        try:
            return int(path.read_text()) / 1e6
        except (OSError, ValueError):
            return None

    return read


def intel_energy_sources() -> list[EnergySource]:
    """Both tiles of every card on the node, plus each whole-card counter.

    Tile counters are what sum to a node total, one per torch device. The card
    counter additionally covers HBM and uncore, so it is larger than its two
    tiles combined -- recorded as an aggregate so the difference can finally be
    quantified instead of left as a caveat, but kept out of the totals so
    nothing is counted twice.

    device_index is the tile's index under ZE_FLAT_DEVICE_HIERARCHY=FLAT, which
    is what makes each tile a separate device. It is arithmetic on the card and
    tile numbers, not a torch query, so this function stays torch-free and the
    sidecar can call it too.
    """
    sources = []
    for card, name, counter in intel_energy_counters():
        if name.endswith(("_gt0", "_gt1")):
            tile = int(name[-1])
            sources.append(
                EnergySource(
                    key=f"card{card}.gt{tile}",
                    scope=f"xpu tile {tile} of card {card} (hwmon energy1_input)",
                    read=microjoule_reader(counter),
                    device_index=card * 2 + tile,
                )
            )
        elif "_gt" not in name:
            sources.append(
                EnergySource(
                    key=f"card{card}",
                    scope=f"whole card {card}, both tiles + HBM (hwmon energy1_input)",
                    read=microjoule_reader(counter),
                    device_index=None,
                    aggregate=True,
                )
            )
    return sources
