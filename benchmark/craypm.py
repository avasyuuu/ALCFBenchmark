"""HPE Cray EX node-level energy counters, /sys/cray/pm_counters.

The one place a compute node reports power for silicon that is not an
accelerator: the whole node, the CPUs, and the memory, straight from the
chassis controller. hwmon can never see these -- which is why the component
breakdown on the site has been impossible -- and no ALCF ticket is needed if
the files simply exist, since they are world-readable where they are present
at all.

NOT YET VERIFIED ON AURORA. Aurora is HPE Cray EX, the platform these
counters belong to, but whether ALCF mounts them for user jobs is exactly
what the next allocation should check:

    ls /sys/cray/pm_counters/ && cat /sys/cray/pm_counters/energy

Every source is self-detecting: a node without the directory yields an empty
list and nothing else changes. Values read like "27725 J 1571230862955436 us"
-- number, unit, timestamp -- and only the number is taken.

Sources are marked aggregate=True on purpose. The node counter covers the
accelerators the hwmon/NVML sources already report, and cpu/memory overlap
the node; summing any of them into the accelerator total would double count.
They ride along in the timeline as labelled series, which is all a component
breakdown needs.
"""

from __future__ import annotations

from pathlib import Path

PM_DIR = Path("/sys/cray/pm_counters")

# Cumulative-joule files worth a series, and the label each gets. accel*
# variants exist on some Cray EX blades; globbed separately below.
_ENERGY_FILES = ("energy", "cpu_energy", "cpu0_energy", "cpu1_energy",
                 "memory_energy")


def _reader(path: Path):
    def read():
        try:
            return float(path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None
    return read


def cray_pm_energy_sources(node_is_aggregate: bool = True) -> list:
    """One EnergySource per readable pm_counters energy file.

    node_is_aggregate is the whole design decision. On a machine with
    accelerator counters the node figure overlaps them, so it must stay out of
    every total or the accelerator energy is counted twice -- that is Aurora,
    and the default. On a machine with NO accelerator counter there is nothing
    to overlap and the node figure IS the measurement; leaving it aggregate
    would exclude every source from the total and produce a run that sampled
    diligently and reported zero joules. That is Crux.

    The component counters (cpu, memory, accel*) are always aggregate: they are
    parts of the node figure, informative as series and wrong to add to it.
    """
    from .power import EnergySource

    if not PM_DIR.is_dir():
        return []
    names = [n for n in _ENERGY_FILES if (PM_DIR / n).is_file()]
    names += sorted(p.name for p in PM_DIR.glob("accel*_energy"))
    sources = []
    for name in names:
        read = _reader(PM_DIR / name)
        if read() is None:
            continue
        label = name.removesuffix("_energy") or "node"
        label = "node" if label == "energy" else label
        sources.append(EnergySource(
            key=f"pm.{label}",
            scope=f"cray pm_counters {name} (node-level, {label})",
            read=read,
            device_index=None,
            aggregate=node_is_aggregate if label == "node" else True,
        ))
    return sources
