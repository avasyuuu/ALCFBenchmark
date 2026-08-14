"""NVIDIA GPU energy counters, read through NVML.

The CUDA twin of hwmon.py, and split out for the same reason: platform.py
imports torch at module scope, which is right for a file about binding ranks to
devices and wrong for one that only wants to know what a node is drawing.
analysis/power_sidecar.py runs inside an AIPerf virtualenv where torch may be a
different build than the server's, and it has no business importing one to read
a power counter.

Stdlib plus pynvml, and no import from this package except EnergySource.
"""

from __future__ import annotations

import time

from .power import EnergySource


def load_nvml():
    """An initialised pynvml module, or None when there is no library or driver.

    Returns None rather than raising because "this machine has no NVIDIA GPU" is
    an ordinary answer here -- Aurora and Crux both give it -- and the caller
    decides whether that is fatal.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def energy_reader(nvml, handle):
    """(read, kind) for one GPU, in joules.

    nvmlDeviceGetTotalEnergyConsumption is a cumulative millijoule counter since
    the last driver reload -- the same shape as Aurora's hwmon node, so the
    subtract-two-readings design carries over unchanged. It is Volta and newer;
    anything older falls back to integrating instantaneous power, which is named
    in the scope string because it is the weaker measurement: it misses whatever
    happens between samples.
    """
    try:
        nvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        def read():
            try:
                return nvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1e3
            except Exception:
                return None

        return read, "nvml energy counter"
    except Exception:
        pass

    state = {"t": None, "j": 0.0}

    def read():
        try:
            watts = nvml.nvmlDeviceGetPowerUsage(handle) / 1e3
        except Exception:
            return None
        now = time.perf_counter()
        if state["t"] is not None:
            state["j"] += watts * (now - state["t"])
        state["t"] = now
        return state["j"]

    return read, "nvml power sampling integrated (no energy counter)"


def nvml_energy_sources(visible: list | None = None) -> list[EnergySource]:
    """One source per GPU NVML can see, whether this job is using it or not.

    NVML enumerates every GPU on the node regardless of CUDA_VISIBLE_DEVICES,
    which is precisely what makes an unused device visible -- and it also means
    the two numberings diverge the moment that variable is set.

    `visible` is the CUDA_VISIBLE_DEVICES list, and passing it remaps
    device_index into torch's numbering so a rank's energy lands on the device
    that rank is actually bound to; a GPU masked out of the job gets None, since
    torch cannot address it at all but it is still drawing power. Pass None to
    label sources by their NVML index instead, which is what a sidecar wants:
    it never binds anything and its --bound-devices are physical.

    The two orderings agree only under CUDA_DEVICE_ORDER=PCI_BUS_ID. CUDA
    defaults to FASTEST_FIRST, which matches on a homogeneous node but is not
    guaranteed to; the submit scripts set it.
    """
    nvml = load_nvml()
    if nvml is None:
        return []
    try:
        count = nvml.nvmlDeviceGetCount()
    except Exception:
        return []

    sources = []
    for i in range(count):
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue
        read, kind = energy_reader(nvml, handle)
        if visible is None:
            index = i
        elif i in visible:
            index = visible.index(i)
        else:
            index = None  # masked out of this job, still drawing power
        sources.append(
            EnergySource(
                key=f"gpu{i}",
                scope=f"whole gpu {i} incl. HBM ({kind})",
                read=read,
                device_index=index,
            )
        )
    return sources
