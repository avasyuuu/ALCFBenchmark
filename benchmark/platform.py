"""Platform abstraction — the only file that should know machine-specific details.

Adding a new machine means adding a Platform subclass here and nothing else.
Aurora is the reference implementation; Polaris/Sophia/Crux follow the same shape.
"""

from __future__ import annotations

import inspect
import os
import re
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .power import EnergySource

# Env vars that carry rank info, in priority order. Different launchers set
# different ones: torchrun sets RANK/LOCAL_RANK, Aurora's mpiexec (MPICH+PALS)
# sets PMI_RANK/PALS_LOCAL_RANKID, OpenMPI sets OMPI_*. Try them all rather
# than hard-coding one launcher.
_RANK_VARS = (
    "RANK",
    "PALS_RANKID",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
)
_WORLD_VARS = (
    "WORLD_SIZE",
    "PALS_NRANKS",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
)
_LOCAL_VARS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    # Polaris' MPICH sets this and none of the others. Without it _env_int falls
    # through to the default of 0, every rank on a node binds to cuda:0, and the
    # job runs -- four ranks oversubscribing one A100 and three idle, looking
    # like a mysterious 4x slowdown rather than a configuration error.
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)


def _env_int(names, default):
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    return default


def get_rank_info():
    """(rank, world_size, local_rank), resolved from whichever launcher is in use."""
    return (
        _env_int(_RANK_VARS, 0),
        _env_int(_WORLD_VARS, 1),
        _env_int(_LOCAL_VARS, 0),
    )


class Platform:
    """Base: single-process CPU. Also the Crux implementation."""

    name = "cpu"
    dist_backend = "gloo"

    def __init__(self, local_rank: int = 0):
        self.local_rank = local_rank
        self.device = torch.device("cpu")

    # --- device management -------------------------------------------------
    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        """Block until queued device work finishes. Required before any timing
        read — device kernels are asynchronous, so an un-synced timer measures
        launch time, not execution time."""

    def empty_cache(self) -> None:
        pass

    def peak_memory_bytes(self) -> int:
        return 0

    def reset_peak_memory(self) -> None:
        pass

    # --- compute -----------------------------------------------------------
    def autocast(self, dtype):
        if dtype in (None, torch.float32):
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def optimize(self, model, optimizer, dtype):
        """Hook for vendor graph optimizers (IPEX on Aurora). No-op by default."""
        return model, optimizer

    # --- energy ------------------------------------------------------------
    def energy_joules(self) -> float | None:
        """Cumulative device energy in joules, or None where unsupported.

        A running counter, not a rate: subtract two readings to get the energy
        of the interval between them. Counters reset when the driver reloads,
        so only differences are meaningful.

        Sampling watts and integrating would miss what happens between samples,
        and a training step here is 13 ms -- a counter has no such gap.
        """
        return None

    def energy_scope(self) -> str:
        """What the counter actually covers, recorded alongside the number so
        an accelerator-only figure is never mistaken for node power."""
        return "unsupported"

    def node_energy_sources(self) -> list[EnergySource]:
        """Every readable energy counter on this NODE, not just this rank's.

        energy_joules() above is deliberately per-rank: it answers "what did my
        device cost". This answers "what did the node's accelerators cost",
        including the ones no rank ever bound to -- which is the only way an
        idle device shows up in a result at all. Sampled by one rank per node,
        so a device with no rank on it still gets read.
        """
        return []

    # --- profiling ---------------------------------------------------------
    def profiler_activities(self) -> list:
        """Activities for torch.profiler. Each backend names its device
        activity differently (XPU / CUDA), and asking for one the build does
        not have raises, so every subclass adds its own on top of CPU."""
        from torch.profiler import ProfilerActivity

        return [ProfilerActivity.CPU]

    def profiler_sort_key(self) -> str:
        """Column to rank the profiler summary by. Self time on the device
        where the work actually happened."""
        return "self_cpu_time_total"

    # --- reporting ---------------------------------------------------------
    def device_name(self) -> str:
        return "cpu"

    def environment(self) -> dict:
        return {
            "platform": self.name,
            "hostname": socket.gethostname(),
            "device_name": self.device_name(),
            "torch_version": torch.__version__,
        }


_DRM_CARD = re.compile(r"^card(\d+)$")


def _hwmon_counters():
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


def _microjoule_reader(path: Path):
    """Closure over one counter path. A factory rather than a lambda in a loop,
    which would capture the loop variable and make every source read the last
    path."""

    def read():
        try:
            return int(path.read_text()) / 1e6
        except (OSError, ValueError):
            return None

    return read


class AuroraPlatform(Platform):
    """Aurora — Intel Data Center GPU Max (Ponte Vecchio) via IPEX + oneCCL.

    Device model: the `frameworks` module sets ZE_FLAT_DEVICE_HIERARCHY=FLAT,
    which makes each *tile* a separate device. So a node exposes 12 devices,
    not 6, and you bind ranks by index here rather than with
    gpu_tile_compact.sh — ALCF explicitly recommends against combining that
    script with the frameworks module.
    """

    name = "aurora"
    dist_backend = "ccl"

    def __init__(self, local_rank: int = 0):
        import intel_extension_for_pytorch as ipex  # noqa: F401

        self.ipex = ipex
        self.local_rank = local_rank
        if not torch.xpu.is_available():
            raise RuntimeError(
                "torch.xpu is not available. Did you `module load frameworks`?"
            )
        # One rank per tile: rank N on this node takes device N.
        self.device = torch.device(f"xpu:{local_rank % torch.xpu.device_count()}")
        torch.xpu.set_device(self.device)

        # PyTorch >= 2.7 ships XCCL as a native XPU collective backend. Older
        # Aurora stacks needed the separate oneccl_bindings_for_pytorch package,
        # which registered itself as "ccl" and is absent from the 2025.3
        # frameworks module. Prefer the built-in wherever it exists.
        if getattr(torch.distributed, "is_xccl_available", lambda: False)():
            self.dist_backend = "xccl"

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def synchronize(self) -> None:
        torch.xpu.synchronize(self.device)

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def peak_memory_bytes(self) -> int:
        return torch.xpu.max_memory_allocated(self.device)

    def reset_peak_memory(self) -> None:
        torch.xpu.reset_peak_memory_stats(self.device)

    def optimize(self, model, optimizer, dtype):
        return self.ipex.optimize(model, optimizer=optimizer, dtype=dtype)

    # --- energy ------------------------------------------------------------
    # Level Zero sysman is present and zesPowerGetEnergyCounter returns
    # ZE_RESULT_SUCCESS, but writes zeros for every power domain -- energy
    # telemetry is gated for unprivileged processes. The driver's hwmon nodes
    # expose the same counters and ARE readable, so read those directly.
    #
    #   /sys/class/drm/card<N>/device/hwmon/hwmon*/name -> i915 | i915_gt0 | i915_gt1
    #                                          energy1_input -> microjoules
    #
    # i915 is the whole card; i915_gt<T> is one tile. Tile-level is what we
    # want: one rank owns one tile, so each rank reads its own and nothing is
    # counted twice. The card figure additionally covers HBM and uncore, so it
    # is larger than the two tiles summed.
    def _energy_path(self):
        if hasattr(self, "_energy_file"):
            return self._energy_file

        self._energy_file = None
        self._energy_scope = "no readable hwmon energy counter"
        index = self.device.index or 0
        card, tile = index // 2, index % 2
        mine = [(n, p) for c, n, p in _hwmon_counters() if c == card]

        # Prefer this rank's tile; fall back to the whole card, which is still
        # a real measurement as long as the result says so.
        for want, scope in (
            (f"_gt{tile}", f"xpu tile {tile} of card {card} (hwmon energy1_input)"),
            ("", f"whole card {card}, both tiles + HBM (hwmon energy1_input)"),
        ):
            for name, counter in mine:
                matches = name.endswith(want) if want else "_gt" not in name
                if matches:
                    self._energy_file = counter
                    self._energy_scope = scope
                    return self._energy_file
        return None

    def node_energy_sources(self) -> list[EnergySource]:
        """Both tiles of every card on the node, plus each whole-card counter.

        Tile counters are what sum to a node total, one per torch device. The
        card counter additionally covers HBM and uncore, so it is larger than
        its two tiles combined -- recorded as an aggregate so the difference can
        finally be quantified instead of left as a caveat, but kept out of the
        totals so nothing is counted twice.
        """
        sources = []
        for card, name, counter in _hwmon_counters():
            if name.endswith(("_gt0", "_gt1")):
                tile = int(name[-1])
                sources.append(
                    EnergySource(
                        key=f"card{card}.gt{tile}",
                        scope=f"xpu tile {tile} of card {card} (hwmon energy1_input)",
                        read=_microjoule_reader(counter),
                        # Inverse of the index -> (card, tile) split in
                        # _energy_path, and correct only under
                        # ZE_FLAT_DEVICE_HIERARCHY=FLAT, which this class
                        # already documents relying on.
                        device_index=card * 2 + tile,
                    )
                )
            elif "_gt" not in name:
                sources.append(
                    EnergySource(
                        key=f"card{card}",
                        scope=f"whole card {card}, both tiles + HBM (hwmon energy1_input)",
                        read=_microjoule_reader(counter),
                        device_index=None,
                        aggregate=True,
                    )
                )
        return sources

    def energy_joules(self) -> float | None:
        path = self._energy_path()
        if path is None:
            return None
        try:
            return int(path.read_text()) / 1e6  # microjoules -> joules
        except (OSError, ValueError):
            return None

    def energy_scope(self) -> str:
        self._energy_path()
        return self._energy_scope

    def profiler_activities(self) -> list:
        from torch.profiler import ProfilerActivity

        activities = super().profiler_activities()
        xpu = getattr(ProfilerActivity, "XPU", None)
        if xpu is not None:
            activities.append(xpu)
        return activities

    def profiler_sort_key(self) -> str:
        return "self_xpu_time_total"

    def device_name(self) -> str:
        try:
            return torch.xpu.get_device_name(self.device)
        except Exception:
            return "intel-xpu"

    def environment(self) -> dict:
        env = super().environment()
        env.update(
            {
                "ipex_version": getattr(self.ipex, "__version__", "unknown"),
                "xpu_device_count": self.device_count(),
                "ze_flat_device_hierarchy": os.environ.get(
                    "ZE_FLAT_DEVICE_HIERARCHY", "unset"
                ),
                "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", "unset"),
            }
        )
        return env


def _nvml_energy_reader(nvml, handle):
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


class CudaPlatform(Platform):
    """Polaris / Sophia — NVIDIA A100 via CUDA + NCCL."""

    name = "cuda"
    dist_backend = "nccl"

    def __init__(self, local_rank: int = 0):
        self.local_rank = local_rank
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this node.")
        self.device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(self.device)
        # TF32 silently changes fp32 matmul precision on A100 and would make
        # an fp32 run not actually fp32. Pin it off so precision means what
        # the --precision flag says.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def peak_memory_bytes(self) -> int:
        return torch.cuda.max_memory_allocated(self.device)

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    # --- energy ------------------------------------------------------------
    # NOT YET RUN ON HARDWARE: written against the NVML docs, needs a Polaris or
    # Sophia run to confirm before any number from it is reported.
    def _nvml(self):
        if hasattr(self, "_nvml_mod"):
            return self._nvml_mod
        self._nvml_mod = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_mod = pynvml
        except Exception:
            pass  # no pynvml, or no driver: energy stays unsupported
        return self._nvml_mod

    def _visible_devices(self):
        """NVML index -> torch index mapping, or None when they coincide.

        NVML enumerates every GPU on the node regardless of
        CUDA_VISIBLE_DEVICES, which is precisely what makes an unused device
        visible -- but it also means the two numberings diverge the moment that
        variable is set, so a job pinned to one GPU would otherwise attribute
        its energy to the wrong device.

        The two agree on ordering only under CUDA_DEVICE_ORDER=PCI_BUS_ID; CUDA
        defaults to FASTEST_FIRST, which matches on a homogeneous node but is
        not guaranteed to. Set it in the submit script.
        """
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not raw:
            return None
        return [int(t) for t in raw.split(",") if t.strip().isdigit()] or None

    def _energy_sources(self):
        if hasattr(self, "_sources"):
            return self._sources

        self._sources = []
        nvml = self._nvml()
        if nvml is None:
            return self._sources
        visible = self._visible_devices()
        try:
            count = nvml.nvmlDeviceGetCount()
        except Exception:
            return self._sources

        for i in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
            except Exception:
                continue
            read, kind = _nvml_energy_reader(nvml, handle)
            if visible is None:
                torch_index = i
            elif i in visible:
                torch_index = visible.index(i)
            else:
                torch_index = None  # masked out of this job, still drawing power
            self._sources.append(
                EnergySource(
                    key=f"gpu{i}",
                    scope=f"whole gpu {i} incl. HBM ({kind})",
                    read=read,
                    device_index=torch_index,
                )
            )
        return self._sources

    def node_energy_sources(self) -> list[EnergySource]:
        return self._energy_sources()

    def _my_source(self):
        index = self.device.index or 0
        for source in self._energy_sources():
            if source.device_index == index:
                return source
        return None

    def energy_joules(self) -> float | None:
        # Deliberately the same EnergySource object the sampler holds: on the
        # power-integration fallback that closure only accumulates when it is
        # read, so sharing it means the start/end reads inherit the sampler's
        # 10 Hz cadence instead of integrating over two points.
        source = self._my_source()
        return source.read() if source else None

    def energy_scope(self) -> str:
        source = self._my_source()
        return source.scope if source else "nvml unavailable"

    def profiler_activities(self) -> list:
        from torch.profiler import ProfilerActivity

        return super().profiler_activities() + [ProfilerActivity.CUDA]

    def profiler_sort_key(self) -> str:
        return "self_cuda_time_total"

    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)


class PolarisPlatform(CudaPlatform):
    """Polaris — 4x A100 40GB per node, PBS + MPICH.

    Behaviourally identical to CudaPlatform today. It exists so a result says
    which MACHINE produced it: `cuda` names a backend, and Polaris and Sophia
    share one. Without the distinction both record machine="cuda", land in the
    same filenames, and add_scaling_efficiency() groups by machine -- so two
    different machines would silently be compared against each other as if they
    were one system scaling up.
    """

    name = "polaris"


class SophiaPlatform(CudaPlatform):
    """Sophia — DGX A100, 8x A100 per node, NVLink within the node.

    Same backend as Polaris but twice the devices per node and a different
    interconnect topology, so collective cost and per-node energy are not
    comparable between the two. Separate name, separate group.

    The `by-node`/`by-gpu` queues serve 40GB A100s, same part as Polaris; only
    the two `bigmem` nodes are 80GB. So the "A100" entry in peak_flops.json (a
    40GB SXM4 figure) is the right one for MFU here, but a bigmem run would need
    its own key before its MFU means anything.
    """

    name = "sophia"


_CUDA_MACHINES = {"polaris": PolarisPlatform, "sophia": SophiaPlatform}


def _alcf_machine() -> str | None:
    """Which ALCF CUDA machine this is, from the scheduler.

    PBS_JOBID carries the server name -- 7405999.polaris-pbs-01.hsn.cm.polaris
    .alcf.anl.gov -- which is more reliable than the hostname, whose compute
    node names (x3112c0s37b0n0) say nothing about the system. Hostname is still
    checked as a fallback for runs outside PBS.
    """
    haystack = f"{os.environ.get('PBS_JOBID', '')} {socket.gethostname()}".lower()
    return next((name for name in _CUDA_MACHINES if name in haystack), None)


def detect_platform(local_rank: int = 0, force: str | None = None) -> Platform:
    """Pick a platform, or build the one named by --platform."""
    if force and force != "auto":
        return {
            "aurora": AuroraPlatform,
            "cuda": CudaPlatform,
            "cpu": Platform,
            **_CUDA_MACHINES,
        }[force](local_rank)

    try:
        import intel_extension_for_pytorch  # noqa: F401

        if torch.xpu.is_available():
            return AuroraPlatform(local_rank)
    except Exception:
        pass

    if torch.cuda.is_available():
        # Named machine where we can tell, generic CUDA where we cannot -- a run
        # labelled "cuda" then means "some NVIDIA box we could not identify",
        # which is honest, rather than a wrong machine name.
        machine = _alcf_machine()
        return (_CUDA_MACHINES.get(machine) or CudaPlatform)(local_rank)

    return Platform(local_rank)


def init_distributed(platform: Platform, rank: int, world_size: int) -> bool:
    """Bring up the process group. Returns True if distributed is active.

    MASTER_ADDR is set by the submit script from the first line of
    $PBS_NODEFILE; every rank must agree on it.
    """
    if world_size <= 1:
        return False

    if platform.dist_backend == "ccl":
        import oneccl_bindings_for_pytorch  # noqa: F401

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Name the device the process group belongs to. Without it collectives run
    # on "whatever device is current", which is right here only because the
    # Platform constructors call set_device -- and is a documented way to get a
    # communicator built on the wrong GPU, where the symptom is a hang rather
    # than an error. Passing it also lets the backend eagerly initialize.
    #
    # Checked by signature rather than assumed: Aurora and Polaris ship
    # different torch builds from different vendor stacks, and neither is ours
    # to pin.
    kwargs = {}
    if platform.device.type != "cpu" and (
        "device_id"
        in inspect.signature(torch.distributed.init_process_group).parameters
    ):
        kwargs["device_id"] = platform.device

    torch.distributed.init_process_group(
        backend=platform.dist_backend, rank=rank, world_size=world_size, **kwargs
    )
    return True
