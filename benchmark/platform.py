"""Platform abstraction — the only file that should know machine-specific details.

Adding a new machine means adding a Platform subclass here and nothing else.
Aurora is the reference implementation; Polaris/Sophia/Crux follow the same shape.
"""

from __future__ import annotations

import os
import socket
from contextlib import nullcontext
from pathlib import Path

import torch

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
        hwmon = Path(f"/sys/class/drm/card{card}/device/hwmon")
        try:
            entries = sorted(hwmon.glob("hwmon*"))
        except OSError:
            entries = []

        # Prefer this rank's tile; fall back to the whole card, which is still
        # a real measurement as long as the result says so.
        for want, scope in (
            (f"_gt{tile}", f"xpu tile {tile} of card {card} (hwmon energy1_input)"),
            ("", f"whole card {card}, both tiles + HBM (hwmon energy1_input)"),
        ):
            for h in entries:
                try:
                    name = (h / "name").read_text().strip()
                except OSError:
                    continue
                matches = name.endswith(want) if want else "_gt" not in name
                if matches and (h / "energy1_input").exists():
                    try:
                        int((h / "energy1_input").read_text())
                    except (OSError, ValueError):
                        continue
                    self._energy_file = h / "energy1_input"
                    self._energy_scope = scope
                    return self._energy_file
        return None

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

    def profiler_activities(self) -> list:
        from torch.profiler import ProfilerActivity

        return super().profiler_activities() + [ProfilerActivity.CUDA]

    def profiler_sort_key(self) -> str:
        return "self_cuda_time_total"

    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)


def detect_platform(local_rank: int = 0, force: str | None = None) -> Platform:
    """Pick a platform, or build the one named by --platform."""
    if force and force != "auto":
        return {"aurora": AuroraPlatform, "cuda": CudaPlatform, "cpu": Platform}[force](
            local_rank
        )

    try:
        import intel_extension_for_pytorch  # noqa: F401

        if torch.xpu.is_available():
            return AuroraPlatform(local_rank)
    except Exception:
        pass

    if torch.cuda.is_available():
        return CudaPlatform(local_rank)

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

    torch.distributed.init_process_group(
        backend=platform.dist_backend, rank=rank, world_size=world_size
    )
    return True
