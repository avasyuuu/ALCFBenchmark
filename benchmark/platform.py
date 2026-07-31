"""Platform abstraction — the only file that should know machine-specific details.

Adding a new machine means adding a Platform subclass here and nothing else.
Aurora is the reference implementation; Polaris/Sophia/Crux follow the same shape.
"""

from __future__ import annotations

import os
import socket
from contextlib import nullcontext

import torch

# Env vars that carry rank info, in priority order. Different launchers set
# different ones: torchrun sets RANK/LOCAL_RANK, Aurora's mpiexec (MPICH+PALS)
# sets PMI_RANK/PALS_LOCAL_RANKID, OpenMPI sets OMPI_*. Try them all rather
# than hard-coding one launcher.
_RANK_VARS = ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK")
_WORLD_VARS = ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE")
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
