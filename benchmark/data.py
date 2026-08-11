"""CIFAR-10 loading, plus a synthetic mode that removes I/O from the measurement.

The synthetic path matters more than it looks: when a machine benchmarks slow,
--synthetic-data tells you in one run whether the bottleneck is the accelerator
or Lustre. Same shapes, same batch count, zero filesystem traffic.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


class SyntheticDataset(Dataset):
    """Random images with deterministic labels. Generated once and indexed, so
    per-step cost is a tensor copy rather than RNG work that would show up as
    fake compute time."""

    def __init__(self, num_samples: int, seed: int = 0):
        self.num_samples = num_samples
        g = torch.Generator().manual_seed(seed)
        self.pool = torch.randn(256, *INPUT_SHAPE, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (256,), generator=g)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i = idx % 256
        return self.pool[i], self.labels[i]


def _cifar_datasets(data_dir: str, download: bool):
    # torchvision is a separate wheel from torch, and a hand-built venv -- which
    # is the only option on Crux -- gets one without the other unless you ask.
    # The bare ModuleNotFoundError names the module but not the fix, and it
    # surfaces once per rank, so eight tracebacks scroll the useful line away.
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchvision is required for CIFAR-10 and is not installed.\n"
            "  crux (CPU):  pip install torchvision --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "  elsewhere:   pip install torchvision\n"
            "Or run with --synthetic-data, which needs no dataset at all."
        ) from exc

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_tf)
    val = datasets.CIFAR10(data_dir, train=False, download=download, transform=val_tf)
    return train, val


def build_loaders(
    *,
    data_dir: str,
    local_batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    synthetic: bool,
    download: bool = False,
    seed: int = 1234,
):
    """Returns (train_loader, val_loader, train_sampler).

    train_sampler is None when not distributed; callers must call
    set_epoch() on it each epoch or every rank reshuffles identically.
    """
    if synthetic:
        train_ds = SyntheticDataset(50_000, seed=seed)
        val_ds = SyntheticDataset(10_000, seed=seed + 1)
    else:
        train_ds, val_ds = _cifar_datasets(data_dir, download)

    train_sampler = val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    common = dict(
        num_workers=workers,
        pin_memory=False,  # pinned host memory is CUDA-specific; XPU ignores it
        persistent_workers=workers > 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,  # ragged final batch would skew step-time statistics
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


def prefetch_cifar(data_dir: str) -> None:
    """Download CIFAR-10 once, from a login node.

    Never let 12+ ranks download concurrently onto Lustre — you get a thundering
    herd on a shared filesystem and, with luck, only a corrupt archive.
    """
    _cifar_datasets(data_dir, download=True)
