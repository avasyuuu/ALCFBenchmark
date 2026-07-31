"""Allreduce microbenchmark — measures the interconnect on its own.

DDP overlaps gradient allreduce with the backward pass, so communication cost
cannot be cleanly separated from inside the training loop. This measures it
directly instead, which is what explains scaling-efficiency losses in the
training results.

    mpiexec -n 24 -ppn 12 python -m benchmark.comm_bench --platform aurora

Algorithmic bandwidth ("algbw") is bytes / time. Bus bandwidth ("busbw")
multiplies by 2(N-1)/N, the traffic a ring allreduce actually moves per rank —
that is the number comparable against the link's rated bandwidth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .metrics import anonymize_environment, pbs_environment
from .platform import detect_platform, get_rank_info, init_distributed

SIZES_MB = [0.001, 0.01, 0.1, 1, 4, 16, 64, 256]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", default="auto", choices=["auto", "aurora", "cuda", "cpu"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--anonymize", action="store_true", help="hash hostnames, redact job id")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()
    platform = detect_platform(local_rank, force=args.platform)
    if not init_distributed(platform, rank, world_size):
        raise SystemExit("comm_bench needs more than one rank")

    is_main = rank == 0
    results = []

    for size_mb in SIZES_MB:
        n_elems = max(1, int(size_mb * 1024 * 1024 / 4))  # fp32
        tensor = torch.ones(n_elems, dtype=torch.float32, device=platform.device)

        for _ in range(args.warmup):
            torch.distributed.all_reduce(tensor)
        platform.synchronize()
        torch.distributed.barrier()

        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            torch.distributed.all_reduce(tensor)
            platform.synchronize()
            times.append(time.perf_counter() - start)

        median = statistics.median(times)
        nbytes = n_elems * 4
        algbw = nbytes / median
        busbw = algbw * 2 * (world_size - 1) / world_size

        results.append(
            {
                "size_mb": size_mb,
                "bytes": nbytes,
                "median_s": median,
                "min_s": min(times),
                "p90_s": sorted(times)[int(0.9 * len(times))],
                "algbw_gbps": algbw / 1e9,
                "busbw_gbps": busbw / 1e9,
            }
        )
        if is_main:
            print(
                f"[comm] {size_mb:>8.3f} MB  {median * 1e6:>10.1f} us  "
                f"algbw {algbw / 1e9:>7.2f} GB/s  busbw {busbw / 1e9:>7.2f} GB/s",
                flush=True,
            )

    if is_main:
        out = Path(args.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "allreduce_microbenchmark",
            "machine": platform.name,
            "backend": platform.dist_backend,
            "world_size": world_size,
            "device": platform.device_name(),
            "environment": (
                anonymize_environment(pbs_environment())
                if args.anonymize
                else pbs_environment()
            ),
            "results": results,
        }
        path = out / f"comm_{platform.name}_ws{world_size}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"[comm] wrote {path}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
