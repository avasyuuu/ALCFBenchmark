"""One-time dataset download. Run from a LOGIN node, never inside a job.

    python -m benchmark.prepare --data-dir ./data

Login nodes (UANs) have outbound network access; compute nodes do not. And 12+
ranks hitting the same Lustre path concurrently is a good way to get a corrupt
archive rather than a fast download.
"""

import argparse

from .data import prefetch_cifar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()

    print(f"downloading CIFAR-10 into {args.data_dir} ...")
    prefetch_cifar(args.data_dir)
    print("done — safe to submit jobs now")


if __name__ == "__main__":
    main()
