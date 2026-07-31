# ALCF Machine Benchmark

A portable neural-network benchmark for comparing ALCF machines — Aurora, Polaris,
Sophia, Crux, and (later) the Cerebras and Graphcore AI Testbed systems — on
time-to-accuracy, throughput, scaling efficiency, and cost in node-hours.

Aurora is the reference implementation. Everything machine-specific lives in
`benchmark/platform.py`; adding a machine means adding a class there.

## Layout

```
benchmark/
  platform.py    device + distributed backend abstraction (the only machine-aware file)
  models.py      CIFAR ResNet-20/32/56 and an analytic FLOP counter
  data.py        CIFAR-10 loaders + synthetic mode
  metrics.py     timers and the results JSON schema
  train.py       benchmark entrypoint
  comm_bench.py  allreduce microbenchmark
  prepare.py     one-time dataset download (login node only)
configs/
  peak_flops.json   vendor peak FLOP/s for MFU — YOU MUST FILL THIS IN
scripts/
  submit_aurora.sh  PBS batch script
analysis/
  summarize.py      results/*.json -> comparison table
results/            one JSON per run
```

## First run on Aurora

```bash
# 1. from a LOGIN node — download the dataset once
module load frameworks
python -m benchmark.prepare --data-dir ./data

# 2. set your project name in scripts/submit_aurora.sh (-A), then
qsub scripts/submit_aurora.sh

# 3. when it finishes
python analysis/summarize.py
```

Before that, sanity-check the whole pipeline on a single tile in an interactive
job — it takes about a minute and catches every setup problem:

```bash
qsub -I -l select=1,walltime=00:30:00,place=scatter -l filesystems=flare:home -A <PROJECT> -q debug
module load frameworks
python -m benchmark.train --platform aurora --epochs 1 --max-steps 30 \
    --synthetic-data --global-batch-size 128 --warmup-steps 5
```

## What gets measured

**Headline** — time-to-accuracy (wall seconds to first reach the target top-1),
steady-state throughput in samples/s, best top-1 at a fixed epoch budget, and
parallel efficiency across node counts.

**Explanatory** — step-time median/p90/stdev (jitter exposes interconnect
contention and unstable nodes, which a mean hides), dataloader wait time,
MFU against vendor peak, startup and warmup cost reported separately from
steady state, and peak memory per device.

**Practical** — node-hours consumed, PBS job id, and the exact node list, so a
result can always be traced back to the hardware it ran on.

## Rules the harness enforces

These exist so the numbers compare machines rather than accidents of setup.

- **Warmup steps are excluded** from steady-state stats (`--warmup-steps`, default 20).
  The first steps carry lazy kernel compilation, allocator growth, and oneCCL
  bootstrap; leaving them in would rank machines by JIT speed.
- **Precision is pinned.** TF32 is explicitly disabled on NVIDIA so `fp32`
  means fp32 on every machine.
- **Strong vs weak scaling is a flag, not an accident.** `--scaling strong`
  holds global batch fixed so accuracy stays comparable as nodes increase;
  `--scaling weak` holds per-rank batch fixed and measures throughput only.
- **LR follows the linear scaling rule with warmup** (Goyal et al. 2017).
  Without it, changing node count changes effective batch size and you would be
  measuring hyperparameters instead of hardware.
- **Device work is synchronized before every timer read.** Accelerator kernels
  are async; an un-synced timer measures launch, not execution.
- **`--synthetic-data`** removes all filesystem I/O. When a machine looks slow,
  this tells you in one run whether the bottleneck is the accelerator or Lustre.

Run each configuration **at least 3 times** and report median plus spread.
Aurora node variability makes single runs untrustworthy.

## Before publishing any MFU number

`configs/peak_flops.json` ships with every value `null`, so MFU is omitted
rather than computed from a number nobody checked. Fill in per-device peak
dense FLOP/s from vendor spec sheets and record the source in each entry.

On Aurora the device is a **tile**, not a GPU — the `frameworks` module sets
`ZE_FLAT_DEVICE_HIERARCHY=FLAT`, so a node presents 12 devices. Peak values
there must be per-tile, roughly half the per-GPU figure.

## A note on `gpu_tile_compact.sh`

Not used here, and not something you write — ALCF ships it. It sets
`ZE_AFFINITY_MASK` per rank for compiled MPI codes, but ALCF advises against
combining it with the `frameworks` module, which already exposes one device per
tile. `AuroraPlatform` binds rank → `xpu:local_rank` directly instead.

## Porting to another machine

1. Add a `Platform` subclass in `platform.py` (device selection, sync, memory,
   distributed backend).
2. Add it to `detect_platform()`.
3. Copy `scripts/submit_aurora.sh` and change the module loads and rank count.

Nothing in `models.py`, `data.py`, `metrics.py`, or `train.py` should need to
change. Cerebras and Graphcore are the exception — they need their own compile
and execution wrappers (`cerebras_pytorch`, `poptorch`) and do not use DDP, so
they get their own runner sharing only the model and data definitions.

## Status

Aurora path is written but **not yet validated on hardware.** Verify on first run:

- [ ] rank/local-rank env vars (`PMI_RANK`, `PALS_LOCAL_RANKID`) resolve correctly
- [ ] `ccl` backend initializes across nodes with `MASTER_ADDR` from `$PBS_NODEFILE`
- [ ] all 12 tiles are actually busy — cross-check with `xpu-smi`
- [ ] `ipex.optimize` plays well with DDP in this frameworks version
