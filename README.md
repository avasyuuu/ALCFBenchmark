# ALCF Machine Benchmark

> **Disclaimer.** This is an independent student project carried out during an
> internship. It is **not** an official Argonne National Laboratory or ALCF
> product, is not endorsed by Argonne, Intel, NVIDIA, or any other vendor, and
> has not been reviewed by any of them.
>
> Any performance numbers here come from a small, deliberately simple workload
> on a shared production system, run with default software settings and no
> vendor tuning. They measure *this benchmark under these conditions* — not the
> peak capability of any machine, and not a fair vendor-versus-vendor
> comparison. Results are also affected by other users' jobs, node health, and
> software versions on the day of the run.
>
> **Do not cite these as official facility performance figures.** For those,
> see ALCF's own documentation and published results.

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
  submit_polaris_aiperf.sh  LLM inference power sweep (separate benchmark, see below)
analysis/
  summarize.py      results/*.json -> comparison table
  summarize_aiperf.py  AIPerf artifact dirs -> power/token/efficiency tables
  nvml_idle.py      idle GPU power floor, to subtract from a measured run
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

## Power: two different measurements

The results JSON carries two energy blocks, and they answer different questions.
Do not add them together.

**`energy`** is per rank: one counter read before training, one after, on the
device this rank owns. It is what `samples_per_joule` and `joules_to_target`
(the energy twin of time-to-accuracy) are computed from.

**`power`** is per node, from a sampler thread that one rank per node runs at
`--power-interval` seconds (default 0.1, `0` disables). It reads **every**
accelerator counter on the node, including devices no rank ever bound to — the
only way an idle device appears in a result at all, since a per-rank read has
nobody to take it. `bound` comes from the rank-to-device binding rule, not from
a power threshold, because a device can idle at a third of its peak.

```bash
python analysis/summarize.py --devices
```

The full time series goes to `results/power/` as one file per node, with epoch
and eval boundaries recorded as marks so power can be attributed to a phase
rather than averaged over the run. The result JSON keeps only the rollup.

On Aurora each tile is counted once; the whole-card counter is recorded
alongside but flagged `aggregate`, since it spans both tiles plus HBM and would
double count. **Scope is not comparable across machines** — an Aurora tile and
an A100 are different measurement boundaries. Compare per node, and read
`scope` before quoting any figure.

## LLM inference power (a separate benchmark)

`scripts/submit_polaris_aiperf.sh` runs NVIDIA AIPerf against vLLM on one A100
and reports **tokens/joule**, the inference-side twin of `samples_per_joule`. It
shares the energy methodology with the training benchmark and nothing else — its
results do not belong in the machine-comparison table, and it only covers
NVIDIA (AIPerf's collectors are DCGM, pynvml and amdsmi; there is no XPU path,
so Aurora is out).

```bash
qsub scripts/submit_polaris_aiperf.sh
python analysis/summarize_aiperf.py results/aiperf/polaris-*/c* --idle results/aiperf/polaris-*/idle.json
```

Three things this harness does deliberately, each because skipping it produced a
wrong answer on the shakedown run:

- **`ignore_eos` + `min_tokens`** pin every generation to exactly `--osl` tokens.
  Left to stop at EOS, output length drifts with load, and a longer generation
  amortises the fixed prefill over more decode tokens — which moves tokens/joule
  on its own and confounds it with concurrency. Ollama cannot do this; vLLM can,
  and that is the whole reason for vLLM here.
- **Request count is fixed across concurrency levels**, so rows are equal work.
  Scale requests with concurrency and absolute joules stop being comparable
  between rows.
- **The idle floor is sampled before the server starts**, on a quiet node.
  Measured after a run instead it reads high, because clocks and fan state have
  not settled.

`summarize_aiperf.py` re-checks the first two on whatever it is given and prints
a warning if either was violated.

The headline result from the laptop shakedown, which is the part expected to
survive onto real hardware: **GPU power is nearly load-invariant** (54.3 → 57.1 W
across an 8x concurrency range while throughput rose 44%), so energy-to-solution
is driven by finishing sooner far more than by drawing fewer watts.

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

## Publishing results

Results JSON records the PBS job id and real node hostnames, which are facility
detail rather than science. `--anonymize` replaces them with salted hashes and
redacts the job id:

```bash
python -m benchmark.train --anonymize ...
```

The salt is generated once, stored at `~/.alcf_bench_salt`, and reused — so
hashes stay stable across runs (you can still tell whether two runs shared a
node) while remaining irreversible to anyone else. It is salted rather than
plain-hashed because Aurora hostnames follow a known enumerable scheme and an
unsalted hash would be trivially reversible.

Kept deliberately: node count, queue, and all software versions. Those are
methodologically necessary and not sensitive.

**`--anonymize` cannot scrub `--note` text.** That's free-form and yours to
write; keep project names out of it if you intend to publish.

Two things it also does not do:

- **Your project name should never enter git at all.** Leave
  `-A CHANGEME_PROJECT` in the submit script and override it at submit time,
  since command-line flags beat `#PBS` directives:
  `qsub -A YOUR_PROJECT scripts/submit_aurora.sh`
- **Get your mentor's approval** before publishing cross-machine performance
  comparisons. That's the actual gate; the repo setting is just a toggle.

Note that making a repo public exposes its **entire history**, not just current
files — so keep sensitive values out from the first commit rather than deleting
them later.

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

Polaris energy (`CudaPlatform`) was **validated on hardware on 2026-08-10** by
the three runs in `results/polaris_*.json`, which settle every item this list
used to ask about:

- [x] `nvmlDeviceGetTotalEnergyConsumption` is present on the A100 driver — every
      Polaris run records `scope: whole gpu 0 incl. HBM (nvml energy counter)`,
      so the counter path is taken and not the power-integration fallback
- [x] NVML and torch indices agree under `CUDA_DEVICE_ORDER=PCI_BUS_ID` — per-rank
      joules are non-null and consistent between runs
- [x] all four A100s appear in the node sampler — `power.devices_total` is 4

Sophia has never run, so its `SophiaPlatform` is still unexercised. It shares
`CudaPlatform` with Polaris, so the energy path above is the same code; what is
untested there is the 8-GPU node, the `torchrun` launcher branch and the
`by-node` queue.

The AIPerf inference sweep (`submit_polaris_aiperf.sh`) is validated only on a
consumer GPU under Ollama. Verify on first run:

- [ ] the conda module and proxy hostname are still current
- [ ] vLLM becomes healthy inside the 15-minute startup budget
- [ ] `summarize_aiperf.py` prints **no** warnings — with `ignore_eos` the
      achieved OSL should now match the requested one exactly
