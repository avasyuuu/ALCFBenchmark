#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_bench_resnet
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -q debug
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# ResNet-20 / CIFAR-10 benchmark on Polaris (4x A100 40GB per node).
#
#   qsub scripts/submit_polaris.sh
#   qsub -l select=2 -q debug-scaling scripts/submit_polaris.sh
#
# Before the first run:
#   1. set -A above to your real project name (my.alcf.anl.gov)
#   2. confirm -l filesystems= names the filesystem your data actually sits on;
#      Polaris REJECTS jobs that do not declare it, and the name differs from
#      Aurora's `flare`
#   3. from a LOGIN node, download CIFAR-10 once:
#        python -m benchmark.prepare --data-dir ./data
#
# Keep every --flag below identical to submit_aurora.sh. The point of this
# script is a machine comparison, and a different batch size or target accuracy
# makes the two columns incomparable.

set -euo pipefail

cd "${PBS_O_WORKDIR}"
if [[ ! -d benchmark ]]; then
    echo "error: no benchmark/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_polaris.sh" >&2
    exit 1
fi

# --- environment -------------------------------------------------------------
# Polaris ships PyTorch through the conda module rather than Aurora's
# `frameworks`. Verify the current name with `module avail` -- ALCF renames
# these across maintenance windows and a stale name fails at load, not at run.
set +u
module use /soft/modulefiles
module load conda
conda activate base
set -u

NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
RANKS_PER_NODE=4                       # one rank per A100
NTOTRANKS=$(( NNODES * RANKS_PER_NODE ))

export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
export MASTER_PORT=29500
export WORLD_SIZE="${NTOTRANKS}"

# 32 EPYC cores per node over 4 ranks.
export OMP_NUM_THREADS=8

# Same AF_UNIX ceiling as Aurora: PBS builds a long TMPDIR, DataLoader workers
# open a multiprocessing socket under it, and the path can exceed 108 bytes.
# Polaris' hostnames are shorter so it may not trip, but the cost of setting it
# is nothing and the failure mode is a hang until walltime.
export TMPDIR=/tmp

# NVML enumerates GPUs in PCI bus order; CUDA defaults to FASTEST_FIRST. On a
# homogeneous node the two agree, but nothing guarantees it, and CudaPlatform
# maps NVML indices to torch indices to attribute energy to the right device.
# Pin the ordering so that mapping is exact rather than coincidental.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Deliberately NOT using ALCF's set_affinity_gpu_polaris.sh, for the same reason
# submit_aurora.sh skips gpu_tile_compact.sh: it sets a per-rank
# CUDA_VISIBLE_DEVICES, which would leave each rank able to see exactly one GPU.
# The power sampler runs on one rank per node and reads every device on it, so a
# rank that can only see its own GPU would report the other three as unused --
# they are in use, by ranks it cannot observe. Leaving all four visible lets
# CudaPlatform bind rank -> cuda:local_rank itself and keeps the idle-device
# accounting honest.

echo "=== job ${PBS_JOBID} ==="
echo "nodes=${NNODES} ranks=${NTOTRANKS} ranks/node=${RANKS_PER_NODE}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
python -c "import torch; print('torch', torch.__version__, \
'| cuda devices', torch.cuda.device_count(), \
'|', torch.cuda.get_device_name(0))"
# pynvml carries the energy counter. Without it the run still completes but
# every energy and power field is null, which is easy to miss until you are
# looking at an empty column days later -- so say so loudly here.
python -c "import pynvml; pynvml.nvmlInit(); \
print('pynvml OK |', pynvml.nvmlDeviceGetCount(), 'GPUs visible to NVML')" \
    || echo "WARNING: pynvml missing -- no energy or power data. pip install nvidia-ml-py"
echo "========================"

# --- run ---------------------------------------------------------------------
mpiexec -n "${NTOTRANKS}" --ppn "${RANKS_PER_NODE}" --depth="${OMP_NUM_THREADS}" \
    --cpu-bind depth \
    python -m benchmark.train \
        --platform polaris \
        --model resnet20 \
        --scaling strong \
        --global-batch-size 1536 \
        --epochs 100 \
        --precision bf16 \
        --target-accuracy 0.90 \
        --warmup-steps 20 \
        --power-interval 0.1 \
        --data-dir ./data \
        --results-dir ./results \
        --note "polaris ${NNODES}-node bf16 strong-scaling 100ep"

# Global batch stays 1536 so accuracy-per-step matches Aurora exactly; only the
# split changes (384 per rank over 4 ranks here, 128 over 12 there). That is the
# whole point of --scaling strong: time-to-accuracy stays comparable across
# machines with different device counts.
