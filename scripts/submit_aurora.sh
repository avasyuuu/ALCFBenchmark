#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_bench_resnet
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -q debug
#PBS -k doe

# ResNet-20 / CIFAR-10 benchmark on Aurora.
#
#   qsub scripts/submit_aurora.sh
#   qsub -l select=2 -q debug-scaling scripts/submit_aurora.sh   # override at submit
#
# Before the first run:
#   1. set -A above to your real project name (find it at my.alcf.anl.gov)
#   2. from a LOGIN node, download CIFAR-10 once:
#        python -m benchmark.prepare --data-dir ./data
#      Never let 12 ranks download concurrently onto Lustre.

set -euo pipefail

cd "${PBS_O_WORKDIR}"

# --- environment -------------------------------------------------------------
# `frameworks` provides PyTorch + IPEX + oneCCL bindings. It also sets
# ZE_FLAT_DEVICE_HIERARCHY=FLAT, which exposes each TILE as its own device --
# so a node has 12 devices, not 6.
module load frameworks

# Deliberately NOT using gpu_tile_compact.sh. That script sets ZE_AFFINITY_MASK
# per rank for compiled MPI codes, but ALCF advises against combining it with
# the frameworks module, because FLAT hierarchy already gives one device per
# tile. benchmark/platform.py binds rank -> xpu:local_rank instead.

NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
RANKS_PER_NODE=12                      # one rank per GPU tile
NTOTRANKS=$(( NNODES * RANKS_PER_NODE ))

# Rendezvous host for torch.distributed: first node in the allocation.
export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
export MASTER_PORT=29500

# oneCCL wants to know how many workers per node it is coordinating.
export CCL_WORKER_COUNT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072    # Slingshot completion queue; avoids
                                        # overflow stalls at higher rank counts

# Each rank gets its own slice of the 104 physical cores, leaving the
# dataloader workers room to run without fighting the training process.
export OMP_NUM_THREADS=8

echo "=== job ${PBS_JOBID} ==="
echo "nodes=${NNODES} ranks=${NTOTRANKS} ranks/node=${RANKS_PER_NODE}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
python -c "import torch, intel_extension_for_pytorch as ipex; \
print('torch', torch.__version__, '| ipex', ipex.__version__, \
'| xpu devices', torch.xpu.device_count())"
echo "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset}"
echo "========================"

# --- run ---------------------------------------------------------------------
# --cpu-bind=depth spreads ranks across cores so they don't contend.
mpiexec -n "${NTOTRANKS}" -ppn "${RANKS_PER_NODE}" --depth="${OMP_NUM_THREADS}" \
    --cpu-bind=depth \
    python -m benchmark.train \
        --platform aurora \
        --model resnet20 \
        --scaling strong \
        --global-batch-size 1536 \
        --epochs 20 \
        --precision bf16 \
        --target-accuracy 0.90 \
        --warmup-steps 20 \
        --data-dir ./data \
        --results-dir ./results \
        --note "aurora ${NNODES}-node bf16 strong-scaling"
