#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_bench_resnet
#PBS -l select=1:system=crux
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -q workq-route
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# ResNet-20 / CIFAR-10 benchmark on Crux (dual AMD EPYC 7742 Rome, no GPU).
#
#   qsub scripts/submit_crux.sh
#   qsub -l select=2 scripts/submit_crux.sh
#
# Before the first run:
#   1. set -A above to your real project name (my.alcf.anl.gov)
#   2. build the venv described under "environment" below -- Crux has no conda
#      module carrying torch, unlike Polaris and Sophia
#   3. from a LOGIN node, download CIFAR-10 once:
#        python -m benchmark.prepare --data-dir ./data
#
# Crux is the CPU baseline, so this is the one submit script that does NOT keep
# every flag identical to the others. Three deliberate deviations, each one a
# thing that would otherwise be measured wrong:
#
#   --precision fp32   EPYC 7742 is Zen 2: AVX2, no AVX-512 and no bf16
#                      instructions. bf16 there is emulated in software, so a
#                      bf16 run would time the emulation rather than the
#                      machine. fp32 is the honest CPU number. This does mean
#                      the Crux row is not directly comparable to the bf16
#                      accelerator rows, which is why it is stated here and in
#                      --note rather than left for a reader to notice.
#   --power-interval 0 there is no accelerator energy counter to read, and
#                      sampling nothing at 10 Hz only adds overhead. Crux will
#                      be absent from both energy tables rather than sitting in
#                      them full of zeros.
#   walltime 06:00:00  a CPU epoch is far slower than a GPU one, and a run
#                      killed at walltime writes no result at all -- the JSON is
#                      written after the last epoch. Better to over-book here
#                      than to lose the whole run. Use the debug queue with
#                      --max-steps for a quick check; see the smoke test at the
#                      bottom of this file.
#
# Queue: workq-route is the routing queue (1-184 nodes, up to 24 h). debug caps
# at 2 h, which the full 100-epoch config will not fit into.

set -euo pipefail

cd "${PBS_O_WORKDIR}"
if [[ ! -d benchmark ]]; then
    echo "error: no benchmark/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_crux.sh" >&2
    exit 1
fi

# --- environment -------------------------------------------------------------
# Crux ships no conda environment with torch in it -- ALCF documents building
# your own and says CPU-optimized prebuilt environments are still to come. So
# this activates one you made rather than a module, and says exactly how to make
# it if it is missing, because the alternative is a stack trace six lines into
# the first import.
#
#   module use /soft/modulefiles && module load conda && conda activate base
#   python -m venv --system-site-packages ~/venvs/crux-bench
#   . ~/venvs/crux-bench/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
VENV="${BENCH_VENV:-${HOME}/venvs/crux-bench}"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "error: no virtualenv at ${VENV}" >&2
    echo "       Crux has no conda module carrying torch; build one first --" >&2
    echo "       see the environment comment in $0, or set BENCH_VENV." >&2
    exit 1
fi
set +u
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
# One rank per NUMA domain. A Crux node is 128 cores in 8 domains of 16, and a
# rank spanning two domains pays cross-socket memory latency on every batch.
RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
NTOTRANKS=$(( NNODES * RANKS_PER_NODE ))
export OMP_NUM_THREADS=$(( 128 / RANKS_PER_NODE ))

export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
export MASTER_PORT=29500
export WORLD_SIZE="${NTOTRANKS}"

# Same AF_UNIX path-length ceiling as the other machines: PBS builds a long
# TMPDIR, DataLoader workers open a multiprocessing socket under it, and the
# failure mode is a hang until walltime.
export TMPDIR=/tmp

echo "=== job ${PBS_JOBID} ==="
echo "nodes=${NNODES} ranks=${NTOTRANKS} ranks/node=${RANKS_PER_NODE} threads/rank=${OMP_NUM_THREADS}"
echo "venv=${VENV}"
python -c "import torch; print('torch', torch.__version__, '| threads', torch.get_num_threads())"
echo "========================"

# --- run ---------------------------------------------------------------------
# --platform crux, not the "cpu" default. CruxPlatform is behaviourally the CPU
# base class, but "cpu" names a device rather than a machine -- a result
# labelled cpu could have come from a laptop, and every CPU machine would group
# under one name in summarize.py.
mpiexec -n "${NTOTRANKS}" --ppn "${RANKS_PER_NODE}" --depth="${OMP_NUM_THREADS}" \
    --cpu-bind depth \
    python -m benchmark.train \
        --platform crux \
        --model resnet20 \
        --scaling strong \
        --global-batch-size 1536 \
        --epochs 100 \
        --precision fp32 \
        --target-accuracy 0.90 \
        --warmup-steps 20 \
        --power-interval 0 \
        --data-dir ./data \
        --results-dir ./results \
        --note "crux ${NNODES}-node fp32 strong-scaling 100ep (CPU baseline, not bf16-comparable)"

# Smoke test, for the debug queue -- proves the venv, the launcher and the crux
# label in a couple of minutes rather than a couple of hours:
#
#   qsub -q debug -l select=1:system=crux -l walltime=00:20:00 \
#        -l filesystems=home:eagle -A datascience_collab -I
#   # then, on the node:
#   mpiexec -n 8 --ppn 8 --depth=16 --cpu-bind depth python -m benchmark.train \
#       --platform crux --precision fp32 --max-steps 30 --epochs 1 \
#       --power-interval 0 --results-dir ./results/smoke --note "crux smoke"
