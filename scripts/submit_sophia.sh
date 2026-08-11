#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_bench_resnet
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -q by-node
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# ResNet-20 / CIFAR-10 benchmark on Sophia (DGX A100, 8x A100 per node).
#
#   qsub scripts/submit_sophia.sh
#   qsub -l select=2 scripts/submit_sophia.sh
#
# Before the first run:
#   1. set -A above to your real project name (my.alcf.anl.gov)
#   2. confirm -l filesystems= names the filesystem your data actually sits on
#   3. from a LOGIN node, download CIFAR-10 once:
#        python -m benchmark.prepare --data-dir ./data
#
# Keep every --flag below identical to submit_aurora.sh and submit_polaris.sh.
# The point of this script is a machine comparison, and a different batch size
# or target accuracy makes the columns incomparable.
#
# Queue: `by-node` gives whole DGX nodes, 1 to 8. The DEFAULT queue is `by-gpu`,
# which allocates individual GPUs -- submitting without -q above would silently
# hand back a single A100 and produce a 1-rank "8-GPU" result. There is no debug
# queue here, unlike Polaris.
#
# Note there is no `system=sophia` in the select. ALCF sets a `system` resource
# per node and submit_polaris.sh uses it, but the Sophia queues already scope to
# this machine, and a resource name the scheduler does not recognise does not
# error -- the job just queues forever waiting for a node that can never match.

set -euo pipefail

cd "${PBS_O_WORKDIR}"
if [[ ! -d benchmark ]]; then
    echo "error: no benchmark/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_sophia.sh" >&2
    exit 1
fi

# --- environment -------------------------------------------------------------
# Same conda module as Polaris; the base environment carries a CUDA-enabled
# torch. Verify the name with `module avail` -- ALCF renames these across
# maintenance windows and a stale name fails at load, not at run.
set +u
module use /soft/modulefiles
module load conda
conda activate base
set -u

NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
RANKS_PER_NODE=8                       # one rank per A100; a DGX node has eight
NTOTRANKS=$(( NNODES * RANKS_PER_NODE ))

# Two 64-core EPYC 7742 per DGX node, 128 cores over 8 ranks.
export OMP_NUM_THREADS=16

# Same AF_UNIX ceiling as Aurora and Polaris: PBS builds a long TMPDIR,
# DataLoader workers open a multiprocessing socket under it, and the path can
# exceed 108 bytes. The failure mode is a hang until walltime, so set it.
export TMPDIR=/tmp

# NVML enumerates GPUs in PCI bus order; CUDA defaults to FASTEST_FIRST. On a
# homogeneous node the two agree, but nothing guarantees it, and CudaPlatform
# maps NVML indices to torch indices to attribute energy to the right device.
# Pin the ordering so that mapping is exact rather than coincidental.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Deliberately NOT setting a per-rank CUDA_VISIBLE_DEVICES, for the same reason
# submit_polaris.sh skips set_affinity_gpu_polaris.sh: the power sampler runs on
# one rank per node and reads every device on it, so a rank that could only see
# its own GPU would report the other seven as unused -- they are in use, by ranks
# it cannot observe. Leaving all eight visible lets CudaPlatform bind
# rank -> cuda:local_rank itself and keeps the idle-device accounting honest.

echo "=== job ${PBS_JOBID} ==="
echo "nodes=${NNODES} ranks=${NTOTRANKS} ranks/node=${RANKS_PER_NODE}"
python -c "import torch; print('torch', torch.__version__, \
'| cuda devices', torch.cuda.device_count(), \
'|', torch.cuda.get_device_name(0))"
# pynvml carries the energy counter. Without it the run still completes but
# every energy and power field is null, which is easy to miss until you are
# looking at an empty column days later -- so say so loudly here.
python -c "import pynvml; pynvml.nvmlInit(); \
print('pynvml OK |', pynvml.nvmlDeviceGetCount(), 'GPUs visible to NVML')" \
    || echo "WARNING: pynvml missing -- no energy or power data. pip install nvidia-ml-py"

# --- launcher ----------------------------------------------------------------
# Sophia is a DGX cluster rather than a Cray, and ALCF's Sophia docs do not
# commit to an MPI launcher the way the Polaris and Aurora ones do. So pick a
# launcher instead of assuming one:
#
#   1 node  -> torchrun. No MPI in the picture at all, and it sets RANK /
#              WORLD_SIZE / LOCAL_RANK directly, which is the first thing
#              get_rank_info() looks for. This is the common case (a DGX node is
#              already 8 GPUs) so the common case has no launcher risk.
#   N nodes -> mpiexec if it exists, else OpenMPI's mpirun. --depth and
#              --cpu-bind are MPICH/PALS spellings that mpirun rejects, hence
#              the separate argument lists rather than one with a swapped
#              binary; platform.py already reads both PMI_* and OMPI_* rank vars.
if [[ "${NNODES}" -eq 1 ]]; then
    LAUNCH=(torchrun --standalone --nnodes=1 --nproc-per-node="${RANKS_PER_NODE}")
elif command -v mpiexec >/dev/null 2>&1; then
    export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
    export MASTER_PORT=29500
    export WORLD_SIZE="${NTOTRANKS}"
    LAUNCH=(mpiexec -n "${NTOTRANKS}" --ppn "${RANKS_PER_NODE}"
            --depth="${OMP_NUM_THREADS}" --cpu-bind depth python)
else
    export MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}")
    export MASTER_PORT=29500
    export WORLD_SIZE="${NTOTRANKS}"
    LAUNCH=(mpirun -n "${NTOTRANKS}" --npernode "${RANKS_PER_NODE}"
            --hostfile "${PBS_NODEFILE}" python)
fi
echo "launcher=${LAUNCH[0]}"
echo "========================"

# --- run ---------------------------------------------------------------------
# --platform sophia is passed explicitly rather than left to auto-detection.
# CudaPlatform.name is "cuda" -- a backend, not a machine -- and Polaris and
# Sophia share it, so an unlabelled run records machine="cuda", lands in
# results/cuda_*.json, and summarize.py groups it with whatever else was called
# cuda. detect_platform() does look for "sophia" in PBS_JOBID and the hostname,
# but that is a fallback for runs outside PBS, not something to depend on for
# the deliverable.
"${LAUNCH[@]}" -m benchmark.train \
        --platform sophia \
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
        --note "sophia ${NNODES}-node bf16 strong-scaling 100ep"

# Global batch stays 1536 so accuracy-per-step matches Aurora and Polaris
# exactly; only the split changes (192 per rank over 8 ranks here, 384 over 4 on
# Polaris, 128 over 12 on Aurora). That is the point of --scaling strong:
# time-to-accuracy stays comparable across machines with different device counts.
