#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_aiperf_power
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -q debug
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# LLM *inference* power sweep on Polaris: NVIDIA AIPerf driving vLLM on one A100,
# with per-GPU power read through NVML.
#
#   qsub scripts/submit_polaris_aiperf.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct qsub -v MODEL scripts/submit_polaris_aiperf.sh
#
# NOT YET RUN ON HARDWARE. Written against the AIPerf 0.12 CLI and vLLM's OpenAI
# server, validated end-to-end only on a consumer GPU under Ollama. The module
# names, the proxy hostname and the vLLM startup time are the parts most likely
# to need a fix on first contact.
#
# This is a DIFFERENT benchmark from submit_polaris.sh, not a variant of it: that
# one trains ResNet-20 and reports samples/joule, this one serves an LLM and
# reports tokens/joule. They share only the energy methodology. Nothing here
# feeds the machine-comparison table in analysis/summarize.py.
#
# Before the first run, on a LOGIN node:
#   1. set -A above to your real project name (my.alcf.anl.gov)
#   2. build the venv (vLLM is not in the conda module):
#        module use /soft/modulefiles && module load conda && conda activate base
#        python -m venv --system-site-packages .venv-aiperf
#        source .venv-aiperf/bin/activate
#        pip install vllm aiperf nvidia-ml-py
#   3. pre-download the model while you still have easy outbound network:
#        HF_HOME=/eagle/<project>/$USER/hf python -c \
#          "from huggingface_hub import snapshot_download as d; d('ibm-granite/granite-4.0-350m')"

set -euo pipefail

cd "${PBS_O_WORKDIR}"
if [[ ! -d analysis ]]; then
    echo "error: no analysis/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_polaris_aiperf.sh" >&2
    exit 1
fi

# --- knobs -------------------------------------------------------------------
# Default model matches the laptop shakedown run so the two are directly
# comparable. It is far too small to saturate an A100 -- for a number that says
# something about the hardware rather than about launch overhead, override with
# a 7B+ model, which is also where the ~40 GB of HBM starts to matter.
MODEL="${MODEL:-ibm-granite/granite-4.0-350m}"
PORT="${PORT:-8000}"
ISL="${ISL:-256}"
OSL="${OSL:-128}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"

# Fixed across every concurrency level ON PURPOSE. Scaling requests with
# concurrency makes each row a different amount of work, and then absolute
# joules and durations cannot be compared between rows -- only ratios can. Equal
# work costs wall time at low concurrency and is worth it.
REQUESTS="${REQUESTS:-256}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="results/aiperf/polaris-${STAMP}"
mkdir -p "${OUTROOT}" logs

# --- environment -------------------------------------------------------------
set +u
module use /soft/modulefiles
module load conda
conda activate base
source .venv-aiperf/bin/activate
set -u

# Polaris compute nodes have no direct route off-site. Without these, the first
# HuggingFace tokenizer fetch hangs until it times out rather than failing fast,
# which looks like vLLM being slow to start. Harmless when the model is already
# cached, so they are set unconditionally. Verify the hostname in ALCF's docs --
# it has changed before.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export no_proxy="localhost,127.0.0.1,.alcf.anl.gov"

# Home is small and backed up; model weights are neither precious nor small.
export HF_HOME="${HF_HOME:-${PWD}/.hf-cache}"

# Same reason as submit_polaris.sh: NVML enumerates in PCI bus order and CUDA
# does not by default, so pin the ordering before anything maps one to the other.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== job ${PBS_JOBID} ==="
echo "model=${MODEL}  isl=${ISL} osl=${OSL}  requests=${REQUESTS}"
echo "concurrencies=${CONCURRENCIES}"
echo "out=${OUTROOT}"
python -c "import torch; print('torch', torch.__version__, '|', torch.cuda.device_count(), 'GPUs')"
python -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK |', pynvml.nvmlDeviceGetCount(), 'GPUs visible to NVML')" \
    || echo "WARNING: pynvml missing -- no power data at all. pip install nvidia-ml-py"
echo "========================"

# --- idle floor ---------------------------------------------------------------
# Measured BEFORE vLLM starts and with nothing else on the node. Taken after a
# run instead, it reads high: clocks and fan state have not settled, which on a
# laptop was worth 6 W of pure error.
echo "sampling idle floor (30 s, node must be quiet)..."
python analysis/nvml_idle.py --seconds 30 -o "${OUTROOT}/idle.json" >/dev/null
python -c "
import json; d=json.load(open('${OUTROOT}/idle.json'))
print(f\"idle: {d['node_idle_w']} W over {d['device_count']} GPU(s)\")"

# --- serve --------------------------------------------------------------------
# vLLM is pinned to GPU 0 while AIPerf's collector still reads all four. That
# asymmetry is deliberate: the other three A100s are idle but powered, and a
# node-hour bills for them too. Masking them in the collector as well would hide
# exactly the cost this benchmark exists to expose.
echo "starting vLLM on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len 4096 \
    --disable-log-requests \
    > "${OUTROOT}/vllm.log" 2>&1 &
VLLM_PID=$!

# Kill the server on any exit path. Without this a failed sweep leaves vLLM
# holding the GPU for the rest of the walltime.
cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "stopping vLLM (pid ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -n "waiting for /health"
for _ in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " -- up"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo " -- vLLM died during startup; last lines:" >&2
        tail -n 40 "${OUTROOT}/vllm.log" >&2
        exit 1
    fi
    echo -n "."
    sleep 5
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
    echo "vLLM never became healthy; see ${OUTROOT}/vllm.log" >&2
    exit 1
}

# --- sweep --------------------------------------------------------------------
for c in ${CONCURRENCIES}; do
    echo "=== concurrency ${c} ==="
    # ignore_eos + min_tokens force every generation to exactly OSL tokens.
    # Without them the model stops at EOS, output length drifts with load, and
    # tokens/joule moves for reasons that have nothing to do with concurrency --
    # a longer generation amortises the fixed prefill over more decode tokens.
    # This is the single most important flag on the command, and Ollama has no
    # equivalent, which is why this benchmark uses vLLM.
    aiperf profile \
        --model "${MODEL}" \
        --tokenizer "${MODEL}" \
        --url "http://localhost:${PORT}" \
        --endpoint-type chat \
        --streaming \
        --concurrency "${c}" \
        --request-count "${REQUESTS}" \
        --warmup-request-count 8 \
        --isl "${ISL}" \
        --osl "${OSL}" \
        --osl-stddev 0 \
        --extra-inputs ignore_eos:true \
        --extra-inputs "min_tokens:${OSL}" \
        --gpu-telemetry pynvml \
        --ui none \
        --output-artifact-dir "${OUTROOT}/c${c}" \
        > "${OUTROOT}/aiperf-c${c}.log" 2>&1 \
        || { echo "  FAILED (see ${OUTROOT}/aiperf-c${c}.log)"; continue; }
    echo "  ok -> ${OUTROOT}/c${c}"
done

# --- summarise ----------------------------------------------------------------
echo
python analysis/summarize_aiperf.py "${OUTROOT}"/c* \
    --idle "${OUTROOT}/idle.json" \
    --json "${OUTROOT}/summary.json" \
    | tee "${OUTROOT}/summary.txt"

echo
echo "artifacts in ${OUTROOT}"
