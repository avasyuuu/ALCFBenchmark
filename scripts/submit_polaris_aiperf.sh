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
#   2. build the venv. The conda module DOES carry vLLM and torch -- the
#      2025-09-25 module has vLLM 0.11.0rc2.dev147 and torch 2.8.0, alongside
#      sglang -- so the venv exists for aiperf, and pip will report vllm as
#      already satisfied rather than installing it:
#        module use /soft/modulefiles && module load conda && conda activate base
#        python -m venv --system-site-packages .venv-aiperf
#        source .venv-aiperf/bin/activate
#        pip install vllm aiperf nvidia-ml-py
#
#      `which vllm` should then point into the conda tree, not the venv. That is
#      what keeps vLLM on the module's stack: its shebang is conda's python, so
#      the packages aiperf downgrades in the venv (pyzmq, psutil, pillow) are
#      invisible to the server. AIPerf runs on the venv's python, vLLM on the
#      module's, and neither disturbs the other.
#
#      NOTE FOR THE COMPARISON: Aurora's frameworks module carries vLLM 0.15.0.
#      Four minor versions and a release candidate apart, across which vLLM's
#      scheduler and batching changed. Forcing a match by pip-installing 0.15.0
#      here would drag its own torch in and risk the CUDA stack, so the version
#      is recorded in run_meta.json instead and treated as a stated confound --
#      the same way --enforce-eager is.
#   3. pre-download the model while you still have easy outbound network.
#      meta-llama is GATED: accept the licence on huggingface.co once, then
#      `huggingface-cli login`. Compute nodes cannot fetch it themselves.
#
#        export HF_HUB_CACHE=$PWD/.hf-cache/hub
#        python -c "from huggingface_hub import snapshot_download as d; \
#          d('meta-llama/Llama-3.1-8B-Instruct', ignore_patterns=['original/*'])"
#
#      HF_HUB_CACHE, not HF_HOME. HF_HOME relocates the token too, so setting it
#      before logging in sends an anonymous request and the gate returns 401 --
#      which reads as "access not granted" when access is fine. This puts the
#      weights where HF_HOME below will look for them ($HF_HOME/hub) and leaves
#      the credential in ~/.cache/huggingface, outside this repo.
#
#      ignore_patterns skips original/consolidated.00.pth: 16 GB of Meta's own
#      checkpoint format that vLLM never reads. Without it the download is twice
#      the size for no benefit -- the safetensors shards are the whole model.
#
#      Aurora needs none of this -- ALCF stages the weights under
#      /flare/datasets/model-weights and the run sets HF_HUB_OFFLINE=1. Polaris
#      has no equivalent, so this is the one manual step the two do not share.

set -euo pipefail

# PBS_O_WORKDIR only exists under qsub. Falling back to $PWD lets the same
# script run inside an interactive allocation, which is where anything gets
# run the first time on new hardware.
cd "${PBS_O_WORKDIR:-$PWD}"
if [[ ! -d analysis ]]; then
    echo "error: no analysis/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_polaris_aiperf.sh" >&2
    exit 1
fi

# --- knobs -------------------------------------------------------------------
# Every value below matches submit_aurora_aiperf.sh, for the same reason the
# training scripts keep their flags identical: the point is a machine
# comparison, and a different model or sequence length makes the columns
# incomparable. Change one here and change it there.
#
# These were the laptop shakedown's settings until 2026-08-12 -- a 350M model at
# ISL 256 / OSL 128, far too small to say anything about an A100. Aurora ran
# Llama-3.1-8B at ISL 1024 / OSL 256, so that is the experiment now.
#
# The model is NOT pre-staged on Polaris the way ALCF stages it on Aurora under
# /flare/datasets/model-weights, and meta-llama is gated on HuggingFace. It has
# to be downloaded once from a login node with a token -- see the header.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"

# Fixed across every concurrency level ON PURPOSE. Scaling requests with
# concurrency makes each row a different amount of work, and then absolute
# joules and durations cannot be compared between rows -- only ratios can. Equal
# work costs wall time at low concurrency and is worth it.
#
# 128 rather than 256 to match Aurora, where the debug queue's one-hour cap set
# the number. An A100 at TP=1 should be quicker than an Aurora tile, so this may
# leave room -- but equal work across machines is worth more than a tighter fit.
REQUESTS="${REQUESTS:-128}"

# GPUs this job serves on, of the node's four. One by default, matching Aurora's
# TP=1 and leaving three powered and idle -- which is the cost this benchmark
# exists to expose, and the reason the collector below still reads all four.
TP="${TP:-1}"

# vLLM rejects a request longer than this. Sized from the sweep rather than
# pinned at 4096, so raising ISL does not silently start failing every request.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 512 ))}"
if (( MAX_MODEL_LEN < 8192 )); then MAX_MODEL_LEN=8192; fi

# On by default, and the reason is comparability rather than correctness here.
# ALCF's Aurora vLLM page uses --enforce-eager in every example, so the Aurora
# sweep runs eager; CUDA graphs would make Polaris faster and the two machines
# would then be measuring different things. Set ENFORCE_EAGER=0 for vLLM's real
# number on an A100 -- a legitimate run, just not one to put beside Aurora's.
EAGER=()
if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then EAGER=(--enforce-eager); fi

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

# Polaris compute nodes have no direct route off-site. Nothing in this script
# should need the network -- the model is cached and HF_HUB_OFFLINE is set below
# -- but AIPerf and vLLM both reach for a tokenizer on paths that are hard to
# predict, and without a proxy such a call hangs until it times out rather than
# failing fast, which reads as vLLM being slow to start. Harmless otherwise.
# Verify the hostname in ALCF's docs -- it has changed before.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export no_proxy="localhost,127.0.0.1,.alcf.anl.gov"

# Home is small and backed up; model weights are neither precious nor small.
export HF_HOME="${HF_HOME:-${PWD}/.hf-cache}"

# Use the cache and nothing else. Not an optimisation -- without it the run dies
# during vLLM startup on a gated repo, for a file it does not need.
#
# transformers probes for tokenizer.model at the repo root. Llama-3.1 has no such
# file (it ships tokenizer.json; the SentencePiece copy lives under original/),
# so the probe misses the cache and becomes an HTTP request. The compute node has
# no HF token -- the token stays on the login node deliberately -- so meta-llama
# answers 401 and an optional file turns into a fatal error several minutes into
# startup. Offline, the same probe is simply a miss.
#
# It also means a genuinely missing model fails immediately and says so, rather
# than hanging on a fetch through the proxy below.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Same reason as submit_polaris.sh: NVML enumerates in PCI bus order and CUDA
# does not by default, so pin the ordering before anything maps one to the other.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== job ${PBS_JOBID} ==="
echo "model=${MODEL}  TP=${TP}  isl=${ISL} osl=${OSL} max_len=${MAX_MODEL_LEN}"
echo "requests=${REQUESTS}  eager=${ENFORCE_EAGER:-1}"
echo "concurrencies=${CONCURRENCIES}"
echo "out=${OUTROOT}"
python -c "import torch; print('torch', torch.__version__, '|', torch.cuda.device_count(), 'GPUs')"
python -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK |', pynvml.nvmlDeviceGetCount(), 'GPUs visible to NVML')" \
    || echo "WARNING: pynvml missing -- no power data at all. pip install nvidia-ml-py"
echo "========================"

# --- provenance ---------------------------------------------------------------
# What the comparison needs to stay honest about. The vLLM version is the reason
# this exists: Aurora's frameworks module carries 0.15.0 and Polaris' conda
# carries 0.11.0rc2 -- four minor versions and a release candidate apart, across
# which vLLM's scheduler and batching changed. It appears nowhere else in the
# committed artifacts, since vllm.log is gitignored, so a reader comparing the
# two machines would have no way to know. Same for the eager flag and the model.
python -c '
import json, socket, subprocess, sys
def ver(dist):
    # importlib.metadata over __version__: it works for a package that never
    # exposes one, and it keeps the local tag -- Aurora reports "0.15.0+xpu",
    # which says more than "0.15.0" about which build produced the numbers.
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        try:
            return __import__(dist).__version__
        except Exception:
            return None
machine, model, tp, isl, osl, reqs, conc, eager, job = sys.argv[1:10]
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
except Exception:
    commit = None
print(json.dumps({
    "kind": "aiperf_run_meta", "machine": machine, "hostname": socket.gethostname(),
    "pbs_jobid": job or None, "git_commit": commit,
    "vllm": ver("vllm"), "torch": ver("torch"), "aiperf": ver("aiperf"),
    "model": model, "tensor_parallel": int(tp), "isl": int(isl), "osl": int(osl),
    "requests_per_level": int(reqs), "concurrencies": [int(c) for c in conc.split()],
    "enforce_eager": eager == "1",
}, indent=2))
' "polaris" "${MODEL}" "${TP}" "${ISL}" "${OSL}" "${REQUESTS}"   "${CONCURRENCIES}" "${ENFORCE_EAGER:-1}" "${PBS_JOBID:-}"   > "${OUTROOT}/run_meta.json"
echo "provenance -> ${OUTROOT}/run_meta.json"

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
# vLLM sees the first TP GPUs while AIPerf's collector still reads all four. That
# asymmetry is deliberate: the remaining A100s are idle but powered, and a
# node-hour bills for them too. Masking them in the collector as well would hide
# exactly the cost this benchmark exists to expose -- on Aurora that unused share
# came to 82.5% of node energy at TP=1, and four GPUs should show the same shape
# less starkly than twelve tiles.
echo "starting vLLM on GPU(s) $(seq -s, 0 $(( TP - 1 ))) of 4..."
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( TP - 1 )))" vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    "${EAGER[@]}" \
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
for _ in $(seq 1 360); do
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
