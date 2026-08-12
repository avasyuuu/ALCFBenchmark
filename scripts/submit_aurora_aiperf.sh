#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_aiperf_power
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -q debug
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# LLM *inference* power sweep on Aurora: AIPerf driving vLLM on Intel Max GPUs,
# with tile energy read by analysis/power_sidecar.py.
#
#   qsub scripts/submit_aurora_aiperf.sh
#   MODEL=meta-llama/Llama-3.3-70B-Instruct TP=8 qsub -v MODEL,TP scripts/submit_aurora_aiperf.sh
#
# Queue: `debug`, which is where a 1-node Aurora job belongs -- it caps at
# nodect 2 and walltime 01:00:00, and the sweep below is sized to fit that hour.
# It also allows one running job per user, so an interactive allocation you are
# still sitting in will block this one rather than run beside it.
#
# NOT YET RUN ON HARDWARE. Written against ALCF's Aurora vLLM page
# (docs.alcf.anl.gov/aurora/data-science/inference/vllm/, vLLM 0.15.0 in the
# frameworks module) and the AIPerf 0.12 CLI. The parts most likely to need a fix
# on first contact are the aiperf venv layering, vLLM's startup time on PVC, and
# whether the chosen model is actually staged under HF_HOME.
#
# This is a DIFFERENT benchmark from submit_aurora.sh, not a variant of it: that
# one trains ResNet-20 and reports samples/joule, this one serves an LLM and
# reports tokens/joule. They share only the energy methodology. Nothing here
# feeds the machine-comparison table in analysis/summarize.py.
#
# Why a sidecar at all: AIPerf's power collectors are DCGM, pynvml and amdsmi.
# There is no XPU path, so every nvidia_* field in its export is empty here. The
# i915 hwmon counters are readable anyway and do not require being the process
# under test, so power_sidecar.py samples them from beside each aiperf run and
# summarize_aiperf.py divides the tokens by those joules.
#
# Before the first run, on a LOGIN node:
#   1. set -A above to your real project name (my.alcf.anl.gov)
#   2. build the aiperf venv ON TOP of frameworks, so vLLM stays importable:
#        module load frameworks
#        python -m venv --system-site-packages .venv-aiperf-aurora
#        source .venv-aiperf-aurora/bin/activate
#        pip install aiperf
#   3. confirm the model is staged -- HF_HUB_OFFLINE=1 below means a missing
#      model fails at load rather than downloading:
#        ls /flare/datasets/model-weights/hub | grep -i llama

set -euo pipefail

# PBS_O_WORKDIR only exists under qsub. Falling back to $PWD lets the same
# script run inside an interactive allocation, which is where anything gets run
# the first time on new hardware -- a queued job that fails in the first ten
# seconds costs a queue wait to learn what a shell says immediately.
cd "${PBS_O_WORKDIR:-$PWD}"
if [[ ! -d analysis ]]; then
    echo "error: no analysis/ package in ${PBS_O_WORKDIR}" >&2
    echo "       submit from the repo root: cd <repo> && qsub scripts/submit_aurora_aiperf.sh" >&2
    exit 1
fi

# --- knobs -------------------------------------------------------------------
# One tile by default. Not because it is the interesting configuration, but
# because it is the one with no failure modes: tensor parallelism has to divide
# the model's attention head count evenly, and TP=1 divides everything. It also
# produces the starkest version of the finding this benchmark exists for -- 11
# of 12 tiles powered and idle while one serves, which the training side already
# measured at over 90% of node energy.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"

# Fixed across every concurrency level ON PURPOSE. Scaling requests with
# concurrency makes each row a different amount of work, and then absolute
# joules and durations cannot be compared between rows -- only ratios can.
#
# 128 rather than the 256 submit_polaris_aiperf.sh uses, because a 1-node Aurora
# job runs in `debug` and that queue caps at one hour. Sized from the first run's
# measured 0.676 req/s at concurrency 4: 128 requests is ~23 min of profiling
# across the six levels, ~7 min of AIPerf startup and warmup, ~3 min of vLLM load
# and 30 s of idle floor -- about 34 minutes. 256 is ~57 min of profiling alone
# and does not fit.
#
# The cost is that absolute joules here are not comparable to a Polaris sweep at
# 256. Rates and ratios still are, and those are what the comparison rests on --
# but if the two ever need to sit in one table, run both at the same count on a
# queue that allows it.
REQUESTS="${REQUESTS:-128}"

# vLLM rejects a request longer than this, and AIPerf's ISL is a prompt length
# it will actually send. Sized from the sweep rather than pinned, so raising ISL
# does not silently start failing every request. ALCF's own examples use 8192;
# this only exceeds it if you ask for more.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 512 ))}"
if (( MAX_MODEL_LEN < 8192 )); then MAX_MODEL_LEN=8192; fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="results/aiperf/aurora-${STAMP}"
mkdir -p "${OUTROOT}" logs

# --- environment -------------------------------------------------------------
# vLLM 0.15.0 ships inside frameworks on Aurora -- no source build, unlike every
# other XPU vLLM story. The venv layers aiperf on top with --system-site-packages
# so `vllm` and `torch` still resolve to the module's copies.
set +u
module load frameworks
source .venv-aiperf-aurora/bin/activate
set -u

# Straight from ALCF's Aurora vLLM page. Not trimmed to the ones that look
# necessary: several of these only matter under Ray or multi-tile, and a sweep
# that grows a TP dimension later should not have to rediscover them.
export HF_HOME="${HF_HOME:-/flare/datasets/model-weights}"
export HF_DATASETS_CACHE="${HF_HOME}"
export HF_MODULES_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=hydra
export OMP_NUM_THREADS=8
export TORCH_LLM_ALLREDUCE=1

# FLAT makes each tile its own device, so a node is 12 devices rather than 6.
# The sidecar depends on this exactly as AuroraPlatform does: it derives a
# tile's device index as card*2+tile, which is only the right answer under FLAT.
# Setting it here keeps the energy attribution and vLLM's device numbering on
# the same model of the machine.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# frameworks/2025.3.1 sets ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" on
# load and prints a warning saying so -- it is what enables vLLM, Ray, Triton-XPU
# and dpctl. Left alone deliberately. ALCF documents "level_zero:gpu" as the
# revert if something misbehaves, so set ONEAPI_REVERT=1 to take it, and email
# support@alcf.anl.gov if you have to, which is what they ask for.
if [[ "${ONEAPI_REVERT:-0}" == "1" ]]; then
    export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
    echo "NOTE: ONEAPI_DEVICE_SELECTOR reverted to level_zero:gpu"
fi

export VLLM_HOST_IP=$(getent hosts "$(hostname).hsn.cm.aurora.alcf.anl.gov" \
    | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
export no_proxy="localhost,127.0.0.1"

# vLLM's model-info cache. Must match the root scripts/vllm_prepopulate_aurora.sh
# filled, because the cache is keyed to it. Checked rather than assumed: without
# it, `vllm serve` dies several minutes in with a pydantic ValidationError that
# names neither the cache nor the segfault behind it, and the job has by then
# spent its startup budget to learn nothing.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PWD}/.vllm_cache}"
if ! find "${VLLM_CACHE_ROOT}" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "error: vLLM model-info cache is empty at ${VLLM_CACHE_ROOT}" >&2
    echo "       vllm serve will fail on Aurora without it -- model inspection" >&2
    echo "       segfaults and surfaces as a ModelConfig validation error." >&2
    echo "       Fill it once, on a compute node:" >&2
    echo "         bash scripts/vllm_prepopulate_aurora.sh" >&2
    exit 1
fi

# Tiles vLLM will occupy, for the sidecar's bound/idle split. vLLM takes the
# first TP devices, so this is 0..TP-1. Everything else on the node is powered
# and unused, and saying so explicitly is what makes the idle columns mean
# something -- the sidecar cannot infer it, since a device can idle at a third
# of its peak and it never sees vLLM's placement.
BOUND="$(seq -s, 0 $(( TP - 1 )))"

echo "=== job ${PBS_JOBID} ==="
echo "model=${MODEL}  TP=${TP}  isl=${ISL} osl=${OSL} max_len=${MAX_MODEL_LEN}"
echo "requests=${REQUESTS}  concurrencies=${CONCURRENCIES}"
# Spelled out because "bound tiles=0" at TP=1 reads as "no tiles are bound"
# when it means "tile index 0" -- the one number on this line where the correct
# value and an alarming misreading are the same character.
echo "tiles serving=[${BOUND}] (${TP} of 12, $(( 12 - TP )) idle)   out=${OUTROOT}"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import torch; print('torch', torch.__version__, '| xpu devices', torch.xpu.device_count())"
python -c "
from benchmark.hwmon import intel_energy_sources
s = intel_energy_sources()
print(f'hwmon: {len(s)} counter(s),', sum(1 for x in s if not x.aggregate), 'per-tile')
assert s, 'no readable i915 energy counters -- the sidecar would measure nothing'"
echo "========================"

# --- idle floor ---------------------------------------------------------------
# Measured BEFORE vLLM starts and with nothing else on the node. Taken after a
# run instead it reads high, because clocks and fan state have not settled.
# nvml_idle.py cannot do this here -- there is no NVML -- so the sidecar samples
# its own floor, which also guarantees the floor and the rows share a scope.
echo "sampling idle floor (30 s, node must be quiet)..."
python analysis/power_sidecar.py \
    --out "${OUTROOT}/idle.json" \
    --bound-devices none \
    --label "idle floor" \
    -- sleep 30

# --- serve --------------------------------------------------------------------
# --enforce-eager per ALCF's page. It costs throughput, so treat these numbers as
# eager-mode numbers rather than as vLLM's best -- and do not compare them to a
# Polaris run made without it.
echo "starting vLLM (TP=${TP})..."
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enforce-eager \
    > "${OUTROOT}/vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "stopping vLLM (pid ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# 30 minutes. A 70B at TP=8 loads far slower than the 8B default, and the whole
# job is wasted if the wait gives up first -- so the budget is generous and the
# liveness check below is what actually catches a dead server.
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
    # The sidecar writes sidecar_power.json into the artifact dir, which is where
    # summarize_aiperf.py looks for it. It exits with aiperf's own status, so a
    # failed level is still a failed level rather than a successful measurement
    # of a failure.
    #
    # ignore_eos + min_tokens force every generation to exactly OSL tokens.
    # Without them output length drifts with load and tokens/joule moves for
    # reasons that have nothing to do with concurrency.
    python analysis/power_sidecar.py \
        --out "${OUTROOT}/c${c}/sidecar_power.json" \
        --bound-devices "${BOUND}" \
        --label "concurrency ${c}" \
        -- aiperf profile \
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
