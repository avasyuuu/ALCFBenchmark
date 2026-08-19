#!/bin/bash -l
#PBS -A datascience_collab
#PBS -N alcf_aiperf_power
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -q by-node
#PBS -k doe
#PBS -o logs/
#PBS -e logs/

# LLM *inference* power sweep on Sophia: NVIDIA AIPerf driving vLLM on a DGX
# A100 node, with per-GPU power read through NVML.
#
#   qsub scripts/submit_sophia_aiperf.sh
#   TP=8 qsub -v TP scripts/submit_sophia_aiperf.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct qsub -v MODEL scripts/submit_sophia_aiperf.sh
#
# NOT YET RUN ON HARDWARE, and Sophia has never produced a run of any kind --
# a job queued on 2026-08-11 never started with 21 of 24 nodes free and no
# scheduler comment, and the queue question is still open. Prove the machine
# first with the training smoke test, which is one batch job and answers
# whether this is the queue or the environment:
#
#   SMOKE=1 qsub -l walltime=00:20:00 -v SMOKE scripts/submit_sophia.sh
#
# QUEUE: `by-node` is REQUIRED here, and not for the reason submit_sophia.sh
# needs it. The default queue is `by-gpu`, which hands out individual GPUs and
# will schedule far more easily -- but NVML enumerates every physical GPU on
# the node regardless of what this job was allocated, so on a shared node the
# "node energy" this benchmark reports would include other people's jobs. Every
# samples/J and tokens/J number on the site assumes exclusive node access. A
# by-gpu run is not a worse measurement, it is a wrong one.
#
# WHY SOPHIA IS WORTH THE TROUBLE: 8 x 40 GB = 320 GB per node against Polaris'
# 160 GB. Llama-3.3-70B is ~141 GB in bf16, which leaves Polaris about 3 GB for
# KV cache -- unusable, which is why the 70B there needs a Ray cluster across
# two nodes that has never been built. On one Sophia node at TP=8 it fits with
# ~147 GB to spare. This is the only NVIDIA machine here that can serve the 70B
# without multi-node serving, and 70B head counts divide by 8 (64 attention, 8
# KV -> 8 and 1 per GPU).
#
# Validated on hardware 2026-08-13 (Llama-3.1-8B at TP=1) and 2026-08-14
# (gemma-3-27b at TP=4), against the AIPerf 0.12 CLI and vLLM's OpenAI server.
# A six-level concurrency sweep takes ~20 min at TP=1 and ~30 min at TP=4,
# both inside the debug queue's one-hour cap.
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
#      Sophia has never had this venv built -- it is an open item from
#      2026-08-11, alongside an ssh key, the same pair Crux needed. Neither
#      exists yet, so budget a login-node session before the first sweep.
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
#      /flare/datasets/model-weights and the run sets HF_HUB_OFFLINE=1. Neither
#      Polaris nor Sophia has an equivalent, so this is the one manual step
#      Aurora does not share. Sophia and Polaris both mount eagle; if the cache
#      already exists there from the Polaris work, point HF_HOME at it rather
#      than downloading 16 GB twice.

set -euo pipefail

# PBS_O_WORKDIR only exists under qsub. Falling back to $PWD lets the same
# script run inside an interactive allocation, which is where anything gets
# run the first time on new hardware.
cd "${PBS_O_WORKDIR:-$PWD}"
if [[ ! -d analysis ]]; then
    echo "error: no analysis/ package in $(pwd)" >&2
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

# Prompt and generation length as ISL:OSL pairs, when the thing being swept is
# the shape of the work rather than how much of it arrives at once. Matches
# submit_aurora_aiperf.sh, which is the only way the two stay comparable.
#
#   SHAPES="1024:64 1024:256 1024:1024" CONCURRENCIES=32 qsub -v SHAPES,CONCURRENCIES ...
#
# The server loads once and does not care what shape it is sent, so five shapes
# in one allocation pay vLLM startup once instead of five times.
#
# Empty means the single ISL/OSL pair above and the level directories keep their
# c<N> names, which is what every sweep already on disk is called.
SHAPES="${SHAPES:-}"
SHAPE_LIST="${SHAPES:-${ISL}:${OSL}}"

# Fixed across every concurrency level ON PURPOSE. Scaling requests with
# concurrency makes each row a different amount of work, and then absolute
# joules and durations cannot be compared between rows -- only ratios can. Equal
# work costs wall time at low concurrency and is worth it.
#
# 128 rather than 256 to match Aurora, where the debug queue's one-hour cap set
# the number. Sophia has no debug queue and so no one-hour cap -- the walltime
# above is 90 minutes -- but the request count stays at Aurora's anyway: equal
# work across machines is the whole point, and a Sophia-only number would not
# belong in the comparison.
REQUESTS="${REQUESTS:-128}"

# GPUs this job serves on, of the node's eight. One by default, matching
# Aurora's and Polaris' TP=1 and leaving seven powered and idle -- which is the
# cost this benchmark exists to expose, and the reason the collector below
# still reads every GPU on the node. Sophia shows that cost more starkly than
# anywhere else here: 7 of 8 idle against Polaris' 3 of 4.
TP="${TP:-1}"

# Asked rather than hardcoded, for the reason submit_sophia.sh asks: the bigmem
# nodes differ from the rest, and a wrong constant here would mask a real GPU
# or claim one that is not there. Falls back to 8 only if torch cannot answer.
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count() or 8)' \
    2>/dev/null || echo 8)}"
if (( TP > NGPU )); then
    echo "error: TP=${TP} but this node reports ${NGPU} GPU(s)" >&2
    exit 1
fi

# vLLM rejects a request longer than this. Sized from the sweep rather than
# pinned at 4096, so raising ISL does not silently start failing every request.
# Sized from the longest shape in the sweep, not the default pair: the server
# loads once and serves all of them.
_longest=0
for _shape in ${SHAPE_LIST}; do
    _need=$(( ${_shape%%:*} + ${_shape##*:} + 512 ))
    if (( _need > _longest )); then _longest=${_need}; fi
done
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${_longest}}"
if (( MAX_MODEL_LEN < 8192 )); then MAX_MODEL_LEN=8192; fi

# On by default, and the reason is comparability rather than correctness here.
# ALCF's Aurora vLLM page uses --enforce-eager in every example, so the Aurora
# sweep runs eager; CUDA graphs would make Sophia faster and the machines would
# then be measuring different things. Set ENFORCE_EAGER=0 for vLLM's real
# number on a DGX A100 -- a legitimate run, just not one to put beside the
# Aurora or Polaris sweeps.
# Weight/activation dtype handed to vLLM. bfloat16 by default, which is what
# every dense model here has been measured in and what keeps the machines
# comparable.
#
# Set DTYPE=auto for a model that ships its own quantization. gpt-oss-120b is
# mxfp4 for the MoE weights (its config names the quant_method and exempts
# attention, router and embeddings); forcing bfloat16 there asks vLLM to
# ignore that and hold ~234 GB of dequantized experts instead of ~61 GB, which
# turns a run that fits four tiles into one that barely fits eight. "auto"
# honours whatever the checkpoint declares.
DTYPE="${DTYPE:-bfloat16}"

EAGER=()
if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then EAGER=(--enforce-eager); fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="results/aiperf/sophia-${STAMP}"
mkdir -p "${OUTROOT}" logs

# --- environment -------------------------------------------------------------
set +u
module use /soft/modulefiles
module load conda
conda activate base
source .venv-aiperf/bin/activate
set -u

# Sophia compute nodes have no direct route off-site (assumed to match
# Polaris; verify against ALCF's Sophia docs). Nothing in this script
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

# Same reason as submit_sophia.sh: NVML enumerates in PCI bus order and CUDA
# does not by default, so pin the ordering before anything maps one to the other.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# The AF_UNIX 108-byte path ceiling, same as every other script here. PBS builds
# TMPDIR from the full job id and hostname --
#   /var/tmp/pbs.172717.sophia-pbs-01.lab.alcf.anl.gov/tmpXXXXXXXX
# -- and AIPerf opens ZMQ IPC sockets under it for its internal event bus, which
# puts the socket path over the limit before AIPerf has sent a single request.
#
# Worse than a crash: AIPerf logs the failure and exits 0, so the sweep reports
# every level "ok" and produces no exports at all. The check after each level
# below exists because of that.
export TMPDIR=/tmp

# :-interactive because this script is meant to run inside an
# allocation as well as under qsub, and `set -u` makes a bare
# ${PBS_JOBID} fatal outside one -- for a line that only labels the log.
echo "=== job ${PBS_JOBID:-interactive} ==="
echo "model=${MODEL}  TP=${TP}  isl=${ISL} osl=${OSL} max_len=${MAX_MODEL_LEN}"
echo "requests=${REQUESTS}  eager=${ENFORCE_EAGER:-1}  dtype=${DTYPE}"
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
machine, model, tp, shapes, reqs, conc, eager, job, dtype = sys.argv[1:10]
pairs = [tuple(int(v) for v in s.split(":")) for s in shapes.split()]
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
except Exception:
    commit = None
print(json.dumps({
    "kind": "aiperf_run_meta", "machine": machine, "hostname": socket.gethostname(),
    "pbs_jobid": job or None, "git_commit": commit,
    "vllm": ver("vllm"), "torch": ver("torch"), "aiperf": ver("aiperf"),
    "model": model, "tensor_parallel": int(tp),
    # isl/osl stay scalar for a single-shape sweep, which is what every reader
    # of this file expects and what every sweep on disk has. A sweep that varied
    # the shape has no single answer, and says so with null rather than naming
    # whichever level happened to be first.
    "isl": pairs[0][0] if len(pairs) == 1 else None,
    "osl": pairs[0][1] if len(pairs) == 1 else None,
    "shapes": [{"isl": i, "osl": o} for i, o in pairs],
    "requests_per_level": int(reqs), "concurrencies": [int(c) for c in conc.split()],
    "enforce_eager": eager == "1", "dtype": dtype,
}, indent=2))
' "sophia" "${MODEL}" "${TP}" "${SHAPE_LIST}" "${REQUESTS}"   "${CONCURRENCIES}" "${ENFORCE_EAGER:-1}" "${PBS_JOBID:-}" "${DTYPE}"   > "${OUTROOT}/run_meta.json"
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
# vLLM sees the first TP GPUs while AIPerf's collector still reads all of them.
# That asymmetry is deliberate: the remaining A100s are idle but powered, and a
# node-hour bills for them too. Masking them in the collector as well would hide
# exactly the cost this benchmark exists to expose -- on Aurora that unused
# share came to 82.5% of node energy at TP=1, and eight GPUs sit between
# Polaris' four and Aurora's twelve tiles.
echo "starting vLLM on GPU(s) $(seq -s, 0 $(( TP - 1 ))) of ${NGPU}..."
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( TP - 1 )))" vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype "${DTYPE}" \
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
# Level directories stay c<N> while there is one shape, and only take the
# _i<ISL>o<OSL> suffix when there is more than one -- a suffix on every run
# would rename every level of every future single-shape sweep.
_nshapes=0
for _shape in ${SHAPE_LIST}; do _nshapes=$(( _nshapes + 1 )); done

for shape in ${SHAPE_LIST}; do
    s_isl="${shape%%:*}"
    s_osl="${shape##*:}"
    for c in ${CONCURRENCIES}; do
        if (( _nshapes > 1 )); then
            lvl="c${c}_i${s_isl}o${s_osl}"
            echo "=== concurrency ${c}, isl ${s_isl} osl ${s_osl} ==="
        else
            lvl="c${c}"
            echo "=== concurrency ${c} ==="
        fi
        # ignore_eos + min_tokens force every generation to exactly OSL tokens.
        # Without them the model stops at EOS, output length drifts with load, and
        # tokens/joule moves for reasons that have nothing to do with concurrency --
        # a longer generation amortises the fixed prefill over more decode tokens.
        # This is the single most important flag on the command, and Ollama has no
        # equivalent, which is why this benchmark uses vLLM.
        # Wrapped in the sidecar even though AIPerf reads NVML itself. Two
        # instruments on the same A100s is the only cross-check available
        # anywhere in this project -- Aurora has no second opinion and its numbers
        # rest entirely on this method being right. The sidecar also reports the
        # bound/idle device split, which AIPerf does not, so the idle columns stop
        # coming back blank on this machine.
    #
        # summarize_aiperf keeps AIPerf's own figures when both exist and reports
        # the disagreement rather than averaging: the two measure the same silicon,
        # so a gap between them is an error in one of them, not a range.
        python analysis/power_sidecar.py \
            --out "${OUTROOT}/${lvl}/sidecar_power.json" \
            --machine sophia \
            --bound-devices "$(seq -s, 0 $(( TP - 1 )))" \
            --label "concurrency ${c} isl ${s_isl} osl ${s_osl}" \
            -- aiperf profile \
                --model "${MODEL}" \
                --tokenizer "${MODEL}" \
                --url "http://localhost:${PORT}" \
                --endpoint-type chat \
                --streaming \
                --concurrency "${c}" \
                --request-count "${REQUESTS}" \
                --warmup-request-count 8 \
                --isl "${s_isl}" \
                --osl "${s_osl}" \
                --osl-stddev 0 \
                --extra-inputs ignore_eos:true \
                --extra-inputs "min_tokens:${s_osl}" \
                --gpu-telemetry pynvml \
                --ui none \
                --output-artifact-dir "${OUTROOT}/${lvl}" \
            > "${OUTROOT}/aiperf-${lvl}.log" 2>&1 \
            || { echo "  FAILED (see ${OUTROOT}/aiperf-${lvl}.log)"; continue; }
        # Exit code is not enough. AIPerf can log a fatal error and still exit 0 --
        # a ZMQ socket path over the AF_UNIX limit did exactly that, and the sweep
        # reported six cheerful "ok" lines and wrote no exports. The artifact is the
        # only honest evidence a level ran.
        if [[ -f "${OUTROOT}/${lvl}/profile_export_aiperf.json" ]]; then
            echo "  ok -> ${OUTROOT}/c${c}"
        else
            echo "  FAILED: exited 0 but wrote no export (see ${OUTROOT}/aiperf-${lvl}.log)"
        fi
    done
done

# --- summarise ----------------------------------------------------------------
echo
python analysis/summarize_aiperf.py "${OUTROOT}"/c* \
    --idle "${OUTROOT}/idle.json" \
    --json "${OUTROOT}/summary.json" \
    | tee "${OUTROOT}/summary.txt"

echo
echo "artifacts in ${OUTROOT}"
