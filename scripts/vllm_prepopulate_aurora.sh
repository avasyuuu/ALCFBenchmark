#!/bin/bash -l
# One-time: fill vLLM's model-info cache on Aurora, so `vllm serve` can start.
#
#   qsub -I -q debug -A datascience_collab -l select=1,walltime=00:30:00,filesystems=flare:home
#   bash scripts/vllm_prepopulate_aurora.sh              # LlamaForCausalLM only
#   ARCHES="LlamaForCausalLM Qwen2ForCausalLM" bash scripts/vllm_prepopulate_aurora.sh
#   ARCHES=all bash scripts/vllm_prepopulate_aurora.sh   # the whole registry, slow
#
# RUN THIS ON A COMPUTE NODE, not a UAN. The cache is keyed to what vLLM can
# actually inspect on the hardware it will serve on, and the failure this works
# around is a segfault in that inspection.
#
# Why it exists: on Aurora, a first-time `vllm serve` dies with
#
#   pydantic_core.ValidationError: 1 validation error for ModelConfig
#     Value error, Model architectures ['LlamaForCausalLM'] failed to be inspected.
#
# vLLM inspects a model architecture in a subprocess to build its config, that
# subprocess segfaults on XPU, and the crash surfaces as a validation error
# naming neither the signal nor the cause. ALCF ships a workaround script that
# walks the model registry and writes the ModelInfo cache directly, skipping the
# inspection that crashes. Their own directory for it is named
# xpu-model-inspection-hidden-sigsegv, which is the clearest statement of the
# bug available. Expected to be fixed in a future frameworks module -- when a
# `vllm serve` works without this, it can go.
#
# The cache persists once built. It is keyed to VLLM_CACHE_ROOT, so moving that
# means doing this again -- which is why the path is set in one place here and
# read by submit_aurora_aiperf.sh rather than defaulting to ~/.cache/vllm in
# both.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

WA_URL="https://raw.githubusercontent.com/argonne-lcf/frameworks-sdk/main/tests/single_node/functionality/vllm/xpu-model-inspection-hidden-sigsegv/WA/vllm_build_all_modelinfo_caches.py"
WA_SCRIPT="vllm_build_all_modelinfo_caches.py"

# Under the repo on /flare, so it is visible from every compute node and outlives
# the job. Home would work too and is smaller; /flare is where the rest of this
# project's regenerable bulk already lives.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PWD}/.vllm_cache}"

# Only the architectures actually served. The registry is large and each entry
# costs an import; the default model needs exactly one of them.
ARCHES="${ARCHES:-LlamaForCausalLM}"

set +u
module load frameworks
set -u

# Compute nodes have no direct route off-site, and this step fetches a script.
# ALCF's page calls this out specifically -- without it curl hangs until it
# times out rather than failing, which reads as a slow download.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export no_proxy="localhost,127.0.0.1"

echo "cache root: ${VLLM_CACHE_ROOT}"
echo "arches:     ${ARCHES}"

if [[ ! -f "${WA_SCRIPT}" ]]; then
    echo "fetching ALCF's workaround script..."
    curl -sSL --fail -o "${WA_SCRIPT}" "${WA_URL}" || {
        echo "could not fetch ${WA_URL}" >&2
        echo "  check the proxy, or download it on a login node and copy it here" >&2
        exit 1
    }
fi

# Not vendored into the repo: it is ALCF's file for an ALCF bug, and a stale
# copy of someone else's workaround is worse than fetching the current one.
mkdir -p "${VLLM_CACHE_ROOT}"

if [[ "${ARCHES}" == "all" ]]; then
    python "${WA_SCRIPT}" --verbose
else
    ARGS=()
    for arch in ${ARCHES}; do ARGS+=(--arch "${arch}"); done
    python "${WA_SCRIPT}" "${ARGS[@]}" --verbose
fi

echo
echo "cache contents:"
find "${VLLM_CACHE_ROOT}" -type f | head -20
echo
echo "done. Now run, with the same cache root:"
echo "  VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT} bash scripts/submit_aurora_aiperf.sh"
