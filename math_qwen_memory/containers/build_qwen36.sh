#!/usr/bin/env bash
# Build a private LUMI PyTorch container with extra pip packages.
#
# Canonical LUMI workflow per:
#   https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/
#   https://docs.lumi-supercomputer.eu/software/installing/easybuild/
#
# Two files:
#   containers/qwen36_requirements.txt   (definition: extra pip packages)
#   containers/build_qwen36.sh           (this script: how to build)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REQS="$HERE/qwen36_requirements.txt"

# 1) Set EBU_USER_PREFIX BEFORE any `module load`. Doing it after has side
#    effects per LUMI docs: "Changing the value of EBU_USER_PREFIX while one
#    of the LUMI modules is loaded has side effects".
#    Pin to a project-local path under containers/ (containerised cleanup:
#    `rm -rf containers/easybuild` removes everything).
PRIVATE_EBU_PREFIX="$HERE/easybuild"
if [ -n "${EBU_USER_PREFIX:-}" ] && [ "$EBU_USER_PREFIX" != "$PRIVATE_EBU_PREFIX" ]; then
    echo "[build_qwen36] inherited EBU_USER_PREFIX=$EBU_USER_PREFIX (shared)"
    echo "[build_qwen36] overriding to project-local: $PRIVATE_EBU_PREFIX"
fi
export EBU_USER_PREFIX="$PRIVATE_EBU_PREFIX"
mkdir -p "$EBU_USER_PREFIX"

# 2) Clear any previously-loaded LUMI modules from the user's shell so they
#    don't carry stale CONTAINERROOT / SINGULARITY_BIND values.
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true

# 3) LUMI docs verbatim: "use the dummy partition `container`, e.g.:
#       module load LUMI partition/container EasyBuild-user
#       eb PyTorch-2.7.1-rocm-6.2.4-python-3.12-singularity-20250827.eb"
module load LUMI
module load partition/container
module load EasyBuild-user

# 4) Pick the newest mainline PyTorch container recipe. Recipes live at this
#    canonical path; reading directly avoids depending on `eb --search`.
LUMI_PT_RECIPES=/appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch
if [ -z "${PYTORCH_MODULE:-}" ]; then
    NEWEST_EB=$(ls "$LUMI_PT_RECIPES"/PyTorch-*-rocm-*-python-*-singularity-*.eb 2>/dev/null \
        | grep -E 'PyTorch-[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-python-.*-singularity-[0-9]{8}\.eb$' \
        | sort -r | head -1 || true)
    if [ -z "$NEWEST_EB" ]; then
        echo "ERROR: no PyTorch container recipe at $LUMI_PT_RECIPES" >&2
        exit 1
    fi
    EB_FILE=$(basename "$NEWEST_EB")
    PYTORCH_MODULE="PyTorch/${EB_FILE#PyTorch-}"
    PYTORCH_MODULE="${PYTORCH_MODULE%.eb}"
else
    EB_FILE="PyTorch-${PYTORCH_MODULE#PyTorch/}.eb"
fi

echo "[build_qwen36] EBU_USER_PREFIX=$EBU_USER_PREFIX"
echo "[build_qwen36] target module:  $PYTORCH_MODULE"
echo "[build_qwen36] eb recipe:      $EB_FILE"
echo "[build_qwen36] requirements:   $REQS"

# 5) Install the PyTorch container module under our private prefix (idempotent
#    — `eb` skips if already installed, unless --rebuild).
if ! module load "$PYTORCH_MODULE" 2>/dev/null; then
    echo "[build_qwen36] $PYTORCH_MODULE not yet installed. Running EasyBuild..."
    # --skip-sanity-check: the recipe's own sanity step pins exact version of
    # transformers / huggingface-cli that the public LUMI SIF doesn't always
    # match. Install proceeds otherwise; we run our own post-install asserts
    # later. Override via EB_FLAGS env var if you need stricter behaviour.
    eb "$EB_FILE" ${EB_FLAGS:---skip-sanity-check}
fi

# 6) Per LUMI docs: "To use the container after installation, the
#    EasyBuild-user module is not needed nor is the container partition."
#    Drop them, then load the freshly installed PyTorch module.
module unload EasyBuild-user 2>/dev/null || true
module unload partition/container 2>/dev/null || true
module load "$PYTORCH_MODULE"

echo "[build_qwen36] loaded $PYTORCH_MODULE"
echo "[build_qwen36] CONTAINERROOT=$CONTAINERROOT (module-managed, private)"
echo "[build_qwen36] SIF=$SIF"

# 7) Sanity-check ROCm torch from host shell (PyTorch ≥ 2.6 wraps python).
python -c "
import torch
assert torch.version.hip is not None, 'base torch is not ROCm — wrong module?'
print('base torch:', torch.__version__, 'hip:', torch.version.hip)
"

# 8) Install our pip packages. PyTorch ≥ 2.6 wraps pip into the container's
#    venv automatically. torch is already provided by the SIF → not pulled.
pip install --upgrade -r "$REQS"

# 9) Final import-symbol assertions for everything generate_traces/inject use.
python -c "
import torch
assert torch.version.hip is not None, 'ROCm torch lost! refuse to ship'
print('torch:', torch.__version__, 'hip:', torch.version.hip)

import transformers, accelerate, tokenizers, safetensors, huggingface_hub
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('tokenizers:', tokenizers.__version__)
print('safetensors:', safetensors.__version__)
print('huggingface_hub:', huggingface_hub.__version__)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LogitsProcessorList, TemperatureLogitsWarper,
    TopPLogitsWarper, TopKLogitsWarper, MinPLogitsWarper,
)
print('transformers symbols: OK')

from math_verify import parse, verify
print('math_verify: OK')
"

# 10) Bake the venv into a SquashFS for fast Lustre access. After this the
#     user-software dir is redundant and would shadow the squashfs; remove it.
make-squashfs
rm -rf "$CONTAINERROOT/user-software"

# 11) Reload module so the squashfs is bound; final post-squashfs check.
module unload "$PYTORCH_MODULE"
module load   "$PYTORCH_MODULE"
python -c "
import torch, transformers, accelerate, math_verify
print('post-squashfs:',
      'torch', torch.__version__,
      'transformers', transformers.__version__,
      'accelerate', accelerate.__version__,
      'math_verify OK')
"

echo
echo "[build_qwen36] DONE."
echo "[build_qwen36] artifact: $CONTAINERROOT/user-software.squashfs"
echo "[build_qwen36] back this up before any 'eb' re-install: cp -a $CONTAINERROOT /project/<id>/"
echo
echo "Run with:  sbatch sbatch/generate_traces_27b.sbatch"
