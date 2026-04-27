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
#    A full purge is the safest way to eliminate EB-user's python 3.11
#    leftover that would shadow the container's pip/python wrappers.
module purge -f 2>/dev/null || true
module load LUMI
# Load the partition that the module was installed under (login = L).
PT_PARTITION=$(find "$EBU_USER_PREFIX/modules/LUMI/" -name "${PYTORCH_MODULE#PyTorch/}.lua" 2>/dev/null \
    | head -1 \
    | sed -nE 's|.*/partition/([^/]+)/PyTorch/.*|\1|p')
[ -n "$PT_PARTITION" ] && module load "partition/$PT_PARTITION"
module load "$PYTORCH_MODULE"

echo "[build_qwen36] loaded $PYTORCH_MODULE (partition/$PT_PARTITION)"
echo "[build_qwen36] CONTAINERROOT=$CONTAINERROOT (module-managed, private)"
echo "[build_qwen36] SIF=$SIF"

# 7) Confirm `python` and `pip` are the CONTAINER WRAPPERS, not EB-user's.
echo "[build_qwen36] which python -> $(which python)"
echo "[build_qwen36] which pip    -> $(which pip)"
PYBIN=$(which python)
PIPBIN=$(which pip)
if [[ "$PYBIN" != "$CONTAINERROOT/"* || "$PIPBIN" != "$CONTAINERROOT/"* ]]; then
    echo "ERROR: python/pip not under \$CONTAINERROOT — wrong wrapper." >&2
    echo "       expected prefix: $CONTAINERROOT/" >&2
    echo "       got python: $PYBIN" >&2
    echo "       got pip:    $PIPBIN" >&2
    exit 1
fi

# 8) Sanity-check ROCm torch from host shell (PyTorch ≥ 2.6 wraps python).
python -c "
import sys, torch
print('python:', sys.version)
assert torch.version.hip is not None, 'base torch is not ROCm — wrong module?'
print('base torch:', torch.__version__, 'hip:', torch.version.hip)
"

# 8) Install our pip packages. PyTorch ≥ 2.6 wraps pip into the container's
#    venv automatically. torch is already provided by the SIF → not pulled.
pip install --upgrade -r "$REQS"

# 8b) Diagnostics: where did pip actually put things? Container's base SIF
#     ships transformers/etc; "Requirement already satisfied" means pip
#     skipped. We need at least one of our requirements to live in
#     $CONTAINERROOT/user-software so make-squashfs has something to pack.
echo "[build_qwen36] pip install locations:"
for pkg in transformers accelerate math_verify huggingface_hub; do
    loc=$(pip show "$pkg" 2>/dev/null | awk '/^Location:/ {print $2}')
    echo "    $pkg -> $loc"
done

if [ ! -d "$CONTAINERROOT/user-software" ]; then
    echo "[build_qwen36] user-software/ empty — base SIF already had everything."
    echo "[build_qwen36] forcing reinstall of math_verify into the user venv to seed it"
    # math_verify is the package least likely to already be in the base SIF.
    # --force-reinstall makes pip write a fresh copy regardless of cached state.
    pip install --upgrade --force-reinstall --no-deps math_verify
fi

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
