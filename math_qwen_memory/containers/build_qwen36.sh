#!/usr/bin/env bash
# Build a private LUMI PyTorch container with extra pip packages.
#
# LUMI-canonical workflow per
#   https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/
#   https://docs.lumi-supercomputer.eu/software/installing/
#
# 1. Set EBU_USER_PREFIX so EasyBuild installs into YOUR scratch (private).
# 2. Install LUMI's EasyBuild PyTorch 2.7.1 module via `eb`. It uses the
#    official ROCm pytorch SIF and a writable per-user user-software venv.
# 3. Load the module → CONTAINERROOT is now your private dir.
# 4. `pip install -r qwen36_requirements.txt` (host shell — 2.7.1+ wraps pip
#    so it routes into the container venv automatically).
# 5. `make-squashfs` → packs the venv into user-software.squashfs.
# 6. Reload module → squashfs is bound; venv dir can be deleted.
#
# Two files:
#   containers/qwen36_requirements.txt  (definition)
#   containers/build_qwen36.sh          (this script)
#
# Run on LUMI login node:
#   bash containers/build_qwen36.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REQS="$HERE/qwen36_requirements.txt"

# --- per-user EasyBuild prefix: inside this script's own containers/ dir ----
#
# The shell's inherited EBU_USER_PREFIX usually points at a project-shared
# dir (e.g. /project/project_465002050/PH/EASYBUILD); using it would make
# `eb` install into shared state. We force a path under THIS project, next
# to the script itself, so everything (modules + SIF + squashfs) is
# self-contained and removable with `rm -rf containers/easybuild`.

PRIVATE_EBU_PREFIX="$HERE/easybuild"

if [ -n "${EBU_USER_PREFIX:-}" ] && [ "$EBU_USER_PREFIX" != "$PRIVATE_EBU_PREFIX" ]; then
    echo "[build_qwen36] WARNING: inherited EBU_USER_PREFIX=$EBU_USER_PREFIX"
    echo "[build_qwen36] overriding to project-local: $PRIVATE_EBU_PREFIX"
fi
export EBU_USER_PREFIX="$PRIVATE_EBU_PREFIX"
mkdir -p "$EBU_USER_PREFIX"

# --- bring up LUMI module system + EasyBuild-user (needed for `eb`) --------

unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI
module load EasyBuild-user

# --- discover newest available PyTorch recipe via `eb --search` -----------

if [ -z "${PYTORCH_MODULE:-}" ]; then
    echo "[build_qwen36] searching EasyBuild for PyTorch+rocm+singularity recipes..."
    # `eb --search PyTorch` prints all recipes. Filter to mainline (not vllm,
    # not exampleVenv) rocm+python+singularity ones, pick newest by date suffix.
    NEWEST_EB=$(eb --search PyTorch 2>&1 \
        | grep -oE '/\S+\.eb' \
        | grep -E 'PyTorch-[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-python-.*-singularity-[0-9]{8}\.eb$' \
        | sort -r | head -1 || true)
    if [ -z "$NEWEST_EB" ]; then
        echo "ERROR: no matching PyTorch recipe via eb --search." >&2
        echo "Run manually:  eb --search PyTorch | grep singularity" >&2
        echo "Then:  PYTORCH_MODULE=PyTorch/<ver> bash $0" >&2
        exit 1
    fi
    EB_FILE=$(basename "$NEWEST_EB")
    EB_BASE=${EB_FILE%.eb}                       # PyTorch-X.Y.Z-rocm-...
    MODULE_VER=${EB_BASE#PyTorch-}               # X.Y.Z-rocm-...
    PYTORCH_MODULE="PyTorch/$MODULE_VER"
else
    EB_FILE="PyTorch-${PYTORCH_MODULE#PyTorch/}.eb"
fi

echo "[build_qwen36] EBU_USER_PREFIX=$EBU_USER_PREFIX"
echo "[build_qwen36] target module:  $PYTORCH_MODULE"
echo "[build_qwen36] eb recipe:      $EB_FILE"
echo "[build_qwen36] requirements:   $REQS"

# --- install module privately if not already loaded -------------------------

# Try direct load first; if not yet installed under EBU_USER_PREFIX, run `eb`.
if ! module load "$PYTORCH_MODULE" 2>/dev/null; then
    echo "[build_qwen36] $PYTORCH_MODULE not yet installed. Running EasyBuild..."
    eb "$EB_FILE" -r
    module load "$PYTORCH_MODULE"
fi

echo "[build_qwen36] loaded $PYTORCH_MODULE"
echo "[build_qwen36] CONTAINERROOT=$CONTAINERROOT  (private; module-managed)"
echo "[build_qwen36] SIF=$SIF"

# --- post-load sanity --------------------------------------------------------

# 2.7.1 puts python/pip on host PATH directly.
python -c "
import torch
assert torch.version.hip is not None, 'base torch is not ROCm — wrong module?'
print('base torch:', torch.__version__, 'hip:', torch.version.hip)
"

# --- install our pip packages ------------------------------------------------

# pip install on host shell routes into $CONTAINERROOT/user-software/venv
# (module wrapper). Torch is already provided by the SIF and the resolver
# treats it as satisfied — no CUDA wheel pulled.
pip install --upgrade -r "$REQS"

# --- post-install assertions -------------------------------------------------

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

# --- bake overlay into squashfs (Lustre-friendly) ---------------------------

make-squashfs

# After make-squashfs, the user-software dir is redundant (the squashfs is
# bound on next module load). Removing it prevents accidental shadow installs.
rm -rf "$CONTAINERROOT/user-software"

# Reload so the squashfs is mounted.
module unload "$PYTORCH_MODULE"
module load   "$PYTORCH_MODULE"

# Final import check (this time against the squashfs-mounted overlay).
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
echo "[build_qwen36] back this up before 'eb' re-installs: cp it to /project/<id>/"
echo
echo "Run with:  sbatch sbatch/generate_traces_27b.sbatch"
