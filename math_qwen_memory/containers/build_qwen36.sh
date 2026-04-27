#!/usr/bin/env bash
# Build a customised LUMI PyTorch container with the project's extra deps.
#
# LUMI does NOT allow `apptainer build --fakeroot` for ordinary users
# (no /etc/subuid mapping). The supported path is:
#   1. `module load PyTorch/<version>-singularity-<date>` — exposes the SIF
#      and a writable Python venv overlay at $CONTAINERROOT/user-software.
#   2. `pip install -r containers/qwen36_requirements.txt` — adds our packages
#      into that overlay.
#   3. `make-squashfs` — bakes the overlay into a squashfs file which the
#      module wrapper auto-attaches at runtime (faster on Lustre).
#
# Two files:
#   - containers/qwen36_requirements.txt  (definition: what to install)
#   - containers/build_qwen36.sh          (this script: how to build it)
#
# Usage on LUMI login node:
#   bash containers/build_qwen36.sh
#
# Then in your sbatch you set CONTAINERROOT to the same path and module-load
# the same PyTorch version. The provided sbatch (sbatch/generate_traces_27b.sbatch)
# already does this.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REQS="$HERE/qwen36_requirements.txt"

# --- paths -------------------------------------------------------------------

# LUMI HOME has 25 GB quota — overlay goes to scratch.
SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
export CONTAINERROOT="${CONTAINERROOT:-$SCRATCH/qwen36_container}"
mkdir -p "$CONTAINERROOT"

echo "[build_qwen36] CONTAINERROOT=$CONTAINERROOT"
echo "[build_qwen36] requirements: $REQS"

# --- module load -------------------------------------------------------------

# LUMI EasyBuild PyTorch module. Override PYTORCH_MODULE if a newer one ships.
PYTORCH_MODULE=${PYTORCH_MODULE:-PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527}

# CSC contributed module path (where the LUMI EasyBuild PyTorch lives).
module use /appl/local/csc/modulefiles
module load "$PYTORCH_MODULE"

echo "[build_qwen36] loaded $PYTORCH_MODULE"
echo "[build_qwen36] SIF=$SIF"

# --- verify base torch is ROCm before touching anything ---------------------

python -c "
import torch
assert torch.version.hip is not None, 'base torch is not ROCm — wrong module?'
print('base torch:', torch.__version__, 'hip:', torch.version.hip)
"

# --- install pip packages into the user-software overlay --------------------

# `pip install` from inside the loaded module writes to
# $CONTAINERROOT/user-software/venv/pytorch (writable; the SIF stays read-only).
# No --no-deps here: the overlay can hold transitive deps safely. We avoid
# torch shadowing because torch is already provided by the SIF and pip's
# resolver sees it as satisfied.
pip install --upgrade -r "$REQS"

# --- post-install assertions: every import path that runtime needs ----------

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

# --- bake overlay into squashfs (fast Lustre access) ------------------------

# `make-squashfs` is provided by the LUMI EasyBuild PyTorch module. It packs
# $CONTAINERROOT/user-software into a SquashFS file the module wrapper
# auto-binds at runtime.
make-squashfs

echo
echo "[build_qwen36] DONE."
echo
echo "Next:"
echo "  sbatch sbatch/generate_traces_27b.sbatch"
echo
echo "The sbatch will:  module use /appl/local/csc/modulefiles"
echo "                  module load $PYTORCH_MODULE"
echo "                  export CONTAINERROOT=$CONTAINERROOT"
echo "and then run python directly (no explicit singularity exec)."
