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

# Refuse to write into a shared install path. The base SIF inside any LUMI
# PyTorch module is read-only (safe), but its overlay at
# $CONTAINERROOT/user-software is writable — pip install there mutates state
# others may depend on. Force CONTAINERROOT to per-user scratch.
SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
PROJECT_USER_ROOT="$SCRATCH/qwen36_container"

if [ -n "${CONTAINERROOT:-}" ] && [[ "$CONTAINERROOT" != "$PROJECT_USER_ROOT" ]]; then
    echo "[build_qwen36] WARNING: existing CONTAINERROOT=$CONTAINERROOT"
    echo "[build_qwen36] looks shared/inherited; overriding to a private path"
    echo "[build_qwen36] under your scratch so pip install does not mutate"
    echo "[build_qwen36] anyone else's environment."
fi
export CONTAINERROOT="$PROJECT_USER_ROOT"
mkdir -p "$CONTAINERROOT"

echo "[build_qwen36] CONTAINERROOT=$CONTAINERROOT  (private overlay path)"
echo "[build_qwen36] requirements: $REQS"
echo "[build_qwen36] (the base SIF inside the module is read-only — never modified)"

# --- module load -------------------------------------------------------------

# Auto-detect the module name. Priority:
#   1. If user already loaded a PyTorch module in their shell, skip reload.
#   2. If user passed PYTORCH_MODULE env var, use it.
#   3. If CONTAINERROOT path embeds a version (e.g. ".../PyTorch/2.5.1-rocm-…"),
#      derive PYTORCH_MODULE from it.
#   4. Hard fallback to a known-good name; if it's not on this system, the
#      `module load` will print a list of available versions and we abort
#      with a clear message.

if module list 2>&1 | grep -qi PyTorch; then
    echo "[build_qwen36] PyTorch module already loaded:"
    module list 2>&1 | grep -i PyTorch | sed 's/^/    /'
else
    if [ -z "${PYTORCH_MODULE:-}" ]; then
        # try to derive from CONTAINERROOT path
        CR_VER=$(basename "$CONTAINERROOT")
        if [[ "$CR_VER" == *rocm*python*singularity* ]]; then
            PYTORCH_MODULE="PyTorch/$CR_VER"
            echo "[build_qwen36] derived module from CONTAINERROOT: $PYTORCH_MODULE"
        else
            PYTORCH_MODULE=PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20241125
            echo "[build_qwen36] using fallback default: $PYTORCH_MODULE"
        fi
    fi

    # CSC contributed module path is the canonical home for LUMI EasyBuild
    # PyTorch. Project-local installs may live elsewhere — set
    # MODULEPATH_EXTRA to add another path if needed.
    module use /appl/local/csc/modulefiles
    [ -n "${MODULEPATH_EXTRA:-}" ] && module use "$MODULEPATH_EXTRA"

    if ! module load "$PYTORCH_MODULE" 2>/dev/null; then
        echo "ERROR: cannot load $PYTORCH_MODULE." >&2
        echo "Available PyTorch modules:" >&2
        module avail PyTorch 2>&1 | sed 's/^/  /' >&2 || true
        echo >&2
        echo "Re-run with PYTORCH_MODULE=<name>  bash containers/build_qwen36.sh" >&2
        exit 1
    fi
    echo "[build_qwen36] loaded $PYTORCH_MODULE"
fi

echo "[build_qwen36] SIF=${SIF:-(not set; module did not export it)}"

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
