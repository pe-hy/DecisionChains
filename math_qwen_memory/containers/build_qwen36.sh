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

# --- per-user EasyBuild prefix (in scratch — HOME has 25 GB quota) ----------

SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
export EBU_USER_PREFIX="${EBU_USER_PREFIX:-$SCRATCH/EasyBuild}"
mkdir -p "$EBU_USER_PREFIX"

# --- module name (override via PYTORCH_MODULE if a newer one ships) ---------

PYTORCH_MODULE=${PYTORCH_MODULE:-PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827}
EB_FILE=${EB_FILE:-${PYTORCH_MODULE#PyTorch/}.eb}
EB_FILE="PyTorch-${EB_FILE}"

echo "[build_qwen36] EBU_USER_PREFIX=$EBU_USER_PREFIX"
echo "[build_qwen36] target module:  $PYTORCH_MODULE"
echo "[build_qwen36] eb recipe:      $EB_FILE"
echo "[build_qwen36] requirements:   $REQS"

# Drop any inherited PyTorch / shared-CONTAINERROOT to avoid bind-leak.
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI

# --- install module privately if not already installed ----------------------

# `module is-avail` doesn't reliably distinguish private vs system modules,
# so just try `module load`. If it fails, run `eb` to install.
module load EasyBuild-user
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
