#!/bin/bash
#SBATCH --job-name=dl_gemma4_26b
#SBATCH --output=sbatch/logs/_raw/dl_gemma4_26b_%j.out
#SBATCH --error=sbatch/logs/_raw/dl_gemma4_26b_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# One-shot download of google/gemma-4-26B-A4B-it into ./hf_cache (~52 GB bf16).
# Gated repo — accept license at https://huggingface.co/google/gemma-4-26B-A4B-it
# and submit with: HF_TOKEN=hf_... sbatch sbatch/download_gemma4_26b.sbatch
#
# Run ONCE before submitting sbatch/generate_traces_gemma4_26b.sbatch.

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/_raw hf_cache

LOG_DIR="sbatch/logs/gemma-4-26b/_download"
mkdir -p "$LOG_DIR"
LOG_BASE="$LOG_DIR/download_${SLURM_JOB_ID}"
exec >"${LOG_BASE}.out" 2>"${LOG_BASE}.err"
echo "[sbatch] log -> $LOG_BASE.{out,err}"

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "Get a token at https://huggingface.co/settings/tokens after accepting" >&2
    echo "the gemma-4 license, then resubmit with:" >&2
    echo "  HF_TOKEN=hf_... sbatch sbatch/download_gemma4_26b.sbatch" >&2
    exit 1
fi

# Use the same private PyTorch container that the gen jobs use, so the
# download environment matches (transformers >=5.5 needed for gemma4 type).
export EBU_USER_PREFIX="$SLURM_SUBMIT_DIR/containers/easybuild"
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI

if [ -z "${PYTORCH_MODULE:-}" ]; then
    NEWEST=$(find "$EBU_USER_PREFIX/modules" -name '[0-9]*.lua' 2>/dev/null \
        | grep -E '/PyTorch/[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-singularity-[0-9]{8}\.lua$' \
        | xargs -r -n1 basename \
        | sort -ru | head -1 || true)
    if [ -z "$NEWEST" ]; then
        echo "ERROR: no PyTorch .lua module under $EBU_USER_PREFIX/modules/" >&2
        exit 1
    fi
    PYTORCH_MODULE="PyTorch/${NEWEST%.lua}"
fi
echo "[sbatch] using $PYTORCH_MODULE"
module load "$PYTORCH_MODULE"

python -u - <<PY
import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache"
os.environ["HF_TOKEN"] = "${HF_TOKEN}"
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="google/gemma-4-26B-A4B-it",
    cache_dir="./hf_cache",
    token="${HF_TOKEN}",
    # Gemma 4 ships flax + onnx + tf weights too — skip them, we only need PT.
    allow_patterns=[
        "*.json",
        "*.txt",
        "*.md",
        "tokenizer.model",
        "*.safetensors",
        "*.safetensors.index.json",
    ],
)
print(f"downloaded: {p}", flush=True)
PY
