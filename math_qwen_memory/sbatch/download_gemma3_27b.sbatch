#!/bin/bash
#SBATCH --job-name=dl_gemma3_27b
#SBATCH --output=sbatch/logs/dl_gemma3_27b_%j.out
#SBATCH --error=sbatch/logs/dl_gemma3_27b_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# One-shot download of google/gemma-3-27b-it into ./hf_cache (~54 GB bf16).
# Gemma 3 is GATED — you must:
#   1. Accept the license at https://huggingface.co/google/gemma-3-27b-it
#   2. Provide a token via:  HF_TOKEN=hf_... sbatch sbatch/download_gemma3_27b.sbatch
#
# Run ONCE before submitting sbatch/generate_traces_gemma_27b.sbatch.

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs hf_cache
export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "Get a token at https://huggingface.co/settings/tokens after accepting" >&2
    echo "the gemma-3 license, then resubmit with:" >&2
    echo "  HF_TOKEN=hf_... sbatch sbatch/download_gemma3_27b.sbatch" >&2
    exit 1
fi

singularity exec $SIF python -u - <<PY
import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache"
os.environ["HF_TOKEN"] = "${HF_TOKEN}"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="google/gemma-3-27b-it", cache_dir="./hf_cache",
                      token="${HF_TOKEN}")
print(f"downloaded: {p}", flush=True)
PY
