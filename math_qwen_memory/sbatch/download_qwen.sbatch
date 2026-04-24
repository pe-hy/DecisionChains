#!/bin/bash
#SBATCH --job-name=dl_qwen
#SBATCH --output=logs/dl_qwen_%j.out
#SBATCH --error=logs/dl_qwen_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# One-shot download of Qwen3-8B into ./hf_cache via singularity python.
# Run ONCE before submitting sbatch/inject.sbatch:
#   sbatch sbatch/download_qwen.sbatch

mkdir -p logs hf_cache
export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache

singularity exec $SIF python -u - <<'PY'
import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen3-8B", cache_dir="./hf_cache")
print(f"downloaded: {p}", flush=True)
PY
