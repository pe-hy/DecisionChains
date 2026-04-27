#!/bin/bash
#SBATCH --job-name=dl_qwen36_27b
#SBATCH --output=sbatch/logs/dl_qwen36_27b_%j.out
#SBATCH --error=sbatch/logs/dl_qwen36_27b_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# One-shot download of Qwen3.6-27B into ./hf_cache (~56 GB bf16 weights).
# Run ONCE before submitting sbatch/generate_traces_27b.sbatch.

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs hf_cache
export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache

singularity exec $SIF python -u - <<'PY'
import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen3.6-27B", cache_dir="./hf_cache")
print(f"downloaded: {p}", flush=True)
PY
