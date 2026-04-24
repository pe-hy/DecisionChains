#!/bin/bash
# Pre-download Qwen3-8B weights into ./hf_cache on LUMI login node.
# Run this ONCE on the login node before submitting sbatch jobs:
#   bash sbatch/download_qwen.sh
# Compute nodes may not have internet.

set -e

cd "$(dirname "$0")/.."

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
mkdir -p hf_cache

python - <<'PY'
import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache"
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    cache_dir="./hf_cache",
)
print(f"Downloaded to: {path}")
PY

echo "Done. Weights in ./hf_cache/"
