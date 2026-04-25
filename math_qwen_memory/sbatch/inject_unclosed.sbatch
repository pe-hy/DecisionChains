#!/bin/bash
#SBATCH --job-name=inject_unclosed
#SBATCH --output=logs/inject_uncl_%A_%a.out
#SBATCH --error=logs/inject_uncl_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# Run injection ONLY on the 93 unclosed-think spans (problems 2,8,9,13,16,17,
# 22,23,26,27,28 — baseline truncated mid-thinking).
# Output: outputs/pilot/aime2026_injections_unclosed_shard{0..7}of8.jsonl
#
# After all array tasks finish, merge with existing baseline-injection file:
#   cd math_qwen_memory && cat outputs/pilot/aime2026_injections.jsonl \
#       outputs/pilot/aime2026_injections_unclosed_shard*of8.jsonl \
#       > outputs/pilot/aime2026_injections.full.jsonl

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

ALIGNED=${ALIGNED:-outputs/pilot/aime2026_test_n30.aligned.jsonl}

singularity exec $SIF \
    python -u inject.py \
        --aligned "$ALIGNED" \
        --out outputs/pilot/aime2026_injections_unclosed.jsonl \
        --only-mode unclosed \
        --max-new-tokens 16384 \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
