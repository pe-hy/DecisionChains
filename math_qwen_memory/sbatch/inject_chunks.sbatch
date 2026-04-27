#!/bin/bash
#SBATCH --job-name=qwen_inject_chunks
#SBATCH --output=sbatch/logs/inject_chunks_%A_%a.out
#SBATCH --error=sbatch/logs/inject_chunks_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# 8-way shard chunk injection: ~587 (problem, chunk) tasks where
# unification_status in {grouped, unmatched}.
# Outputs: outputs/pilot/aime2026_chunk_injections_shard{0..7}of8.jsonl
# Merge:
#   cat outputs/pilot/aime2026_chunk_injections_shard*of8.jsonl \
#       > outputs/pilot/aime2026_chunk_injections.jsonl
#
# Prereq: sbatch sbatch/download_qwen.sbatch (populates ./hf_cache)

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

REWRITES_DIR=${REWRITES_DIR:-outputs/pilot/chunk_rewrites_unified}
BASELINE=${BASELINE:-outputs/pilot/aime2026_test_n30.jsonl}
MAX_NEW=${MAX_NEW:-16384}

singularity exec $SIF \
    python -u inject_chunks.py \
        --rewrites-dir "$REWRITES_DIR" \
        --baseline "$BASELINE" \
        --max-new-tokens "$MAX_NEW" \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
