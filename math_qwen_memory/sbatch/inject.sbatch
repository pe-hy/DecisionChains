#!/bin/bash
#SBATCH --job-name=qwen_inject
#SBATCH --output=logs/inject_%j_%t.out
#SBATCH --error=logs/inject_%j_%t.err
#SBATCH --time=24:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

# Run thinking-block prefill + algorithm injection across all in-think spans,
# 8-way sharded on 1 node. Each task gets (task_idx % 8 == SLURM_PROCID).
# Output: outputs/pilot/aime2026_injections_shard{0..7}of8.jsonl
# Merge: cat outputs/pilot/aime2026_injections_shard*of8.jsonl > outputs/pilot/aime2026_injections.jsonl
#
# Input: --aligned path must already exist (pre-computed by Claude annotator).

mkdir -p logs

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

ALIGNED=${ALIGNED:-outputs/pilot/aime2026_test_n30.aligned.jsonl}

srun -l --kill-on-bad-exit=1 \
    singularity exec $SIF \
        python -u inject.py \
            --aligned "$ALIGNED" \
            --max-new-tokens 16384 \
            --shard-idx $SLURM_PROCID \
            --n-shards $SLURM_NTASKS
