#!/bin/bash
#SBATCH --job-name=qwen_gen_jarnik
#SBATCH --output=sbatch/logs/gen_jarnik_%A_%a.out
#SBATCH --error=sbatch/logs/gen_jarnik_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# Jarník 2018-2026 (j25..j33): 72 problems × 5 traces, 8-way shard.
# 32k token budget (Qwen3 model-card recommendation for math/programming
# competitions; proofs run long).
# Outputs: outputs/pilot/jarnik_test_n72_t5_shard{0..7}of8.jsonl
# Merge:
#   cat outputs/pilot/jarnik_test_n72_t5_shard*of8.jsonl \
#       > outputs/pilot/jarnik_test_n72_t5.jsonl
#
# Submit: sbatch sbatch/generate_traces_jarnik.sbatch
# Override seed: SEED=1 sbatch sbatch/generate_traces_jarnik.sbatch
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

SEED=${SEED:-0}

singularity exec $SIF \
    python -u generate_traces.py \
        --dataset jarnik \
        --split test \
        --n 72 \
        --n-traces 5 \
        --max_new_tokens 16384 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 20 \
        --seed "$SEED" \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
