#!/bin/bash
#SBATCH --job-name=qwen_gen_aime
#SBATCH --output=sbatch/logs/_raw/qwen3-8b-aime_%A_%a.out
#SBATCH --error=sbatch/logs/_raw/qwen3-8b-aime_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# AIME 2026: 30 problems × 5 traces, 8-way shard, 16k token budget.
# Outputs: outputs/pilot/aime2026_test_n30_t5_shard{0..7}of8.jsonl
# Merge:
#   cat outputs/pilot/aime2026_test_n30_t5_shard*of8.jsonl \
#       > outputs/pilot/aime2026_test_n30_t5.jsonl
#
# Submit: sbatch sbatch/generate_traces_aime.sbatch
# Override seed: SEED=1 sbatch sbatch/generate_traces_aime.sbatch
#
# Prereq: sbatch sbatch/download_qwen.sbatch (populates ./hf_cache)

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/_raw

LOG_DIR="sbatch/logs/qwen3-8b/aime2026"
mkdir -p "$LOG_DIR"
LOG_BASE="$LOG_DIR/shard${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}"
exec >"${LOG_BASE}.out" 2>"${LOG_BASE}.err"
echo "[sbatch] log -> $LOG_BASE.{out,err}"

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

SEED=${SEED:-0}

singularity exec $SIF \
    python -u generate_traces.py \
        --dataset aime2026 \
        --split test \
        --n 30 \
        --n-traces 5 \
        --max_new_tokens 16384 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 20 \
        --seed "$SEED" \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
