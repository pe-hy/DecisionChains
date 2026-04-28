#!/bin/bash
#SBATCH --job-name=qwen_gen_traces
#SBATCH --output=sbatch/logs/_raw/qwen3-8b_%A_%a.out
#SBATCH --error=sbatch/logs/_raw/qwen3-8b_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# 8-way shard generation: each array task processes problems where
# (i % 8 == SLURM_ARRAY_TASK_ID), all n_traces per problem on its shard.
# Outputs: outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_shard{0..7}of8.jsonl
# Merge:
#   cat outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_shard*of8.jsonl \
#       > outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}.jsonl
#
# Submit examples:
#   sbatch sbatch/generate_traces.sbatch
#   DATASET=algebra sbatch sbatch/generate_traces.sbatch
#   DATASET=aime2026 N=30 MAX_NEW=12288 sbatch sbatch/generate_traces.sbatch
#
# Prereq: sbatch sbatch/download_qwen.sbatch (populates ./hf_cache)

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/_raw

DATASET=${DATASET:-math}
LOG_DIR="sbatch/logs/qwen3-8b/${DATASET}"
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
SPLIT=${SPLIT:-test}
N=${N:-500}
N_TRACES=${N_TRACES:-5}
MAX_NEW=${MAX_NEW:-8192}
SEED=${SEED:-0}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}

singularity exec $SIF \
    python -u generate_traces.py \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --n "$N" \
        --n-traces "$N_TRACES" \
        --max_new_tokens "$MAX_NEW" \
        --temperature "$TEMP" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --seed "$SEED" \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
