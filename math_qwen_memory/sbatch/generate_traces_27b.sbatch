#!/bin/bash
#SBATCH --job-name=qwen36_27b_gen
#SBATCH --output=sbatch/logs/gen_27b_%A_%a.out
#SBATCH --error=sbatch/logs/gen_27b_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# Qwen3.6-27B on LUMI MI250X (one GCD, 64 GB).
# bf16 weights ~54 GB + KV ~5 GB (16k context, GQA) + framework ~4 GB =
# ~63 GB peak. Fits one 64 GB GCD with ~1-5 GB headroom.
# If you need >20k context output, weights+KV will spill; either:
#   - bump `--gpus=2` and switch model device_map to "auto" (TP across both
#     GCDs of the MI250X module = 128 GB aggregate)
#   - quantize fp8/int8
#
# Outputs: outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_qwen3.6-27b_shard{0..7}of8.jsonl
# Merge:
#   cat outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_qwen3.6-27b_shard*of8.jsonl \
#       > outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_qwen3.6-27b.jsonl
#
# Submit examples:
#   DATASET=jarnik_boxable N=27 sbatch sbatch/generate_traces_27b.sbatch
#   DATASET=aime2026 N=30 sbatch sbatch/generate_traces_27b.sbatch
#   DATASET=jarnik N=72 sbatch sbatch/generate_traces_27b.sbatch
#
# Prereq: Qwen3.6-27B downloaded into ./hf_cache (re-run download_qwen.sbatch
# adapted for the 27B model id, or pre-fetch with huggingface_hub).

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

DATASET=${DATASET:-jarnik_boxable}
SPLIT=${SPLIT:-test}
N=${N:-27}
N_TRACES=${N_TRACES:-5}
SEED=${SEED:-0}
# leave max_new_tokens, temperature, top_p, top_k, min_p unset →
# generate_traces.py uses Qwen3.6-27B card defaults (T=1.0, max_new=32768).

singularity exec $SIF \
    python -u generate_traces.py \
        --model qwen3.6-27b \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --n "$N" \
        --n-traces "$N_TRACES" \
        --seed "$SEED" \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 8
