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

# Qwen3.6-27B on LUMI MI250X, single GCD (~64 GB).
# Uses the per-user PyTorch 2.7.1 module installed by containers/build_qwen36.sh
# (which adds transformers>=4.50, accelerate, math_verify on top of the
# official LUMI ROCm pytorch container as a user-software.squashfs overlay).
#
# Build the module ONCE before submitting:
#   bash containers/build_qwen36.sh
#
# Outputs: outputs/pilot/${DATASET}_..._qwen3.6-27b_shard{0..7}of8.jsonl

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

# --- private LUMI EasyBuild prefix + PyTorch module -------------------------

SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
export EBU_USER_PREFIX=${EBU_USER_PREFIX:-$SCRATCH/EasyBuild}
PYTORCH_MODULE=${PYTORCH_MODULE:-PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827}

unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI
module load EasyBuild-user
module load "$PYTORCH_MODULE"

# --- standard env -----------------------------------------------------------

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

# --- run --------------------------------------------------------------------

DATASET=${DATASET:-jarnik_boxable}
SPLIT=${SPLIT:-test}
N=${N:-27}
N_TRACES=${N_TRACES:-5}
SEED=${SEED:-0}

# 2.7.1+ module wraps `python` to run inside the container with the
# user-software.squashfs overlay attached. No explicit `singularity exec`.
srun python -u generate_traces.py \
    --model qwen3.6-27b \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --n "$N" \
    --n-traces "$N_TRACES" \
    --seed "$SEED" \
    --shard-idx "$SLURM_ARRAY_TASK_ID" \
    --n-shards 8
