#!/bin/bash
#SBATCH --job-name=gemma4_26b_gen
#SBATCH --output=sbatch/logs/_raw/gemma4-26b_%A_%a.out
#SBATCH --error=sbatch/logs/_raw/gemma4-26b_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --account=project_465002631
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=0-7

# Gemma 4 26B-A4B-it (MoE, 4B active) on LUMI MI250X, single GCD (~64 GB).
# bf16 weights ~52 GB → leaves ~10 GB for KV cache + activations.
# Reuses the same private PyTorch container as Qwen3.6-27B / Gemma 3 27B.
# Requires transformers >= 5.5 for `gemma4` model type.
#
# Outputs: outputs/pilot/${DATASET}_..._gemma-4-26b_shard{0..7}of8.jsonl
#
# Submit examples:
#   # AIME (default)
#   DATASET=aime2026 N=30 N_TRACES=5 sbatch sbatch/generate_traces_gemma4_26b.sbatch
#   # Jarnik (boxable)
#   DATASET=jarnik_boxable N=27 N_TRACES=5 sbatch sbatch/generate_traces_gemma4_26b.sbatch
#
# Prereq: sbatch sbatch/download_gemma4_26b.sbatch  (populates ./hf_cache)

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/_raw

# 0) Redirect stdout/stderr into a structured per-model/per-dataset folder.
DATASET=${DATASET:-aime2026}
LOG_DIR="sbatch/logs/gemma-4-26b/${DATASET}"
mkdir -p "$LOG_DIR"
LOG_BASE="$LOG_DIR/shard${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}"
exec >"${LOG_BASE}.out" 2>"${LOG_BASE}.err"
echo "[sbatch] log -> $LOG_BASE.{out,err}"

# 1) Set EBU_USER_PREFIX BEFORE any module load (LUMI doc requirement).
export EBU_USER_PREFIX="$SLURM_SUBMIT_DIR/containers/easybuild"

# 2) Clean any inherited container env, then load LUMI + the PyTorch module.
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI

# 3) Pick newest module installed under the private prefix.
if [ -z "${PYTORCH_MODULE:-}" ]; then
    NEWEST=$(find "$EBU_USER_PREFIX/modules" -name '[0-9]*.lua' 2>/dev/null \
        | grep -E '/PyTorch/[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-singularity-[0-9]{8}\.lua$' \
        | xargs -r -n1 basename \
        | sort -ru | head -1 || true)
    if [ -z "$NEWEST" ]; then
        echo "ERROR: no PyTorch .lua module under $EBU_USER_PREFIX/modules/" >&2
        echo "Run: bash containers/build_qwen36.sh" >&2
        exit 1
    fi
    PYTORCH_MODULE="PyTorch/${NEWEST%.lua}"
fi
echo "[sbatch] using $PYTORCH_MODULE"

module load "$PYTORCH_MODULE"

# 4) Standard env.
export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
# Mirror Gemma-3 setup: skip the chat_template.jinja probe that triggers
# 401s on gated repos when no token is in env.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

# 5) Run.
SPLIT=${SPLIT:-test}
N=${N:-30}
N_TRACES=${N_TRACES:-5}
SEED=${SEED:-0}

srun python -u generate_traces.py \
    --model gemma-4-26b \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --n "$N" \
    --n-traces "$N_TRACES" \
    --seed "$SEED" \
    --shard-idx "$SLURM_ARRAY_TASK_ID" \
    --n-shards 8
