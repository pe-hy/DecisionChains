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
# Uses the per-user PyTorch container module installed by
#   bash containers/build_qwen36.sh
# (which baked transformers>=4.50, accelerate, math_verify into a
# user-software.squashfs overlay on top of the LUMI ROCm pytorch container).
#
# Outputs: outputs/pilot/${DATASET}_..._qwen3.6-27b_shard{0..7}of8.jsonl

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

# 1) Set EBU_USER_PREFIX BEFORE any module load (LUMI doc requirement).
#    Pin to the same private path the build script used.
export EBU_USER_PREFIX="$SLURM_SUBMIT_DIR/containers/easybuild"

# 2) Clean any inherited container env, then load LUMI + the PyTorch module.
#    Per LUMI docs: at runtime the partition/container partition is NOT needed.
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI

# 3) Pick newest module installed under the private prefix. The module may
#    live under partition/L (login-node install) or partition/G; search all.
if [ -z "${PYTORCH_MODULE:-}" ]; then
    NEWEST=$(find "$EBU_USER_PREFIX/modules/LUMI/"*/partition/*/PyTorch \
               -maxdepth 1 -name '[0-9]*.lua' 2>/dev/null \
        | xargs -r -n1 basename \
        | grep -E '^[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-singularity-[0-9]{8}\.lua$' \
        | sort -ru | head -1 || true)
    if [ -z "$NEWEST" ]; then
        echo "ERROR: no PyTorch .lua module under $EBU_USER_PREFIX/modules/" >&2
        echo "Run: bash containers/build_qwen36.sh" >&2
        exit 1
    fi
    PYTORCH_MODULE="PyTorch/${NEWEST%.lua}"
fi

# Module was likely installed under a different partition than the runtime
# node's default. Switch into the partition that holds it before loading.
PT_PARTITION=$(find "$EBU_USER_PREFIX/modules/LUMI/" -name "${PYTORCH_MODULE#PyTorch/}.lua" 2>/dev/null \
    | head -1 \
    | sed -nE 's|.*/partition/([^/]+)/PyTorch/.*|\1|p')
if [ -n "$PT_PARTITION" ]; then
    echo "[sbatch] PyTorch module lives under partition/$PT_PARTITION; switching"
    module load "partition/$PT_PARTITION" 2>/dev/null || true
fi

module load "$PYTORCH_MODULE"

# 4) Standard env.
export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

# 5) Run. PyTorch 2.6+ module wraps `python` so it executes inside the
#    container with the user-software.squashfs overlay attached. No
#    explicit `singularity exec` needed.
DATASET=${DATASET:-jarnik_boxable}
SPLIT=${SPLIT:-test}
N=${N:-27}
N_TRACES=${N_TRACES:-5}
SEED=${SEED:-0}

srun python -u generate_traces.py \
    --model qwen3.6-27b \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --n "$N" \
    --n-traces "$N_TRACES" \
    --seed "$SEED" \
    --shard-idx "$SLURM_ARRAY_TASK_ID" \
    --n-shards 8
