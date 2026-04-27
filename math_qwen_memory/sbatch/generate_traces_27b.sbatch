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
# Uses the LUMI EasyBuild PyTorch module + the squashfs overlay built by
# containers/build_qwen36.sh (which adds transformers>=4.50, accelerate,
# math_verify on top of the ROCm pytorch base — required for Qwen3.6's
# `qwen3_5` model_type).
#
# Build the overlay ONCE before submitting (login node):
#   bash containers/build_qwen36.sh
#
# Memory budget at 16k context: ~54 GB weights + ~5 GB KV (GQA) + ~4 GB
# framework = ~63 GB peak → fits one 64 GB GCD.
#
# Outputs: outputs/pilot/${DATASET}_${SPLIT}_n${N}_t${N_TRACES}_qwen3.6-27b_shard{0..7}of8.jsonl

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs

# --- load LUMI PyTorch module + attach the overlay built earlier ------------

# CONTAINERROOT must match the per-user scratch path the build script used.
# We pin it explicitly to avoid using a shared/inherited overlay.
SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
export CONTAINERROOT="$SCRATCH/qwen36_container"

if [ ! -d "$CONTAINERROOT/user-software" ] && [ ! -f "$CONTAINERROOT/user-software.squashfs" ]; then
    echo "ERROR: $CONTAINERROOT has no overlay. Run containers/build_qwen36.sh first." >&2
    exit 1
fi

# Derive PYTORCH_MODULE from CONTAINERROOT path when not explicitly set.
if [ -z "${PYTORCH_MODULE:-}" ]; then
    CR_VER=$(basename "$CONTAINERROOT")
    if [[ "$CR_VER" == *rocm*python*singularity* ]]; then
        PYTORCH_MODULE="PyTorch/$CR_VER"
    else
        PYTORCH_MODULE=PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20241125
    fi
fi

module use /appl/local/csc/modulefiles
[ -n "${MODULEPATH_EXTRA:-}" ] && module use "$MODULEPATH_EXTRA"
module load "$PYTORCH_MODULE"

# --- standard env -----------------------------------------------------------

export HF_HOME=./hf_cache
export HF_HUB_CACHE=./hf_cache
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# --- run --------------------------------------------------------------------

DATASET=${DATASET:-jarnik_boxable}
SPLIT=${SPLIT:-test}
N=${N:-27}
N_TRACES=${N_TRACES:-5}
SEED=${SEED:-0}

# Module's wrapper makes `python` the container's python with the overlay
# bound — no explicit `singularity exec $SIF` needed.
python -u generate_traces.py \
    --model qwen3.6-27b \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --n "$N" \
    --n-traces "$N_TRACES" \
    --seed "$SEED" \
    --shard-idx $SLURM_ARRAY_TASK_ID \
    --n-shards 8
