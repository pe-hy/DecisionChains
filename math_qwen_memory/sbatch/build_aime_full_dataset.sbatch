#!/bin/bash
#SBATCH --job-name=build_aime_full
#SBATCH --output=sbatch/logs/_raw/build_aime_full_%j.out
#SBATCH --error=sbatch/logs/_raw/build_aime_full_%j.err
#SBATCH --time=00:15:00
#SBATCH --account=project_465002631
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Builds data/AIME_2021_2025/test.jsonl by merging:
#   - di-zhang-fdu/AIME_1983_2024 (CSV, filtered to year>=2016)
#   - opencompass/AIME2025         (JSONLs, 30 problems for 2025)
#   - AI-MO/aimo-validation-aime   (parquet, fills 2024-I + 2023-I-15 gaps)
#
# All three raw inputs are pre-staged under data/AIME_2021_2025/_raw/.
#
# Submit: sbatch sbatch/build_aime_full_dataset.sbatch

cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch/logs/_raw

# Hierarchical log routing.
LOG_DIR="sbatch/logs/data_prep/aime_2021_2025"
mkdir -p "$LOG_DIR"
LOG_BASE="$LOG_DIR/build_${SLURM_JOB_ID}"
exec >"${LOG_BASE}.out" 2>"${LOG_BASE}.err"
echo "[sbatch] log -> $LOG_BASE.{out,err}"

# Use the same private container as gen jobs (has pyarrow, pandas).
export EBU_USER_PREFIX="$SLURM_SUBMIT_DIR/containers/easybuild"
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true
module load LUMI

if [ -z "${PYTORCH_MODULE:-}" ]; then
    NEWEST=$(find "$EBU_USER_PREFIX/modules" -name '[0-9]*.lua' 2>/dev/null \
        | grep -E '/PyTorch/[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-singularity-[0-9]{8}\.lua$' \
        | xargs -r -n1 basename \
        | sort -ru | head -1 || true)
    if [ -z "$NEWEST" ]; then
        echo "ERROR: no PyTorch .lua module under $EBU_USER_PREFIX/modules/" >&2
        exit 1
    fi
    PYTORCH_MODULE="PyTorch/${NEWEST%.lua}"
fi
echo "[sbatch] using $PYTORCH_MODULE"
module load "$PYTORCH_MODULE"

python -u scripts/build_aime_full_dataset.py
