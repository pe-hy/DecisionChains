#!/usr/bin/env bash
# All-in-one: baseline validation → sweep of single-letter perturbations
# across (step, letter) pairs → visualizations.
#
# The sweep is ordered by step first, then letter: for each step S in
# INTERVENE_STEPS, run every letter in INTERVENE_LETTERS, so you get e.g.
#   step 0: b, d, t, j
#   step 1: b, d, t, j
#   step 2: b, d, t, j
# Each run is a single perturbation (one forced letter at one step).
#
# Results are saved under validate_pretrained/output/:
#   output/res.jsonl                             — baseline data
#   output/viz_res.pdf                           — baseline visualisation
#   output/step{S}_letter{L}/res_perturb.jsonl   — perturbed data
#   output/step{S}_letter{L}/viz_*.pdf           — perturbed visualisations

set -euo pipefail

# --- paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
MODEL_PATH="${REPO_ROOT}/outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M"
TOKENIZER_PATH="${REPO_ROOT}/outputs/tokenizer/tokenizer_decision_chains_extended.json"
VAL_FILE="${REPO_ROOT}/outputs/data/decision_chains_extended/val.json"

# --- perturbation sweep -----------------------------------------------------
INTERVENE_STEPS=(0 1 2)             # 0-indexed positions to force a letter at
INTERVENE_LETTERS=(b d t j)         # letters forced (one of a..t)

# --- run knobs --------------------------------------------------------------
NUM_EXAMPLES=2048
BATCH_SIZE=256
MAX_LENGTH=512
DEVICE="cuda"
SHOW_SAMPLES=0                      # sample traces (0 = off)

# --- derived paths ----------------------------------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/output"
BASELINE_JSONL="${OUTPUT_DIR}/res.jsonl"

TOTAL=$(( ${#INTERVENE_STEPS[@]} * ${#INTERVENE_LETTERS[@]} ))
# Steps: 1 baseline run + TOTAL perturbed runs + 1 baseline viz + TOTAL perturb viz.
TOTAL_STAGES=$(( 2 + 2 * TOTAL ))
STAGE=0

_banner() {
  STAGE=$(( STAGE + 1 ))
  echo ""
  echo "================================================================"
  echo "[${STAGE}/${TOTAL_STAGES}] $1"
  echo "================================================================"
}

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

# ===========================================================================
# 1. Baseline validation
# ===========================================================================
_banner "Baseline validation"

CMD=(
  python validate_pretrained/validate.py
  --model_path      "${MODEL_PATH}"
  --tokenizer_path  "${TOKENIZER_PATH}"
  --val_file        "${VAL_FILE}"
  --num_examples    "${NUM_EXAMPLES}"
  --batch_size      "${BATCH_SIZE}"
  --max_length      "${MAX_LENGTH}"
  --device          "${DEVICE}"
  --show_samples    "${SHOW_SAMPLES}"
  --dump_jsonl      "${BASELINE_JSONL}"
)
echo "+ ${CMD[*]}"
"${CMD[@]}"

# ===========================================================================
# 2. Perturbed validation sweep (step-major: step 0 × all letters, step 1 × …)
# ===========================================================================
for S in "${INTERVENE_STEPS[@]}"; do
  for L in "${INTERVENE_LETTERS[@]}"; do
    _banner "Perturbed validation  step=${S}  letter=${L}"

    PERTURB_DIR="${OUTPUT_DIR}/step${S}_letter${L}"
    PERTURB_JSONL="${PERTURB_DIR}/res_perturb.jsonl"
    mkdir -p "${PERTURB_DIR}"

    CMD=(
      python validate_pretrained/validate.py
      --model_path       "${MODEL_PATH}"
      --tokenizer_path   "${TOKENIZER_PATH}"
      --val_file         "${VAL_FILE}"
      --num_examples     "${NUM_EXAMPLES}"
      --batch_size       "${BATCH_SIZE}"
      --max_length       "${MAX_LENGTH}"
      --device           "${DEVICE}"
      --show_samples     "${SHOW_SAMPLES}"
      --intervene_step   "${S}"
      --intervene_letter "${L}"
      --dump_jsonl       "${PERTURB_JSONL}"
    )
    echo "+ ${CMD[*]}"
    "${CMD[@]}"
  done
done

# ===========================================================================
# 3. Baseline visualization
# ===========================================================================
_banner "Baseline visualization"

CMD=(
  python validate_pretrained/visualize_results.py
  --baseline "${BASELINE_JSONL}"
)
echo "+ ${CMD[*]}"
"${CMD[@]}"

# ===========================================================================
# 4. Perturbed visualizations (one PDF per (step, letter))
# ===========================================================================
for S in "${INTERVENE_STEPS[@]}"; do
  for L in "${INTERVENE_LETTERS[@]}"; do
    _banner "Perturbed visualization  step=${S}  letter=${L}"

    PERTURB_DIR="${OUTPUT_DIR}/step${S}_letter${L}"
    PERTURB_JSONL="${PERTURB_DIR}/res_perturb.jsonl"

    CMD=(
      python validate_pretrained/visualize_results.py
      --baseline "${BASELINE_JSONL}"
      --perturb  "${PERTURB_JSONL}"
    )
    echo "+ ${CMD[*]}"
    "${CMD[@]}"
  done
done

echo ""
echo "================================================================"
echo "Done. Sweep covered ${TOTAL} (step, letter) pairs."
echo "  baseline:  ${BASELINE_JSONL}"
echo "  perturbed: ${OUTPUT_DIR}/step{${INTERVENE_STEPS[*]// /,}}_letter{${INTERVENE_LETTERS[*]// /,}}/"
echo "================================================================"
