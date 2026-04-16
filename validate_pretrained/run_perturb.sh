#!/usr/bin/env bash
# Trace perturbation: force a replacement letter at a chosen step during
# generation and re-score the model's continuation with the same metrics.
#
# How it works under the hood (see validate.py::run_validation):
#   1. Run a normal greedy generation on each val example.
#   2. Locate the token of the letter the model emitted at INTERVENE_STEP
#      (step 0 = the first letter after [TRACE]; step k = the token right
#      after the k-th ';').
#   3. Splice a new prefix = original_prompt + generated_tokens_before_letter
#      + [INTERVENE_LETTER], then let the model continue from there.
#   4. Score the resulting trace (pre-letter + forced letter + continuation).
#
# Edit INTERVENE_STEP / INTERVENE_LETTER below and re-run.

set -euo pipefail

# --- paths ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M"
TOKENIZER_PATH="${REPO_ROOT}/outputs/tokenizer/tokenizer_decision_chains_extended.json"
VAL_FILE="${REPO_ROOT}/outputs/data/decision_chains_extended/val.json"

# --- perturbation knobs -----------------------------------------------------
INTERVENE_STEP=2          # 0-indexed step at which to replace the letter
INTERVENE_LETTER="t"      # letter forced at that step (one of a..t)

# --- run knobs --------------------------------------------------------------
NUM_EXAMPLES=2048          # how many val.json examples to evaluate
BATCH_SIZE=256             # generation batch size
MAX_LENGTH=512            # model block_size; upper bound for generation
DEVICE="cuda"             # "cuda" or "cpu"
SHOW_SAMPLES=2            # print this many sample traces (0 to disable)
PERTURB_DIR="validate_pretrained/output/step${INTERVENE_STEP}_letter${INTERVENE_LETTER}"
DUMP_JSONL="${PERTURB_DIR}/res_perturb.jsonl"
                          # leave empty to skip the per-example dump

# --- run ---------------------------------------------------------------------------------------------------------
cd "${REPO_ROOT}"

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
  --intervene_step   "${INTERVENE_STEP}"
  --intervene_letter "${INTERVENE_LETTER}"
)

if [[ -n "${DUMP_JSONL}" ]]; then
  CMD+=(--dump_jsonl "${DUMP_JSONL}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
