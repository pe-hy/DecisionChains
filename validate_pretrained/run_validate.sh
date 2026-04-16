#!/usr/bin/env bash
# Baseline validation of a pretrained decision-chain model on val.json.
# Edit the variables below and re-run. Everything here maps 1:1 onto the
# CLI flags of validate.py.

set -euo pipefail

# --- paths ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M"
TOKENIZER_PATH="${REPO_ROOT}/outputs/tokenizer/tokenizer_decision_chains_extended.json"
VAL_FILE="${REPO_ROOT}/outputs/data/decision_chains_extended/val.json"

# --- run knobs --------------------------------------------------------------
NUM_EXAMPLES=2048          # how many val.json examples to evaluate
BATCH_SIZE=256             # generation batch size
MAX_LENGTH=512            # model block_size; upper bound for generation
DEVICE="cuda"             # "cuda" or "cpu"
SHOW_SAMPLES=2            # print this many sample traces (0 to disable)
DUMP_JSONL="validate_pretrained/output/res.jsonl"
                          # leave empty to skip the per-example dump

# --- run --------------------------------------------------------------------
cd "${REPO_ROOT}"

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
)

if [[ -n "${DUMP_JSONL}" ]]; then
  CMD+=(--dump_jsonl "${DUMP_JSONL}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
