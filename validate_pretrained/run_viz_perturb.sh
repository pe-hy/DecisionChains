#!/usr/bin/env bash
# Visualize baseline + perturbed validation results side by side.
#
# The perturbed results are read from output/step{S}_letter{L}/res_perturb.jsonl.
# Edit INTERVENE_STEP / INTERVENE_LETTER to match the perturbation you ran.
#
# Produces:
#   output/step{S}_letter{L}/viz_res_perturb_with_perturb.pdf
#   output/step{S}_letter{L}/viz_res_perturb_with_perturb_comparison.pdf

set -euo pipefail

# --- paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${SCRIPT_DIR}/output/res.jsonl"

# --- perturbation knobs (must match run_perturb.sh) -------------------------
INTERVENE_STEP=1
INTERVENE_LETTER="t"

PERTURB_DIR="${SCRIPT_DIR}/output/step${INTERVENE_STEP}_letter${INTERVENE_LETTER}"
PERTURB="${PERTURB_DIR}/res_perturb.jsonl"

# --- knobs ------------------------------------------------------------------
LABEL=""                    # custom title (default: auto-derived)
OUTPUT=""                   # custom output path (default: auto in PERTURB_DIR)

# --- sanity check -----------------------------------------------------------
if [[ ! -f "${PERTURB}" ]]; then
  echo "ERROR: ${PERTURB} not found. Run run_perturb.sh first." >&2
  exit 1
fi

# --- run --------------------------------------------------------------------
cd "${SCRIPT_DIR}/.."

CMD=(
  python validate_pretrained/visualize_results.py
  --baseline "${BASELINE}"
  --perturb  "${PERTURB}"
)

if [[ -n "${LABEL}" ]]; then
  CMD+=(--label "${LABEL}")
fi
if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
