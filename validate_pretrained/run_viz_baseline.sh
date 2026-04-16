#!/usr/bin/env bash
# Visualize baseline validation results (output/res.jsonl).
#
# Produces a 3x3 PDF panel: overall metrics, per-chain-length breakdown,
# per-step accuracy curves, length distribution, letter confusion heatmaps,
# and per-letter operation accuracy.

set -euo pipefail

# --- paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${SCRIPT_DIR}/output/res.jsonl"

# --- knobs ------------------------------------------------------------------
LABEL=""                    # custom title (default: derived from filename)
OUTPUT=""                   # custom output path (default: auto in output/)

# --- run --------------------------------------------------------------------
cd "${SCRIPT_DIR}/.."

CMD=(
  python validate_pretrained/visualize_results.py
  --baseline "${BASELINE}"
)

if [[ -n "${LABEL}" ]]; then
  CMD+=(--label "${LABEL}")
fi
if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
