#!/usr/bin/env bash
# Build the Qwen3.6 extended SIF on LUMI.
#
# Two-file workflow:
#   1. containers/qwen36.def    — apptainer definition (Bootstrap: localimage)
#   2. containers/build_qwen36.sh (this file) — resolves base image path,
#      substitutes it into the .def, runs `apptainer build --fakeroot`.
#
# Usage:
#   bash containers/build_qwen36.sh
#   # or override paths/version:
#   BASE_SIF=/appl/local/containers/sif-images/<name>.sif \
#   OUT_SIF=$SCRATCH/sif/qwen36.sif \
#       bash containers/build_qwen36.sh
#
# After build, point your sbatch at the new SIF:
#   export QWEN36_SIF=$OUT_SIF
#   sbatch sbatch/generate_traces_27b.sbatch

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEF_TEMPLATE="$HERE/qwen36.def"

# --- locate base image -------------------------------------------------------

# Default: pick the most recent LUMI ROCm PyTorch sif (python 3.12, torch 2.7).
# Override BASE_SIF env var to force a different one.
SIF_DIR=/appl/local/containers/sif-images
DEFAULT_GLOB="$SIF_DIR/lumi-pytorch-rocm-*-python-3.12-pytorch-v2.7.0*.sif"

if [ -z "${BASE_SIF:-}" ]; then
    # newest matching path
    BASE_SIF=$(ls -1 $DEFAULT_GLOB 2>/dev/null | sort -r | head -1 || true)
fi

if [ -z "${BASE_SIF:-}" ] || [ ! -f "$BASE_SIF" ]; then
    echo "ERROR: could not locate LUMI ROCm PyTorch base sif." >&2
    echo "Looked under: $DEFAULT_GLOB" >&2
    echo "Available LUMI containers:" >&2
    ls -la $SIF_DIR 2>&1 | sed 's/^/  /' >&2
    echo >&2
    echo "Set BASE_SIF=<path> and re-run." >&2
    exit 1
fi

echo "[build_qwen36] base image: $BASE_SIF"

# --- output path -------------------------------------------------------------

# LUMI HOME has 25 GB quota — write SIF to scratch.
SCRATCH=${SCRATCH:-/pfs/lustrep4/scratch/project_465002631/Petr}
OUT_SIF=${OUT_SIF:-$SCRATCH/sif/qwen36.sif}
mkdir -p "$(dirname "$OUT_SIF")"

echo "[build_qwen36] output: $OUT_SIF"

# --- materialise the .def with substituted base path ------------------------

WORK_DEF="$(mktemp -t qwen36-XXXXXX.def)"
trap 'rm -f "$WORK_DEF"' EXIT

# escape forward slashes in BASE_SIF for sed
ESCAPED_BASE=$(printf '%s\n' "$BASE_SIF" | sed 's/[\/&]/\\&/g')
sed "s/{{BASE_SIF}}/$ESCAPED_BASE/" "$DEF_TEMPLATE" > "$WORK_DEF"

echo "[build_qwen36] materialised .def → $WORK_DEF"
echo "[build_qwen36] head of resolved def:"
head -3 "$WORK_DEF" | sed 's/^/    /'

# --- build -------------------------------------------------------------------

# `apptainer` is the same binary as `singularity` on LUMI; either name works.
BUILDER=$(command -v apptainer || command -v singularity || true)
if [ -z "$BUILDER" ]; then
    echo "ERROR: neither apptainer nor singularity found in PATH" >&2
    exit 1
fi

echo "[build_qwen36] using builder: $BUILDER"
echo "[build_qwen36] running build (this takes a few minutes)..."

# --fakeroot is required on LUMI login nodes (no real root). If --fakeroot
# is unavailable in your environment, fall back to building on a node where
# you have it, or use `cotainr build` instead (LUMI's recommended path).
$BUILDER build --fakeroot --force "$OUT_SIF" "$WORK_DEF"

echo "[build_qwen36] DONE: $OUT_SIF"
echo
echo "Next step:"
echo "  export QWEN36_SIF=$OUT_SIF"
echo "  sbatch sbatch/generate_traces_27b.sbatch"
