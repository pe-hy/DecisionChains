#!/usr/bin/env bash
#
# Full GRPO pipeline: train GRPO → evaluate (standard + pass@k) → visualize
#
# Assumes a pretrained base model already exists.
#
# Usage:
#   ./run_grpo_pipeline.sh                           # defaults
#   ./run_grpo_pipeline.sh --iterations 100          # quick run
#   ./run_grpo_pipeline.sh --skip-train              # eval existing GRPO checkpoint
#   ./run_grpo_pipeline.sh --train-args 'grpo.lr=5e-7 grpo.group_size=32'
#

set -euo pipefail

# ── Defaults ──
ITERATIONS=200
SKIP_TRAIN=false
SKIP_INFERENCE=false
PASS_K_N_SAMPLES=256
PASS_K_TEMPERATURE=0.8
PASS_K_K_VALUES="1,2,4,8,16,32,64,128,256"
PASS_K_BATCH_SIZE=128
PASS_K_MAX_INPUTS=""
EXTRA_TRAIN_ARGS=""

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)         ITERATIONS="$2"; shift 2 ;;
        --pass-k-samples)     PASS_K_N_SAMPLES="$2"; shift 2 ;;
        --pass-k-temp)        PASS_K_TEMPERATURE="$2"; shift 2 ;;
        --pass-k-max-inputs)  PASS_K_MAX_INPUTS="$2"; shift 2 ;;
        --skip-train)         SKIP_TRAIN=true; shift ;;
        --skip-inference)     SKIP_INFERENCE=true; shift ;;
        --train-args)         EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --iterations N          GRPO iterations (default: 500)"
            echo "  --pass-k-samples N      Samples per input for pass@k (default: 256)"
            echo "  --pass-k-temp T         Sampling temperature (default: 0.8)"
            echo "  --pass-k-max-inputs N   Limit test inputs for pass@k"
            echo "  --skip-train            Skip GRPO training"
            echo "  --skip-inference        Skip standard inference"
            echo "  --train-args 'ARGS'     Extra Hydra overrides for GRPO training"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Derived paths (must match config/base_grpo.yaml) ──
GRPO_MODEL_NAME="grpo_12l-8h-512d-decision-chains-ext_6_2M"
GRPO_MODEL_PATH="outputs/temp/hf_${GRPO_MODEL_NAME}"
BASE_MODEL_PATH="outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M"
TOKENIZER_PATH="outputs/tokenizer/tokenizer_decision_chains_extended.json"
TEST_FILE="outputs/data/decision_chains_extended/val.json"

PASS_K_OUTPUT_DIR="outputs/eval_results/pass_at_k_grpo"

# ── Logging ──
log() { echo -e "\n\033[1;35m[grpo-pipeline]\033[0m $1"; }
err() { echo -e "\n\033[1;31m[grpo-pipeline ERROR]\033[0m $1" >&2; }

# ── Verify base model exists ──
if [ ! -d "$BASE_MODEL_PATH" ]; then
    err "Base model not found at: $BASE_MODEL_PATH"
    err "Run the pretraining pipeline first (./run_pipeline.sh)"
    exit 1
fi
log "Base model: $BASE_MODEL_PATH"

# ── Step 1: GRPO training ──
if [ "$SKIP_TRAIN" = false ]; then
    log "Step 1/4: GRPO training (iterations=${ITERATIONS})"
    # shellcheck disable=SC2086
    python grpo_train.py grpo.n_iterations="$ITERATIONS" $EXTRA_TRAIN_ARGS
else
    log "Step 1/4: Skipping GRPO training (--skip-train)"
fi

if [ ! -d "$GRPO_MODEL_PATH" ]; then
    err "GRPO model not found at: $GRPO_MODEL_PATH"
    exit 1
fi
log "  GRPO model: $GRPO_MODEL_PATH"

# ── Step 2: Standard inference ──
if [ "$SKIP_INFERENCE" = false ]; then
    log "Step 2/4: Running standard inference on GRPO model"
    python scripts/inference_decision_chains_extended.py \
        inference.modelpath="$GRPO_MODEL_PATH"
else
    log "Step 2/4: Skipping standard inference (--skip-inference)"
fi

# ── Step 3: Pass@k evaluation ──
log "Step 3/4: Pass@k evaluation (n_samples=${PASS_K_N_SAMPLES}, T=${PASS_K_TEMPERATURE})"

PASS_K_CMD="python scripts/eval_pass_at_k.py \
    --model_path $GRPO_MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --test_file $TEST_FILE \
    --k_values $PASS_K_K_VALUES \
    --n_samples $PASS_K_N_SAMPLES \
    --temperature $PASS_K_TEMPERATURE \
    --batch_size $PASS_K_BATCH_SIZE \
    --output_dir $PASS_K_OUTPUT_DIR"

if [ -n "$PASS_K_MAX_INPUTS" ]; then
    PASS_K_CMD="$PASS_K_CMD --max_inputs $PASS_K_MAX_INPUTS"
fi

eval "$PASS_K_CMD"

# ── Step 4: Plot pass@k (compare base vs GRPO if both exist) ──
GRPO_JSON="${PASS_K_OUTPUT_DIR}/pass_at_k_T${PASS_K_TEMPERATURE}_n${PASS_K_N_SAMPLES}.json"
BASE_JSON="outputs/eval_results/pass_at_k/pass_at_k_T${PASS_K_TEMPERATURE}_n${PASS_K_N_SAMPLES}.json"
PLOT_PATH="${PASS_K_OUTPUT_DIR}/pass_at_k.png"

if [ -f "$GRPO_JSON" ]; then
    if [ -f "$BASE_JSON" ]; then
        log "Step 4/4: Plotting pass@k comparison (base vs GRPO)"
        python visualization/plot_pass_at_k.py \
            --input "$BASE_JSON" "$GRPO_JSON" \
            --labels "Base (pretrained)" "GRPO" \
            --output "$PLOT_PATH" \
            --title "Base vs GRPO pass@k"
    else
        log "Step 4/4: Plotting pass@k (GRPO only; run base pipeline for comparison)"
        python visualization/plot_pass_at_k.py \
            --input "$GRPO_JSON" \
            --output "$PLOT_PATH"
    fi
    log "Done! Results:"
    log "  GRPO model:   $GRPO_MODEL_PATH"
    log "  Pass@k JSON:  $GRPO_JSON"
    log "  Pass@k plot:  $PLOT_PATH"
else
    err "Pass@k JSON not found at $GRPO_JSON"
    exit 1
fi
