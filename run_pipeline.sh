#!/usr/bin/env bash
#
# Full pipeline: generate data → train → evaluate (standard + pass@k) → visualize
#
# Usage:
#   ./run_pipeline.sh                    # defaults: 1M train, 50 epochs, 256 pass@k samples
#   ./run_pipeline.sh --epochs 10        # quick run
#   ./run_pipeline.sh --skip-datagen     # reuse existing data
#   ./run_pipeline.sh --skip-train       # evaluate existing checkpoint
#
# All outputs go under outputs/. Key results:
#   outputs/eval_results/pass_at_k/pass_at_k_T{T}_n{N}.json
#   outputs/eval_results/pass_at_k/pass_at_k.png
#

set -euo pipefail

# ── Defaults ──
TARGET_TRAIN=2000000
TARGET_TEST=10000
EPOCHS=50
PASS_K_N_SAMPLES=256
PASS_K_TEMPERATURE=0.8
PASS_K_K_VALUES="1,2,4,8,16,32,64,128,256"
PASS_K_BATCH_SIZE=128
PASS_K_MAX_INPUTS=""  # empty = all
SKIP_DATAGEN=false
SKIP_TRAIN=false
SKIP_INFERENCE=false
EXTRA_TRAIN_ARGS=""

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --target-train)   TARGET_TRAIN="$2"; shift 2 ;;
        --target-test)    TARGET_TEST="$2"; shift 2 ;;
        --pass-k-samples) PASS_K_N_SAMPLES="$2"; shift 2 ;;
        --pass-k-temp)    PASS_K_TEMPERATURE="$2"; shift 2 ;;
        --pass-k-values)  PASS_K_K_VALUES="$2"; shift 2 ;;
        --pass-k-batch)   PASS_K_BATCH_SIZE="$2"; shift 2 ;;
        --pass-k-max-inputs) PASS_K_MAX_INPUTS="$2"; shift 2 ;;
        --skip-datagen)   SKIP_DATAGEN=true; shift ;;
        --skip-train)     SKIP_TRAIN=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --train-args)     EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --epochs N             Training epochs (default: 50)"
            echo "  --target-train N       Training examples (default: 1000000)"
            echo "  --target-test N        Test examples (default: 10000)"
            echo "  --pass-k-samples N     Samples per input for pass@k (default: 256)"
            echo "  --pass-k-temp T        Sampling temperature (default: 0.8)"
            echo "  --pass-k-values S      Comma-separated k values (default: 1,2,4,...,256)"
            echo "  --pass-k-batch N       Generation batch size (default: 128)"
            echo "  --pass-k-max-inputs N  Limit test inputs for pass@k (default: all)"
            echo "  --skip-datagen         Skip data generation"
            echo "  --skip-train           Skip training (use existing checkpoint)"
            echo "  --skip-inference       Skip standard inference"
            echo "  --train-args 'ARGS'    Extra Hydra overrides for training"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Derived paths (must match config/base_decision_chains_extended.yaml) ──
# The config resolves model.name from wandb.model_name, which uses n_layer/n_head/n_embd.
# Default: 12l-8h-256d-decision-chains-ext_6_2M
# We read these from the config defaults, but training overrides may change them.
DATA_DIR="outputs/data/decision_chains_extended"
TOKENIZER_PATH="outputs/tokenizer/tokenizer_decision_chains_extended.json"
TEST_FILE="${DATA_DIR}/val.json"

# Model name depends on config — we'll detect it after training
# Default from config:
MODEL_NAME="12l-8h-256d-decision-chains-ext_6_2M"
HF_MODEL_PATH="outputs/temp/hf_${MODEL_NAME}"

PASS_K_OUTPUT_DIR="outputs/eval_results/pass_at_k"

# ── Logging ──
log() { echo -e "\n\033[1;36m[pipeline]\033[0m $1"; }
err() { echo -e "\n\033[1;31m[pipeline ERROR]\033[0m $1" >&2; }

# ── Step 1: Generate data ──
if [ "$SKIP_DATAGEN" = false ]; then
    log "Step 1/5: Generating data (train=${TARGET_TRAIN}, test=${TARGET_TEST})"
    python scripts/generate_decision_chains_extended.py \
        --target_train "$TARGET_TRAIN" \
        --target_test "$TARGET_TEST"
else
    log "Step 1/5: Skipping data generation (--skip-datagen)"
fi

# Verify data exists
if [ ! -f "$TEST_FILE" ]; then
    err "Test file not found: $TEST_FILE"
    err "Run without --skip-datagen to generate data first."
    exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
    err "Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi
log "  Data OK: $(wc -l < /dev/null; python -c "import json; print(len(json.load(open('${TEST_FILE}'))))" 2>/dev/null || echo '?') test examples"

# ── Step 2: Train ──
if [ "$SKIP_TRAIN" = false ]; then
    log "Step 2/5: Training (epochs=${EPOCHS}) ${EXTRA_TRAIN_ARGS:+with overrides: $EXTRA_TRAIN_ARGS}"
    # shellcheck disable=SC2086
    python train_chains.py model.epochs="$EPOCHS" $EXTRA_TRAIN_ARGS
else
    log "Step 2/5: Skipping training (--skip-train)"
fi

# Verify HF checkpoint exists
if [ ! -d "$HF_MODEL_PATH" ]; then
    err "HF model not found at: $HF_MODEL_PATH"
    err "Expected the training script to produce it. Check model name."
    exit 1
fi
log "  Model OK: $HF_MODEL_PATH"

# ── Step 3: Standard inference ──
if [ "$SKIP_INFERENCE" = false ]; then
    log "Step 3/5: Running standard inference"
    python scripts/inference_decision_chains_extended.py
else
    log "Step 3/5: Skipping standard inference (--skip-inference)"
fi

# ── Step 4: Pass@k evaluation ──
log "Step 4/5: Pass@k evaluation (n_samples=${PASS_K_N_SAMPLES}, T=${PASS_K_TEMPERATURE})"

PASS_K_CMD="python scripts/eval_pass_at_k.py \
    --model_path $HF_MODEL_PATH \
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

# ── Step 5: Plot pass@k ──
PASS_K_JSON="${PASS_K_OUTPUT_DIR}/pass_at_k_T${PASS_K_TEMPERATURE}_n${PASS_K_N_SAMPLES}.json"
PASS_K_PLOT="${PASS_K_OUTPUT_DIR}/pass_at_k.png"

if [ -f "$PASS_K_JSON" ]; then
    log "Step 5/5: Plotting pass@k curves"
    python visualization/plot_pass_at_k.py \
        --input "$PASS_K_JSON" \
        --output "$PASS_K_PLOT"
    log "Done! Results:"
    log "  Pass@k JSON: $PASS_K_JSON"
    log "  Pass@k plot: $PASS_K_PLOT"
else
    err "Pass@k JSON not found at $PASS_K_JSON — plotting skipped"
    exit 1
fi
