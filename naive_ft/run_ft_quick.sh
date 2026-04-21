#!/usr/bin/env bash
#
# run_ft_quick.sh — 4-run minimum pass: one per variant.
# Matches the memory winner's compute budget (n_train=3000, epochs=10).
# Each run also logs to W&B under project "memory-experiment" with variant
# group, so cross-folder comparison is straightforward.

set +e
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=logs
mkdir -p logs
cd "$(dirname "$0")"

run() {
    local NAME=$1; shift
    if [ -f "outputs/experiments/$NAME.json" ]; then
        echo ">>> SKIP $NAME  (already done)"
        return 0
    fi
    echo "==== START $NAME ===="
    python ft.py --name "$NAME" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "==== FAILED $NAME  (exit $rc) ===="
    else
        echo "==== DONE $NAME ===="
    fi
}

COMMON="--n_train 3000 --epochs 10 --batch_size 4 --n_eval 200 \
        --wandb --wandb_project memory-experiment"

run FT_full_A_n3000_e10   --method full --mode A --lr 5e-5 $COMMON --wandb_group ft_full_A
run FT_full_B_n3000_e10   --method full --mode B --lr 5e-5 $COMMON --wandb_group ft_full_B
run FT_lora_A_n3000_e10   --method lora --mode A --lr 5e-4 --lora_rank 8 $COMMON --wandb_group ft_lora_A
run FT_lora_B_n3000_e10   --method lora --mode B --lr 5e-4 --lora_rank 8 $COMMON --wandb_group ft_lora_B

echo ""
echo "==== QUICK PASS DONE ===="
