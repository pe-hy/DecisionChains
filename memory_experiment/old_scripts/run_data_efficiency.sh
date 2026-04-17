#!/usr/bin/env bash
#
# run_data_efficiency.sh
#
# How few training examples does the memory actually need to align the model?
# Base recipe = E4 winner (data=all, ce=full_seq, dp_target=f, layer=10,
# gate, no sparsity). Only n_train and epochs (and memory size at the end)
# vary. GPU 0, W&B logging, baseline cached across all runs.
#
# Time budget: ~6 min per run × 18 runs ≈ 2 hours.

set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=logs
cd "$(dirname "$0")"

# Idempotent restart: skip if the run's JSON already exists.
run() {
    local NAME=$1; shift
    if [ -f "outputs/experiments/$NAME.json" ]; then
        echo ">>> SKIP $NAME  (outputs/experiments/$NAME.json already exists)"
        return
    fi
    python exp.py --name "$NAME" "$@"
}

N_EVAL=150
WBP="memory-experiment"
COMMON_BASE="--data_filter all --ce_mode full_seq --dp_target f --layers 10 --gate --sparsity none --wandb --wandb_project $WBP --n_eval $N_EVAL"

# ─── Stage 1: n_train × epochs sweep (mem_entries fixed at 32) ───────────────
# Two epoch settings per n_train to span over/under-training. For very small
# n_train, more epochs because each epoch = few gradient steps.

run D_n10_e50       --n_train   10 --epochs  50 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n10   --seed 0
run D_n10_e200      --n_train   10 --epochs 200 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n10   --seed 0
run D_n10_e200_s1   --n_train   10 --epochs 200 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n10   --seed 1
run D_n10_e200_s2   --n_train   10 --epochs 200 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n10   --seed 2

run D_n30_e30       --n_train   30 --epochs  30 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n30
run D_n30_e100      --n_train   30 --epochs 100 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n30

run D_n100_e20      --n_train  100 --epochs  20 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n100
run D_n100_e50      --n_train  100 --epochs  50 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n100

run D_n300_e15      --n_train  300 --epochs  15 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n300
run D_n300_e30      --n_train  300 --epochs  30 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n300

run D_n1000_e10     --n_train 1000 --epochs  10 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n1000
run D_n1000_e20     --n_train 1000 --epochs  20 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n1000

run D_n3000_e5      --n_train 3000 --epochs   5 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n3000
run D_n3000_e10     --n_train 3000 --epochs  10 --mem_entries 32 $COMMON_BASE --wandb_group data_eff_n3000


# ─── Stage 2: memory-size sweep at a small n_train (100 / 30 epochs) ─────────
# Tests whether a smaller memory suffices once data is limited. If n=100 is
# still too much, we'll rerun at n=30 after looking at Stage 1.

run M_mem2_n100_e30   --n_train 100 --epochs 30 --mem_entries 2  $COMMON_BASE --wandb_group mem_size_n100
run M_mem4_n100_e30   --n_train 100 --epochs 30 --mem_entries 4  $COMMON_BASE --wandb_group mem_size_n100
run M_mem8_n100_e30   --n_train 100 --epochs 30 --mem_entries 8  $COMMON_BASE --wandb_group mem_size_n100
run M_mem16_n100_e30  --n_train 100 --epochs 30 --mem_entries 16 $COMMON_BASE --wandb_group mem_size_n100


python compare.py
