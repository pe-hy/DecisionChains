#!/usr/bin/env bash
#
# Phase 1 — Screening sweep.
# Single-seed runs at n_eval=50. Tests every dimension at the new working point.
# Results feed Phase 2 (top-5 × 3 seeds) and Phase 3 (compound best).

set +e   # keep going on errors
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=logs
cd "$(dirname "$0")"

run() {
    local NAME=$1; shift
    if [ -f "outputs/experiments/$NAME.json" ]; then
        echo ">>> SKIP $NAME  (already done)"
        return 0
    fi
    echo "==== START $NAME ===="
    python exp.py --name "$NAME" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "==== FAILED $NAME  (exit $rc) ===="
    else
        echo "==== DONE $NAME ===="
    fi
}

NEVAL=50
WBP="memory-experiment"
COMMON="--wandb --wandb_project $WBP --n_eval $NEVAL"
BASE="--data_filter all --ce_mode full_seq --dp_target f --gate --sparsity none"


# ─── A. Ingredient ablations at n=3000 × 10ep (5 runs) ────────────────────

run A1_ref                $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_ingredient
run A2_no_mix             --data_filter ffff --ce_mode full_seq --dp_target gt --layers 10 --mem_entries 32 --gate --sparsity none --n_train 3000 --epochs 10 $COMMON --wandb_group p1_ingredient
run A3_no_fullseq         --data_filter all  --ce_mode dp_only  --dp_target f  --layers 10 --mem_entries 32 --gate --sparsity none --n_train 3000 --epochs 10 $COMMON --wandb_group p1_ingredient
run A4_no_gate            --data_filter all  --ce_mode full_seq --dp_target f  --layers 10 --mem_entries 32         --sparsity none --n_train 3000 --epochs 10 $COMMON --wandb_group p1_ingredient
run A5_no_override        --data_filter all  --ce_mode full_seq --dp_target gt --layers 10 --mem_entries 32 --gate --sparsity none --n_train 3000 --epochs 10 $COMMON --wandb_group p1_ingredient


# ─── B. Layer sweep at n=3000 × 10ep (8 runs) ────────────────────────────

for L in 2 4 6 8 11; do
    run "B_layer${L}"     $BASE --layers $L --mem_entries 32 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_layer
done
run B_layers_6_10         $BASE --layers 6,10 --mem_entries 32 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_layer
run B_layers_6_9_12       $BASE --layers 6,9,12 --mem_entries 32 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_layer
run B_layers_8_10         $BASE --layers 8,10 --mem_entries 32 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_layer


# ─── C. Memory size at working regimes (8 runs) ──────────────────────────

# At n=1000 × 10ep:
for M in 2 4 8 16 64; do
    run "C_mem${M}_n1000" $BASE --layers 10 --mem_entries $M --n_train 1000 --epochs 10 $COMMON --wandb_group p1_mem_n1000
done
# At n=3000 × 10ep:
for M in 4 16 64; do
    run "C_mem${M}_n3000" $BASE --layers 10 --mem_entries $M --n_train 3000 --epochs 10 $COMMON --wandb_group p1_mem_n3000
done


# ─── D. Sparsity sweep at n=3000 × 10ep (5 runs) ─────────────────────────

for LAMB in 0.001 0.01 0.1 0.3 1.0; do
    run "D_sp${LAMB}"     --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity l2_nondp --sparsity_coeff $LAMB --n_train 3000 --epochs 10 $COMMON --wandb_group p1_sparsity
done


# ─── E. Learning rate sweep at n=3000 × 10ep (4 runs) ────────────────────

for LR in 1e-4 3e-4 3e-3 1e-2; do
    run "E_lr${LR}"       $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs 10 --lr $LR $COMMON --wandb_group p1_lr
done


# ─── F. Batch size sweep at n=3000 × 10ep (3 runs) ───────────────────────

for BS in 4 16 32; do
    run "F_bs${BS}"       $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs 10 --batch_size $BS $COMMON --wandb_group p1_batch
done


# ─── G. Epochs at n=1000 (3 runs) ────────────────────────────────────────

for EP in 20 30 50; do
    run "G_n1000_e${EP}"  $BASE --layers 10 --mem_entries 32 --n_train 1000 --epochs $EP $COMMON --wandb_group p1_epochs_n1000
done


# ─── H. Epochs at n=3000 (2 runs) ────────────────────────────────────────

for EP in 20 30; do
    run "H_n3000_e${EP}"  $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs $EP $COMMON --wandb_group p1_epochs_n3000
done


# ─── I. Compound small-memory + sparsity + layer (6 runs) ────────────────
# Tests whether small-mem + surgicality gives back op_acc without breaking alignment.

run I_mem8_sp0p1_l10      --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries  8 --gate --sparsity l2_nondp --sparsity_coeff 0.1  --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound
run I_mem8_sp0p1_l6       --data_filter all --ce_mode full_seq --dp_target f --layers 6  --mem_entries  8 --gate --sparsity l2_nondp --sparsity_coeff 0.1  --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound
run I_mem4_sp0p01_l10     --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries  4 --gate --sparsity l2_nondp --sparsity_coeff 0.01 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound
run I_mem16_sp0p01_l10    --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 16 --gate --sparsity l2_nondp --sparsity_coeff 0.01 --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound
run I_mem16_sp0p1_l10     --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 16 --gate --sparsity l2_nondp --sparsity_coeff 0.1  --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound
run I_mem16_sp0p1_l6      --data_filter all --ce_mode full_seq --dp_target f --layers 6  --mem_entries 16 --gate --sparsity l2_nondp --sparsity_coeff 0.1  --n_train 3000 --epochs 10 $COMMON --wandb_group p1_compound


# ─── J. Extra seeds on A1_ref (2 runs) ───────────────────────────────────

run A1_ref_s1             $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs 10 --seed 1 $COMMON --wandb_group p1_ingredient
run A1_ref_s2             $BASE --layers 10 --mem_entries 32 --n_train 3000 --epochs 10 --seed 2 $COMMON --wandb_group p1_ingredient


echo "==== PHASE 1 DONE ===="
