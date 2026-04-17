#!/usr/bin/env bash
#
# run_experiments.sh — 27-variant alignment sweep on GPU 0.
#
# Goal: find the simplest, fastest recipe that aligns the frozen model
# toward f at decision points. Each run logs to W&B (project below)
# AND writes outputs/experiments/<name>.json.
#
# Time budget: ~6 min per run (baseline cached after run 1) → ~3 hours total.
# All runs share defaults; each block overrides one dimension at a time
# (single-axis ablations) so each result is interpretable in isolation.
#
# Reference recipe (the winner from the E1-E4 pilot): W01.

set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=logs
cd "$(dirname "$0")"

# Common defaults — tuned for fast iteration.
N_TRAIN=2048
N_EVAL=150
EPOCHS=10
WBP="memory-experiment"
COMMON="--n_train $N_TRAIN --n_eval $N_EVAL --epochs $EPOCHS --wandb --wandb_project $WBP"

# Reference recipe: data=all, ce=full_seq, dp_target=f, layer=10,
#                   mem_entries=32, gate, sparsity=none.
REF="--data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none"


# ─── A. Reference + ingredient ablations (does each E4 piece matter?) ────────

python exp.py --name W01_reference         $REF $COMMON --wandb_group ingredient_ablation
python exp.py --name W02_data_ffff         --data_filter ffff --ce_mode full_seq --dp_target gt --layers 10 --mem_entries 32 --gate --sparsity none $COMMON --wandb_group ingredient_ablation
python exp.py --name W03_ce_dponly         --data_filter all  --ce_mode dp_only  --dp_target f  --layers 10 --mem_entries 32 --gate --sparsity none $COMMON --wandb_group ingredient_ablation
python exp.py --name W04_no_gate           --data_filter all  --ce_mode full_seq --dp_target f  --layers 10 --mem_entries 32         --sparsity none $COMMON --wandb_group ingredient_ablation


# ─── B. Sparsity sweep (does penalizing off-DP correction recover op_acc?) ──

python exp.py --name W05_sparsity_0p01     --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity l2_nondp --sparsity_coeff 0.01 $COMMON --wandb_group sparsity
python exp.py --name W06_sparsity_0p1      --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity l2_nondp --sparsity_coeff 0.1  $COMMON --wandb_group sparsity
python exp.py --name W07_sparsity_1p0      --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity l2_nondp --sparsity_coeff 1.0  $COMMON --wandb_group sparsity


# ─── C. Layer placement (where in the network is the edit most effective?) ──

python exp.py --name W08_layer3            --data_filter all --ce_mode full_seq --dp_target f --layers 3  --mem_entries 32 --gate --sparsity none $COMMON --wandb_group layer
python exp.py --name W09_layer6            --data_filter all --ce_mode full_seq --dp_target f --layers 6  --mem_entries 32 --gate --sparsity none $COMMON --wandb_group layer
python exp.py --name W10_layer8            --data_filter all --ce_mode full_seq --dp_target f --layers 8  --mem_entries 32 --gate --sparsity none $COMMON --wandb_group layer
python exp.py --name W11_layer11           --data_filter all --ce_mode full_seq --dp_target f --layers 11 --mem_entries 32 --gate --sparsity none $COMMON --wandb_group layer


# ─── D. Memory size (capacity sweep, ~constant compute per step) ─────────────

python exp.py --name W12_mem4              --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 4   --gate --sparsity none $COMMON --wandb_group mem_size
python exp.py --name W13_mem8              --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 8   --gate --sparsity none $COMMON --wandb_group mem_size
python exp.py --name W14_mem16             --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 16  --gate --sparsity none $COMMON --wandb_group mem_size
python exp.py --name W15_mem64             --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 64  --gate --sparsity none $COMMON --wandb_group mem_size
python exp.py --name W16_mem128            --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 128 --gate --sparsity none $COMMON --wandb_group mem_size


# ─── E. Multi-layer injection (one shared memory, multiple hooks) ────────────

python exp.py --name W17_layers_6_10       --data_filter all --ce_mode full_seq --dp_target f --layers 6,10   --mem_entries 32 --gate --sparsity none $COMMON --wandb_group multi_layer
python exp.py --name W18_layers_6_9_12     --data_filter all --ce_mode full_seq --dp_target f --layers 6,9,12 --mem_entries 32 --gate --sparsity none $COMMON --wandb_group multi_layer


# ─── F. Speed levers (how cheap can we make this?) ───────────────────────────

python exp.py --name W19_epochs3           --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --n_train $N_TRAIN --n_eval $N_EVAL --epochs 3  --wandb --wandb_project $WBP --wandb_group speed
python exp.py --name W20_epochs5           --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --n_train $N_TRAIN --n_eval $N_EVAL --epochs 5  --wandb --wandb_project $WBP --wandb_group speed
python exp.py --name W21_data512           --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --n_train 512  --n_eval $N_EVAL --epochs $EPOCHS --wandb --wandb_project $WBP --wandb_group speed
python exp.py --name W22_data1024          --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --n_train 1024 --n_eval $N_EVAL --epochs $EPOCHS --wandb --wandb_project $WBP --wandb_group speed
python exp.py --name W23_data8192          --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --n_train 8192 --n_eval $N_EVAL --epochs $EPOCHS --wandb --wandb_project $WBP --wandb_group speed


# ─── G. Learning rate ───────────────────────────────────────────────────────

python exp.py --name W24_lr_3e-4           --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --lr 3e-4 $COMMON --wandb_group lr
python exp.py --name W25_lr_3e-3           --data_filter all --ce_mode full_seq --dp_target f --layers 10 --mem_entries 32 --gate --sparsity none --lr 3e-3 $COMMON --wandb_group lr


# ─── H. Seed reproducibility (noise floor) ──────────────────────────────────

python exp.py --name W26_seed1             $REF $COMMON --seed 1 --wandb_group seeds
python exp.py --name W27_seed2             $REF $COMMON --seed 2 --wandb_group seeds


# ─── Final comparison ───────────────────────────────────────────────────────
python compare.py
