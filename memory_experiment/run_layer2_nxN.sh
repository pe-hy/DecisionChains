#!/usr/bin/env bash
#
# run_layer2_nxN.sh — n_train × mem_entries grid at the winning config (layer 2).
#
# Everything else held at the compound winner from Phase 3:
#   layer=2, lr=3e-3, bs=4, gate, no sparsity, data=all, ce=full_seq, dp_target=f, epochs=10
#
# Grid: n_train ∈ {100, 1000, 2000, 3000}, N ∈ {1, 2, 4, 8, 16, 32, 64}
# 4 × 7 = 28 runs. n_eval=200 for clean numbers.
# ~8 min/run × 28 ≈ 3.7 hours.

set +e
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

# Winning-recipe flags (everything except n_train, mem_entries)
BASE="--data_filter all --ce_mode full_seq --dp_target f --gate --sparsity none \
      --layers 2 --lr 3e-3 --batch_size 4 --epochs 10"
COMMON="--wandb --wandb_project memory-experiment --n_eval 200"

# Grid: outer loop over n_train, inner over N. Run small-n first so you
# see the "can it align with 100 examples at layer 2?" answer earliest.
for NTR in 100 1000 2000 3000; do
    for N in 1 2 4 8 16 32 64; do
        run "L2_n${NTR}_N${N}"  $BASE --n_train $NTR --mem_entries $N  \
            $COMMON --wandb_group layer2_nxN
    done
done

echo ""
echo "==== SWEEP DONE ===="
python compare.py
