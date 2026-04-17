#!/usr/bin/env bash
#
# run_all_phases.sh — Master overnight sweep.
#
# Chains Phase 1 (46 screening runs) → Phase 2 (top-5 × 3 seeds = 15 runs at
# n_eval=200) → Phase 3 (compound best × 3 compute budgets × 3 seeds + bonus
# = ~12 runs at n_eval=300) → summarize_sweep.py.
#
# Each step is idempotent (skips if JSON exists). Safe to rerun.

set +e
export CUDA_VISIBLE_DEVICES=0
export WANDB_DIR=logs
cd "$(dirname "$0")"

echo "=========================================="
echo " OVERNIGHT SWEEP — $(date)"
echo "=========================================="

echo ""
echo ">>> PHASE 1: screening ($(date))"
bash run_phase1.sh
echo "<<< PHASE 1 exit=$?"

echo ""
echo ">>> PHASE 2: top-5 × 3 seeds ($(date))"
python run_phase2.py
echo "<<< PHASE 2 exit=$?"

echo ""
echo ">>> PHASE 3: compound best ($(date))"
python run_phase3.py
echo "<<< PHASE 3 exit=$?"

echo ""
echo ">>> SUMMARY ($(date))"
python summarize_sweep.py
echo "<<< SUMMARY exit=$?"

echo ""
echo "=========================================="
echo " ALL PHASES DONE — $(date)"
echo " See outputs/final_results.md"
echo "=========================================="
