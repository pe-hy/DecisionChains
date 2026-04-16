#!/bin/bash
set -e

# GRPO with per-step advantage decomposition + hybrid KL
#
# Key design:
#   - Policy gradient on DECISION tokens only (per-step advantages)
#   - KL penalty on ALL response tokens (anchors arithmetic to reference)
#
# This decouples f-steering (decision mask) from arithmetic protection (response mask KL).

python grpo_train_step.py \
    model.name="grpo-step_12l-8h-512d-decision-chains-ext_6_2M" \
    grpo.lr=1e-5 \
    grpo.kl_coeff=0.01 \
    grpo.n_iterations=500 \
    grpo.group_size=32 \
    grpo.temperature=1.0 \
    grpo.mini_epochs=1 \
    grpo.eval_every=25 \
    grpo.save_every=250
