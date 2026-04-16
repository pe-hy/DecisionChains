# External Memory Experiment — Plan

## Context

We have a pretrained 12L-8H-512D GPT-NeoX transformer that learned to execute decision chains (sequences of 3-5 vector transformations, where f/g decision functions select which letter a-t to apply at each step). The next research step is adding **external key-value memory** to surgically modify decisions. Before building the memory, we need to verify that the model's hidden states contain a **linearly separable signal** for "this position is about to predict a decision letter." If a single `nn.Linear(512, 2)` can classify this, its weight vector is directly usable as the key vector for external memory's dot-product attention.

## Step 1: Decision-Point Linear Probe (`classify_decision_points.py`)

Single script that:

1. **Load model** — `AutoModelForCausalLM.from_pretrained()` from `checkpoint/12l-8h-512d-decision-chains-ext_6_2M/`
2. **Load data** — Read N examples from `../../outputs/data/decision_chains_extended/val.json`
3. **Tokenize** — Concatenate as `[BOS] {input} [TRACE] {output} [EOS]`, pad
4. **Forward pass** — `output_hidden_states=True` → hidden states at all 13 layers (embedding + 12 transformer)
5. **Label decision points** — Position `i` is a decision point if `token[i+1]` is a letter (a-t) AND `i` is after `[TRACE]`
6. **Train linear probe per layer** — `nn.Linear(512, 2)` with Adam + cross-entropy, handle class imbalance with weights
7. **Report** — Per-layer accuracy, precision, recall, F1 table

## Step 2 (future): Perturbation Environment

Build evaluation harness: take N traces, override one letter choice via external memory, measure if target changed correctly and rest of chain preserved.
