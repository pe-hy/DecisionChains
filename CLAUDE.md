# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training small GPT-NeoX transformers on **decision chain** tasks. A model learns to apply chains of 3-5 vector transformations, where the transformation at each step is selected by a decision function applied to the current intermediate vector. Tests compositional generalization on novel function tuples and novel input vectors.

Two experiment variants:
- **Extended** (`train_chains.py`): Standard chains with 3-5 steps, evaluated on block validity, probability distributions at decision points, and valid chain/solution fractions.
- **STOP** (`train_chains_stop.py`): Same as extended, but training data includes negative examples where a wrong decision-function branch is taken and terminated with a `STOP` token. Evaluates whether the model learns to assign high P(STOP) after wrong branches.

## Commands

### Data generation
```bash
python scripts/generate_decision_chains_extended.py --target_train 1000000 --target_test 10000
python scripts/generate_decision_chains_stop.py --target_train 1000000 --target_test 10000 --neg_ratio 1.0
```
Outputs go to `outputs/data/decision_chains_{extended,stop}/` and tokenizers to `outputs/tokenizer/`.

### Training
```bash
python train_chains.py                  # extended experiment (config: base_decision_chains_extended)
python train_chains_stop.py             # STOP experiment (config: base_decision_chains_stop)
python train_chains.py model.epochs=100 # Hydra override example
```

### Inference (post-training analysis)
```bash
python scripts/inference_decision_chains_extended.py
python scripts/inference_decision_chains_stop.py
```
Loads the HF-converted checkpoint, runs free generation on unique test vectors, extracts probability distributions at decision points, labels trajectories with 2^N-way match, and generates HTML visualizations.

### Config overrides
All scripts use Hydra. Override any config value on the command line:
```bash
python train_chains.py model.batch_size=256 model.n_layer=6 eval.num_examples=1024
```

## Architecture

### Pipeline flow
1. **Generate data** (`scripts/generate_*.py`) → JSON train/val files + tokenizer JSON
2. **Train** (`train_chains.py` / `train_chains_stop.py`) → LitGPT checkpoint, converted to HF at each validation epoch for evaluation
3. **Evaluate during training** (`evaluation/evaluator_chains_*.py`) → metrics logged to WandB
4. **Post-training inference** (`scripts/inference_*.py`) → probability analysis NPZ, labeled trajectories JSONL, HTML visualizations

### Key design decisions

**Model**: LitGPT `GPT` (GPT-NeoX architecture, default 12L-8H-256D) wrapped in a Lightning module (`LitChains`). During validation, checkpoints are converted from LitGPT → HuggingFace format via `convert_litgpt_to_hf()` to use HF's `generate()` with `output_scores=True`.

**Tokenizer**: Custom word-level (`WhitespaceSplit`) tokenizer built during data generation. Special tokens: `[BOS]`, `[EOS]`, `[PAD]`, `[MASK]`, `[UNK]`, `[TRACE]`. The `[TRACE]` token separates input from output and is used as the masking delimiter during training (loss computed only after `[TRACE]`).

**Loss masking**: `mask_targets()` in `LitChains` masks all tokens before the first `[TRACE]` delimiter, so the model is only trained to predict the output trace.

**Data format**: Each example has `input` ("INPUT : [ v ] OUTPUT : [ v ]") and `output` (semicolon-separated trace blocks: "letter trace R [ result ] ; ..."). The 20 transformation functions (letters a-t) are defined in `ops/transformations.py`.

**Decision functions**: `func_f(L) = is_even(L[0]) * 10 + L[1]` and `func_g(L) = is_even(L[3]) * 10 + L[4]`, mapping vectors to indices 0-19 → letters a-t. At each chain step, a coin toss selects f or g.

**Train/test split**: Function tuples (the ordered sequence of letters) are split into disjoint sets per chain length. Vectors are also split. This tests compositional generalization on unseen function combinations.

**STOP variant**: Negative examples use the wrong decision function at one step, compute the full (wrong) trace block, then append `; STOP`. The `StopChainEvaluator` probes P(STOP) after correct vs. wrong prefixes.

### Module dependencies
- `framework/data.py` — `get_data()`, `get_tokenizer()`, `Datamodule` (Lightning data module)
- `framework/hf_config.py` — `get_configs()` returns LitGPT `Config` + HF config dict
- `ops/transformations.py` — All 20+ transformation functions, each takes `(vec, k, trace_list)` and returns result vec
- `evaluation/evaluator_chains_extended.py` — `ExtendedChainEvaluator` with free generation + probability extraction at decision points
- `evaluation/evaluator_chains_stop.py` — `StopChainEvaluator` extends above with STOP probing
- `evaluation/trajectory_labeler.py` — Labels predictions against all 2^N valid ground-truth traces
- `visualization/visualize_superposition.py` — Generates interactive HTML bar charts of letter probability distributions

### Config structure (Hydra YAML)
Configs live in `config/`. Key sections: `model` (architecture + training), `data` (paths, tokenizer, sampling), `optim`, `eval`, `inference`, `wandb`, `convert_hf` (LitGPT→HF conversion paths).
