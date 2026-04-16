# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training small GPT-NeoX transformers on **decision chain** tasks. A model learns to apply chains of 3-5 vector transformations, where the transformation at each step is selected by a decision function applied to the current intermediate vector. Tests compositional generalization on novel function tuples and novel input vectors.

Four experiment variants:
- **Extended** (`train_chains.py`): Standard chains with 3-5 steps, evaluated on block validity, probability distributions at decision points, and valid chain/solution fractions.
- **STOP** (`train_chains_stop.py`): Same as extended, but training data includes negative examples where a wrong decision-function branch is taken and terminated with a `STOP` token. Evaluates whether the model learns to assign high P(STOP) after wrong branches.
- **Balanced** (`train_chains_balanced.py`, `train_chains_balanced_v2.py`): Combines healthy partial/full traces with unhealthy STOP-terminated traces. For each GT chain of length N, produces 2N training variants (N healthy + N unhealthy). Only uses chains where f != g at every step for exact balance.
- **GRPO** (`grpo_train.py`, `grpo_train_step.py`): Post-training via Group Relative Policy Optimization on a pretrained HF checkpoint. Binary correctness reward with group-normalized advantages.

## Commands

### Data generation
```bash
python scripts/generate_decision_chains_extended.py --target_train 1000000 --target_test 10000
python scripts/generate_decision_chains_stop.py --target_train 1000000 --target_test 10000 --neg_ratio 1.0
python scripts/generate_decision_chains_balanced.py --target_train 1000000 --target_test 10000
```
Outputs go to `outputs/data/decision_chains_{extended,stop,balanced}/` and tokenizers to `outputs/tokenizer/`.

### Training
```bash
python train_chains.py                  # extended (config: base_decision_chains_extended)
python train_chains_stop.py             # STOP (config: base_decision_chains_stop)
python train_chains_balanced.py         # balanced (config: base_decision_chains_balanced)
python grpo_train.py                    # GRPO post-training (config: base_grpo)
python train_chains.py model.epochs=100 # Hydra override example
```

### Inference (post-training analysis)
```bash
python scripts/inference_decision_chains_extended.py
python scripts/inference_decision_chains_stop.py
```
Loads the HF-converted checkpoint, runs free generation on unique test vectors, extracts probability distributions at decision points, labels trajectories with 2^N-way match, and generates HTML visualizations.

### Pass@k evaluation (standalone)
```bash
python scripts/eval_pass_at_k.py \
    --model_path outputs/temp/hf_<model_name> \
    --tokenizer_path outputs/tokenizer/tokenizer_decision_chains_extended.json \
    --test_file outputs/data/decision_chains_extended/val.json \
    --k_values 1,2,4,8,16,32,64,128,256 \
    --n_samples 256 --temperature 0.8
python visualization/plot_pass_at_k.py --input results.json --output pass_at_k.png
```

### End-to-end pipeline
```bash
./run_pipeline.sh                        # full: generate → train → infer → pass@k → plot
./run_pipeline.sh --epochs 5 --skip-datagen
./run_pipeline.sh --skip-train           # eval existing checkpoint
./run_pipeline.sh --pass-k-max-inputs 100  # quick pass@k test
```

### Config overrides
All training/inference scripts use Hydra. Override any config value on the command line:
```bash
python train_chains.py model.batch_size=256 model.n_layer=6 eval.num_examples=1024
```
Config names: `base_decision_chains_extended`, `base_decision_chains_stop`, `base_decision_chains_balanced`, `base_grpo`.

## Architecture

### Pipeline flow
1. **Generate data** (`scripts/generate_*.py`) → JSON train/val files + tokenizer JSON
2. **Train** (`train_chains*.py`) → LitGPT checkpoint, converted to HF at each validation epoch for evaluation
3. **Evaluate during training** (`evaluation/evaluator_chains_*.py`) → metrics logged to WandB
4. **Post-training inference** (`scripts/inference_*.py`) → probability analysis NPZ, labeled trajectories JSONL, HTML visualizations
5. **Pass@k** (`scripts/eval_pass_at_k.py` + `visualization/plot_pass_at_k.py`) → sampling-based correctness curves

### Key design decisions

**Model**: LitGPT `GPT` (GPT-NeoX architecture, default 12L-8H-512D) wrapped in a Lightning module (`LitChains`). During validation, checkpoints are converted from LitGPT → HuggingFace format via `convert_litgpt_to_hf()` to use HF's `generate()` with `output_scores=True`. Uses `flash_attention_2` and `bf16-true` precision.

**Tokenizer**: Custom word-level (`WhitespaceSplit`) tokenizer built during data generation. Special tokens: `[BOS]`, `[EOS]`, `[PAD]`, ` Padres`, `[UNK]`, `[TRACE]`. The `[TRACE]` token separates input from output and is used as the masking delimiter during training (loss computed only after `[TRACE]`).

**Loss masking**: `mask_targets()` in `LitChains` masks all tokens before the first `[TRACE]` delimiter, so the model is only trained to predict the output trace.

**Tokenized cache**: `get_data()` in `framework/data.py` caches tokenized datasets to `outputs/data/decision_chains_*/tokenized_cache/`. Delete this directory to force re-tokenization after data regeneration.

**Data format**: Each example has `input` ("INPUT : [ v ] OUTPUT : [ v ]") and `output` (semicolon-separated trace blocks: "letter trace R [ result ] ; ..."). The 20 transformation functions (letters a-t) are defined in `ops/transformations.py`.

**Decision functions**: `func_f(L) = is_even(L[0]) * 10 + L[1]` and `func_g(L) = is_even(L[3]) * 10 + L[4]`, mapping vectors to indices 0-19 → letters a-t. At each chain step, a coin toss selects f or g. This yields 2^N valid traces per input.

**Train/test split**: Function tuples (the ordered sequence of letters) are split into disjoint sets per chain length. Vectors are also split. This tests compositional generalization on unseen function combinations.

**STOP variant**: Negative examples use the wrong decision function at one step, compute the full (wrong) trace block, then append `; STOP`. The `StopChainEvaluator` probes P(STOP) after correct vs. wrong prefixes.

**Balanced variant**: Combines healthy (partial peeked + full) and unhealthy (STOP-terminated) traces in a 1:1 ratio per GT chain. Uses a `mask_last_n` field in training JSON to selectively mask loss on the peeked letter in partial healthy traces. `LitChains` in `train_chains_balanced*.py` reads this field.

**GRPO**: Loads an existing HF checkpoint, generates K completions per input with temperature sampling, scores with `check_completion_correct()`, computes group-normalized advantages, and applies clipped policy gradient. Key diagnostic: fraction of groups with mixed rewards (non-zero gradient).

### Module dependencies
- `framework/data.py` — `get_data()` (with tokenized caching), `get_tokenizer()`, `Datamodule` (Lightning data module)
- `framework/hf_config.py` — `get_configs()` returns LitGPT `Config` + HF config dict
- `ops/transformations.py` — All 32 transformation functions, each takes `(vec, k, trace_list)` and returns result vec. Only 20 are used in chains (a-t); the rest are unused.
- `evaluation/evaluator_chains_extended.py` — `ExtendedChainEvaluator` with free generation + probability extraction at decision points. Also exports `parse_chain_output_n`, `apply_and_trace`, `extract_input_vector`, decision functions, and `LETTER_TO_FUNC`.
- `evaluation/evaluator_chains_stop.py` — `StopChainEvaluator` extends above with STOP probing
- `evaluation/evaluator_chains_balanced.py` — Evaluator for the balanced experiment
- `evaluation/correctness.py` — Shared `check_completion_correct()` used by pass@k and GRPO
- `evaluation/trajectory_labeler.py` — Labels predictions against all 2^N valid ground-truth traces
- `visualization/visualize_superposition.py` — Generates interactive HTML bar charts of letter probability distributions
- `visualization/plot_pass_at_k.py` — Matplotlib pass@k curves

### Config structure (Hydra YAML)
Configs live in `config/`: `base_decision_chains_extended.yaml`, `base_decision_chains_stop.yaml`, `base_decision_chains_balanced.yaml`, `base_grpo.yaml`. Key sections: `model` (architecture + training), `data` (paths, tokenizer, sampling), `optim`, `eval`, `inference`, `wandb`, `convert_hf` (LitGPT→HF conversion paths), `grpo` (GRPO-specific hyperparams), `output` (GRPO output paths).

### Output paths
- LitGPT checkpoint: `outputs/temp/<model_name>/`
- HF checkpoint: `outputs/temp/hf_<model_name>/`
- Lightning best checkpoints: `outputs/temp/checkpoints/<model_name>/`
- Inference results: `outputs/eval_results/<model_name>/`
- Pass@k results: `outputs/eval_results/pass_at_k/`

### Key dependencies
LitGPT (for GPT model + config), PyTorch Lightning, Hydra/OmegaConf, HuggingFace transformers (for HF model + tokenizer + generation), flash-attn (flash_attention_2), WandB (logging).
