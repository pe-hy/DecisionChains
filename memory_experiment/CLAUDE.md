# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experiment in **surgical memory injection** into a pretrained decision-chain transformer. The goal is to modify the model's decision at a specific chain step by adding a learned vector to its residual stream, without retraining the model. This is a stepping stone toward external key-value memory that can steer model behavior at inference time.

The pretrained model (12L-8H-512D GPT-NeoX) generates traces of 3-5 vector transformations, where at each step a decision function (f or g) selects which transformation letter (a-t) to apply. At each decision point, there are (typically) two valid letter choices. We learn a 512-dim vector V that, when added to a hidden state at a chosen layer, flips the model's prediction from one valid letter to the other.

## Commands

```bash
# Classify decision points via linear probe on hidden states
python classify_decision_points.py --n_train 500 --n_test 500 --probe_epochs 100

# Main experiment: learn V, inject, generate, score
python memory_inject.py --n_examples 50 --layers 1,6,12 --steps 200
python memory_inject.py --n_examples 10 --layers all --steps 300

# Both scripts auto-convert the LitGPT checkpoint to HF on first run
```

## Architecture

### What exists

**`metrics.py`** — Self-contained scoring library with no external model dependencies. Defines all 20 transformations (a-t), the two decision functions (`decision_f`, `decision_g`), trace parsing, and three core metrics:
- **operation_accuracy**: Is the arithmetic in each trace block correct? (Re-execute the letter's transformation, compare block text.)
- **operation_selection**: Is each letter a valid f/g output given the current intermediate vector?
- **complete_solution**: All ops correct + all selections correct + final vector matches the OUTPUT in the prompt.
- Key entry points: `score_trace()`, `aggregate_scores()`, `extract_input_output()`, `decision_letters()`.

**`classify_decision_points.py`** — Linear probe experiment. Tests whether a single `nn.Linear(512, 2)` can classify "is this token position a decision point?" from hidden states at each layer. Trains on train.json, evaluates on val.json (proper example-level split). High accuracy means the model's hidden states contain a linearly separable signal for decision points — a prerequisite for dot-product-based external memory retrieval.

**`memory_inject.py`** — Core injection experiment. For each test example at each layer:
1. Generate baseline trace, score it.
2. Determine the two valid f/g letters. Pick the one the model didn't choose as the target.
3. Teacher-force the full sequence (prompt + GT output), learn V via gradient descent (200 steps, Adam, CE loss targeting the alternative letter). V is the only optimized parameter; model weights are frozen.
4. Inject V at the `[TRACE]` position via a forward hook on `model.gpt_neox.layers[layer_idx]`.
5. Generate from the prompt with V injected, score the perturbed trace.
6. Report target-hit rate, operation accuracy, selection accuracy, complete-solution rate per layer.

**`ALGORITHM.md`** — Detailed description of the injection mechanism, concrete example, and notes on extending to arbitrary decision points (not just the first one).

### Mechanism

```
At layer L, position P ([TRACE] or ;):
    h'[P] = h[P] + V          ← V is a learned 512-dim vector
Logits at position P now predict a different letter.
V is optimized via CrossEntropy(logits[P], target_letter_id).
Model weights stay frozen.
```

The hook is registered on the residual stream output of a transformer layer. During generation, the first forward pass processes the full prompt; the hook adds V at the trace position. Subsequent tokens are generated one-by-one, and since the trace position is in the KV cache, the injection only affects the initial pass.

### Currently scoped to first decision point only

The `[TRACE]` position is at a fixed index in the prompt (known before generation), which simplifies the experiment. Correcting errors at later decision points requires a two-phase approach: generate freely up to the target `;`, then inject at that position. See `ALGORITHM.md` for the extension sketch.

### Key constants (hardcoded token IDs)

| Token | ID |
|-------|----|
| Letters a-t | 67-86 |
| `[TRACE]` | 87 |
| `[BOS]` | 88 |
| `[PAD]` | 89 |
| `[EOS]` | 92 |

### Data and model paths

Both scripts use relative paths from `SCRIPT_DIR`:
- Checkpoint: `checkpoint/12l-8h-512d-decision-chains-ext_6_2M/` (LitGPT format)
- HF checkpoint: `checkpoint/12l-8h-512d-decision-chains-ext_6_2M/hf/` (auto-converted on first run)
- Test data: `../../outputs/data/decision_chains_extended/val.json` (relative to this directory)

### The cascading problem

Changing one letter is not a local edit — the intermediate vector changes, which changes which letters f/g select at all subsequent steps. This is expected and correct behavior. The right metric is `operation_selection`: does the model still make valid f/g choices given the new intermediate vectors? Not "are the downstream letters the same as before?"

### Decision functions

```
f(L) = is_even(L[0]) * 10 + L[1]   →  index 0-19  →  letter a-t
g(L) = is_even(L[3]) * 10 + L[4]   →  index 0-19  →  letter a-t
```

When f and g produce the same letter for a vector, there's no alternative branch to switch to — those examples are skipped in `memory_inject.py`.

## Plans

The `plans/` directory contains design documents:
- `plan.md` — Original plan for the injection experiment with concrete example
- `plan_metrics.md` — Metrics design, mapping to the parent repo's `validate.py`
- `plan_memory.md` — Step-by-step plan for the linear probe + injection experiments
- `data_flow.md` — Detailed data flow for `classify_decision_points.py`
- `discussion.md` — Raw brainstorming notes (Czech) about the approach
