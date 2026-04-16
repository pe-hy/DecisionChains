# Decision Chains: Full Pipeline Documentation

## Overview

This codebase trains small GPT-NeoX transformers on **decision chain** tasks. A model receives an input vector and a target output vector, and must produce the trace of 3-5 sequential vector transformations that connects them. At each step, the correct transformation is determined by applying a **decision function** (f or g) to the current intermediate vector.

There are two experiment variants:
- **Extended**: Standard chains, evaluated on block validity and probability distributions at decision points.
- **STOP**: Adds negative examples where a wrong decision branch is taken and terminated with a STOP token.

---

## 1. Task Definition

### Vectors and Arithmetic

All vectors have 6 integer elements in [0, 9]. All operations use modular arithmetic with k=10.

### Decision Functions

Two functions map a 6-element vector to an index in [0, 19]:

```
f(L) = is_even(L[0]) * 10 + L[1]
g(L) = is_even(L[3]) * 10 + L[4]
```

where `is_even(x) = 1 if x % 2 == 0 else 0`.

The index maps to one of 20 letters (a-t), each corresponding to a transformation function.

### Chain Execution

For a chain of length n (3, 4, or 5):
1. Start with input vector v_0.
2. At each step i: a coin toss selects decision function f or g.
3. Apply the selected function to the current vector to get a letter (a-t).
4. Apply the letter's transformation to get the next vector.
5. After n steps, the final vector v_n is the output.

Since f or g can be chosen at each step, there are **2^n valid traces** per input (all combinations produce correct chains). The model sees both the INPUT and OUTPUT vectors and must produce any valid trace.

### Example

```
INPUT : [ 2 , 3 , 4 , 5 , 6 , 7 ] OUTPUT : [ 4 , 0 , 0 , 0 , 0 , 0 ]
[TRACE]
n : 2 + 2 = 4 , 3 + 2 = 5 , ... R [ 4 , 5 , 6 , 7 , 8 , 9 ] ; p : 4 , 4*5=0 , ... R [ 4 , 0 , 0 , 0 , 0 , 0 ] ; k : ...
```

Each trace block: `letter : computation_trace R [ result_vector ]`, separated by ` ; `.

---

## 2. Transformation Functions (a-t)

All functions have signature `(vec: List[int], k: int, trace: List[str]) -> List[int]` and append a trace string describing the computation.

| Letter | Name | Operation |
|--------|------|-----------|
| a | reverse | Reverse vector order |
| b | add_1 | Each element + 1 mod k |
| c | double | Each element * 2 mod k |
| d | negate | Each element: (k - x) mod k |
| e | cumsum | Cumulative sum mod k (left to right) |
| f | scale_by_first | Each element * vec[0] mod k |
| g | rotate_left | Shift all elements one position left (wrap) |
| h | swap_pairs | Swap adjacent pairs; odd-length keeps last |
| i | position_multiply | Each element * its index mod k |
| j | diff | First element unchanged, rest: vec[i] - vec[i-1] mod k |
| k | add_last_to_all | Each element + vec[-1] mod k |
| l | square | Each element^2 mod k |
| m | rotate_right | Shift all elements one position right (wrap) |
| n | add_first_to_all | Each element + vec[0] mod k |
| o | cumsum_reverse | Cumulative sum right-to-left, then reverse |
| p | prefix_product | Cumulative product mod k |
| q | sliding_sum | Each: vec[i] + vec[(i+1) % len] mod k (circular) |
| r | position_add | Each element + its index mod k |
| s | conditional_double | If x >= k/2: 2*x mod k, else x |
| t | interleave_sum_diff | Adjacent pairs -> (sum, diff) mod k |

Defined in `ops/transformations.py`. The file also contains ~10 additional transformations (sort_asc, add_2, etc.) that are **not used** in the chain experiments.

---

## 3. Data Generation

### Extended (`scripts/generate_decision_chains_extended.py`)

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_vectors` | 95000 | Distinct input vectors for training |
| `--test_vectors` | 5000 | Distinct input vectors for testing |
| `--target_train` | 1,000,000 | Target training examples |
| `--target_test` | 10,000 | Target test examples |
| `--test_fraction` | 0.25 | Fraction of tuples held out for test |
| `--chain_lengths` | [3, 4, 5] | Allowed chain lengths |
| `--vector_length` | 6 | Vector dimension |
| `--reachable_sample` | 200,000 | Vectors sampled for tuple discovery |
| `--seed` | 42 | Random seed |
| `--output_dir` | None | Output dir (auto: `outputs/data/decision_chains_extended/`) |

**Algorithm:**

1. **Vector pool**: Generate all 10^6 possible 6-element vectors. Split into 95K train + 5K test (disjoint).

2. **Tuple discovery**: Sample 200K vectors. For each vector and each chain length, enumerate all 2^n decision-function combinations to discover all reachable letter tuples (ordered sequences of n letters).

3. **Train/test tuple split**: Per chain length, split discovered tuples so ~75% go to train and ~25% to test. Uses a coverage-driven greedy algorithm ensuring every letter appears at every position in the train set. Train and test tuples are **disjoint** — this is how compositional generalization is tested.

4. **Example generation**: For each example, pick a random chain length, random vector from the appropriate pool, random f/g combination. If the resulting letter tuple belongs to the correct split, keep the example. Targets uniform distribution across chain lengths.

5. **Tokenizer building**: Sample 3000 train + 1000 test examples, collect all unique whitespace-delimited tokens, build a WordLevel tokenizer with special tokens [BOS], [EOS], [PAD], [MASK], [UNK].

**Outputs:**

| File | Description |
|------|-------------|
| `train.json` | Array of `{input, output, letters, decision_funcs}` |
| `val.json` | Same format, disjoint vectors and tuples |
| `metadata.json` | Train/test tuples per length, letter mappings, decision function definitions, reachable counts |
| `tokenizer_decision_chains_extended.json` | HuggingFace tokenizer JSON |

**Example JSON entry:**
```json
{
  "input": "INPUT : [ 1 , 2 , 3 , 4 , 5 , 6 ] OUTPUT : [ 8 , 7 , 6 , 5 , 4 , 3 ]",
  "output": "a reverse R [ 6 , 5 , 4 , 3 , 2 , 1 ] ; b : 6 + 1 = 7 , ... R [ 7 , 6 , 5 , 4 , 3 , 2 ] ; ...",
  "letters": ["a", "b", "c"],
  "decision_funcs": ["f", "g", "f"]
}
```

### STOP Variant (`scripts/generate_decision_chains_stop.py`)

Extends the extended generator with negative examples.

**Additional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--neg_ratio` | 1.0 | Max ratio of negative to positive examples |
| `--mistake_at` | None | Explicit step indices for mistakes |
| `--num_mistakes` | None | Random number of mistakes per example |

**Negative example generation:**

For each positive (GT) example:
1. Replay the chain to get intermediate vectors at each step.
2. Identify **eligible steps**: where f and g produce different letters (so a wrong choice exists).
3. At the selected mistake position, use the **opposite** decision function.
4. Compute the (wrong) trace block for the wrong letter.
5. Concatenate: correct prefix + wrong block + ` ; STOP`.

**Negative example format:**
```json
{
  "input": "INPUT : [ ... ] OUTPUT : [ ... ]",
  "output": "a reverse R [ ... ] ; d : ... R [ ... ] ; STOP",
  "letters": ["a", "d"],
  "decision_funcs": ["f", "wrong_g"],
  "is_negative": true,
  "mistake_at": 1,
  "gt_letter_at_mistake": "b",
  "wrong_letter_at_mistake": "d",
  "gt_chain_length": 3
}
```

---

## 4. Training

### Model Architecture

GPT-NeoX via LitGPT library. Default config (`config/base_decision_chains_extended.yaml`):

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Heads | 8 |
| Embedding dim | 512 |
| Block size (max seq len) | 512 |
| Batch size | 256 |

### Training Script (`train_chains.py`)

**Lightning module `LitChains`:**
- Wraps LitGPT `GPT` model.
- **Loss masking**: `mask_targets()` masks all tokens before the first `[TRACE]` delimiter. The model is only trained to predict the output trace, not the input.
- Optimizer: AdamW with cosine warmup schedule.

**Validation loop (`on_validation_epoch_end`):**
1. Saves LitGPT checkpoint.
2. Converts to HuggingFace format via `convert_litgpt_to_hf()`.
3. Runs `ExtendedChainEvaluator` on the test set (greedy generation).
4. Logs metrics to WandB.

**Output paths:**
- LitGPT checkpoint: `outputs/temp/<model_name>/`
- HF checkpoint: `outputs/temp/hf_<model_name>/`
- Lightning checkpoints: `outputs/temp/checkpoints/<model_name>/`

### Tokenization (`framework/data.py`)

Each example is tokenized as:
```
[BOS] INPUT : [ v ] OUTPUT : [ v ] [TRACE] letter trace R [ vec ] ; ... [EOS]
```

Padded to `block_size` (512). The `[TRACE]` token is the delimiter: everything before it is input context, everything after is the prediction target.

### STOP Training (`train_chains_stop.py`)

Identical structure but uses `StopChainEvaluator` and verifies the STOP token exists in the vocabulary at startup.

---

## 5. Evaluation During Training

### `ExtendedChainEvaluator` (`evaluation/evaluator_chains_extended.py`)

Runs **greedy** generation (do_sample=False) with `output_scores=True` to extract logit distributions.

**Metrics computed:**

| Metric | Definition |
|--------|-----------|
| `exact_match` | Predicted string == ground-truth string exactly |
| `parseable_fraction` | Output successfully parsed into trace blocks |
| `valid_chain_frac` | Correct length + all letters from f/g + all blocks mathematically valid |
| `valid_solution_frac` | Valid chain + final vector matches OUTPUT |
| `step{N}_block_valid` | Per-step block validity (recomputed trace matches) |
| `step{N}_avg_prob_f` | Average P(f's letter) at decision point N |
| `step{N}_avg_prob_g` | Average P(g's letter) at decision point N |
| `avg_prob_f`, `avg_prob_g` | Grand averages across all steps |

**Block validity check**: For each step, recompute `apply_and_trace(letter, current_vec)` and compare the full block string (including the computation trace) against the model's output.

### `StopChainEvaluator` (`evaluation/evaluator_chains_stop.py`)

Extends `ExtendedChainEvaluator` with:
- STOP detection in free generation output.
- **STOP probing**: Constructs correct-branch and wrong-branch prefixes, runs a forward pass, and measures P(STOP) after each. Key metric: `stop_discrimination = P(STOP | wrong) - P(STOP | correct)`.

### Trajectory Labeler (`evaluation/trajectory_labeler.py`)

Labels each prediction against all 2^N valid ground-truth traces. A prediction is correct if:
1. The predicted letter sequence matches some GT trace's letters.
2. All intermediate vectors match exactly.

Outputs `labeled_dataset.jsonl` with per-example correctness labels.

### Correctness Checker (`evaluation/correctness.py`)

Shared module used by pass@k evaluation. A completion is correct if:
1. Every letter is one of the 2 valid choices (from f or g).
2. Every trace block is mathematically valid.
3. The final vector matches the target OUTPUT.

---

## 6. Post-Training Inference

### Extended (`scripts/inference_decision_chains_extended.py`)

Full analysis pipeline run after training completes. Uses Hydra config.

**Steps:**
1. Load HF model and tokenizer.
2. Load test data, deduplicate by input string.
3. Build prompts: `[BOS] input_str [TRACE]`.
4. Generate with `output_scores=True`: extract per-step softmax distributions over the 20 letters.
5. Extract last-layer embeddings at OUTPUT vector and intermediate result positions.
6. Compute metrics (including n-way match, tuple classification, per-step probabilities).
7. Label trajectories.
8. Save all results.

**Outputs:**

| File | Format | Content |
|------|--------|---------|
| `inference_results_extended.json` | JSON | Metrics, per-example results, tuple distributions |
| `decision_point_probs_extended.npz` | NumPy | Per-step probability arrays, vectors, decision indices |
| `labeled_dataset.jsonl` | JSONL | Trajectory correctness labels |
| `labeled_dataset_pure.npz` | NumPy | Embeddings, labels, chain lengths, input vecs |
| `letter_probs_extended.html` | HTML | Interactive visualization |

**Additional metrics beyond training-time evaluation:**

| Metric | Definition |
|--------|-----------|
| `n_way_match` | Matches any of 2^N GT traces (letters + vectors) |
| `all_steps_valid_frac` | All steps chose f/g + all blocks valid (ignores chain length) |
| `chosen_train_tuple_frac` | Model's chosen letter tuple was seen in training |
| `chosen_test_tuple_frac` | Model's chosen letter tuple is from test set |
| Token-level entropy | Per-token entropy and top-k alternatives |

### STOP (`scripts/inference_decision_chains_stop.py`)

Extends the above with:
- STOP detection in free generation.
- STOP probability probing (wrong vs correct branches).
- Token-level P(STOP) tracking.

---

## 7. Pass@k Evaluation

### `scripts/eval_pass_at_k.py`

Measures the **exploration ceiling**: what fraction of test problems the model can solve given k independent sampling attempts.

**Key difference from training evaluation**: Uses **temperature sampling** (not greedy), generating n_samples completions per input.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | required | Path to HF checkpoint |
| `--tokenizer_path` | required | Path to tokenizer JSON |
| `--test_file` | required | Path to val.json |
| `--k_values` | 1,2,4,...,256 | k values to compute |
| `--n_samples` | 256 | Samples generated per input |
| `--temperature` | 0.8 | Sampling temperature |
| `--batch_size` | 128 | Generation batch size |
| `--max_inputs` | None | Limit test inputs |
| `--seed` | 42 | Random seed |
| `--output_dir` | outputs/eval_results/pass_at_k/ | Output directory |

**Estimator**: Uses the unbiased pass@k estimator from Chen et al. (Codex, 2021):
```
pass@k = 1 - C(n-c, k) / C(n, k)
```
where n = total samples generated, c = correct samples. This allows computing pass@k for all k <= n from a single set of n samples, with each estimate unbiased.

**Correctness criterion**: Same as `valid_solution_frac` — every letter from f/g, every block mathematically valid, final vector matches target.

**Output JSON:**
```json
{
  "model_path": "...",
  "temperature": 0.8,
  "n_test_inputs": 5000,
  "n_samples_per_input": 256,
  "pass_at_k": {"1": 0.42, "4": 0.61, "16": 0.78, ...},
  "pass_at_k_by_length": {
    "3": {"1": 0.65, ...},
    "4": {"1": 0.41, ...},
    "5": {"1": 0.22, ...}
  },
  "saturation_k": 64,
  "completion_stats": {
    "avg_correct_frac": 0.34,
    "n_never_correct": 150,
    "n_always_correct": 820
  },
  "per_input": [...]
}
```

---

## 8. Visualization

### Interactive HTML (`visualization/visualize_superposition.py`)

Three view modes:

**Bar Chart View**: Per-example bar charts showing the softmax probability distribution over 20 letters at each decision point. Color-coded: green = f's letter, blue = g's letter, cyan = both, gray = other. Red border on model's chosen letter. Supports log scale, filtering to unseen tuples.

**Entropy View**: Token-by-token display with entropy bars (blue=low, red=high). Decision points highlighted with orange glow. Hover for top-5 alternatives. Filters: valid-but-wrong-final, has-invalid-letter.

**Summary View**: Grid of metric cards with overall, per-length, and (for STOP) stop-specific statistics.

**Usage:**
```bash
# Generated automatically by inference scripts
# Or standalone:
python visualization/visualize_superposition.py \
    --npz_path decision_point_probs_extended.npz --output viz.html
```

### Pass@k Plots (`visualization/plot_pass_at_k.py`)

Matplotlib figures with log-x axis.

**Single model**: Two subplots — overall pass@k curve + per-chain-length breakdown.

**Model comparison**: Overlaid curves with distinct colors/markers/linestyles.

```bash
# Single
python visualization/plot_pass_at_k.py \
    --input pass_at_k_T0.8_n256.json --output pass_at_k.png

# Compare
python visualization/plot_pass_at_k.py \
    --input model_A.json model_B.json \
    --labels "Extended" "STOP" --output comparison.png
```

---

## 9. Full Pipeline Script

`run_pipeline.sh` runs all 5 steps sequentially:

```
Step 1: Data generation  (generate_decision_chains_extended.py)
Step 2: Training          (train_chains.py)
Step 3: Standard inference (inference_decision_chains_extended.py)
Step 4: Pass@k evaluation (eval_pass_at_k.py)
Step 5: Pass@k plotting   (plot_pass_at_k.py)
```

**Key flags:**

| Flag | Effect |
|------|--------|
| `--epochs N` | Set training epochs |
| `--skip-datagen` | Reuse existing data |
| `--skip-train` | Use existing checkpoint |
| `--skip-inference` | Skip standard inference |
| `--pass-k-samples N` | Samples per input for pass@k |
| `--pass-k-temp T` | Sampling temperature |
| `--pass-k-max-inputs N` | Limit test inputs (for quick runs) |
| `--train-args 'ARGS'` | Extra Hydra overrides |

**Examples:**
```bash
./run_pipeline.sh                                           # full run
./run_pipeline.sh --epochs 5 --pass-k-samples 16            # quick test
./run_pipeline.sh --skip-datagen --skip-train                # eval only
CUDA_VISIBLE_DEVICES=2 ./run_pipeline.sh                     # specific GPU
```

---

## 10. Configuration

All training/inference scripts use Hydra. Configs live in `config/`.

### `config/base_decision_chains_extended.yaml`

```yaml
model:
  batch_size: 256
  block_size: 512       # max sequence length
  epochs: 50
  n_layer: 12
  n_head: 8
  n_embd: 512

optim:
  lr: 2e-3
  warmup_steps: 10
  weight_decay: 0.01

data:
  train_file: outputs/data/decision_chains_extended/train.json
  test_file: outputs/data/decision_chains_extended/val.json
  tokenizer_path: outputs/tokenizer/tokenizer_decision_chains_extended.json
  split_str: "[TRACE]"

eval:
  num_examples: 512
  batch_size: 128

inference:
  batch_size: 1024
  html_samples: 500
```

Override any value on the command line: `python train_chains.py model.epochs=100 model.n_layer=6`

### `config/base_decision_chains_stop.yaml`

Same structure with additional fields:
- `eval.stop_probe_examples: 256`
- `eval.stop_probe_batch_size: 64`

---

## 11. Directory Structure

```
DecisionChains/
  config/
    base_decision_chains_extended.yaml
    base_decision_chains_stop.yaml
  evaluation/
    evaluator_chains_extended.py    # Training-time evaluator
    evaluator_chains_stop.py        # STOP evaluator (extends extended)
    trajectory_labeler.py           # 2^N-way match labeling
    correctness.py                  # Shared correctness checking
  framework/
    data.py                         # Data loading, tokenization, Lightning DataModule
    hf_config.py                    # LitGPT/HF config generation
  ops/
    transformations.py              # All 20+ transformation functions
  scripts/
    generate_decision_chains_extended.py
    generate_decision_chains_stop.py
    inference_decision_chains_extended.py
    inference_decision_chains_stop.py
    eval_pass_at_k.py               # Pass@k evaluation
  visualization/
    visualize_superposition.py      # Interactive HTML (bar charts + entropy)
    plot_pass_at_k.py               # Matplotlib pass@k curves
  train_chains.py                   # Training (extended)
  train_chains_stop.py              # Training (STOP)
  run_pipeline.sh                   # Full pipeline script
  CLAUDE.md                         # Claude Code instructions
  plan.md                           # Research plan

  outputs/                          # Generated at runtime
    data/
      decision_chains_extended/     # train.json, val.json, metadata.json
      decision_chains_stop/
    tokenizer/
      tokenizer_decision_chains_extended.json
      tokenizer_decision_chains_stop.json
    temp/
      <model_name>/                 # LitGPT checkpoint
      hf_<model_name>/             # HuggingFace checkpoint
      checkpoints/<model_name>/    # Lightning best checkpoints
    eval_results/
      pass_at_k/                   # Pass@k JSON + plots
```

---

## 12. Train/Test Split Design

The split is designed to test **compositional generalization**:

1. **Vectors are split**: 95K train vectors, 5K test vectors, no overlap.
2. **Function tuples are split per chain length**: For each chain length (3, 4, 5), the set of reachable letter tuples is divided into ~75% train and ~25% test, with no overlap.

This means the test set contains:
- **Unseen vectors**: The model never saw these specific inputs.
- **Unseen function compositions**: The specific ordered sequence of transformations was never seen during training.

The model must generalize compositionally: it has seen each individual transformation and each decision function, but not these particular combinations on these particular inputs.
