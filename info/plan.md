# Research Plan: Pretraining Composition and the RL Exploration Ceiling

## Research Question

**Does the diversity of reasoning patterns in pretraining data determine the exploration ceiling of RL post-training?**

Specifically: a model pretrained only on correct decision chains has a measurable pass@k ceiling — the maximum fraction of test problems it can solve given unlimited sampling attempts. We hypothesize that:

1. RL post-training (GRPO) cannot exceed this ceiling — it flattens the pass@k curve toward pass@1 but does not extend it.
2. Pretraining on richer reasoning patterns (backtracking, error detection via STOP) raises the ceiling by teaching the model meta-patterns that pure correct-only data does not contain.
3. Mixing easy and hard problems (short vs. long chains) during RL creates gradient interference that disproportionately harms hard-problem performance.

This is a *Physics of LLMs*-style study: controlled synthetic environment, small models, precise claims, clean measurements.

## Why This Environment

The decision chain task has properties that make it ideal for this study:

- **Fully controlled data composition**: we generate all training data, so pretraining mixtures are exact.
- **Known ground truth**: every input has exactly 2^N valid traces (all f/g combinations), so correctness is unambiguous.
- **Natural difficulty axis**: chain length (3, 4, 5 steps) maps directly to problem difficulty (more decision points, longer credit assignment, more ways to fail).
- **No accidental recovery**: after a wrong step, the probability of randomly reaching the correct final vector is ~10^-6 per step (state space = 10^6). Mistakes are genuinely fatal, making error detection a meaningful capability.
- **Compositional generalization**: test-set function tuples are disjoint from training, so we measure genuine generalization, not memorization.

## Connection to Prior Work

- **SGE paper (Strategy-Guided Exploration)**: shows RL for LLM agents is limited by the base model's pass@k ceiling. We test whether this ceiling is a property of the *algorithm* (fixable with better exploration) or the *pretraining data* (fixable only by changing what the model learned).
- **Physics of LLMs**: methodological template — small models, synthetic tasks, precise claims about capabilities.
- **Ray interference** (arXiv:2601.18779): mixing easy/hard problems during RL is counterproductive. We can test this precisely since chain length = difficulty.
- **DAPO / gradient imbalance** (OpenReview:jIeJJqG7dz): negative-advantage samples suppress useful behaviors. Observable in our setting since we know exactly which samples are easy/hard.

---

## Step 1: Pass@k Evaluation Infrastructure

**Goal**: Given a trained model, sample k independent completions per test input, check correctness of each, and compute pass@k across the test set. This is the foundational measurement for everything that follows.

### Definition

pass@k = fraction of test problems where at least one of k independent samples is correct.

A sample is **correct** if:
1. All N trace blocks are valid (letter → transformation → result vector checks out), AND
2. The final vector matches the target OUTPUT vector.

This matches the existing `valid_solution_frac` metric but computed over k independent samples per input.

### Implementation: `scripts/eval_pass_at_k.py`

A standalone script (not integrated into the training loop) that:
1. Loads a trained HF model checkpoint and tokenizer
2. Loads the test dataset
3. For each unique test input, generates k completions using temperature sampling
4. Checks correctness of each completion using existing `apply_and_trace` + vector comparison
5. Computes pass@k for various k values (1, 2, 4, 8, 16, 32, 64, 128, 256)
6. Breaks down results by chain length (3, 4, 5)
7. Outputs results as JSON and prints a summary table

```
python scripts/eval_pass_at_k.py \
    --model_path outputs/temp/hf_<model_name> \
    --tokenizer_path outputs/tokenizer/tokenizer_decision_chains_extended.json \
    --test_file outputs/data/decision_chains_extended/val.json \
    --k_values 1,4,16,64,256 \
    --temperature 0.8 \
    --batch_size 128 \
    --output_dir outputs/eval_results/pass_at_k/
```

### Key Design Decisions

**Reuse existing evaluation logic**: The correctness check reuses `apply_and_trace()` and `parse_chain_output_n()` from `evaluation/evaluator_chains_extended.py`. These are already well-tested. Import them rather than duplicating.

**Separate from training loop**: pass@k evaluation is expensive (k generations per input) and should run post-training, not every validation epoch. Keep it as a standalone script.

**Temperature as a parameter**: The exploration ceiling depends on temperature. We'll sweep over temperatures (0.6, 0.8, 1.0, 1.2) to find the ceiling for each model.

**Unbiased estimator**: For large test sets where generating all k samples is expensive, use the unbiased pass@k estimator from Chen et al. (Codex paper): if n samples are generated and c are correct, pass@k = 1 - C(n-c, k) / C(n, k). This lets us generate n samples and estimate pass@k for all k ≤ n.

**Group by chain length**: Report pass@k separately for 3-step, 4-step, and 5-step chains. This is critical for the ray interference analysis later.

### File Structure

```
scripts/
  eval_pass_at_k.py          # Main pass@k evaluation script

evaluation/
  evaluator_chains_extended.py  # (existing) — reuse apply_and_trace, parse helpers
  correctness.py                # Extract correctness-checking logic into a shared module
```

The `correctness.py` module extracts the "is this completion correct?" logic into a clean function:

```python
def check_completion_correct(
    generated_text: str,
    input_vec: list[int],
    target_output_vec: list[int],
) -> dict:
    """Check if a single generated completion is correct.
    
    Returns dict with:
      - correct: bool (all blocks valid AND final vec matches target)
      - n_blocks: int (number of parsed trace blocks)
      - n_valid_blocks: int (number of individually valid blocks)
      - final_vec: list[int] | None (the result vector of the last block)
    """
```

This function is then used by both `eval_pass_at_k.py` and (optionally) the existing evaluators.

### Output Format

```json
{
    "model_path": "outputs/temp/hf_...",
    "temperature": 0.8,
    "n_test_inputs": 5000,
    "n_samples_per_input": 256,
    "pass_at_k": {
        "1": 0.42, "4": 0.61, "16": 0.78, "64": 0.89, "256": 0.94
    },
    "pass_at_k_by_length": {
        "3": {"1": 0.65, "4": 0.82, ...},
        "4": {"1": 0.41, "4": 0.60, ...},
        "5": {"1": 0.22, "4": 0.38, ...}
    },
    "saturation_k": 64,  // k at which pass@k plateaus (< 1% gain from doubling)
    "completion_stats": {
        "avg_valid_blocks": 3.8,
        "avg_correct_frac": 0.34
    }
}
```

### Definition of Done

- [ ] `evaluation/correctness.py` exists with `check_completion_correct()` tested against known examples
- [ ] `scripts/eval_pass_at_k.py` runs end-to-end on a trained checkpoint
- [ ] Output JSON includes per-chain-length breakdown
- [ ] Results are reproducible (seeded sampling)
- [ ] Pass@k curve is monotonically non-decreasing (sanity check)

---

## Future Steps (not implemented now)

### Step 2: Pretraining Composition Sweep
Train models with 4 data compositions:
- A: correct traces only (existing extended setup)
- B: correct + STOP after wrong branches (existing STOP setup)
- C: correct + full backtracking (wrong → detect → retry → correct) — new data generation needed
- D: correct only, but with more data (control for data volume)

Measure pass@k for each. The hypothesis: C > B > A ≈ D.

### Step 3: RL Post-Training (GRPO)
Apply GRPO to each pretrained model. Measure whether pass@1 after RL converges to pass@k or exceeds it. Replicate the SGE paper's finding in our controlled setting.

### Step 4: Difficulty Mixing Analysis
During RL, compare:
- Training on all chain lengths simultaneously
- Training on each chain length separately
- Curriculum (easy → hard)

Measure whether mixing causes ray interference on 5-step chains.

### Step 5: Gradient Diagnostics
Instrument the RL training loop to log:
- Fraction of negative-advantage vs positive-advantage samples per batch
- Gradient norms from easy vs hard samples
- Per-chain-length learning curves

Test whether adaptive clipping (DAPO-style) alleviates the imbalance.
