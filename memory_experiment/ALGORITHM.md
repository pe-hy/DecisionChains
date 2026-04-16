# Residual Stream Injection — Algorithm

## What it does

We have a pretrained transformer that generates decision chains. At each decision point, it picks a letter (a-t) representing a transformation function. Sometimes the model picks the wrong letter. We want to **surgically correct** one decision point by adding a learned vector to the model's internal representation, without retraining the model.

## The mechanism

At a decision point, the token before the letter (`;` or `[TRACE]`) has a hidden state h at each layer. The logits computed from h determine which letter comes next. We learn a single vector V (512 dims) and add it to h at one specific layer:

```
h' = h + V
```

This shifts the logits so the model predicts a different letter. V is optimized via gradient descent (model weights stay frozen).

## Concrete example

```
Input:  [ 4 , 3 , 8 , 7 , 7 , 4 ]
Output: [ 2 , 0 , 8 , 4 , 3 , 9 ]

Decision functions on this vector:
  f([4,3,8,7,7,4]) → letter "n"  (add_first_to_all)
  g([4,3,8,7,7,4]) → letter "h"  (swap_pairs)

Baseline generation (no injection):
  n : 4+4=8, 3+4=7, 8+4=2, 7+4=1, 7+4=1, 4+4=8 R [8,7,2,1,1,8]
  ; r : ... R [...] ; s : ... R [...] ; q : ... R [2,0,8,4,3,9]
  ↑ model chose "n" (the f-branch)

We want to switch to "h" (the g-branch):
  1. Tokenize full sequence. [TRACE] is at position 31.
  2. Initialize V = zeros(512).
  3. Register hook on layer 6: h[31] = h[31] + V
  4. Forward pass → logits at pos 31 → loss = CrossEntropy(logits, "h")
  5. Backprop through loss → update V (only V, model frozen)
  6. Repeat for 200 steps. V converges to |V| ≈ 3.0, P("h") ≈ 1.0.

Generation with V injected:
  h : (4,3)→(3,4), (8,7)→(7,8), (7,4)→(4,7) R [3,4,7,8,4,7]
  ; ... (model continues freely with its own f/g logic on the new vector)

Result: target hit ✓, downstream op_accuracy=6/9, sel_accuracy=8/9
```

## Can this correct errors at arbitrary decision points?

**Yes, with a modification.** Currently `memory_inject.py` only targets the first decision point (the `[TRACE]` position) because it's at a fixed position in the prompt.

To correct an error at decision point k (e.g., step 3), the approach would be:

### Step 1: Identify the divergence point

Run baseline generation, score with `metrics.py`. The `per_step_sel` field tells you exactly where the model picked a letter that isn't a valid f/g output:

```python
score = score_trace(input_vec, output_vec, generated_text)
# score.per_step_sel = [True, True, False, True, True]
#                                    ↑ divergence at step 2 (0-indexed)
```

### Step 2: Locate the token position

The divergence is at step k. In the token sequence, the decision point for step k is the `;` token before the k-th letter. During generation, we need to:

1. Let the model generate freely up to the `;` before step k
2. At that `;` position, inject V to steer the next letter

### Step 3: Two-phase generation

This is the same approach `validate_pretrained/validate.py` already uses for intervention:

```
Phase 1: Generate normally up to the ; before step k
          → gives us prompt_ids + gen_ids[:semicolon_pos]

Phase 2: Build a new prefix = prompt + generated_up_to_semicolon
          Learn V on this prefix (teacher-forced with correct continuation)
          Generate from this prefix with V injected at the semicolon position
```

The key difference from the current script: the injection position isn't fixed — it depends on how long the model's generation is up to step k. But the mechanism (learn V, add to residual stream, generate) is identical.

### Implementation sketch

```python
# 1. Generate baseline
baseline_text = generate_baseline(model, tokenizer, prompt_ids, device)
baseline_score = score_trace(input_vec, output_vec, baseline_text)

# 2. Find first wrong step
for step, sel_ok in enumerate(baseline_score.per_step_sel):
    if not sel_ok:
        break  # step = divergence point

# 3. Find the ; position before that step in the generated tokens
gen_ids = tokenizer.encode(baseline_text, add_special_tokens=False)
semicolon_positions = [i for i, t in enumerate(gen_ids) if t == semicolon_id]
inject_pos = len(prompt_ids) + semicolon_positions[step - 1]  # position of ; before step k

# 4. Build prefix up to that point, learn V, generate with V
prefix = prompt_ids + gen_ids[:semicolon_positions[step - 1] + 1]
# ... same learn_value_vector / generate_with_injection logic
```

This is a natural extension — the current script proves the mechanism works at the first decision point, and the same approach generalizes to any step.
