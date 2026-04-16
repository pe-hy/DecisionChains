# Plan: External Memory Injection Experiment

## Concrete example

Take this trace from val.json:

```
Prompt:  [BOS] INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ] OUTPUT : [ 2 , 0 , 8 , 4 , 3 , 9 ] [TRACE]
                                                                                        ↑ pos 31

Model generates:
  n : 4 + 4 = 8 , 3 + 4 = 7 , ... R [ 8 , 7 , 2 , 1 , 1 , 8 ] ; r : ... R [ ... ] ; s : ... ; q : ...
  ↑ pos 32 (first letter)
```

In a causal LM, **the hidden state at position 31 (`[TRACE]`) produces the logits that predict position 32 (letter `n`)**. The decision functions give:
- `f([4,3,8,7,7,4])` → letter `n` (add_first_to_all)
- `g([4,3,8,7,7,4])` → letter `h` (swap_pairs)

The model chose `n`. Say we want to force it to choose `m` (rotate_right) instead.

### Step 1: Learn the value vector V

We tokenize the FULL sequence (prompt + trace) for teacher forcing. Then:

```
At layer L (say layer 6), position 31:
    h[31]' = h[31] + V          ← V is a learnable 512-dim vector, initialized to zeros

The model forward pass uses h[31]' instead of h[31].
The logits at position 31 now predict a DIFFERENT distribution over the next token.

Loss = CrossEntropy(logits[31], target="m")

We optimize V via Adam for ~200 steps until the model confidently predicts "m" at position 31.
```

V is the ONLY thing being optimized. The model weights are frozen.

### Step 2: Generate with V

Now we generate autoregressively from the prompt, with the same hook active:

```
Prompt: [BOS] INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ] OUTPUT : [ 2 , 0 , 8 , 4 , 3 , 9 ] [TRACE]

Forward pass on prompt → at layer 6, position 31, add V → logits predict "m"

Model generates: m R [ 4 , 4 , 3 , 8 , 7 , 7 ] ; ...
                 ↑ rotate_right applied to [4,3,8,7,7,4]

Now the model continues from [ 4 , 4 , 3 , 8 , 7 , 7 ] — the NEW intermediate vector.
The hook is only active at position 31, so all subsequent tokens are generated normally.
The model must figure out the correct next letter for this new vector using its own f/g logic.
```

### Step 3: Score with metrics.py

```python
score = score_trace(
    input_vec=[4, 3, 8, 7, 7, 4],
    output_vec=[2, 0, 8, 4, 3, 9],
    generated_text="m R [ 4 , 4 , 3 , 8 , 7 , 7 ] ; ..."
)
```

Check:
- **Target success**: Did the model output `m` at step 0? (yes/no)
- **operation_accuracy**: Is each block's arithmetic correct?
- **operation_selection**: At each subsequent step, does the model pick a valid f/g letter for the new intermediate vector?
- **complete_solution**: All correct AND final vec matches OUTPUT `[2,0,8,4,3,9]`?

Note: `complete_solution` will often be False after perturbation because the chain now follows a different path — `m` doesn't lead to the same output as `n`. That's expected. The key metric is whether the model's **program stays intact** (operation_selection and operation_accuracy remain high).

## Scope: first decision point only

For simplicity, we only target the **first decision point** (position of `[TRACE]`). This position is fixed and known before generation — no need to detect `;` tokens mid-generation.

Later decision points would require tracking positions during autoregressive generation, which adds complexity without changing the core experiment.

## What the discussion says

The mechanism (lines 1-3, 8):
1. Take the hidden state at some layer at a decision-point position
2. Add a learned **value vector** to the residual stream at that position
3. This steers the model's next-token prediction toward a different letter
4. The value vector is learned via gradient descent — "it will take maybe a minute" (line 10)

The evaluation (lines 9-10, 13-15, 23, 25):
- Take ~100 traces
- At one decision point, inject the value vector to change the letter
- Measure: did the target letter change? Does the program stay intact? (metrics.py)
- Try multiple layers (line 14)

## Script: `memory_inject.py`

Single file, uses `metrics.py` for scoring.

```
memory_inject.py
├── load_model_and_tokenizer()      # reuse checkpoint loading
├── find_trace_position()           # find [TRACE] token position in tokenized prompt
├── learn_value_vector()            # gradient descent on V for one (example, target_letter, layer)
├── generate_with_injection()       # autoregressive generation with hook adding V at [TRACE] position
├── main()                          # loop over examples × layers, print results
```

### learn_value_vector(model, full_ids, trace_pos, target_letter_id, layer, lr, steps)

```python
V = torch.zeros(hidden_dim, requires_grad=True)
optimizer = Adam([V], lr=lr)

def hook_fn(module, input, output):
    output[0][:, trace_pos, :] = output[0][:, trace_pos, :] + V

handle = model.gpt_neox.layers[layer].register_forward_hook(hook_fn)

for step in range(steps):
    optimizer.zero_grad()
    logits = model(full_ids).logits[0, trace_pos]
    loss = F.cross_entropy(logits.unsqueeze(0), target_letter_id.unsqueeze(0))
    loss.backward()
    optimizer.step()

handle.remove()
return V.detach()
```

### generate_with_injection(model, tokenizer, prompt_ids, V, trace_pos, layer)

```python
def hook_fn(module, input, output):
    output[0][:, trace_pos, :] = output[0][:, trace_pos, :] + V

handle = model.gpt_neox.layers[layer].register_forward_hook(hook_fn)
# prompt_ids ends at [TRACE], so trace_pos = len(prompt_ids) - 1
generated = model.generate(prompt_ids, max_length=512, do_sample=False)
handle.remove()

# decode the part after the prompt
trace_text = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True)
return trace_text
```

Note: during generation, the first forward pass processes the full prompt (positions 0..trace_pos). The hook fires and adds V at trace_pos. After that, tokens are generated one-by-one and the hook still fires but trace_pos is within the KV cache — **the injection only affects the first pass**.

### Target letter selection

For each example, we pick a target letter that is:
1. Different from what the model originally chose
2. A valid f or g output for the input vector (so we know the "program" can support it)

If f→`n` and g→`h`, and the model chose `n`, we target `h` (the other valid choice). This is the gentlest possible perturbation — we're asking the model to switch from one valid branch to the other.

### What to report

Per layer:
```
Layer | Target hit | op_accuracy | sel_accuracy | complete_solution
------|-----------|-------------|-------------|------------------
  1   |   0.XX    |    0.XX     |    0.XX     |      0.XX
 ...
 12   |   0.XX    |    0.XX     |    0.XX     |      0.XX

Baseline (no injection):
      |     -     |    0.XX     |    0.XX     |      0.XX
```

## Args

```
--n_examples 100
--layers 1,6,12        # or "all" for 1-12
--lr 0.01
--steps 200
--device cuda
```

## Dependencies

- `metrics.py` — scoring
- Model checkpoint in `checkpoint/12l-8h-512d-decision-chains-ext_6_2M/hf/`
- Val data in `../../outputs/data/decision_chains_extended/val.json`
