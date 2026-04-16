# Metrics Plan for External Memory Evaluation

## Existing implementation: `validate_pretrained/validate.py`

The metrics we need already exist in `validate.py`. It defines three core per-example metrics via `score_example()`:

1. **operation_accuracy** (`op_correct`): For each block, re-execute the chosen letter's transformation on the previous vector. Does the process text AND result vector match?
2. **operation_selection** (`sel_correct`): Is the chosen letter a valid output of decision function f or g applied to the current vector?
3. **complete_solution** (`full_correct`): All operations correct AND all selections correct AND final result vector equals the OUTPUT vector from the prompt.

It also supports **trace intervention** out of the box:
```bash
python validate.py --intervene_step 1 --intervene_letter m
```
This forces letter `m` at step 1, lets the model free-generate the rest, and re-scores with the same metrics. The JSONL dump includes post-intervention metrics (op/sel accuracy for steps AFTER the intervention).

## How this maps to our external memory experiment

`validate.py` tests **oracle intervention** (directly forcing a token during generation). Our memory experiment will test **soft intervention** (modifying hidden states via external memory to steer the letter choice). The scoring metrics are the same — only the intervention mechanism differs.

### Metrics to use (aligned with validate.py)

| Metric | validate.py field | What it measures |
|--------|-------------------|------------------|
| **Operation accuracy** | `operation_accuracy` | Per block: is the arithmetic correct? |
| **Operation selection** | `operation_selection` | Per decision point: is the letter a valid f or g output? |
| **Complete solution** | `complete_solution` | All ops correct + all selections correct + final vec = OUTPUT |
| **Post-intervention op accuracy** | `post_op_rate` | Op accuracy for steps AFTER the intervention |
| **Post-intervention sel accuracy** | `post_sel_rate` | Selection accuracy for steps AFTER the intervention |

### Additional breakdowns (already in validate.py)

- **By chain length**: metrics per ground-truth chain length (3, 4, 5)
- **By step index**: per-step operation accuracy and selection accuracy
- **By letter**: per-letter operation accuracy (does the model compute each function correctly?)
- **Chain length match**: did the model produce the expected number of steps?

## The cascading problem

Changing one letter is NOT a local edit:
- The computation in block k changes → different result vector v_k'
- Decision point k+1 sees v_k' → may select a DIFFERENT letter (correctly, via f/g)
- This cascades through all subsequent blocks

So we can't check "are other letters the same?" — they won't be. Instead, `operation_selection` checks whether each downstream letter is a valid f-or-g choice given the NEW intermediate vector. This is the right metric.

## What we reuse vs. what we build

**Reuse from validate.py:**
- `score_example()` — per-example scoring
- `aggregate_scores()` — roll up to summary stats
- `format_report()` — human-readable output
- `parse_generated_blocks()` — trace parsing
- `apply_letter()` — re-execute transformations
- `decision_f()`, `decision_g()` — decision functions

**Build new (for external memory):**
- The memory injection mechanism (modify hidden states at a specific layer at decision-point positions)
- A wrapper that: (1) runs forward pass with memory injection, (2) generates the perturbed trace, (3) feeds it to `score_example()` for evaluation
- Comparison: run the same examples with and without memory injection, diff the metrics

## Design decisions to vary

- **Which layer** to inject the memory into (residual stream after layer 1? 6? 12?)
- **How** to learn the value vector (gradient descent to flip one letter)
- These metrics are layer-agnostic — same `score_example()` evaluation regardless of injection method
