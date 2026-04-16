# validate_pretrained

Standalone, single-file validator for a pretrained decision-chain GPT-NeoX
model. Everything needed to load the model, run inference on `val.json`, score
the output, and optionally perturb the trace is in one Python file so it can
be debugged end-to-end without touching the rest of the repo.

## Layout

| File | Purpose |
| --- | --- |
| [validate.py](validate.py) | All logic: inlined transformations, decision functions, parser, scorer, HF loader, batched greedy generation, intervention. |
| [test_validate.py](test_validate.py) | 13 focused unit tests (parser, scorer, letter-position finder, intervention splicing, plus a round-trip against `val.json`). |
| [run_validate.sh](run_validate.sh) | Baseline validation wrapper. Edit the variables at the top and run. |
| [run_perturb.sh](run_perturb.sh) | Same wrapper but with a forced letter at a chosen step (trace perturbation). |
| [README.md](README.md) | This file. |

## The task (recap)

Each `val.json` example has a prompt of the form

```
INPUT : [ v0 , v1 , v2 , v3 , v4 , v5 ] OUTPUT : [ w0 , w1 , w2 , w3 , w4 , w5 ]
```

and a target trace: a `;`-separated sequence of blocks, each starting with a
letter `a..t` that selects one of 20 elementwise / permutation transformations
applied modulo 10. At every step, the letter is chosen by one of two decision
functions applied to the *current* vector:

- `f(L) = is_even(L[0]) * 10 + L[1]`  (0..19 → letter a..t)
- `g(L) = is_even(L[3]) * 10 + L[4]`  (0..19 → letter a..t)

Which of `f` or `g` is used at a given step was a coin flip at data-generation
time, so either answer is valid. Chains are 3, 4, or 5 steps; the final
transformed vector equals the prompt's `OUTPUT` vector.

## How validation works

1. Load the HF checkpoint (defaults to
   `outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M`) and the custom
   word-level tokenizer.
2. For each example, build the prompt `"[BOS] {input} [TRACE]"` and run
   batched **greedy** (`do_sample=False`, `num_beams=1`) generation up to
   `max_length` tokens.
3. Decode the tail after `[TRACE]`, strip at `[EOS]`, and split the trace on
   `" ; "` into blocks.
4. For every block the model produced, score it against a local re-execution
   of the letter's transformation (see metrics below).
5. Aggregate and print a multi-section report.

The Python transformations inlined in `validate.py` are byte-for-byte
reproductions of the ones in [`ops/transformations.py`](../ops/transformations.py).
A round-trip test re-runs every ground-truth step in the first 500 val.json
examples to prevent drift.

## Metrics

All three headline metrics are computed per example and rolled up. The scorer
walks the predicted blocks left-to-right, maintaining a `current_vec` that
starts from the prompt's `INPUT` vector and advances after each validated
step (falling back to the model's parsed result vector if the block was
invalid, so later steps can still be scored).

### operation_accuracy

For each block: re-execute the block's letter on `current_vec` and compare the
**entire block text** (process + result vector) against what the model wrote.
This catches both arithmetic errors ("8 + 4 = 3") and wrong result vectors. A
block is "op-correct" iff the whole string matches exactly.

Reported as:
- `overall.operation_accuracy` = (sum of correct blocks) / (total predicted blocks)
- Per ground-truth chain length
- Per step index
- Per predicted letter (which transformations the model executes best/worst)

### operation_selection

For each block: compute `f(current_vec)` and `g(current_vec)`. A block is
"selection-correct" iff its letter equals either `f`'s or `g`'s answer. This
is the metric that measures whether the model made a *valid branching choice*
at each decision point, independent of whether it then executed the chosen
operation correctly.

Reported as:
- `overall.operation_selection` = (sum of correct selections) / (total predicted blocks)
- Per ground-truth chain length
- Per step index

### complete_solution

A single example is "completely solved" iff **all** of the following hold:

1. At least one block was produced.
2. Every block is operation-correct.
3. Every block is selection-correct.
4. The final parsed result vector equals the prompt's `OUTPUT` vector.

Reported as the fraction of examples where all four hold.

### Supporting metrics

- `parseable_fraction` — examples whose prompt parsed cleanly into INPUT/OUTPUT.
- `chain_matches_output` — fraction with final parsed vector == prompt OUTPUT.
  Decoupling this from `complete_solution` lets you see cases where the chain
  *reaches* the right answer but uses an invalid letter or a broken arithmetic
  step along the way.
- `length_matches_gt` — fraction where the model produced exactly as many
  blocks as the ground truth. In practice this is often much lower than
  `operation_accuracy`: the model tends to generate *extra valid* steps past
  the target, which keeps per-block accuracy high but tanks the complete
  solution rate.

### Report sections

Every run prints four sections (see `format_report` in `validate.py`):

```
=== Overall (...) ===
  num_examples, total_predicted_ops, parseable_fraction,
  operation_accuracy, operation_selection,
  chain_matches_output, length_matches_gt, complete_solution

=== By ground-truth chain length ===
  length | n | op_acc | sel_acc | len_ok | complete

=== By step index (over examples that reached that step) ===
  step | n | op_acc | sel_acc

=== By predicted letter (operation accuracy only) ===
  letter | n | op_acc
```

With `--show_samples N` it also prints up to `N` full traces (half correct,
half failing) with per-step flags. With `--dump_jsonl PATH` it writes every
example's prompt, GT, generation, and per-step booleans to a JSONL file.

## Running baseline validation

```bash
bash validate_pretrained/run_validate.sh
```

Top of the script exposes:

| Variable | Meaning |
| --- | --- |
| `MODEL_PATH`       | HF checkpoint directory |
| `TOKENIZER_PATH`   | Custom word-level tokenizer JSON |
| `VAL_FILE`         | `val.json` to evaluate against |
| `NUM_EXAMPLES`     | How many examples from the head of val.json |
| `BATCH_SIZE`       | Greedy-generation batch size |
| `MAX_LENGTH`       | Upper bound on total sequence length (prompt + generation) |
| `DEVICE`           | `cuda` or `cpu` |
| `SHOW_SAMPLES`     | Print N sample traces after the report (0 to disable) |
| `DUMP_JSONL`       | Optional path for per-example JSONL dump |

Or invoke `validate.py` directly with the same flags — they're all standard
`argparse` options.

## Perturbing the trace

```bash
bash validate_pretrained/run_perturb.sh
```

Set `INTERVENE_STEP` (0-indexed) and `INTERVENE_LETTER` (a..t). The script
runs the model **twice** per batch:

1. **Pass 1 — locate**: greedy generation as normal. The generated tokens are
   scanned to find the letter token at the target step:
   - step 0  = the first token after `[TRACE]`
   - step k  = the token immediately after the k-th `;` in the generation
   If the model's trace has fewer than `k+1` blocks, that example is skipped
   (its baseline output is kept for scoring).
2. **Pass 2 — splice and continue**: build a new prefix
   `prompt_ids + gen_ids[:letter_pos] + [forced_letter_tid]`, left-pad and
   batch these, and call `model.generate()` again. The model writes the rest
   of the trace conditioned on the forced letter.

The final trace for each perturbed example is
`gen_ids[:letter_pos] + [forced_letter] + continuation`. This is then scored
with the same three metrics. Expected behaviour:

- `operation_accuracy` at the intervened step drops *only* if the model fails
  to correctly execute the forced transformation. The model was trained on
  trace blocks for all 20 letters, so it typically executes the process
  correctly even for a forced letter.
- `operation_selection` at the intervened step is 0 whenever the forced
  letter is neither `f`'s nor `g`'s answer (which is the common case).
- `complete_solution` usually drops to ~0 because the final vector no longer
  matches the prompt's `OUTPUT` — which is the point: it tells you how much
  the downstream chain depends on the intervened decision.

Only one intervention step is applied per run; cascading perturbations would
need a follow-up pass.

## Why each design choice

- **Greedy decoding.** Makes everything reproducible and the metrics
  interpretable: any variation between runs is a model/data issue, not
  sampling noise.
- **Block text comparison (not just the result vector).** A model can write a
  correct result vector while fabricating the arithmetic that got there
  (`"8 + 4 = 3 , ... R [correct]"`). Full-block match catches those.
- **Fall back to parsed `vec` when a block is invalid.** If we zeroed out
  `current_vec` on the first error, all downstream steps would trivially
  fail and the per-step and per-letter breakdowns would be useless.
- **`chain_matches_output` reported separately from `complete_solution`.**
  The model often reaches the right answer with an *invalid* letter at some
  earlier step (selection broken, process correct). Separating the two lets
  you see this.
- **Two-pass intervention.** Simpler and cleaner than hand-rolling a
  step-by-step decoder with a state machine, and uses the same batched
  `model.generate()` code path as the baseline.

## Tests

```bash
python -m pytest validate_pretrained/test_validate.py -v
```

13 tests, ~4s:

- `test_all_letters_round_trip_with_val_json` — re-executes every ground-truth
  step in the first 500 val.json examples. If any inlined transformation
  diverges from the data-generation code, this blows up with the exact
  mismatch. **Run this any time you touch the transformations.**
- `test_decision_functions_cover_full_range` — sanity for `f` and `g`.
- `test_extract_input_output` / `test_parse_blocks_mixed_formats` /
  `test_parse_blocks_malformed_block_keeps_slot` — parser.
- `test_score_example_ground_truth_is_fully_correct` — end-to-end scorer on a
  synthetic correct trace.
- `test_score_example_wrong_letter_breaks_selection_not_process` — a forced
  invalid letter correctly executed scores op=1, sel=0, complete=False.
- `test_score_example_tampered_result_vector` — flipping the R-vector drops
  op_correct but keeps selection.
- `test_aggregate_scores_has_all_sections` — verifies all four report
  sections exist with correct counts and keys.
- `test_find_letter_position_step0` / `..._after_semicolons` /
  `..._trailing_semicolon_returns_none` — letter-position finder edge cases.
- `test_build_intervened_prefix_splices_cleanly` — prefix splicing math.

## Notes and caveats

- `val.json` is loaded from the head (`examples[:NUM_EXAMPLES]`). The file is
  already shuffled by the data generator, so this is fine for a
  representative slice, but it's not a random subsample.
- `max_length` is the **total** sequence length (prompt + generation). The
  training config uses 512, which is enough for all chain lengths in this
  dataset; bump it if you retrain on longer vectors.
- `bfloat16` / `float16` generation is available by passing a different
  `dtype` into `load_model_and_tokenizer` — the default is `float32` for
  numerical stability in debugging.
- The tokenizer is deliberately loaded from the standalone JSON rather than
  the HF model directory, to match the training pipeline exactly.
