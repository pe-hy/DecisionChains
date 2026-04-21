# naive_ft — Finetuning Baselines for the Memory Experiment

Standard finetuning baselines to compare against the external-memory
approach in `../memory_experiment/`. We want to know: **does ordinary
finetuning match or beat the memory approach, and at what cost?**

The answer is important before claiming the memory method "works well".
If full FT + standard supervision matches the memory at similar compute,
the memory's value is mostly in parameter efficiency. If full FT even with
our best supervision can't match the memory, the mechanism itself is
doing something nontrivial.

---

## What this folder does

For the same alignment goal (steer the frozen base model toward always
picking `f`'s letter at every decision point), we train the **base
model's own weights** (full FT) or **low-rank adapters on top of it**
(LoRA) instead of an external memory module.

Four variants × one metric table = direct comparison.

### Two mechanisms

- **Full finetune** (`--method full`) — all 30M base-model weights become
  trainable. Standard pretrained-model finetuning.
- **LoRA** (`--method lora`) — wrap the base model with low-rank
  adapters (`peft.LoraConfig`) on the attention's `query_key_value` and
  `dense` projections. Only the adapter params train (~150K at rank=8,
  ~600K at rank=32); base model stays frozen.

### Two supervision modes

- **Mode A** (`--mode A`) — data = ffff only, labels = GT (unchanged).
  This is the "most naive" baseline: show the model only successful
  f-path traces and let it learn by imitation. No custom loss trickery.
- **Mode B** (`--mode B`) — data = all traces (any f/g coin-flip
  pattern), labels = GT except at decision-point letter positions where
  we override the GT token to `f(current_intermediate_vec)`'s letter id.
  This is the exact supervision signal the memory approach uses —
  including it separately here isolates **mechanism** from
  **supervision**: if Mode B (matching supervision) still underperforms
  memory, the memory architecture itself is doing work.

A-vs-B is the interesting axis. On ffff-only data the override is a
no-op (GT is already f), so we only ever pair A with ffff data and B
with all data.

The 2×2 grid:

|  | Mode A (ffff + GT) | Mode B (all + f override) |
|---|---|---|
| Full FT | `FT_full_A` | `FT_full_B` |
| LoRA r=8 | `FT_lora_A` | `FT_lora_B` |

Plus the memory winner from `../memory_experiment/` for context.

---

## How training works — the process in detail

### Shared steps across all four variants

1. **Load base model + tokenizer** from
   `../memory_experiment/checkpoint/12l-8h-512d-decision-chains-ext_6_2M/hf/`.
   Reuses `load_model_and_tokenizer` from
   `../memory_experiment/exp.py` so the setup is bit-identical.

2. **Wrap (LoRA only).** For `--method lora`, wrap the model via
   `peft.LoraConfig(r=rank, lora_alpha=alpha,
   target_modules=["query_key_value", "dense"], task_type="CAUSAL_LM")`
   and `get_peft_model(...)`. All base weights stay frozen. Only the
   adapter matrices `A_i ∈ ℝ^{r×d}` and `B_i ∈ ℝ^{d×r}` for each
   targeted `W_i` train, representing an update `ΔW_i = α·B_i·A_i/r`
   applied additively at each forward pass.

3. **Unfreeze (full FT only).** `for p in model.parameters():
   p.requires_grad = True`. All 30M params train with AdamW.

4. **Prepare data.** Same tokenization and label scheme as
   `../memory_experiment/exp.py`:
   - Tokenize as `[BOS] {input} [TRACE] {output} [EOS]`.
   - Pad to uniform length with `PAD_TOKEN_ID=89`, build
     `attention_mask` from non-pad positions.
   - Labels = input_ids with **−100 on prompt and pad tokens** (so
     cross-entropy ignores them) and with DP target overrides applied
     if Mode B.
   - DP positions = the `[TRACE]` token and every `;` token after it
     (at those indices the *next* token is a letter).
   - In Mode B: at each DP letter position we replace the GT label
     with `f(current_intermediate_vec)`'s token id, where the
     intermediate vector is computed by walking the GT trace under
     teacher forcing.

5. **Train.** AdamW with linear warmup (10% of steps) + cosine decay.
   Gradient clipping at 1.0. Cross-entropy loss over the whole
   supervised slice — `ce_mode="full_seq"` in all four variants so the
   loss includes arithmetic tokens, not just DP letters. That keeps
   the model from breaking arithmetic while it learns to steer letter
   choice.

6. **Evaluate** on two val slices:
   - `val_fonly` — ffff subset (`complete_solution` is meaningful
     here because GT OUTPUT equals the f-path output).
   - `val_full` — all coin-flip patterns (the honest alignment test).
   Both at `n_eval=200` by default. Greedy autoregressive generation
   up to 512 tokens. Same `evaluate()` function as the memory runs.

7. **Save** the result JSON to `outputs/experiments/{name}.json`. Schema
   is **intentionally identical to `../memory_experiment/`** so the
   `compare.py` there works cross-folder. The field names
   `memory_fonly` / `memory_full` in the JSON actually hold
   finetuned-model eval — this is a schema-compat hack.

### The loss — exact formula

For a single training example with sequence of tokens `x_0, x_1, ..., x_{T-1}`:

```
  L = (1/|S|) · Σ_{t ∈ S} CE( logits_t,  label_t )
```

where:
- `logits_t ∈ ℝ^vocab` is the model's next-token distribution at position `t`
- `label_t` is the target for predicting position `t+1`:
  - `-100` (ignored) if `t` is before or equal to the `[TRACE]` token,
    or if `x_{t+1}` is padding
  - `x_{t+1}` (the GT token) for non-DP output positions
  - **Mode B only**: if `t+1` is a decision-point letter position,
    `label_t = f(current_intermediate_vec_at_step_k)`'s token id instead
    of `x_{t+1}`
  - **Mode A**: `label_t = x_{t+1}` at every supervised position, and
    the data is pre-filtered to ffff so `x_{t+1}` is already f's letter
    at DP positions
- `S` = set of positions `t` where `label_t ≠ -100`

In Mode A this is just causal LM loss on ffff traces. In Mode B it's
causal LM loss everywhere *except* at DP letter positions, where the
label has been swapped from the GT letter to `f(v)`'s letter.

With `ce_mode="full_seq"` (always used here), `|S|` is ~150 per
example — 3-5 DP letters + the ~145 arithmetic / result / separator
tokens in the output. So arithmetic tokens dominate the loss (~97%),
which is what keeps the model's arithmetic behavior intact while a
few DP-target tokens pull letter choice toward f.

### Optimizer specifics

- **AdamW** with `weight_decay=0.01` (default), β=(0.9, 0.999).
- **Learning rate schedule**: linear warmup for the first 10% of steps
  (`get_cosine_schedule_with_warmup` from HuggingFace), then cosine
  decay to 0 over the remaining 90%.
- **Gradient clipping**: `clip_grad_norm_` at 1.0.
- **Batch size**: 4 by default (small, because the memory winner used
  bs=4 so we match for comparability).
- **LR defaults**: 5e-5 for full FT, 5e-4 for LoRA. Full FT needs a
  small LR to avoid catastrophic forgetting; LoRA tolerates 10× higher
  because the adapter starts at zero and its effective update is
  scaled by `α/r`.

### Baseline caching

`../memory_experiment/` already computed baseline metrics (no memory,
just the frozen pretrained model) at several `n_eval` settings and
cached them to
`../memory_experiment/outputs/experiments/_baseline_cache_n{N}.json`.
`ft.py` reuses that cache automatically (same base model, same val
data). First run at a new `n_eval` pays the cache cost locally; later
runs reuse it.

---

## Comparison methodology

### The headline table (produced by `compare_vs_memory.py`)

| Method | Trainable params | Train time | full f_sel Δ | full full_align Δ | full op_acc Δ |
|---|---|---|---|---|---|
| Memory (winner, 3 seeds) | 330K | 170s | +0.445 | +0.794 | −0.012 |
| FT_full_A | 30M | ?s | ? | ? | ? |
| FT_full_B | 30M | ?s | ? | ? | ? |
| FT_lora_A | ~150K | ?s | ? | ? | ? |
| FT_lora_B | ~150K | ?s | ? | ? | ? |

All entries at matched compute (`n_train=3000, epochs=10, batch_size=4`).

### What "better" means

- **Full f_selection Δ** — primary alignment metric. Higher is better.
- **Full full_f_alignment Δ** — strict per-example alignment. Higher is
  better.
- **Full operation_accuracy Δ** — arithmetic correctness. Must not
  regress; smaller negative Δ is better.

### What we expect to learn

- If **Mode A of either method** matches the memory, vanilla
  finetuning on restricted data is enough and the memory's override
  trick was unnecessary.
- If **Mode A underperforms but Mode B matches**, the winning
  ingredient is the `dp_target=f` supervision, not the mechanism.
- If **even Mode B underperforms the memory**, the memory's
  layer-2 residual-stream injection is doing something the model
  weights can't easily replicate.
- **Trainable-param cost**: memory ≈ 330K; LoRA r=8 ≈ 150K; LoRA r=32 ≈
  600K; full FT = 30M. The memory and LoRA r=8 are at similar param
  counts — that's the closest apples-to-apples.

---

## Files

| File | Role |
|---|---|
| `README.md` | This file. |
| `PLAN.md` | The minimal-start plan (4 runs) with expansion options. |
| `ft.py` | Unified trainer. Imports data prep + eval from `../memory_experiment/exp.py`. |
| `run_ft_quick.sh` | 4-run minimum pass at `n_train=3000, epochs=10`. |
| `compare_vs_memory.py` | Headline table: memory winner vs each FT variant. |
| `metrics.py` | Symlink to `../memory_experiment/metrics.py`. |
| `outputs/experiments/` | Per-run JSONs (schema identical to memory_experiment). |

## Usage

```bash
# Smoke test (verifies pipeline end-to-end, ~4 min on one GPU)
CUDA_VISIBLE_DEVICES=1 python ft.py --name smoke \
    --method full --mode A --lr 5e-5 \
    --n_train 100 --epochs 1 --n_eval 30

# 4-variant quick pass (~40 min total)
CUDA_VISIBLE_DEVICES=1 bash run_ft_quick.sh

# Headline table vs memory
python compare_vs_memory.py
```

The quick-pass script is hardcoded to GPU 1 (`CUDA_VISIBLE_DEVICES=1`
at the top) because GPU 0 is typically busy running the memory sweeps.
Change the env var if that's not true for you.

## Status

Scaffolding done, smoke test passed (full FT, mode A, n_train=100,
epochs=1 — produced a valid JSON with all metrics). Ready to run
`bash run_ft_quick.sh`.
