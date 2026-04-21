# Memory-Steered Decision Chains

A small external **key-value memory** is added to a frozen pretrained
transformer at inference time. The memory's job is to **bias the model's
choice at decision points** without any base-model fine-tuning.

This README is the single source of truth. Other docs were retired —
read this one before touching code.

---

## 1. The base task

The pretrained model is a **12L-8H-512D GPT-NeoX** trained to execute
**decision chains**: a sequence of 3–5 vector transformations.

At each chain step the model:

1. Reads the current intermediate vector `v`.
2. Picks one of two letters: `f(v)` or `g(v)`. Each maps the vector to one of
   20 transformation letters `a..t` (e.g. `n` = add_first_to_all,
   `m` = rotate_right). The decision functions are
   `f(v)=is_even(v[0])*10+v[1]` and `g(v)=is_even(v[3])*10+v[4]`.
3. Applies the chosen letter's transformation to `v` and emits the new vector.

In pretraining the coin flip between `f` and `g` is uniform, so the model
learns **both** are valid. We want to **align it to always pick `f`**, using
a tiny add-on memory rather than fine-tuning the 30M-param base.

Example prompt (no chain yet):
```
[BOS] INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ] OUTPUT : [ 2 , 0 , 8 , 4 , 3 , 9 ] [TRACE]
```
The model autoregressively produces the trace after `[TRACE]`. At the prompt's
last position (the `[TRACE]` token), the next token is **the first letter**.
After each block, the next position (the `;`) selects the **next** letter.
These are the **decision points (DPs)**.

---

## 2. The memory mechanism

The base model is frozen. We hook a single layer `L` and add a residual
correction:

```
                       frozen layers 0..L-1
                              ↓
                       layer L output  h ∈ ℝ^{B×T×D}    (D=512)
                              │
                              ▼
            ┌──────────────── memory ────────────────┐
            │   q   = W_q · h                  ∈ ℝ^{B×T×D}
            │   attn = softmax(q · Kᵀ / √D)    ∈ ℝ^{B×T×N}    (N entries)
            │   out  = γ · (attn · V)          ∈ ℝ^{B×T×D}
            └────────────────────────────────────────┘
                              │
                              ▼
                       h' = h + out   ← injected back into the residual stream
                              ▼
                       frozen layers L+1..11 → logits
```

What's trained:

| Param | Shape | Role |
|---|---|---|
| `K` (keys) | `(N, D)` | Patterns: "this position needs steering" |
| `V` (values) | `(N, D)` | Correction vectors to add |
| `W_q` (query proj) | `(D, D)` | How to ask the memory from the hidden state |
| `γ` (gate, optional) | `(1,)` | Scalar magnitude on the correction |

Default `N=16` or `32`; total trainable params ≈ 270K (W_q dominates at 262K).
Everything else, including all attention and MLP weights of the base model,
**stays frozen**. Gradients flow through the frozen layers; they're treated as
fixed nonlinearities the gradient passes through.

The hook fires **at every token position** in the forward pass. At eval time,
during autoregressive generation, only the first forward pass over the prompt
adds the correction at the prompt positions; subsequent decoded tokens get the
correction added at their own positions as they're generated.

---

## 3. The loss

```
L = CE_target  +  λ · sparsity_term
```

### CE_target — what the model should output

Two configurable supervision modes (`--ce_mode`):

- **`dp_only`** — Cross-entropy only at DP letter positions. Target is set by
  `--dp_target`:
  - `gt`: the letter actually in the GT trace (mixed f/g if data is unfiltered).
  - `f`:  `f(current_intermediate_vec)` regardless of the GT coin flip.
  Intermediate vectors are computed by walking the GT trace under teacher
  forcing.

- **`full_seq`** — Cross-entropy on every token after `[TRACE]`. The whole
  trace is the target. This gives ~20× more gradient signal than `dp_only`
  because every arithmetic token also contributes. With `--dp_target f` and
  mixed data, the DP labels are overridden to `f`'s letter while arithmetic
  labels follow the GT trace.

### sparsity_term (`--sparsity`)

- **`none`** — no penalty. The memory is free to attend everywhere.
- **`l2_nondp`** — `mean(‖mem_out‖₂)` over **non-DP** positions (multiplied by
  `--sparsity_coeff`, default 0.1). Pushes the correction magnitude toward
  zero at arithmetic / prompt tokens, leaving CE to drive the magnitude up at
  DPs. Goal: make the correction **surgical** (only fires where it matters).

A historical note: an earlier version used `mean(|attn|)` on the softmax
output. Since softmax sums to 1 and is non-negative, that quantity is always
exactly `1/N` regardless of the parameters — zero gradient, a no-op. The
`l2_nondp` variant fixes this by penalizing the actual correction vector.

---

## 4. A worked example

Take `INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ]`. Decision functions give:

- `f([4,3,8,7,7,4]) = is_even(4)·10 + 3 = 13 → letter n` (add_first_to_all)
- `g([4,3,8,7,7,4]) = is_even(7)·10 + 7 =  7 → letter h` (swap_pairs)

Suppose the unaligned base model picks `h` (g-branch) because of how its 50/50
training shaped it. We want the aligned model with memory to pick `n`
(f-branch).

During training, with `--dp_target f`, the label at the `[TRACE]` position is
the token id for `n`. CE pushes the logits at that position toward `n`. The
memory learns `K`, `V`, `W_q` so that:

- At positions whose hidden state looks like a DP, the query matches some key
  strongly, and the corresponding value (added to `h`) shifts the next-token
  logits toward `f`'s letter.
- At arithmetic positions, no key matches (or `mem_out` magnitude is small),
  so the residual stream is unchanged and the arithmetic behavior is
  preserved.

At inference: prompt is fed in, hook is active, the model decodes greedy. At
the `[TRACE]` position the corrected logits make `n` the argmax. The model
then emits the full `n`-block: `n : 4+4=8 , 3+4=7 , 8+4=2 , 7+4=1 , 7+4=1 ,
4+4=8 R [ 8 , 7 , 2 , 1 , 1 , 8 ]`. Next DP is the `;` at the end of that
block; same memory fires, picks `f` at the new vector `[8,7,2,1,1,8]`, and so
on.

---

## 5. Train / val split (important)

The pretrained model's data was generated with **disjoint splits on letter
tuples and input vectors**:

- 144,297 unique letter tuples in train, 4,904 in val — **0 overlap**.
- Sampled input vectors: train and val have **0 overlap**.
- Decision-function patterns (`fff`, `fgfg`, ...): all 56 patterns appear in
  both — uniform random, independent of input/letter axes.

Practical implications:

- `val_fonly` (val examples with all-`f` coin flips): true compositional
  generalization on novel letter tuples & vectors. Use this when you need
  `complete_solution` to be meaningful (final vec = GT OUTPUT only matches on
  the f-path).
- `val_full` (no filter): the honest alignment metric — the same compositional
  generalization, but across **all** coin-flip patterns. This catches the
  off-path generalization issue: does the memory steer correctly even when
  the model is mid-trajectory on a non-f path?

Filtering training data to `ffff` is a coin-flip selection — it's
distributionally clean (input vectors and letter tuples are unaffected) but
it means the memory only ever sees hidden states from f-path trajectories.
Mixed training (`--data_filter all` with `--dp_target f`) is the methodologically
cleaner choice for steering at inference time.

---

## 6. Metrics

All defined in `metrics.py`. Computed by walking the model's own generated
trace; under the alignment-semantic, intermediate vectors are advanced by
correctly executing the model's chosen letter (so the next-step f/g check is
against the true next state, not the model's possibly-buggy arithmetic).

Primary (alignment):

- `f_selection` — fraction of steps where the model picked `f`'s letter.
- `full_f_alignment` — fraction of **examples** where every step picked `f`.
- `per_step_f_selection` — `f_selection` broken out by step index. Diagnoses
  whether the memory is only effective at the first DP (where position is
  fixed) or generalizes to mid-generation DPs.

Sanity:

- `operation_accuracy` — arithmetic correctness, must not drop.
- `f_or_g_valid_selection` — letter is in `{f_letter, g_letter}`. The
  pre-alignment "is the model producing valid decision-function output at
  all" check.

End-to-end (only meaningful on `ffff` val):

- `chain_matches_output` — final vec = GT OUTPUT vec.
- `complete_solution` — all ops correct **and** all sels valid **and** chain
  matches output.

---

## 7. Files

| File | Role |
|---|---|
| `exp.py` | Unified experiment runner. One invocation = one config = one JSON. |
| `compare.py` | Loads `outputs/experiments/*.json` and prints ranked tables. |
| `run_all_phases.sh` | Master overnight sweep: Phase 1 → Phase 2 → Phase 3 → summarize. |
| `run_phase1.sh` | 46-run screening sweep at `n_eval=50`. Tests every dimension at the new working point. |
| `run_phase2.py` | Takes Phase 1 JSONs, picks top-5 by full-val f_selection, confirms with 3 seeds at `n_eval=200`. |
| `run_phase3.py` | Combines best layer/sparsity/mem/lr/bs from Phase 1; runs compound × compute budgets × 3 seeds at `n_eval=300`. |
| `summarize_sweep.py` | Builds `outputs/final_results.md` from all phase JSONs. |
| `visualization/` | Static HTML viewer for all runs (`visualization/index.html`, regenerated via `visualization/build_data.py`). |
| `metrics.py` | Trace parsing, scoring, aggregation. |
| `checkpoint/12l-8h-512d-decision-chains-ext_6_2M/` | Pretrained LitGPT + HF copy. |
| `outputs/experiments/` | Per-experiment result JSONs (config, history, metrics). |
| `outputs/E1-E4_results.{md,txt}` | Pilot 2×2 comparison table. |
| `outputs/data_efficiency_results.txt` | 18-run data-efficiency sweep result. |
| `outputs/final_results.md` | Written by `summarize_sweep.py` after the full sweep. |
| `outputs/BEST_MODEL.md` | Summary of the best recipe from the overnight sweep: config, hyperparam explanations, absolute numbers vs 2nd-best and E4. |
| `old_scripts/` | Archived earlier sweep scripts superseded by `run_all_phases.sh`. |
| `logs/` | All stdout logs + archived `wandb/`. |
| `wandb/` | Current W&B run dir (runs launched via scripts go into `logs/wandb/` via `WANDB_DIR`). |
| `CLAUDE.md` | Pointer file for Claude Code; defers to this README. |
| `README.md` | This file. |
| `DIAGRAM.txt` | One-page ASCII diagram. Open this first. |

---

## 8. Running

```bash
# Single experiment — every dimension is a CLI flag.
CUDA_VISIBLE_DEVICES=0 python exp.py --name E1_ffff_dponly \
    --data_filter ffff --ce_mode dp_only --dp_target f \
    --layers 10 --mem_entries 16 \
    --sparsity l2_nondp --sparsity_coeff 0.1 \
    --n_train 4096 --n_eval 200 --epochs 20

# Full matrix (sequential on one GPU). Each writes outputs/experiments/<name>.json.
python exp.py --name E1_ffff_dponly  --data_filter ffff --ce_mode dp_only --dp_target f --layers 10 --mem_entries 16 --sparsity l2_nondp --sparsity_coeff 0.1 --n_train 4096 --n_eval 200 --epochs 20
python exp.py --name E2_ffff_fullseq --data_filter ffff --ce_mode full_seq --dp_target gt --layers 10 --mem_entries 32 --gate --sparsity none --n_train 4096 --n_eval 200 --epochs 20
python exp.py --name E3_all_dponly_f --data_filter all  --ce_mode dp_only --dp_target f  --layers 10 --mem_entries 16 --sparsity l2_nondp --sparsity_coeff 0.1 --n_train 4096 --n_eval 200 --epochs 20
python exp.py --name E4_all_fullseq_f --data_filter all --ce_mode full_seq --dp_target f  --layers 10 --mem_entries 32 --gate --sparsity none --n_train 4096 --n_eval 200 --epochs 20

# Compare everything you've run.
python compare.py
python compare.py --only E1_ffff_dponly,E4_all_fullseq_f
```

### Key flags reference

| Flag | Choices | Meaning |
|---|---|---|
| `--name` | str | Used as the JSON filename and table label. |
| `--data_filter` | `ffff` | `all` | Training data slice (val is always evaluated both ways). |
| `--ce_mode` | `dp_only` | `full_seq` | CE supervision scope. |
| `--dp_target` | `gt` | `f` | What to predict at decision-point letter positions. |
| `--layers` | comma list of ints | Layers to inject memory at (e.g. `10` or `6,10`). |
| `--mem_entries` | int | `N`, the number of K/V memory rows. |
| `--gate` | flag | Add a learnable scalar `γ` on `mem_out`. |
| `--sparsity` | `none` | `l2_nondp` | Surgicality penalty on `‖mem_out‖` at non-DP. |
| `--sparsity_coeff` | float | `λ` for the sparsity term. |
| `--n_train`, `--n_eval`, `--epochs`, `--lr`, `--batch_size`, `--seed` | usual hyperparams. |
| `--wandb` | flag | Log per-epoch CE / sparsity / norms and final eval metrics to W&B. |
| `--wandb_project` | str | W&B project name (default `memory-experiment`). |
| `--wandb_group` | str | Group label for the run (e.g. `sparsity`, `layer`) — clusters runs in the UI. |

A baseline-eval cache lives at `outputs/experiments/_baseline_cache_n<N_EVAL>.json`
keyed by `--n_eval`. The first run for a given `n_eval` populates the cache;
subsequent runs reuse it, halving wall time across a sweep.

---

## 9. Reading `compare.py` output

For each experiment, the comparison table shows `baseline → memory (delta)`
on both `ffff` and `full` val slices, for the four headline metrics:

- `operation_accuracy` — arithmetic. Should not regress.
- `f_selection` — primary alignment signal, averaged over steps.
- `full_f_alignment` — per-example all-f alignment. Strictly stronger.
- `complete_solution` — only meaningful on `ffff`.

The "OVERALL RANKING" section at the bottom sorts by `f_selection` on
**full val** — the honest test. The ffff-val numbers are the diagnostic /
upper bound because the memory was (potentially) trained on the same
distribution.
