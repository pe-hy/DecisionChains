# GRPO Post-Training Experiment Log

Post-training of the pretrained decision-chain transformer (12L-8H-512D GPT-NeoX, ~76M params)
with the goal of steering it to **always pick decision function f** while keeping arithmetic
traces valid. The pretrained baseline learned ~50/50 f/g from supervised data; GRPO should
concentrate probability mass on the f branch at every decision point.

**Headline result:** none of the 7 training runs surpassed the pretrained baseline on
`eval/valid_solution_frac` (pass@1). Every variant either destroyed the arithmetic
(`blocks_valid` → 5–65 %) or failed to move the all-f rate at all. The latest run
([4z4p3zvi](#run-7--4z4p3zvi--per-step-advantages--hybrid-mask)) is the only one that
reliably raised all-f behaviour (9.6 % → 25 %), but at the cost of dropping pass@1 from
69.1 % → 31.2 % because arithmetic degraded.

## Baseline (pretrained 12L-8H-512D, 6.2M tokens)

Measured on `outputs/data/decision_chains_extended/val.json`, 512 unique inputs:

| metric | overall | len 3 | len 4 | len 5 |
|---|---:|---:|---:|---:|
| `valid_solution_frac`     | **0.6914** | 0.7383 | 0.7208 | 0.6145 |
| `valid_solution_all_f_frac` | **0.0957** | 0.1342 | 0.0914 | 0.0663 |

Pass@k (baseline, T=0.8, n=64 samples/input, 512 inputs):
`pass@1 0.626 · pass@4 0.767 · pass@16 0.853 · pass@64 0.904`

## Shared task / reward setup

- **Prompt**: `[BOS] INPUT : [v] OUTPUT : [v2] [TRACE]`
- **Chain**: 3–5 decision steps. At every step the model picks a letter (a–t). Letters
  from `decision_func_f(L) = is_even(L[0])*10 + L[1]` or `decision_func_g(L) = is_even(L[3])*10 + L[4]`
  are considered “valid”. The chain-level target is *all letters came from f*.
- **Correctness check** (`evaluation/correctness.py :: check_completion_correct`) returns
  `all_blocks_valid` (arithmetic), `chose_f[]` (which step picked f), and `all_chose_f`.
- **Group advantage (standard GRPO)**: for each prompt i, sample K completions, compute
  `A_{i,j} = (r_{i,j} − mean_j) / max(std_j, ε)`. Groups where every completion is
  correct (or every completion is wrong) contribute zero advantage → the **signal-rate**
  metric (`groups_with_signal / n_prompts`) is the best single diagnostic.
- **Mask variants** (both defined in [grpo_train.py](../grpo_train.py)):
  - `make_decision_mask`: gradient only on the 3–5 positions where `input_ids[j] ∈ {[TRACE], ;}`
    (i.e. where the model is about to emit a decision letter). Concentrates the signal
    ~30–40× per sequence but shares the parameters with arithmetic prediction.
  - `make_response_mask`: gradient on every token after `[TRACE]`. Standard PPO behaviour;
    dilutes the f-signal across ~150 tokens per sequence that are otherwise identical
    between correct and incorrect completions.
- **KL penalty**: when `kl_coeff > 0`, a frozen reference copy of the model is loaded and
  `kl = ((cur − ref) * mask).sum() / mask.sum()` is added to the loss.
- **Optimizer**: AdamW, linear warmup (`warmup_iters=5`), `max_grad_norm=1.0`,
  `clip_param=0.2`, `inputs_per_batch=64`, `forward_batch_size=128`.
- **Pretrained model path**: `outputs/temp/hf_12l-8h-512d-decision-chains-ext_6_2M`.

Two training scripts:
- [grpo_train.py](../grpo_train.py) — sequence-level GRPO (one scalar reward per completion).
- [grpo_train_step.py](../grpo_train_step.py) — per-decision-step GRPO (independent
  advantages per decision position).

---

## Run 1 — `zlcok3re` (first attempt, original reward + decision mask)

**Script**: `grpo_train.py` — started 2026-04-07 22:34

**Reward (at the time)**: binary `1 if correct else 0` where *correct* = "all blocks valid
AND final vector matches target" — i.e. any f/g combination was rewarded.

**Config**:
```
K=16  temp=0.8  lr=1e-4  mini_epochs=4  kl_coeff=0  mask=decision
```

**Outcome**:
- First attempt **crashed** with `KeyError: INT_TO_LETTER[20]`. Model generated the
  token `"10"` inside vector brackets, the decision function returned an out-of-range
  index. Fixed by adding a bounds check in `correctness.py` and
  `evaluator_chains_extended.py`.
- After the fix, **catastrophic collapse**: `eval/valid_solution_frac` went
  **0.691 → 0.215** in 200 iterations (final: `len3=0.208, len4=0.228, len5=0.205`).
- Signal rate collapsed to 22 %, `reward_mean` to 5 %. Root cause: with 2^N valid
  ground-truth traces per input, both f and g were rewarded simultaneously, creating
  conflicting gradients; combined with lr=1e-4 and 4 mini-epochs, training diverged.
- Final wandb-summary confirms `eval/valid_solution_frac=0.2148` at iter 200.

**Action**: switched the reward from “any correct” to **all-f-only** (see Run 2+).

---

## Run 2 — `x6l2t047` (all-f reward, decision mask, aggressive LR)

**Script**: `grpo_train.py` — started 2026-04-08 10:16

**Reward**: binary `1 if all_chose_f and all_blocks_valid else 0`. `final_vec_matches`
is intentionally **not** required: the prompt's OUTPUT vector corresponds to whatever
f/g combo the data generator used, not necessarily to the all-f trajectory, so demanding
final-vector match would make the reward almost always zero.

**Config**:
```
K=32  temp=0.8  lr=1e-4  mini_epochs=4  kl_coeff=0  mask=decision
```

**Outcome**: **arithmetic destroyed in <10 iterations.**
- `grpo/completions_blocks_valid` 0.96 → 0.48 (iter 7) → ~0.06 at iter 111 (summary).
- `grpo/completions_all_f=0.054`, `grpo/completions_valid_fg=0.010`.
- `eval/valid_solution_frac=0.041` (was 0.691), `eval/valid_solution_all_f_frac=0.189`.
- The model learnt to *emit f letters* but could no longer do the arithmetic, so
  the final-vector check fails and the pass@1 metric tanks.
- Root cause: `decision mask × lr 1e-4 × 4 mini-epochs × no KL` is too much gradient
  through weights shared with arithmetic prediction.

---

## Run 3 — `aohyhbop` (response mask, conservative LR + KL)

**Script**: `grpo_train.py` — started 2026-04-08 13:10

**Reward**: same as Run 2 (binary all-f).

**Config**:
```
K=32  temp=0.8  lr=5e-6  mini_epochs=1  kl_coeff=0.05  mask=response
```

**Outcome**: **arithmetic preserved, but no f-learning.**
- `grpo/completions_blocks_valid` stayed 0.91–0.97 throughout.
- `grpo/completions_all_f` stuck at 4–16 %, `eval/valid_solution_all_f_frac ≈ 0.094` at
  iter 50 (effectively baseline).
- Run was manually interrupted (KeyboardInterrupt in the log) once it was clear the
  signal wasn’t moving.
- Root cause: response mask spreads the 5 per-sequence decision-step gradients across
  ~150 response tokens. 97 % of the gradient budget is on arithmetic tokens that are
  identical between all-f and any-f completions → near-zero effective signal on the
  actual decision.

---

## Run 4 — `c6cij41x` (response mask, higher temp & LR)

**Script**: `grpo_train.py` — started 2026-04-08 15:24

**Reward**: binary all-f.

**Config**:
```
K=32  temp=1.0  lr=2e-5  mini_epochs=1  kl_coeff=0.05  mask=response
```

**Outcome**: arithmetic stable, f-rate essentially unchanged.
- `grpo/completions_blocks_valid ≈ 0.87`, `grpo/completions_valid_fg ≈ 0.69`.
- `eval/valid_solution_all_f_frac=0.0957` (= baseline), `eval/valid_solution_frac=0.676`
  (≈ baseline).
- Signal rate 23 %, concentrated in len-3 groups (len-5 signal ≈ 13 %).
- Confirms Run 3's diagnosis: response mask + KL dilutes the decision-step gradient
  regardless of LR/temperature.

---

## Run 5 — `at710o5x` (decision mask with safety net)

**Script**: `grpo_train.py` — started 2026-04-08 17:36

**Reward**: binary all-f.

**Config**:
```
K=32  temp=1.0  lr=2e-5  mini_epochs=1  kl_coeff=0.05  mask=decision
```

**Outcome**: arithmetic held (`blk 0.87–0.93`) thanks to the smaller LR + KL penalty,
but the model still didn't learn f.
- `eval/valid_solution_all_f_frac ≈ 0.102` at iter 50 — barely above baseline.
- Per-length signal analysis showed len-5 `group_pass_rate` near 0 %. With K=32
  completions per prompt and P(all-f) ≈ 1–5 %, most groups had **zero** positive
  examples → zero advantage → zero gradient. **Signal starvation**.
- Run was manually interrupted.
- Diagnosis: the core problem is not the mask, it's the base rate of all-f completions
  given `temp=1.0` and the pretrained ~50/50 prior. Need either a much larger K
  (64–128) or a denser reward signal that doesn't require the whole chain to be f
  simultaneously.

---

## Run 6 — `v8492h6x` (dense per-step reward, decision mask)

**Script**: `grpo_train.py` — started 2026-04-08 20:05

**Change**: `compute_rewards` replaced with a **dense reward**:
```python
# evaluation/correctness.py returns chose_f = [bool] per step
if result["all_blocks_valid"] and len(chose_f) > 0:
    r = sum(chose_f) / len(chose_f)   # fraction of steps that chose f
    rewards[i, j] = r
```
i.e. the reward is the *fraction* of decision steps that picked f, gated on the
arithmetic being entirely valid. This removes the binary 0/1 cliff: a completion
that picks f at 4/5 steps still gets 0.8 and contributes to the advantage. All-wrong
and all-right groups are much rarer under this reward.

**Config**:
```
K=32  temp=1.0  lr=2e-5  mini_epochs=1  kl_coeff=0.05  mask=decision
n_iterations=200
```

**Outcome** (200 iterations, final summary):
| metric | baseline | iter 200 | Δ |
|---|---:|---:|---:|
| `eval/valid_solution_frac`       | 0.691 | **0.635** | −0.056 |
| `eval/valid_solution_all_f_frac` | 0.096 | **0.117** | +0.021 |
| `valid_solution_frac_len3`       | 0.738 | 0.671 | |
| `valid_solution_frac_len4`       | 0.721 | 0.706 | |
| `valid_solution_frac_len5`       | 0.614 | 0.518 | |
| `grpo/completions_all_f`         | —     | 0.167 | |
| `grpo/completions_blocks_valid`  | —     | 0.873 | |
| `grpo/signal_rate` (overall)     | —     | 0.953 | |
| `grpo/kl`                        | —     | 0.238 | |

- Dense reward fixed the signal-starvation symptom (`signal_rate 0.95`) and every
  prompt had non-zero gradient (`groups_all_wrong=0`).
- Despite that, all-f barely moved (+2 pts) and overall pass@1 regressed by ~6 pts
  (mostly on len-5 chains). The gradient *could* flow but did not converge on an
  all-f solution in 200 iterations.
- **Pass@k evaluation** (this is the run the saved
  [`outputs/eval_results/pass_at_k_grpo/pass_at_k_T0.8_n256.json`](../outputs/eval_results/pass_at_k_grpo/pass_at_k_T0.8_n256.json)
  corresponds to — written 2026-04-09 00:42, minutes after the model was saved):

  | k | baseline (n=64) | Run 6 GRPO (n=256) |
  |---|---:|---:|
  | 1   | 0.626 | **0.568** |
  | 4   | 0.767 | 0.728 |
  | 16  | 0.853 | 0.827 |
  | 64  | 0.904 | 0.893 |
  | 256 | —     | 0.920 |

  GRPO *flattened* the pass@k curve — pass@1 dropped but the ceiling (pass@256) is
  comparable. Consistent with the SGE hypothesis in [plan.md](../plan.md): RL
  post-training converges pass@1 toward the pre-existing pass@k ceiling but does not
  raise it.

---

## Run 7 — `4z4p3zvi` (per-step advantages + hybrid mask) — CURRENT

**Script**: [grpo_train_step.py](../grpo_train_step.py) — started 2026-04-09 16:42,
500 iterations over ~11 h.

**Change**: completely redesigned credit assignment. Instead of one scalar reward per
completion, compute an **independent advantage per decision step**:

```python
# grpo_train_step.py :: compute_step_rewards
step_chose_f[i, j, k] = 1.0 if completion j chose f at step k (and all blocks valid)
step_valid[i, j, k]   = 1.0 if step k exists and the completion has valid arithmetic

# grpo_train_step.py :: compute_step_advantages
for each prompt i, step k:
    mask = step_valid[i, :, k] > 0
    rewards_k = step_chose_f[i, mask, k]
    A[i, :, k] = (step_chose_f[i, :, k] − rewards_k.mean()) / rewards_k.std()
```

Each decision position gets a direct gradient toward f, **independent of whether other
steps in the same chain got it right**. This bypasses the credit-assignment bottleneck
of sequence-level GRPO: at len 5, a completion with 4/5 f still provides positive
advantage on the 4 correct steps.

**Hybrid mask** (also new):
- **Policy gradient** is applied only on `decision_mask` (step-k advantage at the
  k-th `[TRACE]`/`;` position). `grpo_loss_step` uses a 2-D advantage tensor aligned
  with token positions.
- **KL penalty** is applied on `response_mask` (all post-[TRACE] tokens) against the
  frozen reference model, anchoring the arithmetic distribution.

This decouples f-steering (decision tokens) from arithmetic protection (all response
tokens) — the explicit motivation is in [run_grpo_pipeline_step.sh](../run_grpo_pipeline_step.sh).

**Config**:
```
K=32  temp=1.0  lr=1e-5  mini_epochs=1  kl_coeff=0.01  n_iterations=500
eval_every=25  save_every=250   # (hybrid) PG=decision, KL=response
```

**Outcome** (500 iterations, final summary from
`wandb/run-20260409_164219-4z4p3zvi/files/wandb-summary.json`):

| metric | baseline | iter 500 | Δ |
|---|---:|---:|---:|
| `eval/valid_solution_frac`        | **0.691** | **0.3125** | −0.379 |
| `eval/valid_solution_all_f_frac`  | **0.0957** | **0.2500** | +0.154 |
| `valid_solution_len3`             | 0.738 | 0.389 | |
| `valid_solution_len4`             | 0.721 | 0.294 | |
| `valid_solution_len5`             | 0.614 | 0.265 | |
| `valid_solution_all_f_len3`       | 0.134 | 0.268 | |
| `valid_solution_all_f_len4`       | 0.091 | 0.259 | |
| `valid_solution_all_f_len5`       | 0.066 | 0.223 | |
| `grpo/completions_all_f`          | —     | 0.195 | |
| `grpo/completions_all_f_seq`      | —     | 0.221 | |
| `grpo/completions_blocks_valid`   | —     | 0.652 | |
| `grpo/completions_valid_fg`       | —     | 0.212 | |
| `grpo/step_f_mean`                | —     | 0.758 | |
| `grpo/step_signal_rate`           | —     | 0.479 | |
| `grpo/step_f_mean_len{3,4,5}`     | —     | 0.81 / 0.75 / 0.74 | |
| `grpo/kl`                         | —     | **−1.34** | |

**Observations**:
- This is the only run that clearly *moved* the all-f behaviour: per-step `step_f_mean`
  0.75, all-f `eval` frac 0.25 (2.6× baseline). Per-length all-f frac roughly doubled
  on every length.
- **But** arithmetic collapsed: `completions_blocks_valid 0.65`, `completions_valid_fg 0.21`.
  Pass@1 dropped from 0.691 to 0.313 — worse than Run 6.
- **KL went negative (−1.34)**. The KL term in `grpo_train_step.py:441` is
  `((cur − ref) * resp_mask).sum() / resp_mask.sum()` — this is the mean token-level
  log-ratio, **not** a real KL divergence. A real KL is always ≥ 0; allowing it to
  drift negative means the policy is being rewarded for *lowering* arithmetic-token
  log-probs relative to the reference (as long as the PG term on decision tokens
  compensates), which is exactly the observed pathology. This is a likely
  implementation bug and worth investigating before the next run.
- No pass@k evaluation was run on this checkpoint
  (`outputs/temp/hf_grpo-step_12l-8h-512d-decision-chains-ext_6_2M/` saved 2026-04-10 03:39
  but `outputs/eval_results/pass_at_k_grpo/` was not updated).
- Two earlier attempts at the same script — `4gsftsid` (2026-04-09 09:56) and
  `08e17pqn` (2026-04-09 15:50) — were both killed early with KeyboardInterrupt during
  generation (no useful metrics logged).

---

## Summary table

| run | script | reward | mask | K | T | lr | KL | iters | pass@1 | all-f | blocks_valid | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline    | —               | —                    | —        | —  | —   | —     | —    | —   | **0.691** | 0.096 | —     | — |
| 1  `zlcok3re` | grpo_train      | binary, any f/g      | decision | 16 | 0.8 | 1e-4  | 0    | 200 | 0.215 | —     | —     | collapse (conflicting gradients) |
| 2  `x6l2t047` | grpo_train      | binary, all-f        | decision | 32 | 0.8 | 1e-4  | 0    | 111 | 0.041 | 0.189 | 0.062 | arithmetic destroyed |
| 3  `aohyhbop` | grpo_train      | binary, all-f        | response | 32 | 0.8 | 5e-6  | 0.05 | ~50 | ≈ base | ≈ base | 0.95  | no f-learning (signal diluted) |
| 4  `c6cij41x` | grpo_train      | binary, all-f        | response | 32 | 1.0 | 2e-5  | 0.05 | 100 | 0.676 | 0.096 | 0.87  | stable, no learning |
| 5  `at710o5x` | grpo_train      | binary, all-f        | decision | 32 | 1.0 | 2e-5  | 0.05 | ~80 | ≈ base | 0.102 | 0.90  | signal starvation |
| 6  `v8492h6x` | grpo_train      | **dense** frac-of-f  | decision | 32 | 1.0 | 2e-5  | 0.05 | 200 | 0.635 | 0.117 | 0.87  | flat pass@k, no real gain |
| 7  `4z4p3zvi` | grpo_train_step | **per-step** ±1 on f | hybrid† | 32 | 1.0 | 1e-5  | 0.01 | 500 | 0.313 | 0.250 | 0.65  | f-rate ↑, arithmetic ↓, KL<0 |

† Run 7: policy gradient on `decision_mask`, KL penalty on `response_mask`.

## Cross-run lessons

1. **Any f/g correctness reward is self-defeating** (Run 1). With 2^N equally-valid
   targets per prompt, standard "is this trace correct?" rewards both f and g and the
   gradients cancel. You must pick one target policy (all-f) and reward only that.

2. **Decision mask by itself is fragile** (Runs 2, 5). Concentrating 30× more gradient
   on 5 tokens per sequence that share weights with arithmetic breaks arithmetic unless
   either (a) the LR is conservative or (b) a KL anchor constrains the non-decision
   tokens.

3. **Response mask by itself learns nothing** (Runs 3, 4). The f-signal is drowned by
   ~150 arithmetic tokens per sequence whose log-probs are nearly identical across
   completions and dilute the effective gradient by 30–40×.

4. **Dense per-step rewards beat binary all-f for signal, but do not by themselves
   produce learning** (Run 6). Signal rate went to 95 % with zero all-wrong groups, yet
   all-f only moved +2 pts in 200 iterations and pass@1 regressed. The optimisation
   landscape has a nearby minimum where the model slightly biases toward f without
   committing — clip=0.2 with lr=2e-5 may be too conservative to escape it.

5. **Per-step credit assignment actually moves behaviour** (Run 7). It is the only run
   where all-f meaningfully increased (+15 pts). But the current implementation
   combines it with a **one-sided "KL" term** that is actually a signed log-ratio and
   is allowed to drift negative, which correlates with arithmetic degradation. Fixing
   this (either by using `(cur − ref)²` / proper KL, or by clamping the term to ≥0)
   is the obvious next step before more hyperparameter sweeps.

6. **Signal starvation is a base-rate problem, not a mask problem** (consistent across
   runs). At T=1.0 with pretrained 50/50 f/g prior, P(all-f across a 5-step chain) ≈
   0.03, so at K=32 most len-5 groups have zero positive completions. Either K must
   grow to 64–128, temperature must rise to 1.2+, or credit assignment has to become
   per-step (Run 7).

7. **Pass@k ceiling is real** (Run 6 pass@k evaluation). GRPO flattens the pass@k
   curve toward pass@1 but does not extend it beyond the pretrained ceiling
   (pass@256 ≈ pass@64 baseline ≈ 0.90). This matches the SGE hypothesis stated in
   [plan.md](../plan.md).

## Open questions / next steps

- **Fix the KL term in `grpo_train_step.py`**. Either square it, use
  `F.kl_div(log_softmax(cur), log_softmax(ref))`, or clamp to ≥0 so the optimiser
  cannot lower arithmetic log-probs relative to reference.
- **Evaluate Run 7 with pass@k** (`./run_grpo_pipeline_step.sh` or manual
  `eval_pass_at_k.py` on `outputs/temp/hf_grpo-step_12l-8h-512d-decision-chains-ext_6_2M/`)
  to see whether the all-f shift transfers to sampled rollouts or is greedy-only.
- **Try a larger K or higher T** on Run 6's setup — the dense reward already has 95 %
  signal rate, so the missing ingredient may be optimiser aggression rather than
  signal density.
- **Revisit memory entry** `project_grpo_finetuning.md`: the "current diagnosis" there
  predates Runs 6 and 7 and should be refreshed.
