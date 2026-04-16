# Balanced experiment: wrong-letter label collision

## TL;DR

The "balanced" data scheme is balanced at the **chain** level (N healthy + N unhealthy per GT chain), but **not** at the individual decision-point level. At the last decision point of a length-N chain, the model sees correct-letter and wrong-letter labels in a **1:1 ratio**, which teaches it an ambivalent distribution there. Once greedy free-gen samples the wrong letter at that position, the unhealthy training path bleeds through and the model emits `; STOP`. This is why `stop_in_free_gen_rate ≈ 0.99` and `Health/chain_quality_gt = 0.00`, even though token-level `acc = 0.986`.

The fix is symmetric to the peek-letter mask we already have: **mask the wrong-letter token's label in unhealthy examples**. The wrong letter still appears in the model's context, but no longer competes as a label at the decision point.

---

## The current training recipe (after the peek-mask fix)

For a GT chain of length N, the data generator emits 2N examples:

- **N − 1 healthy_partial** variants: execute blocks 0..s correctly, then append a "peek" letter `letter_{s+1}`, then `[EOS]`. Loss is masked on both the peek letter and `[EOS]` (the recent fix).
- **1 healthy_full** variant: the full correct chain, no masking.
- **N unhealthy** variants: one per step `m`, with the correct prefix up to `m−1`, then the **wrong letter** (picked by the opposite decision function) and its fully-expanded trace, then `; STOP [EOS]`. **Everything here is currently supervised — including the wrong letter itself.**

## A worked example (length 3)

Suppose the GT chain's correct letters are `[c, k, b]` at steps 0, 1, 2, and the opposite decision function would have picked `[d, n, a]`. Abbreviating each block as `<letter> <trace> R [ <vec> ]`:

- `B0` = `c : 2*1=2 , ... R [ v0 ]`
- `B1` = `k : 2+3=5 , ... R [ v1 ]`
- `B2` = `b : 3 , 5 , ... R [ v2 ]`
- `WB0`, `WB1`, `WB2` — same shape but with `d`, `n`, `a`

### What gets supervised right now

(`✓` = loss is computed at this position; `✗` = label is set to `-100`.)

```
ex1  healthy_partial s=0:   c : … R [ v0 ] ;  k [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✗   ✗     ← peek 'k' and EOS masked

ex2  healthy_partial s=1:   c : … R [ v0 ] ;  k : … R [ v1 ] ;  b [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✗   ✗

ex3  healthy_full:          c : … R [ v0 ] ;  k : … R [ v1 ] ;  b : … R [ v2 ] [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓  ✓

ex4  unhealthy m=0:          d : … R [ w0 ] ;  STOP [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓

ex5  unhealthy m=1:          c : … R [ v0 ] ;  n : … R [ w1 ] ;  STOP [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓

ex6  unhealthy m=2:          c : … R [ v0 ] ;  k : … R [ v1 ] ;  a : … R [ w2 ] ;  STOP [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓
```

### Counting supervisions at each "next-token-after-;" position

A position I'll call a "slot" is either right after `[TRACE]` (slot 0) or right after a `;` (slot 1, 2, ...). Each of the 6 examples contributes a labeled target at every slot it spans. **Three kinds of labels show up in these slots**: a correct letter, a wrong letter, or `STOP` — depending on the variant.

For our length-3 example, here's which example contributes what label at each slot:

| slot | ex1 (hp s=0) | ex2 (hp s=1) | ex3 (hf) | ex4 (u m=0) | ex5 (u m=1) | ex6 (u m=2) | aggregate labels |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | `c` | `c` | `c` | **`d`** | `c` | `c` | 5 × `c`, 1 × `d` |
| 1 | ~`k`~ masked | `k` | `k` | **`STOP`** | **`n`** | `k` | 3 × `k`, 1 × `n`, **1 × `STOP`** |
| 2 | — | ~`b`~ masked | `b` | — | **`STOP`** | **`a`** | 1 × `b`, 1 × `a`, **1 × `STOP`** |
| 3 | — | — | — | — | — | **`STOP`** | 1 × `STOP` |

Key things to notice:

1. **`STOP` does show up as a label** — it's the token that unhealthy examples train right after the wrong block's trailing `;`. Each unhealthy with mistake step `m` contributes exactly one `STOP` label, at slot `m + 1` (the slot right after the wrong block).
2. The peek-letter positions in healthy partials are crossed out: those labels are `-100` under the current fix, so they don't contribute.
3. Slot 3 in this length-3 chain only exists for ex6 (and only has a single `STOP` label). There's no healthy supervision there — it's just "after a completed wrong chain, say `STOP`".

### The collision

Look at slot 2: the labels are `b`, `a`, and `STOP` — **one of each**. The model sees the three options in a perfect 1:1:1 ratio at the last real decision point of a length-3 chain. It learns a roughly uniform distribution there, with P(`STOP`) ≈ 1/3 just from the label frequency.

General formula — at slot `k` of a length-N chain, per GT chain:

- **correct letter** count: `2N − 2k − 1`
- **wrong letter** count: `1` (from unhealthy with `mistake_at = k`)
- **`STOP`** count: `1` (from unhealthy with `mistake_at = k − 1`, for `k ≥ 1`)

Plugging in length-5:

| slot `k` | correct | wrong | `STOP` | correct : wrong : STOP |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 9 | 1 | 0 | 9 : 1 : 0 |
| 1 | 7 | 1 | 1 | 7 : 1 : 1 |
| 2 | 5 | 1 | 1 | 5 : 1 : 1 |
| 3 | 3 | 1 | 1 | 3 : 1 : 1 |
| 4 | **1** | **1** | **1** | **1 : 1 : 1** ← collision |

At the last real decision point of a length-5 chain, the model sees correct, wrong, and `STOP` in equal proportion. It has no reason to prefer any of them, and in free generation, greedy decoding over ~5 such positions accumulates enough `STOP` mass to halt the chain almost every time. The observed `P_stop_after_correct` at epoch 2 confirms the shape:

| slot | `P_stop_after_correct` |
|:---:|:---:|
| 0 | 0.12 |
| 1 | 0.16 |
| 2 | 0.30 |
| 3 | 0.45 |
| 4 | **0.91** |

(The slot-4 number is higher than the 1:3 prediction because the last-slot "correct prefix" is actually **zero-shot** — it never appears in training at all, since healthy_full ends at block N−1 with no trailing `;`. The model extrapolates from nearby contexts, and the nearest contexts are the `; STOP` sequences from unhealthies.)

---

## The proposed fix (Option 3)

**One change**: in the unhealthy variants, mask the wrong letter's label the same way the peek letter is masked in healthy partials. Nothing else changes — not the data layout, not the STOP training, not the trace content.

### What gets supervised under the fix

Only lines 4, 5, 6 change. Lines 1, 2, 3 are untouched.

```
ex4  unhealthy m=0:          d : … R [ w0 ] ;  STOP [EOS]
                            ✗✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓
                            ↑ wrong-letter label masked

ex5  unhealthy m=1:          c : … R [ v0 ] ;  n : … R [ w1 ] ;  STOP [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✗✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓
                                             ↑ masked

ex6  unhealthy m=2:          c : … R [ v0 ] ;  k : … R [ v1 ] ;  a : … R [ w2 ] ;  STOP [EOS]
                            ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓✓✓✓✓✓✓✓✓ ✓  ✗✓✓✓✓✓✓✓✓✓✓✓ ✓  ✓✓✓✓  ✓
                                                              ↑ masked
```

### New supervision tally

| Step | ex1 | ex2 | ex3 | ex4 | ex5 | ex6 | **correct : wrong** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `c` | `c` | `c` | — | `c` | `c` | **5 : 0** |
| 1 | — | `k` | `k` | — | — | `k` | **3 : 0** |
| 2 | — | — | `b` | — | — | — | **1 : 0** |

At every decision point the model is trained to predict the correct letter, period. No more collisions.

---

## Why this doesn't break the STOP signal

The wrong letter still appears **in context**. When the model predicts the rest of the wrong block, predicts the `;` after it, and predicts the `STOP` token, the attention still sees the wrong letter at its position — so the conditional "given a wrong letter followed by its block → emit `STOP`" still flows. We're just removing the spurious label "given the current prefix, the next letter should be `a`" from the loss.

Concretely, in each unhealthy example:

- ✓ The correct prefix blocks `B0 .. B_{m−1}` are still trained normally (block-execution supervision).
- ✗ **The wrong letter token is masked.**
- ✓ The wrong block's trace content (`:`, trace steps, `R`, `[`, vec, `]`) is still trained. This is just valid arithmetic — "what does `a` (reverse) do to `[v1]`?" — and provides extra arithmetic coverage.
- ✓ The `;` after the wrong block is trained.
- ✓ `STOP` is trained.
- ✓ `[EOS]` is trained.

Nothing about the STOP emission pipeline is weakened. The only thing removed is the one token the experiment was never supposed to teach in the first place.

---

## Expected effect on the metrics

| metric | current (epoch 2) | expected after fix |
|---|:---:|:---:|
| `acc` (token-level) | 0.986 | ~same or slightly higher |
| `step4_P_stop_after_correct` | 0.91 | should drop sharply (~0.1–0.2) |
| `step4_P_stop_after_wrong` | 0.96 | ~same |
| `stop_discrimination` | 0.18 | should widen, especially at late steps |
| `stop_in_free_gen_rate` | 0.99 | should drop well below 0.5 |
| `valid_chain_frac` | 0.00 | should become nonzero |
| `Health/chain_quality_gt` | 0.00 | should become nonzero |
| `Health/overall` (= `cq_gt × cq_stop × no_spurious`) | 0.00 | should become nonzero |

The token-accuracy metric is essentially unaffected because it was never the bottleneck — this fix changes free-gen behavior, not teacher-forced behavior.

---

## Two ways to implement the fix

### Option A — regenerate the data

In `scripts/generate_decision_chains_balanced.py`, add a `wrong_letter_offset` (or reuse/extend the `mask_last_n` convention) to unhealthy example dicts, pointing at the position of the wrong letter in the output. Then extend `train_chains_balanced.py:mask_targets` to zero that position too.

- **Pros**: explicit, easy to audit.
- **Cons**: requires regenerating and re-tokenizing the full 16M-example dataset.

### Option B — detect at runtime in `mask_targets`

Keep the data as-is. In `mask_targets`, detect unhealthy examples by looking for the `STOP` token id in `input_ids`, then walk backward from `STOP` to find the letter that starts the wrong block (the token right after the second-to-last `;` before `STOP`). Mask that position.

- **Pros**: no data regeneration, no cache invalidation, self-contained change.
- **Cons**: a few lines of slightly fiddly index arithmetic. Needs unit-checking against a handful of real batches before launching.

Either works. Option B is what I'd reach for first — it keeps the 16M-example cache valid and the diff is small.

---

## One-line summary

**Current**: we mask the peek letter in healthy partials but still train on the wrong letter in unhealthies. That's asymmetric and causes a correct/wrong label collision at the last decision point of every chain length.

**Proposed**: mask the wrong letter in unhealthies the same way we mask the peek letter in healthy partials. The experiment then actually balances at every decision point, which is what the name implies.
