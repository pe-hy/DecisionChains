# Hopfield / Associative Memory Steering Plan

## 1. Goal

Bias the pretrained extended-chain model to pick decision function **f** at the first decision step, **without** touching base weights and **without** degrading compositional generalization on novel function tuples or novel input vectors.

Stretch goal (after step 1 works): extend to all decision steps.

## 2. Hard constraints

- Base model stays frozen. No full finetuning, no GRPO, no LoRA on the backbone (in phase 1).
- No retraining from scratch. No architectural changes to LitGPT.
- Intervention must be localized to decision-point positions. It must not fire on non-decision tokens.
- All reported numbers must come from the existing extended evaluator ([evaluation/evaluator_chains_extended.py](evaluation/evaluator_chains_extended.py)) so results are directly comparable to prior runs.

## 3. Why this framing (and not the earlier GRPO attempt)

The GRPO runs failed from signal starvation: the reward was sparse, and whatever policy gradient leaked into the model also leaked into the circuits that compute `func_f` and `func_g` on novel vectors. The real disease is **interference** between the policy update and the compositional computation, not reward sparsity per se.

A frozen-model, inference-time intervention sidesteps this by construction: if we never change the base weights, we cannot damage compositional generalization. The only question becomes whether a localized additive bias at the decision position is expressive enough to flip P(f) without spilling elsewhere.

## 4. Conceptual anchor

A modern Hopfield network (Ramsauer et al. 2021) with a single stored pattern is mathematically equivalent to adding a constant vector to the residual stream, scaled by the similarity between the query and the stored key. With N stored patterns it becomes a single attention head with hand-built keys and values. This is the full structural content of "associative memory" for a frozen transformer.

CAMELoT ([memory_papers/2402.13449v1.pdf](memory_papers/2402.13449v1.pdf)) is the direct precedent in the memory literature: a training-free associative memory module coupled to a frozen attention-based LLM, using the Hopfield read/write primitives. Phase 1 below is literally CAMELoT with one slot and no consolidation/recency. Phases 2-3 grow it toward (but never into) the full CAMELoT machinery.

## 5. Phase 1 — Single-slot steering vector (minimum viable intervention)

### 5.1 Construction of the pattern

1. Load the HF-converted pretrained extended checkpoint (the same artifact that [scripts/inference_decision_chains_extended.py](scripts/inference_decision_chains_extended.py) loads).
2. Sample ~1000 test-split examples. Use the test split explicitly — we want the steering vector to generalize, so we probe on vectors and function tuples the model has never seen.
3. For each example, run a forward pass up to and including the first decision position. The first decision position is the token index right before where the model emits the first trace letter. Identify it deterministically from the tokenizer:
   - The input ends with `[TRACE]`. The first emitted letter is the first output token after `[TRACE]`.
   - The "decision hidden state" is the residual stream at the position that predicts that first letter — i.e. the last token of the prefix before generation.
4. For each example, record:
   - `h_dec`: the residual stream at a chosen layer `L_steer` (see 5.3) at the decision position. Shape `[d_model]`.
   - `picked`: whether the greedy argmax at that position corresponds to the f-letter or the g-letter for this example's ground truth.
5. Compute:
   - `h_f_mean = mean(h_dec | picked == f-letter)`
   - `h_g_mean = mean(h_dec | picked == g-letter)`
   - `delta = h_f_mean - h_g_mean`
6. Store `delta` as a tensor alongside the checkpoint. This is the single stored pattern.

Notes:
- Use **correctly-predicted** examples only when computing means. We want the direction that separates "circuit computed f and committed to it" from "circuit computed g and committed to it", not the direction that separates random errors.
- Balance the two groups. If the model is exactly 50/50, no rebalancing is needed; if it's skewed, subsample to equal counts before averaging.
- Record `n_f`, `n_g`, and the norm of `delta` — if `|delta|` is very small, the decision is not linearly encoded at this layer and phase 1 will not work.

### 5.2 Application at inference

Two equivalent formulations — pick the simpler one first.

**5.2a. Logit bias (simplest).** Project `delta` through the model's unembedding matrix `W_U` once, up front: `logit_bias = alpha * delta @ W_U.T`. At inference, when the position being predicted is the first decision letter, add `logit_bias` to the logits. This is a closed-form, one-line change in the generation loop.

**5.2b. Residual injection.** At inference, hook layer `L_steer` of the model; at the first decision position only, add `alpha * delta` to the residual stream. Let the remaining layers and the unembedding act on it normally. This is strictly more expressive than 5.2a (it lets downstream layers re-process the steered state), but requires a forward hook and a position detector.

Start with 5.2a. Only move to 5.2b if 5.2a cannot reach P(f) > 0.95 without degrading compositional metrics.

### 5.3 Choice of `L_steer`

Sweep: pick `L_steer` as layer 6, 8, 10 (for a 12-layer model). Compute `delta` at each, report `|delta|` and the linear probe accuracy (see 5.4). Pick the layer where the decision is most cleanly linearly encoded.

### 5.4 Sanity probe before running the full evaluator

Before touching the evaluator, run a cheap linear-separability check. Train a logistic regression on `h_dec` to classify "picked f" vs "picked g", using a held-out split of the 1000 probe examples. If the probe cannot reach >90% accuracy, phase 1 is doomed: the decision is not linearly encoded at this layer. Sweep layers. If no layer works, skip to phase 2.

### 5.5 Alpha sweep

Sweep `alpha in {0.5, 1.0, 2.0, 4.0, 8.0, 16.0}` on a small eval set (256 examples). Pick the alpha that maximizes P(first decision == f) subject to `valid_chain_fraction >= baseline - 0.02`. If no alpha satisfies this, phase 1 fails — go to phase 2.

### 5.6 Position detector

The logit bias must fire only at the first decision position, not on subsequent letters, not on non-decision tokens. Implement as: during generation, count the number of `;` tokens emitted so far. The first decision letter is emitted when zero `;` tokens have been emitted yet and the previous token was `[TRACE]` (or, equivalently, when the count of emitted letters-in-decision-positions is zero). Verify on 10 hand-picked examples that the detector fires exactly once per sequence at the intended position.

## 6. Phase 2 — k-slot Hopfield memory (only if phase 1 fails)

### 6.1 Motivation

A single `delta` assumes the f-vs-g decision lives in one direction, uniformly across the input distribution. If the model's internal representation of the decision depends on input cluster (e.g. "even first coord" vs "odd first coord" have different decision circuits), one vector is too blunt and phase 1's alpha sweep will show "bias P(f) up breaks valid-chain fraction".

### 6.2 Construction

1. Collect `h_dec` for ~5000 test examples as in phase 1.
2. Run k-means on `h_dec` with `k in {4, 8, 16, 32}`. For each cluster, compute the local `delta_k = mean_f_in_k - mean_g_in_k`.
3. Store keys = cluster centroids, values = per-cluster `delta_k`, both as frozen tensors.

### 6.3 Application

At the first decision position, compute similarity between the current `h_dec` and each stored key; softmax with temperature `tau`; weighted-sum the values; add `alpha * retrieved_value` to the residual (or project through `W_U` for logit bias).

This is a single attention head with `k` slots, hand-built from activations, never trained. Per Ramsauer et al. this is a modern Hopfield network.

### 6.4 Hyperparameter sweep

`k in {4, 8, 16, 32}`, `tau in {0.1, 1.0, 10.0}`, `alpha` as in phase 1. Same success criterion: max P(f) subject to valid_chain_fraction within 2 points of baseline.

## 7. Phase 3 — Tiny trainable gate (only if phase 2 fails)

### 7.1 Motivation

If even k slots of frozen steering don't separate "push P(f) up" from "break compositional generalization", the intervention needs a small amount of learned nonlinearity. But it must stay tiny.

### 7.2 Design

Add a single trainable scalar gate `g` (or at most a rank-1 LoRA on the output projection at `L_steer`) that multiplies the retrieved steering vector before it is added to the residual. Train only this gate via SFT on a subset of existing training data filtered to `first_decision == f`. Every other parameter in the model is frozen, including the stored keys/values. Use the existing evaluator as the stopping criterion: halt training as soon as P(f) > 0.95 AND valid_chain_fraction holds.

Budget: if phase 3 needs more than a rank-1 LoRA to work, abandon the Hopfield framing and revisit — that would be a signal that the decision-point circuit is genuinely entangled with the compositional circuit and cannot be cleanly intervened on from outside.

## 8. Measurement protocol (non-negotiable, same across all phases)

Every phase reports these exact metrics, computed by [evaluation/evaluator_chains_extended.py](evaluation/evaluator_chains_extended.py) on the held-out test split:

1. **Baseline numbers**, no intervention (run once, reuse): P(first decision == f), per-step letter distributions, valid_chain_fraction, block validity, novel-tuple generalization fraction.
2. **Intervention numbers**, for each hyperparameter setting.
3. **Position-localization check**: P(f) at decision steps 2, 3, 4, 5 must be unchanged from baseline. If these drift, the intervention is leaking.
4. **Novel-tuple check**: valid_chain_fraction split by whether the function tuple appeared in training. If the generalization gap widens under intervention, the steering vector is breaking compositional generalization and the result is invalid even if P(f) at step 1 looks good.
5. **No-op check**: the same intervention applied with `alpha = 0` must reproduce the baseline exactly. This catches implementation bugs in the hook / position detector.

Success criterion for the whole project:
- P(first decision == f) > 0.95
- valid_chain_fraction within 2 points of baseline
- P(f) at decision steps 2-N unchanged from baseline (within noise)
- Novel-tuple generalization gap unchanged from baseline

If any of these four fails at every phase, write up the negative result and move on.

## 9. Implementation map

New files:
- `steering/build_delta.py` — loads HF checkpoint, runs probe examples through it, collects `h_dec`, computes `delta` (and cluster-based memory in phase 2), saves to `outputs/steering/delta_L{layer}.pt`.
- `steering/apply_steering.py` — thin wrapper around HF `generate()` that installs a forward hook at `L_steer` implementing the position detector and adds `alpha * delta` (or the retrieved value). Exposes a `generate_with_steering(model, tokenizer, inputs, delta, alpha, L_steer)` function.
- `steering/eval_steering.py` — runs the existing `ExtendedChainEvaluator` but swaps its generation call for `generate_with_steering`. Emits the full measurement protocol above.

Touched files (minimal):
- Nothing in [framework/](framework/), [ops/](ops/), [train_chains.py](train_chains.py), or [scripts/generate_decision_chains_extended.py](scripts/generate_decision_chains_extended.py). The intervention is a read-only add-on.
- [evaluation/evaluator_chains_extended.py](evaluation/evaluator_chains_extended.py) may need a narrow hook point — a `generate_fn` parameter on `ExtendedChainEvaluator.__init__` that defaults to `model.generate` — so `eval_steering.py` can inject its own generate. If that's too invasive, duplicate the evaluator call path in `eval_steering.py` instead.

Order of work:
1. Linear probe sanity check at layers 6/8/10 — cheap, decides whether phase 1 is viable.
2. Phase 1 delta build + logit-bias application + alpha sweep.
3. Full measurement protocol on the best alpha.
4. Only if phase 1 fails all four success criteria: phase 2, same order.
5. Only if phase 2 also fails: phase 3, same order.

## 10. Risks and what would falsify the approach

- **Linear probe at `h_dec` is below 90%.** The decision is not linearly encoded at any single layer. Phase 1 cannot work. Phase 2 might still work (per-cluster deltas). Phase 3 probably cannot either, because a rank-1 gate is still effectively linear in the stored pattern.
- **P(f) goes up but valid_chain_fraction drops.** The steering vector is entangled with computation. This is the critical failure mode — it's the same disease that killed GRPO, just cheaper to diagnose. If this happens at every k in phase 2, the Hopfield framing is wrong for this problem and the answer is probably circuit-level intervention, not associative memory.
- **Position detector fires on the wrong token.** Catch this with the no-op check (section 8.5) before trusting any intervention numbers.
- **Steering at step 1 bleeds into step 2+.** Catch with section 8.3. If it bleeds, the residual injection is flowing forward through attention at later positions; the logit-bias formulation (5.2a) is immune to this by construction, so fall back to it.
- **`delta` is dominated by input-vector directions rather than the f/g direction.** Symptom: `delta` has large norm but doesn't improve P(f) at any alpha. Fix: orthogonalize `delta` against the top principal components of `h_dec` across all examples (remove the shared input-encoding direction).

## 11. What this plan explicitly does not do

- It does not add a memory layer to the architecture (that was the sparse-memory-finetuning path from [memory_papers/2510.15103v1.pdf](memory_papers/2510.15103v1.pdf), ruled out as overkill).
- It does not retrain the base model.
- It does not use CAMELoT's consolidation or recency machinery. We have one fact to store, not a stream.
- It does not address decision steps 2-N. That is the stretch goal, approached only after step 1 works end-to-end.

## 12. Paper citations for writeup

- CAMELoT — [memory_papers/2402.13449v1.pdf](memory_papers/2402.13449v1.pdf) — structural precedent for associative memory on a frozen attention-based LLM.
- Ramsauer et al. 2021, "Hopfield Networks is All You Need" — equivalence of modern Hopfield networks and attention; justifies the 1-slot and k-slot formulations as degenerate Hopfield networks.
- Larimar — [memory_papers/2407.01437v2.pdf](memory_papers/2407.01437v2.pdf) — only if phase 2 is extended with a proper least-squares write over multiple stored patterns.

The other seven papers in [memory_papers/](memory_papers/) are off-target for this plan (capacity / long-context / retraining) and should not be cited.
