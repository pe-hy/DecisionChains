# Activation Steering & Representation Engineering — Literature Review

## Overview

Activation steering is a family of **inference-time** techniques that modify a frozen transformer's hidden states (usually the residual stream, sometimes per-head attention outputs, and more recently the KV cache) to control high-level behavior without retraining weights. The family can be taxonomized along three axes:

1. **How the vector is computed.** *Static*: mean-difference of activations on a small set of contrastive prompt pairs (ActAdd, CAA, RepE, ITI). *Learned*: vectors fit by backprop (bias-only RL, our KV memory). *Dictionary-based*: directions discovered by a sparse autoencoder trained on the base model (Anthropic SAE steering).
2. **Where it is injected.** Residual stream (ActAdd, CAA, RepE, persona vectors), per-head attention output (ITI), MLP bias (bias-only adaptation), KV cache (cache steering).
3. **How it is applied.** Unconditionally at all positions after the prompt (CAA), only when a learned "condition vector" fires (CAST), or only at specific token positions / layers chosen via probing (LayerNavigator).

Relative to fine-tuning, steering is cheap (often <0.01% params equivalent), composable, and interpretable, at the cost of typically weaker and more brittle behavior changes.

## Key methods

### Static steering vectors

**ActAdd (Turner et al., arXiv 2308.10248, 2023).** Takes two natural-language prompts, e.g. `"Love"` vs `"Hate"`, forwards both through the model, and subtracts the residual-stream activations at a chosen layer and token position to produce a single steering vector `v`. At inference, `v` is added (with a scalar coefficient `c`) to the residual stream at that same layer. As few as 2 prompts can produce a usable vector. State-of-the-art on sentiment shift and detoxification in 2023, with minimal degradation on off-target tasks.

**Contrastive Activation Addition / CAA (Panickssery et al., arXiv 2312.06681, 2023; ACL 2024).** Scales ActAdd up: instead of one prompt pair, average `a_pos - a_neg` over hundreds of matched multiple-choice contrastive examples (e.g. "sycophantic A" vs "honest B"). The vector is added at every post-prompt token at a single layer. On Llama-2-Chat, CAA stacks cleanly on top of system prompts and SFT, suggesting steering targets a different axis of control.

### Multi-vector / learned

**RepE (Zou et al., arXiv 2310.01405, 2023).** Generalizes CAA into a two-step framework: *Representation Reading* (find a direction for a concept, e.g. via PCA on contrastive pairs) and *Representation Control* (add/subtract that direction, or project onto / off it). Delivered +18 pp zero-shot accuracy on TruthfulQA via honesty steering. The `repeng` Python library (Vogel, 2024) trains a full-model control vector in under 60 seconds.

**ITI — Inference-Time Intervention (Li et al., arXiv 2306.03341, NeurIPS 2023).** Probes each attention head independently for a "truthfulness direction," keeps the top-K heads by probe accuracy (~48 heads on Llama-7B), and at inference shifts those heads' outputs along the probe normal with a tuned magnitude. Raised Alpaca TruthfulQA from 32.5% → 65.1%. Key insight: truthfulness information is **sparse across heads**, so targeted per-head intervention beats residual-stream addition for this task.

**Bias-only adaptation (Goncharov et al., arXiv 2505.18706, ICML 2025).** Instead of computing a vector from contrastive pairs, *train* one additive bias vector per MLP layer with RL, freezing all other weights (~0.0016% new params on 8B models). Matches full RL-tuned reasoning models on six math benchmarks (AIME24/25, AMC23, MATH500, MinervaMath, OlympiadBench). Logit-lens analysis shows the trained biases boost tokens for structured language and logical connectors — consistent with the view that RL strengthens latent reasoning circuits rather than teaching new ones. This is the closest published analogue to our learned-memory approach.

### Interpretability-driven

**SAE feature steering — "Scaling Monosemanticity" (Anthropic, 2024).** Train a sparse autoencoder on Claude-3-Sonnet's residual stream to extract ~34M monosemantic features. Clamping a feature's activation at inference produces targeted behavior changes ("Golden Gate Claude"). SAEs are also valuable for *discovery* (find unknown concepts). Caveat: on already-known behaviors, subsequent work found SAE steering is often less effective than well-chosen CAA directions and can break structured output — SAEs are better for discovering concepts than enforcing them (arXiv 2506.23845).

**Persona Vectors (Anthropic, arXiv 2507.21509, 2025).** Automated pipeline: given a natural-language trait definition ("evil", "sycophancy", "hallucination"), generate contrastive prompts, extract the mean-difference direction, then (a) monitor the projection during training to detect persona drift, (b) subtract post-hoc to mitigate, or (c) add *during* fine-tuning as a "vaccine" that prevents persona shift on poisoned data. Bridges CAA-style extraction with training-time intervention.

### Conditional / multi-vector

**CAST — Conditional Activation Steering (Lee et al., arXiv 2409.05907, ICLR 2025 spotlight).** Introduces *condition vectors*: a second set of directions whose cosine similarity with the current hidden state acts as a learned classifier ("is this prompt harmful?"). The steering vector fires only when the condition matches. Supports multi-property control by injecting different vectors at different layers (e.g. refusal at layer 24, formality at layer 28). Released as IBM's `activation-steering` toolkit. This is the cleanest published precedent for a router-gated steering setup.

**KV cache steering (Belitsky et al., arXiv 2507.08799, 2025).** One-shot intervention applied directly to the KV cache rather than the residual stream — more stable at inference, and enables style transfer (stepwise, causal, analogical reasoning). Induces CoT in small models *without* fine-tuning, using steering vectors built from teacher traces.

### Math / reasoning-specific (2024–2026)

- **"Small Vectors, Big Effects"** (arXiv 2509.06608, 2025): mechanistic study showing a handful of RL-induced reasoning changes are captured by low-rank steering directions — steering vectors as a compact *probe* of what RL actually changed.
- **"Knowing Before Saying"** (arXiv 2505.24362, 2025): linear probes on residual states predict CoT success *before* the chain completes, implying a "will-this-be-correct" direction exists and can be steered.
- **Activation Steering for CoT Compression** (ResearchGate 2024/25): reduces CoT length up to 67% with negligible runtime cost.
- **Budget Guidance** (arXiv 2506.13752, 2025): +26% on MATH-500 under tight thinking budgets vs. baselines.
- **Understanding & Steering Cognitive Behaviors** (arXiv 2512.24574, 2025): test-time control of reflection vs. forward progress in reasoning models.

### Task arithmetic (weight-space tangent)

**Task arithmetic (Ilharco et al., ICLR 2023).** A *task vector* is `θ_finetuned − θ_pretrained`. Addition composes capabilities across tasks; negation performs targeted forgetting. Lives in *weight* space, not activation space — much heavier than steering but operates at all tokens and all layers simultaneously. Useful as an outer loop (task arithmetic to pick a backbone; activation steering for fine control).

## Comparison with learned KV memory (our method)

| Axis | CAA / ActAdd / RepE | Our KV memory module |
|---|---|---|
| **Computation** | Mean-diff on contrastive pairs (no grad) | Keys/values learned by gradient descent |
| **Expressivity** | One (or few) fixed directions per behavior | Hundreds of learned key-value pairs, selected by attention |
| **Routing** | None (all positions) or a separate condition vector (CAST) | Intrinsic: attention softmax over keys is the router |
| **Data needed** | 10s–1000s of contrastive pairs | Thousands of (input, target-trace) examples |
| **What it injects** | Single vector, scalar-scaled | Input-dependent convex combination of learned values |
| **Target behavior** | Monolithic traits (honesty, sentiment, persona) | Context-dependent decisions (which transformation next, which proof step) |

**Expected winner by regime.** For a *single* global bias ("be more honest", "use more CoT"), CAA/RepE is likely competitive and far cheaper. Bias-only adaptation (Goncharov 2025) is the strongest static-direction baseline for math reasoning — we should run it as an ablation. For **input-conditional** behavior — the decision function that selects `f` vs `g` at each chain step, or the analogous "which sub-skill do I need for *this* algebra problem" — a single vector is fundamentally insufficient: the needed correction depends on the current hidden state. This is exactly where trainable memory should dominate. CAST is the closest existing method; our memory module generalizes CAST by replacing its single condition→vector mapping with a full attention lookup.

## Actionable takeaways for Qwen3-8B + math

- **Layer to probe first.** For 32-ish-layer decoder models, steering typically peaks at 40–70% depth. On Qwen3-8B (36 layers), start probes at layers 14–24. Use **LayerNavigator** (OpenReview 2025) style scoring — discriminability (class separation on contrastive pairs) and consistency (behavior holds across prompts) — rather than brute-force sweeping.
- **Contrastive data for math correctness.** Three good sources: (i) matched correct/incorrect solutions to the same MATH problems from a weaker model (cheap labels — final-answer check); (ii) early-vs-late step pairs from correct solutions (probes "step-progress" directions, cf. "Knowing Before Saying"); (iii) GRPO-style group pairs — highest-reward vs. lowest-reward completions per prompt — which double as a RepE contrast set and a reward-model signal.
- **Baselines to run before claiming memory helps.** (1) CAA with a "correct vs. incorrect" direction; (2) bias-only adaptation (Goncharov 2025) on our dataset; (3) LoRA matched for parameter count.
- **Libraries.**
  - `repeng` (vgel): fastest path to a CAA/RepE vector. Warning: no MoE support — check Qwen3-8B dense vs. MoE variant.
  - `steering-vectors` (Tan et al.): cleaner abstractions, better for multi-vector experiments.
  - `IBM/activation-steering` (CAST implementation): essential if we want conditional/multi-vector control as an ablation.
  - `nnsight` 0.6: for probing-heavy workflows and scalable remote execution — the best ergonomic option for layer-sweep experiments.
  - `Dialz` (arXiv 2505.06262, 2025): unified toolkit across steering methods.
- **Injection protocol.** Inject after the `[problem]` tokens, during CoT generation (post-prompt), mirroring CAA. Scan coefficients in [−3, +3] standard deviations of the layer's activation norm.

## Open questions

1. Can a *single* learned steering vector match our KV memory on chain-decision tasks, or does the decision structure demand input-conditional routing? This is the central empirical claim to test.
2. Do math-correctness directions transfer across problem *types* (algebra → geometry) or do they fragment, motivating multi-vector (CAST-style) or attention-routed (our) approaches?
3. Is KV cache steering (Belitsky 2025) a cheaper substitute for our memory module, or complementary (inject once into cache + continuous memory lookup during decoding)?
4. SAE features on Qwen3-8B: any public Qwen3 SAEs, or must we train our own? Would SAE-derived math features outperform contrastive-pair directions?
5. Can we monitor persona-vector style drift during memory training — does the memory quietly change the model's general persona, or only the target reasoning behavior?

## Sources

- ActAdd — [arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248)
- CAA (Panickssery et al.) — [arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
- RepE (Zou et al.) — [arxiv.org/abs/2310.01405](https://arxiv.org/abs/2310.01405); code: [github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- ITI (Li et al.) — [arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341); code: [github.com/likenneth/honest_llama](https://github.com/likenneth/honest_llama)
- Scaling Monosemanticity / Golden Gate Claude — [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- Persona Vectors — [anthropic.com/research/persona-vectors](https://www.anthropic.com/research/persona-vectors); [arxiv.org/abs/2507.21509](https://arxiv.org/abs/2507.21509)
- CAST — [arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907); toolkit: [github.com/IBM/activation-steering](https://github.com/IBM/activation-steering)
- KV cache steering — [arxiv.org/abs/2507.08799](https://arxiv.org/abs/2507.08799); code: [github.com/MaxBelitsky/cache-steering](https://github.com/MaxBelitsky/cache-steering)
- Bias-only adaptation — [arxiv.org/abs/2505.18706](https://arxiv.org/abs/2505.18706); code: [github.com/corl-team/steering-reasoning](https://github.com/corl-team/steering-reasoning)
- Small Vectors, Big Effects — [arxiv.org/abs/2509.06608](https://arxiv.org/abs/2509.06608)
- Knowing Before Saying — [arxiv.org/abs/2505.24362](https://arxiv.org/html/2505.24362v2)
- Budget Guidance — [arxiv.org/abs/2506.13752](https://arxiv.org/abs/2506.13752)
- Understanding & Steering Cognitive Behaviors — [arxiv.org/abs/2512.24574](https://arxiv.org/html/2512.24574v2)
- LayerNavigator — [openreview.net/forum?id=wj4lM45xQR](https://openreview.net/forum?id=wj4lM45xQR)
- Task Arithmetic (Ilharco et al.) — [arxiv.org/abs/2212.04089](https://arxiv.org/abs/2212.04089); code: [github.com/mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors)
- SAEs for discovery, not steering — [arxiv.org/abs/2506.23845](https://arxiv.org/html/2506.23845)
- `repeng` library — [github.com/vgel/repeng](https://github.com/vgel/repeng)
- `steering-vectors` library — [github.com/steering-vectors/steering-vectors](https://github.com/steering-vectors/steering-vectors)
- Dialz toolkit — [arxiv.org/abs/2505.06262](https://arxiv.org/html/2505.06262v1)
- `nnsight` 0.6 — [nnsight.net/blog/2026/02/26/introducing-nnsight-06](https://nnsight.net/blog/2026/02/26/introducing-nnsight-06/)
- Awesome Interpretability list — [github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models)
