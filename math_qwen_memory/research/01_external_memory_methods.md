# External Memory & Residual-Stream Interventions — Literature Review

## Overview

Methods that augment a (usually frozen) transformer with external state form a broad family whose unifying claim is: *most knowledge and behavior can be surfaced by a small, trainable module that reads/writes the residual stream, without touching the backbone weights*. The family spans (a) **memory banks** — learned or retrieval-built key/value tables queried by attention (MemNN, PKM, Memorizing Transformers, RETRO, kNN-LM, Memory Layers); (b) **learnable context** — soft prompts/prefixes that live in the input/KV cache (Prompt Tuning, Prefix Tuning, P-tuning v2); (c) **residual-stream edits** — low-rank additive interventions in activation space (steering vectors, CAA, RepE, ITI, activation patching); and (d) **small modular adapters** hanging off frozen weights (LoRA, IA³, adapters).

The axes of variation that matter for our use case are: (i) **where** the module is attached — embedding space, residual stream at a specific layer, attention KV, or FFN; (ii) **what is learned** — keys + values, only values (fixed key index), bias/scale vectors, low-rank deltas, or nothing (retrieval-only); (iii) **how the result is combined** — additive into residual (h + γ·m), gated, concatenated into KV cache, or interpolated with output logits; (iv) **how keys are indexed** — dense dot product, product keys for sub-linear lookup, or ANN over a non-differentiable datastore. Our current design (frozen backbone, ~270K params, h' = h + γ·softmax(qK/√d)V at a single mid-layer) sits at the intersection of "memory layers" (trainable K,V) and "residual-stream steering" (additive, single site).

A second unifying idea: because the frozen backbone already contains strong linguistic/reasoning circuits, the module's job is generally **biasing decisions at bottleneck tokens** rather than re-implementing computation. This reframes our "decision-chain" observation — that the module shifts probabilities at choice tokens — as the expected behavior, not an artifact.

## Key methods

### Classical memory-augmented NNs

**Memory Networks (Weston, Chopra, Bordes — 2014, arXiv:1410.3916).**
- External array of memory slots, addressed by a scoring function of (query, slot).
- Originally required supervision on which memory slot to attend to; brittle.
- Introduced the core vocabulary: input, generalization, output, response modules; multi-hop reading.

**End-to-End Memory Networks (Sukhbaatar, Szlam, Weston, Fergus — NeurIPS 2015).**
- Made the above fully differentiable: softmax attention over memory with K hops.
- Embedding matrices A (keys) and C (values) are learned; queries come from recurrent state.
- Conceptually identical to our module: q·K → softmax → ·V → add into state. Multi-hop reading prefigures multi-layer steering.

**Key-Value Memory Networks (Miller et al. — EMNLP 2016).**
- Separates key (addressing) and value (content) matrices, allowing asymmetric encodings.
- QA from Wikipedia/KB; ~1.6pt F1 gain over memory-net baselines.
- This is the literal name of our architecture at a frozen LM's layer L.

### Modern transformer memory

**Product Key Memory / PKM (Lample et al. — NeurIPS 2019, arXiv:1907.05242).**
- Trainable memory slots (|K|=512² = 262k) addressed via Cartesian-product subkeys → sub-linear top-k.
- 12-layer + PKM beats 24-layer transformer; ~1B extra params, negligible FLOP overhead.
- Directly precedes Meta's Memory Layers (same group of authors).

**kNN-LM (Khandelwal et al. — ICLR 2020, arXiv:1911.00172).**
- Datastore of (context-hidden-state, next-token) pairs from training data. At test, linearly interpolate p_LM(y|x) with p_kNN(y|x).
- No training needed; drop-in. 2.9 perplexity improvement on WikiText-103 (15.79 SOTA at the time).
- Shows that frozen representations + large datastore + interpolation is enough — heavy analog of our "biased decision" finding.

**Memorizing Transformers (Wu, Rabe, Hutchins, Szegedy — ICLR 2022, arXiv:2203.08913).**
- Adds a non-differentiable kNN index over past (K, V) pairs at a single attention layer. External memory of up to 262k tokens.
- Gradients not backpropagated into memory; memory values are just cached activations.
- Gains on arXiv-math, code, theorems. Highly relevant: single-layer attention-over-external-KV, frozen reads. Differs from ours in that the keys are **cached activations**, not trained.

**RETRO (Borgeaud et al. — ICML 2022, arXiv:2112.04426).**
- 2T-token BERT-based retrieval datastore; chunked cross-attention decoder injects retrieved neighbors.
- 25× smaller than GPT-3 at comparable PPL on The Pile.
- Heavier (retriever + CCA layers) and text-retrieval-based; but validates that "inject retrieved tokens into residual via attention" is a scalable recipe.

**Memory3 (Yang et al., Shanghai AI Lab — 2024, arXiv:2407.01178).**
- Externalizes knowledge as KV blocks written from training text, retrieved and read by standard self-attention alongside the context KV cache.
- Sparsification + two-stage pretraining. 2.4B model beats larger dense LLMs and RAG baselines.
- "Third memory" framing: parameters = implicit, context KV = working, explicit external = long-term.

**Memory Layers at Scale (Meta — December 2024, arXiv:2412.09764; github.com/facebookresearch/memory).**
- Replaces some FFN blocks with trainable product-key KV stores (up to 128B memory params, sparse activation).
- Outperforms compute-matched dense and MoE baselines, especially on factual tasks.
- Closest modern analog to our module, at massive scale; their "swap one FFN for a memory layer" is a strong prior for where to attach ours (mid-network, FFN slot).

### Soft / learnable context

**Prefix Tuning (Li & Liang — ACL 2021, arXiv:2101.00190).**
- Learns continuous prefix vectors that are prepended to the KV cache at every layer; 0.1% of params.
- Competitive with full fine-tune on GPT-2/BART generation; extrapolates better in low-data regimes.
- Conceptually: "virtual memory entries attended by every token" vs. ours "memory attended only by one layer's queries."

**Prompt Tuning (Lester, Al-Rfou, Constant — EMNLP 2021, arXiv:2104.08691).**
- Even simpler: learn a short sequence of soft tokens only at the input embedding layer. Matches full fine-tune at ≥10B params.
- A degenerate case of external memory (single write position, fixed read via normal self-attention).

**P-tuning v2 (Liu et al. — 2021, arXiv:2110.07602).**
- Soft prompts at every layer (not just input) — essentially prefix tuning re-branded, but robust across scales/tasks without reparameterization tricks.
- Useful reference for deciding "one layer vs all layers" for our memory hook.

### Mechanistic / activation-level interventions

**Activation patching / causal tracing (Meng et al. 2022, Heimersheim & Nanda 2024 — arXiv:2404.15255).**
- Causal counterfactual: run clean + corrupt inputs, copy activations from one to the other, measure output change. Identifies **which** layers/heads carry specific information.
- Strong evidence that factual/decision circuits localize to narrow layer bands — middle layers for mid-stream features.

**Steering vectors / CAA / RepE / ITI (2023–2024).**
- Add a single learned or contrast-derived direction v into the residual at layer ℓ: h'_ℓ = h_ℓ + α·v.
- Small, composable, inference-only. Effective for behavioral axes (refusal, honesty, sentiment).
- Sibling doc covers in detail. Key contrast with our module: steering vectors are **static** (one vector per concept), our memory is **query-dependent** (softmax over K picks a vector per token). That query-dependency is much closer to actual attention and should be strictly more expressive than CAA for structured decisions.

**Layer-selection empirics (Qwen2.5/Llama activation-steering studies, 2024–2025).**
- Middle layers (10–15 of 28 for Qwen2.5-7B; ~6–18 of 48 for GPT-2-XL) most effective for semantic steering.
- Effects are cleaner and stronger in Qwen than Llama at equal size — relevant for us.
- Depth-scheduled (Gaussian-weighted) multi-layer steering beats single-layer for many tasks.

### IA³ / LoRA / adapters (context only)

**LoRA (Hu et al. — 2021, arXiv:2106.09685).** Low-rank ΔW on weight matrices; 10 000× fewer params than full FT, no inference latency. Not memory, but philosophically related small-module-on-frozen-backbone. Often the strong baseline we must beat.

**IA³ (Liu et al. — 2022).** Three learned scaling vectors (keys, values, FFN activations) that rescale activations; ~0.01% params. Cheapest non-trivial intervention in the PEFT literature. Good sanity-check baseline for "what fraction of our memory's benefit is just multiplicative scaling."

### Most recent (2024–2026) for reasoning

- **Memory Layers at Scale (Meta, Dec 2024)** — see above. Factual-task gains especially large; math gains more modest. Suggests our approach should help memorization-style sub-problems in MATH, less so pure multi-step algebra.
- **MemReasoner (IBM, NeurIPS 2024, arXiv:2412).** Latent memory module external to a transformer, read iteratively with bi-GRU until a stable readout. Generalizes multi-hop reasoning over 128k-token documents better than off-the-shelf LLMs. Read-until-stable echoes multi-hop End-to-End MemNN.
- **Memory-Augmented RL for ≤1B LLMs (Apr 2025, arXiv:2504.02273).** RL fine-tunes a small LLM with an intrinsic reward from a memory of past successful trajectories. +5% GSM8K, +4% MATH-500, +1% AIME24 over no-memory baseline. Strong signal that memory of *past traces* helps math, not just facts.
- **Engram (DeepSeek, 2026, arXiv:2601.07372).** Conditional memory via scalable lookup; sparsity as a new axis. Newer product-key-style memory.

## Actionable takeaways for Qwen3-8B + math reasoning

Qwen3-8B has 36 layers, hidden size 4096, 32 attention heads (128-dim each), GQA with 8 KV heads. 40GB A100 runs bf16 inference comfortably (~16GB model) with room for our memory module plus activation memory.

**Where to hook.** Based on Qwen2.5-7B steering literature and Meta Memory Layers practice:
- **Primary target: layers 16–22** (mid-to-late third). Qwen middle layers carry the cleanest semantic features; late-middle is where "decision" features typically emerge.
- **Consider multi-layer hooks** (Gaussian-weighted, 3–5 adjacent layers) before committing to single-layer. The 2025 depth-scheduling papers consistently beat single-layer in Qwen.
- Avoid layer 0 (too low-level, like input-embedding prompt tuning) and the last 1–2 layers (too close to unembedding — just biases token logits, no reasoning uplift).

**What to learn.** 
- Trainable (K, V) tables with M ≈ 4k–64k slots, head dim 128 (matching Qwen heads). Memory params ≈ 2·M·128 = 1M–16M; still ~0.01–0.2% of 8B — well inside your prior 270K budget by log scale.
- Use **product keys** if M > 16k to keep lookup sub-linear and avoid softmax-over-everything latency.
- Try both **multi-head memory** (8 heads, matching KV) and **single-head** with larger d — PKM paper favors 4–8 heads.

**Injection formula.**
- Start with pure additive: h' = h + γ·Attn(q=W_q h, K, V), γ learned per-layer or scalar, init small (1e-2). This mirrors steering / CAA and keeps the network in its pretraining distribution initially.
- A gated variant h' = h + σ(g(h))·Attn(...) is strictly more expressive and matches adapter-style practice. Worth an ablation.

**Parameter budget.** Practical sweet spot for 8B: **1M–20M trainable params** (0.01–0.25% of model). Beyond ~50M you're in LoRA territory where full LoRA usually wins.

**Training.** bf16 backbone frozen (no grad), memory module in fp32. gradient checkpointing of backbone for a 40GB card; Qwen3-8B + 8k context + batch 1 already uses ~22GB of activations otherwise. Keep memory module small enough to use Adam without trouble.

**Pitfalls specific to 8B natural-language models.**
1. Tokenization dominates math: Qwen3 uses a BPE that splits numbers into digits (usually). The decision-point analogy to our synthetic chains only holds at *semantic* decision tokens (operator choices, case-splits), not every token. Label which tokens to bias carefully.
2. Qwen3 has "thinking mode" with explicit `<think>` tags. Our memory should be hooked inside the thinking span, not on the final answer — otherwise the intervention is downstream of the reasoning.
3. GQA: W_q is per-head but K,V are shared across 4 heads. If you mirror GQA in the memory module (8 KV heads) params stay small; if you use 32 independent memory heads you quadruple cost for likely marginal gain.
4. Flash-attention kernels don't natively support injecting custom (K, V) pairs mid-layer — you'll need to hook on the residual stream *after* attention, not inside it, unless you're willing to write/borrow a custom kernel.
5. Catastrophic interference at high γ: steering literature consistently finds degradation at strengths >2–3. Start γ small, consider regularizing ||Attn(...)|| or KL to unmodified outputs on held-out prompts.

## Open questions

- Does query-dependent external memory give uplift *over* static steering vectors on multi-step math, or is a simple CAA-style direction at the right layer enough? (Expected: memory wins on branching decisions, ties on global style shifts — but needs measurement.)
- Single-layer vs. multi-layer hook: on a 36-layer model, is the optimal regime 1 hook, 3 adjacent, or a full depth schedule?
- How does a learned memory compare to a **retrieval-only** memory (kNN over training chains' hidden states) for math? The 2025 memory-RL paper says retrieval of *past solution trajectories* helps; would our setup improve by initializing V from such trajectories?
- What's the smallest M (memory-slot count) that still works? Our decision-chain result suggests tens; math algebra may need thousands.
- Does freezing the memory after initialization (à la Memorizing Transformers, no grad through V) close the gap, or do trainable values matter?
- Interaction with Qwen's "thinking" / chain-of-thought traces: does injecting memory during `<think>` help the model reach correct branches, or does it corrupt chain coherence?

## Sources

- [Weston et al., Memory Networks (2014)](https://arxiv.org/abs/1410.3916)
- [Sukhbaatar et al., End-To-End Memory Networks (NeurIPS 2015, arXiv:1503.08895)](https://arxiv.org/abs/1503.08895)
- [Miller et al., Key-Value Memory Networks (EMNLP 2016, arXiv:1606.03126)](https://arxiv.org/abs/1606.03126)
- [Lample et al., Large Memory Layers with Product Keys (NeurIPS 2019, arXiv:1907.05242)](https://arxiv.org/abs/1907.05242)
- [Khandelwal et al., Generalization through Memorization: Nearest Neighbor Language Models (ICLR 2020, arXiv:1911.00172)](https://arxiv.org/abs/1911.00172)
- [Wu et al., Memorizing Transformers (ICLR 2022, arXiv:2203.08913)](https://arxiv.org/abs/2203.08913)
- [Borgeaud et al., Improving Language Models by Retrieving from Trillions of Tokens — RETRO (ICML 2022, arXiv:2112.04426)](https://arxiv.org/abs/2112.04426)
- [Yang et al., Memory³: Language Modeling with Explicit Memory (2024, arXiv:2407.01178)](https://arxiv.org/abs/2407.01178)
- [Berges et al. (Meta), Memory Layers at Scale (Dec 2024, arXiv:2412.09764)](https://arxiv.org/abs/2412.09764) — [code: facebookresearch/memory](https://github.com/facebookresearch/memory)
- [Li & Liang, Prefix-Tuning (ACL 2021, arXiv:2101.00190)](https://arxiv.org/abs/2101.00190)
- [Lester, Al-Rfou, Constant, The Power of Scale for Parameter-Efficient Prompt Tuning (EMNLP 2021, arXiv:2104.08691)](https://arxiv.org/abs/2104.08691)
- [Liu et al., P-tuning v2 (arXiv:2110.07602)](https://arxiv.org/abs/2110.07602)
- [Hu et al., LoRA (2021, arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
- [IA³ — Liu et al., Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (arXiv:2205.05638)](https://arxiv.org/abs/2205.05638) — [HF docs](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [Heimersheim & Nanda, How to Use and Interpret Activation Patching (2024, arXiv:2404.15255)](https://arxiv.org/abs/2404.15255)
- [Ko et al., MemReasoner (NeurIPS 2024)](https://research.ibm.com/publications/memreasoner-a-memory-augmented-llm-architecture-for-multi-hop-reasoning)
- [Memory-Augmented RL for sub-1B LLMs (Apr 2025, arXiv:2504.02273)](https://arxiv.org/abs/2504.02273)
- [Memory-Augmented Transformers: A Systematic Review (2025, arXiv:2508.10824)](https://arxiv.org/abs/2508.10824)
- [Qwen3 Technical Report (May 2025, arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3-8B model card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-8B)
- [Scaling laws for activation steering with Llama 2 (2025, arXiv:2507.11771)](https://arxiv.org/abs/2507.11771)
- [Small Vectors, Big Effects: RL-Induced Reasoning via Steering Vectors (2025, arXiv:2509.06608)](https://arxiv.org/abs/2509.06608)
- [Depth-Wise Activation Steering for Honest Language Models (2025, arXiv:2512.07667)](https://www.arxiv.org/abs/2512.07667)
