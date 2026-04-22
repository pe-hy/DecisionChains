# 2026 Advances — External Memory & Activation Steering

## Time horizon
Jan 1 2026 – Apr 22 2026. Dense focus on methods applicable to Qwen3-8B math reasoning. This is a pure update to `research/01` and `research/02`; everything here is Jan–Apr 2026 unless marked as 2025 context.

## New steering & representation-level methods

**RISER — Orchestrating Latent Reasoning Skills for Adaptive Activation Steering** (Ye, Yuan, Bin et al., arXiv 2601.09269, 14 Jan 2026). Builds a library of reusable reasoning steering vectors and an RL-trained lightweight router that **compositionally selects and weights them per input token/step**, an explicit generalisation of CAST from binary gating to a small mixture-of-steerings. +3.4–6.5 pp average zero-shot gain across seven benchmarks and **2–3× token efficiency** vs. CoT. Closest published analogue to our "attention over learned KV memory as router." The router is the shift.

**ODESteer — Unified ODE-Based Steering Framework for LLM Alignment** (Zhao et al., arXiv 2602.17560, 19 Feb 2026, ICLR 2026 poster). Shows conventional activation addition is the first-order approximation of an ODE whose barrier function is the log-density ratio between positive/negative activation sets; replacing the single vector with **multi-step ODE integration** gives adaptive, input-dependent steering. +5.7 pp TruthfulQA, +2.5 pp UltraFeedback, +2.4 pp RealToxicityPrompts over prior SOTA CAA variants. A clean theoretical framing that turns static CAA into a short dynamical system — directly relevant to the "one-step additive bias" vs. "iterative correction" design axis.

**FlowSteer — Steering Large Reasoning Models towards Concise Reasoning via Flow Matching** (Li, Bergner, Zhao et al., arXiv 2602.05539, 5 Feb 2026). Learns a **velocity field** via Flow Matching between verbose and concise reasoning distributions; injects a continuous nonlinear transformation, not a fixed shift. Targets CoT length rather than correctness; reports token savings while holding accuracy. Strong argument that "length-of-thought" is a nonlinear manifold shift, not a direction.

**Spherical Steering — Geometry-Aware Activation Rotation** (You, Deng, Chen, arXiv 2602.08169, 9 Feb 2026). Replaces additive shift with a **geodesic rotation** on the activation sphere, preserving norm; confidence-gated intensity. +10 pp each on TruthfulQA, COPA, Storycloze with markedly less open-ended-generation degradation than addition. Argues representation-collapse concerns of additive CAA are norm-driven.

**DIRECTER — Instruction Following via Activation Steering with Dynamic Rejection** (Kang, Kim, arXiv 2603.06745, 6 Mar 2026). Dynamic rejection decoding loop that **modulates steering strength per step** by comparing steered vs. unsteered distributions, plus attention-sensitivity layer ranking. Up to 6.5 pp accuracy gains. Concrete realisation of "adaptive, not static" steering.

**Control RL (CRL-Token) — Interpretable Token-Level SAE Steering** (Cho, Wu, Koshiyama, arXiv 2602.10437, 11 Feb 2026). RL policy that **chooses which SAE feature to amplify at each token**, with adaptive feature masking for diversity. Evaluated on Gemma-2-2B across MMLU, BBQ, GSM8K, HarmBench, XSTest; produces per-token intervention logs. The first "token-conditional, learned, interpretable" steering policy — same abstract structure as our learned-memory router, but over SAE features instead of trained K,V.

**SAE-Steering — Controllable Reasoning via Sparse Autoencoder Steering** (Fang, Wang, Xue et al., arXiv 2601.03595, 7 Jan 2026). Two-stage pipeline: recall features that amplify strategy-keyword logits (filters >99% of features), then rank by control effectiveness. **+7 pp absolute accuracy** by redirecting models from erroneous to correct reasoning paths; +15% over prior SAE-steering methods on control effectiveness. Evidence SAE steering *can* compete with CAA for reasoning control when feature selection is done right.

**STIR — Self-Distilled Tools for Internal Reasoning** (Shi, Zhu, Shi et al., arXiv 2602.04925, 4 Feb 2026). Treats reasoning enhancement as latent-trajectory control: mines latent actions from successful traces, builds a **sparse control basis library**, then applies value-modulated anchor-gated impulses during generation. +1.9–7.5 pp across six arithmetic/logical benchmarks, −35% tokens. Structurally identical to "learned library of steering motifs + router," but mined from self-distillation rather than backprop.

**HELIX — Tethered Reasoning via Manifold Steering** (arXiv 2602.17691, Feb 2026). Decouples output entropy from hallucination by tethering hidden-state trajectories to a pre-computed truthfulness manifold. Graduated steering vectors fire on **only 0.2–2.5% of tokens** — extremely sparse, decision-point-targeted intervention.

**Steered LLM Activations are Non-Surjective** (Mishra, Khashabi, Liu, arXiv 2604.09839, 10 Apr 2026). Proves that activation steering pushes the residual stream **off the manifold reachable from any discrete prompt** — formal separation between white-box steering and black-box prompting. Practical takeaway: a steering success cannot be mimicked by prompting, and steering effects don't necessarily transfer to prompt-based deployment. Validates our "learned memory is genuinely a different mechanism than prompt engineering" framing.

**Analysing the Safety Pitfalls of Steering Vectors** (Li, Fastowski, Zaradoukas et al., arXiv 2603.24543, 25 Mar 2026). CAA steering can swing jailbreak success rates by **±50–57 pp** because steering directions overlap the refusal direction. Safety caveat — we should report refusal behaviour when claiming our memory improves math.

## New external memory methods

**Trained Persistent Memory for Frozen Encoder–Decoder LLMs: Six Architectural Methods** (Jeong, arXiv 2603.16413, 17 Mar 2026). Pilot on frozen Flan-T5-XL with small trainable adapters across three injection points × four write mechanisms. Stateless baseline scores zero memory recall; at 10× capacity all six adapter designs work, at 1× only three survive — capacity is the design bottleneck. All writes/reads are **differentiable dense-vector ops**, not textual.

**Trained Persistent Memory for Frozen Decoder-Only LLMs** (Jeong, arXiv 2603.22329, 20 Mar 2026). GPT-2 companion: prefix, parallel cross-attention, **KV extension**, Hebbian memory, context-gated branch, slot-based sparse write. At 1× capacity three methods reach 7–18% retained-memory / ΔK 7–10; at 10× all converge. Directly vindicates our "small trainable memory module on a frozen backbone" architecture on a decoder LLM.

**Memory Bank Compression for Continual Adaptation** (Katraouras, Rafailidis, arXiv 2601.00756, 2 Jan 2026, SAC '26). Codebook-optimisation + online reset to prevent collapse, combined with **KV-LoRA** in attention. Memory bank shrinks to **0.3% of the most competitive baseline** while retaining accuracy. Good reference for quantised / compressed memory designs.

**PRIME — Training-Free Proactive Reasoning via Iterative Memory Evolution** (Wang, Jiang, arXiv 2604.07645, 8 Apr 2026). Gradient-free agent memory: structured experiences in {successful strategies, failure patterns, user preferences}, evolved by meta-ops, retrieved via RAG. Competitive with gradient-based fine-tuning on user-centric benchmarks; notable as a **pure non-parametric alternative** baseline our memory must beat.

**Beyond Speedup — Utilizing KV Cache for Sampling and Reasoning** (Xing, Li, Zhen et al., arXiv 2601.20326, 28 Jan 2026). Uses the KV cache itself as a representation for Chain-of-Embedding and fast/slow reasoning switching. **5.7× token reduction with minimal accuracy loss on Qwen3-8B and DeepSeek-R1-Distil-Qwen-14B.** Directly relevant to Qwen3-8B and to "residual/KV as substrate" intuitions.

**How Much Cache Does Reasoning Need? Depth–Cache Tradeoffs** (arXiv 2604.17935, Apr 2026). Theoretical + empirical study of KV cache compression vs. multi-step reasoning degradation. Useful for sizing our memory relative to KV footprint.

## Reasoning-specific steering in 2026

- **RISER** — multi-skill router over reasoning vectors (above).
- **SAE-Steering** — +7 pp by redirecting paths (above).
- **STIR** — latent-action replay, +1.9–7.5 pp, −35% tokens (above).
- **HELIX** — 0.2–2.5% token-sparse steering for hallucination control (above).
- **CRL-Token** — token-level SAE feature selection (above, evaluated on GSM8K).
- **FlowSteer** — CoT-length control via flow matching (above).
- **SPINE** (arXiv 2511.17938, late 2025 — NeurIPS 2025 context) — distribution-aware **forking-token selection** for test-time RL, explicit descendant of "Beyond 80/20."
- **PRA — Process Reward Agents** (Sohn et al., arXiv 2604.09482, 10 Apr 2026). Test-time **online step-wise reward** to a **frozen policy** from a domain-grounded agent; search-based decoding prunes bad trajectories on the fly. 80.8% MedQA with Qwen3-4B (SOTA at 4B); up to +25.7 pp across 0.5B–8B frozen policies. Conceptually: "steer by ranking next-step candidates with an external reward," a cousin of our decision-point approach.
- **PROGRS** (arXiv 2604.02341, Feb 2026). Outcome-conditioned centering of PRM scores, integrated into GRPO. Small PRM steering tweak that materially affects RLVR geometry.
- **Sparse but Critical — Token-Level Analysis of RLVR Distributional Shifts** (Meng et al., arXiv 2603.22446, 23 Mar 2026). **Injecting only a few RL-sampled tokens into base generations recovers the gains; injecting base tokens into RL sequences collapses them** — RLVR's effect is carried by a sparse set of decision tokens. Strong direct follow-up to Wang "Beyond 80/20."

## Parameter-efficient competitors

**Learning to Reason in 13 Parameters (TinyLoRA)** (Morris, Mireshghallah, Ibrahim, Mahloujifar, arXiv 2602.04118, 4 Feb 2026). Scales LoRA down to **13 trained parameters (26 bytes bf16)** on **Qwen2.5-8B** via RL; **91% GSM8K**, recovers ~90% of RL gains on AIME/AMC/MATH500, and notes SFT needs 100–1000× larger updates to match. This is the new "no-parameters-needed" floor we must compare against — our ~270 K-param memory should either decisively beat this or we reframe the contribution.

**Adapter Merging Reactivates Latent Reasoning Traces** (Zou, arXiv 2601.18350, 26 Jan 2026). Merging DAPT + SFT LoRAs resurfaces "explicit reasoning traces" via misaligned directional updates; rank-1 logit-space intervention modulates decision distributions. Evaluated on Qwen3-14B, with Qwen3-8B as separate eval — **same backbone family as us.** Reinforces the "low-rank additive directions can flip reasoning branch" picture.

## Follow-ups to 2025 landmarks

- **Wang "Beyond 80/20" (2506.01939):** Sparse but Critical (2603.22446) provides the cleanest cross-sampling evidence that RLVR gains live in a tiny token set. SPINE (late 2025) operationalises token-selective test-time RL from this. Budget Guidance / PROGRS / CRL-Token all implicitly ride the same assumption.
- **Goncharov 2025 bias-only (2505.18706):** 2026 work has moved past plain bias-only to **learned, input-conditional** shifts (RISER, ODESteer, DIRECTER, CRL-Token). The "per-layer MLP bias" is now the minimal baseline, not the frontier — we should run it as a baseline.
- **Budget Guidance (2506.13752):** Refined and extended by FlowSteer (nonlinear length steering) and STU-PID (PID-controlled steering strength).
- **RepE / CAA / ITI / ActAdd (2023):** The 2026 thread is unambiguously *dynamic* and *input-conditional* — static CAA is the baseline to beat, not the SOTA.
- **Persona Vectors (2507.21509):** 2026 work (2603.24543, 2604.11120) extends them to safety audit and persona-method evaluation. Less relevant for math, but a reminder to log refusal/safety behaviour.
- **KV cache steering — Belitsky (2507.08799):** Now echoed by 2601.20326 using the KV cache as a reasoning substrate, and by the two persistent-memory-on-frozen-LLM papers (2603.16413, 2603.22329) which extend trainable-KV to encoder-decoder and decoder-only.
- **Math-Shepherd / PRM800K:** PRA (2604.09482) and PROGRS (2604.02341) now deploy PRMs as *frozen* online steerers rather than offline rerankers.

## What this changes for our Phase-1 design

- **Router/gating is the 2026 default, not an ablation.** Add the RISER-style "library of K reasoning vectors + learned router" as a named ablation against our attention-over-KV lookup. Both answer "which behaviour, this token?" — we should show our learned K,V router wins.
- **Token-conditional fires are now expected.** HELIX fires on 0.2–2.5% of tokens; SPINE/Sparse-but-Critical show RLVR gains live in <20% of tokens. Our evaluation should report **at which token positions the memory actually changes logits** — this is our analogue of forking tokens and lines up with prior decision-chain findings.
- **Add TinyLoRA (13 params) as the minimum-parameter baseline.** If 13 params give 91% GSM8K on Qwen2.5-8B via RL, our ~270 K-param memory must either (a) clearly beat RL-TinyLoRA on a target benchmark or (b) offer a capability TinyLoRA can't — input-conditional branching, interpretability, or out-of-distribution generalisation on decision chains.
- **Baseline stack to run:** (i) static CAA (2023), (ii) Goncharov bias-only (2505.18706), (iii) SAE-Steering (2601.03595), (iv) RISER-style router over static vectors (2601.09269), (v) TinyLoRA (2602.04118), (vi) KV-cache steering (2507.08799).
- **Geometry check.** Spherical Steering (2602.08169) and ODESteer (2602.17560) argue that additive residual shifts are the crude first-order form; track (a) post-memory activation norm drift and (b) whether multi-step application helps. Cheap to add.
- **Safety sanity check.** Log MT-Bench / refusal-rate before/after memory to avoid the 2603.24543 pitfall.
- **Test on Qwen3-8B, not just smaller models.** 2601.20326 and 2601.18350 both use Qwen3-8B/14B — there is now peer-group evidence that steering-like methods on Qwen3 are viable and publishable.

## Sources

- [RISER — arXiv 2601.09269 (14 Jan 2026)](https://arxiv.org/abs/2601.09269)
- [ODESteer — arXiv 2602.17560 (19 Feb 2026, ICLR 2026)](https://arxiv.org/abs/2602.17560)
- [FlowSteer — arXiv 2602.05539 (5 Feb 2026)](https://arxiv.org/abs/2602.05539)
- [Spherical Steering — arXiv 2602.08169 (9 Feb 2026)](https://arxiv.org/abs/2602.08169)
- [DIRECTER — arXiv 2603.06745 (6 Mar 2026)](https://arxiv.org/abs/2603.06745)
- [Control RL (CRL-Token) — arXiv 2602.10437 (11 Feb 2026)](https://arxiv.org/abs/2602.10437)
- [SAE-Steering — arXiv 2601.03595 (7 Jan 2026)](https://arxiv.org/abs/2601.03595)
- [STIR — arXiv 2602.04925 (4 Feb 2026)](https://arxiv.org/abs/2602.04925)
- [HELIX / Tethered Reasoning — arXiv 2602.17691 (Feb 2026)](https://arxiv.org/abs/2602.17691)
- [Non-Surjective Steering — arXiv 2604.09839 (10 Apr 2026)](https://arxiv.org/abs/2604.09839)
- [Safety Pitfalls of Steering Vectors — arXiv 2603.24543 (25 Mar 2026)](https://arxiv.org/abs/2603.24543)
- [Persistent Memory, Encoder–Decoder — arXiv 2603.16413 (17 Mar 2026)](https://arxiv.org/abs/2603.16413)
- [Persistent Memory, Decoder-Only — arXiv 2603.22329 (20 Mar 2026)](https://arxiv.org/abs/2603.22329)
- [Memory Bank Compression — arXiv 2601.00756 (2 Jan 2026, SAC '26)](https://arxiv.org/abs/2601.00756)
- [PRIME — arXiv 2604.07645 (8 Apr 2026)](https://arxiv.org/abs/2604.07645)
- [Beyond Speedup: KV Cache for Reasoning — arXiv 2601.20326 (28 Jan 2026)](https://arxiv.org/abs/2601.20326)
- [Depth–Cache Tradeoffs — arXiv 2604.17935 (Apr 2026)](https://arxiv.org/abs/2604.17935)
- [Process Reward Agents (PRA) — arXiv 2604.09482 (10 Apr 2026)](https://arxiv.org/abs/2604.09482)
- [PROGRS — arXiv 2604.02341 (Feb 2026)](https://arxiv.org/abs/2604.02341)
- [Sparse but Critical — arXiv 2603.22446 (23 Mar 2026)](https://arxiv.org/abs/2603.22446)
- [TinyLoRA (13 parameters) — arXiv 2602.04118 (4 Feb 2026)](https://arxiv.org/abs/2602.04118)
- [Adapter Merging Reactivates Reasoning — arXiv 2601.18350 (26 Jan 2026)](https://arxiv.org/abs/2601.18350)
- 2025 context: [Wang "Beyond 80/20" 2506.01939](https://arxiv.org/abs/2506.01939) · [Goncharov bias-only 2505.18706](https://arxiv.org/abs/2505.18706) · [Budget Guidance 2506.13752](https://arxiv.org/abs/2506.13752) · [KV Cache Steering / Belitsky 2507.08799](https://arxiv.org/abs/2507.08799) · [Persona Vectors 2507.21509](https://arxiv.org/abs/2507.21509)
