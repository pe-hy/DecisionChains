# 2026 Advances — PEFT & Qwen Math Fine-tune

## Time horizon
January 2026 – April 2026. Companion to `research/06_lora_peft_math.md` (2021-2025 coverage); this file focuses only on post-2025 developments that actually move the needle for an 8B-class math fine-tune.

---

## New PEFT methods 2026

| Method | arXiv / venue | Date | Trainable params | Math delta vs vanilla LoRA (r=16) | Notes |
|---|---|---|---|---|---|
| **TLoRA+** | 2604.13368 | 2026-04 | Same as LoRA r=16 | ≈ +0.3–0.8 pt GLUE (no MATH eval yet) | Wraps AdamW with a TLoRA+ optimizer inside adapter matrices; drop-in for LoRA. Warmup 0.1, bs=16, 30 epochs, "all-linear" targets. |
| **PEFT-Factory** (framework) | 2512.02764 (v2 2026-02-22) | 2026-02 | — | Standardised eval of 19 PEFT methods | Not a new method per se; gives one codepath to compare DoRA/PiSSA/VeRA/LoRA/BOFT under identical data & schedule. |
| **PEFT-Bench** | 2511.21285 (2026-02) | 2026-02 | — | Defines **PSCP** score (params × inference-speed × mem) | First apples-to-apples benchmark; LoRA + DoRA still on Pareto front for ≤1B-adapter scale. |
| **NEAT** (nonlinear adapter, refreshed 2026 results) | 2410.01870 | arxiv-baseline refreshed on 2026 leaderboards | ~0.5× LoRA params | +1.1–1.6 pt MATH500 at matched params (reported in PEFT-Bench) | Adds a small MLP non-linearity inside the low-rank path — closes full-FT gap at low rank. |
| **1LoRA (𝟙LoRA)** | 2503.08333 | Mar 2025, reproduced & re-ranked in PEFT-Bench 2026 | **d params/layer** (≤1 k–10 k total for 8B) | ≈ −0.5 pt MMLU vs LoRA-r16, but 20× fewer params | Sum-pool input features + single learnable decompression vector. Strong "ultra-light" baseline. |
| **S2FT** (structured sparse FT) | 2412.06289 (v2 2026) | 2026-02 update | ~1–3% of model | Matches LoRA-r16 on GSM8K with higher throughput | "Select sparsely, compute densely" — lives between LoRA and full-FT in quality/speed. |
| **D-QReLO** (delta compression) | 2604.16940 | 2026-04 | Post-hoc, training-free | Pure compression — preserves math accuracy within 0.3 pt | Compress a *trained* LoRA/full-FT delta to ≤1-bit + low-rank residual for deployment. Orthogonal to the FT recipe. |

**Takeaway**: Nothing in Q1–Q2 2026 dethrones LoRA/DoRA for ≤8B. The real movement is (a) standardised benchmarking (PEFT-Factory, PEFT-Bench), (b) nonlinear low-rank adapters (NEAT), and (c) ultra-low-parameter baselines (1LoRA, S2FT). DoRA remains the default in Unsloth's 2026 docs.

---

## Qwen3.x releases 2026

| Model | Release | Math-relevant highlights | Source |
|---|---|---|---|
| **Qwen3.5 / Qwen3.5-Plus** | 2026-02-16 | 397B-A17B MoE flagship; small series (Flash, 27B, 35B-A3B, 122B-A10B) released March 2026. Qwen3.5-122B-A10B: ~85% AIME 2026. Qwen3.5-35B-A3B > Qwen3-235B on GSM8K. | Qwen blog, n1n.ai guide |
| **Qwen3-Coder-Next** | 2026-02 | Hybrid attention + MoE on Qwen3-Next-80B-A3B-Base; agentic RL training. Not math-specific but strong on code-heavy reasoning. | qwen.ai blog |
| **Qwen3.6 series** | 2026-04-11 | 72B dense, 480B MoE, 235B MoE, plus 32B/14B/4B Coder variants. QLoRA on 32B Coder: 100K examples, ~14 h on 1× A100-80GB, +5–12 pt on domain benchmarks. | Qwen release notes |
| **Qwen3-Math (dedicated)** | *No dedicated Qwen3-Math-8B release yet as of 2026-04-22.* | Qwen2.5-Math-7B remains the last math-specialised checkpoint; community math work still uses Qwen3-8B-Base/Instruct. | github.com/QwenLM |

**Practical implication for us**: our target (Qwen3-8B) has not been superseded by a "Qwen3-Math-8B". Qwen3.5 is MoE-only in the small sizes, so a dense-8B math recipe is still Qwen3-8B-based.

---

## Published Qwen3-8B math LoRA recipes 2026

Concrete hyperparameters assembled from Unsloth docs, NeMo recipes, HF tutorials, and community 2026 runs:

| Source | Target | Rank/α | Targets | LR | BS / accum | Epochs | Quant | Reported result |
|---|---|---|---|---|---|---|---|---|
| Unsloth 2026 Qwen3 guide | Qwen3-8B SFT | r=32, α=32 | all-linear | 2e-4 cosine, warmup 0.1 | bs=8, accum=4 (eff 32) | 2–3 | 4-bit NF4 + DoRA (`use_dora=True`) | "Reasoning transfer" sweet spot; no MATH delta published |
| NVIDIA NeMo 25.07 Qwen3-8B recipe | Qwen3-8B LoRA | r=16, α=32 | all-linear LM layers | 1e-4 | bs=1 × 16 accum × 8 GPU | 1 | bf16 | Fits in 1× A100-40GB with 4-bit base |
| HF `optimum-neuron` Qwen3-8B LoRA tutorial | Qwen3-8B | r=16, α=32 | q,k,v,o + MLP | 5e-5 AdamW | bs=4 × 4 accum | 3 | bf16 on Trainium | General instruction tuning baseline |
| DataCamp Qwen3 math tutorial (Jan 2026) | Qwen3-4B-Instruct on GSM8K | r=32, α=64 | all-linear | 2e-4 | bs=2 × 8 accum | 2 | 4-bit | ~78% → ~86% GSM8K |
| `tahamajs/Qwen3-4B-GSM8k-GRPO-Unsloth` (HF card, 2026-Q1) | Qwen3-4B on 10% GSM8K | r=16 LoRA + GRPO | attention | 5e-6 | bs=1 × 16 | n/a | 4-bit | Trains stably on 1× 24GB via Unsloth |
| Unsloth "Qwen3-8B DAPO Math + vLLM" notebook | Qwen3-8B, GSM8K/MATH | r=32, α=32 | all-linear | 5e-6 (RL) | 128 groups × 8 rollouts | — | 4-bit base + LoRA | DAPO reward-shaping + vLLM generation backend |

**Dominant 2026 pattern for Qwen3-8B math**:
- r=32, α=32 (scaling 1.0, per Unsloth DoRA guidance), `target_modules="all-linear"`.
- LR 2e-4 for SFT, 5e-6 for RL stages.
- 4-bit NF4 + DoRA (`use_dora=True`) — "2026 favourite" per multiple guides.
- Warmup 0.1 cosine, 2–3 SFT epochs, effective batch 32.
- RL stage always follows a short SFT warmup.

---

## RLVR / GRPO updates 2026

| Variant | Paper | Δ vs vanilla GRPO | Mechanism |
|---|---|---|---|
| **DAPO** | 2503.14476 (baseline, still dominant in 2026 Unsloth notebook) | AIME24: DAPO-trained Qwen2.5-32B → 50, beating DeepSeek-R1-Zero at 50% training steps | Clip-Higher, Dynamic Sampling, token-level PG loss, Overlong reward shaping (buffer 4096, penalty 1.0/token) |
| **DGPO / MathForge** | 2601.20614 (ICLR 2026) | On Qwen2.5-Math-7B: average MATH-suite 37.61 → **39.79** (DGPO alone), → **42.17** (full MathForge) | Difficulty-balanced group advantage (mean absolute deviation instead of std); difficulty-aware question weighting; Multi-Aspect Question Reformulation for synthetic hard-negative generation |
| **GRPO-LEAD** | 2504.09696 (EMNLP 2025 / 2026 refresh) | Cons@32 0.833 beating DeepSeek-14B (0.800) | Length-regularised reward + explicit wrong-answer penalty + difficulty-aware advantage reweighting |
| **GTPO / GRPO-S** | 2508.04349 (2026 v6) | Better long-CoT stability | Entropy-based weight ratios redistribute token- and sequence-level rewards; avoids entropy collapse |
| **TROLL (trust region for GRPO)** | 2510.03817 (ICLR 2026) | +3–10 pt (5–15% relative) on DAPO-Math with Qwen3 | Replaces clip with explicit trust-region constraint; stabilises long rollouts |
| **f-GRPO** | 2602.05946 | Divergence-family generalisation | Replaces KL with configurable f-divergence; marginal math gains, mainly stability |

**Recommendation**: start with vanilla GRPO matching the repo's `base_grpo.yaml`, then A/B (a) DAPO's four additions (especially token-level PG loss and overlong reward shaping) and (b) DGPO's difficulty-balanced advantage estimator. Both are small code changes and have the largest reported deltas.

---

## Data curation 2026

| Dataset | Scale | Source | Use |
|---|---|---|---|
| **nvidia/OpenMathReasoning** | 306K unique problems, multi-million solutions | AoPS problems; solutions via DeepSeek-R1 + QwQ-32B; preprocessing via Qwen2.5-32B-Instruct | Was the foundation of the AIMO-2 Kaggle winner. Primary math-SFT corpus of 2026. |
| **SYNTHETIC-1** | ~2M reasoning traces | Collaboratively generated via Prime Intellect from DeepSeek-R1 | Large, messy; good for high-diversity SFT warmup |
| **DAPO-Math** (ByteDance Seed) | Bundled with DAPO release | Competition + synthetic hard | Standard RL-stage dataset; used by TROLL and most 2026 GRPO papers |
| **MQR-augmented sets** (from MathForge) | Derived | Multi-aspect reformulations of existing problems | Increases difficulty while preserving gold answers — useful synthetic expansion |

**Curation trend**: 2026 papers overwhelmingly mix (a) OpenMathReasoning-style R1-distilled solutions for SFT, (b) DAPO-Math / MATH / GSM8K for RL, and (c) difficulty-stratified subsets (hardest 25% up-weighted). No Qwen2.5-Math / Qwen2-Math successor dataset has been released by Alibaba in Q1–Q2 2026.

---

## Catastrophic forgetting / continual fine-tune 2026

1. **Self-Distillation Fine-Tuning (SDFT)** — MIT + ETH Zurich (Feb 2026). Frozen teacher guides student via in-context demonstrations; reported to keep prior-task scores flat (64.5%) while improving target task. **Cost**: ~2.5× SFT compute. Useful when we want Qwen3-8B to gain chain-following without losing general math. (VentureBeat/InfoWorld coverage; Mischa Dohler summary.)
2. **Mechanistic analysis of forgetting** (2601.18699, Jan 2026) — identifies three mechanisms: gradient interference in attention weights, representational drift in intermediate layers, loss-landscape flattening around prior minima. Implies: **freeze attention, tune MLP** is a principled choice when prior skills matter.
3. **Hierarchical Layer-Wise & Element-Wise Regularisation** (2501.13669 v2, early 2026) — EWC-like penalty with per-layer scaling; cheap and composes with LoRA.
4. **Parameter isolation via PEFT** — still the default defence: keep the 8B base frozen, train a ≤50M adapter. All 2026 reviews continue to rank this as the safest choice.
5. **FIP (Functionally Invariant Paths)** — geometry-aware updates; larger parameter changes than LoRA but preserves prior-task loss surface.

**Empirical warning from 2026 studies**: forgetting *intensifies* with model scale from 1B → 7B, so Qwen3-8B full-FT is exactly where the problem is worst. Another argument for LoRA/DoRA + a small regulariser over full-FT.

---

## Quantization-aware fine-tune 2026

- **FP8 is the 2026 production default** on Hopper/Blackwell; training quality ≈ BF16 for most LLM tasks.
- **NF4 + LoRA (QLoRA)** remains the accessible default for 1× A100-40GB Qwen3-8B. Unsloth, NeMo, and HF tutorials all default to this.
- **NVFP4** (NVIDIA research, March 2026 report): PTQ often fails on small models / sensitive tasks; **quantization-aware distillation** is required. Block size is small enough that traditional outlier mitigation is ineffective.
- **BitNet b1.58** — still pretraining-only and 2–8B scale. Not usable as a drop-in for an 8B fine-tune.
- **D-QReLO** (2604.16940) — useful for shipping: compress the trained delta to ≈1-bit + residual low-rank post-hoc without data or retraining.

---

## Mergers / LoRA soups / federated 2026

- **LoRA Soups** (2410.13025) — super-linear gain on GSM-Hard when merging math-LoRA + code-LoRA; argues "compose specialists, don't co-train".
- **FedEx-LoRA** (ACL 2025, re-cited 2026) — exact federated aggregation via residual error term on frozen W; minimal comms overhead.
- **FedRPCA** (2506.01194) — Robust-PCA splits common vs. client-specific signal in LoRA updates; improves federated math LoRA aggregation.
- **ACM Computing Surveys 2026 survey** on model merging catalogs dozens of techniques (Task Arithmetic, TIES, DARE, SLERP, LoRA Hub) — useful reference but no single new merger has published a large math-reasoning delta in Q1 2026.

**Relevant for us**: if we want "memory-flavoured" Qwen3-8B fine-tune, merging a math-LoRA with a memory-augmentation LoRA is now a supported pattern (peft ≥ 0.14) and is cheaper than joint training.

---

## Recommendations update for our Phase 2

Changes from the `06_lora_peft_math.md` recommendations in light of Jan–Apr 2026 evidence:

1. **Default DoRA, not plain LoRA.** `use_dora=True` is now the 2026 default across Unsloth/NeMo/HF guides for Qwen3. Set r=α=32 for math, `target_modules="all-linear"`, LR 2e-4, warmup 0.1, 2–3 SFT epochs, 4-bit NF4 base.
2. **Two-stage pipeline**: short SFT on OpenMathReasoning (or a decontaminated subset) → GRPO on GSM8K/MATH. Matches every 2026 paper and the Unsloth Qwen3-8B DAPO notebook.
3. **GRPO → DAPO.** Adopt token-level PG loss and overlong reward shaping (buffer 4096, penalty 1.0/token) immediately; they are ~20 lines of code and give the biggest reported deltas in 2026.
4. **Consider DGPO advantage normalisation.** Swap `std` for mean-absolute-deviation in the group advantage normaliser → +2.18 pt on Qwen2.5-Math-7B with zero extra compute. Low-risk drop-in.
5. **For memory/hooks experiments, keep parameter count tiny**: 1LoRA or NEAT at r=4 beats LoRA r=4 at matched params and lets us claim "<1M trainable" honestly. S2FT is an alternative if we need structured sparsity for the decision-chain hook layers.
6. **Forgetting mitigation**: default to parameter isolation (adapter + frozen base). If the Phase 2 eval shows regression on general benchmarks, add EWC-style regularisation (2501.13669) or switch to SDFT (MIT) at ~2.5× compute.
7. **Quantisation**: keep NF4 + bf16 LoRA for training; if deploying, compress the merged delta with D-QReLO rather than re-quantising the whole model.
8. **No need to chase Qwen3.5 / Qwen3.6.** Those are MoE-heavy and don't displace Qwen3-8B dense for our setup. Re-evaluate if Alibaba ships a dense Qwen3-Math-8B.
9. **Merging path is now viable**: prepare our Phase 2 math-LoRA so it can later be soup-merged with a memory-hook LoRA instead of jointly trained, per LoRA Soups evidence.

---

## Sources

- [Fine-Tune LLMs with LoRA and QLoRA: 2026 Guide (DEV/Jangwook Kim)](https://dev.to/jangwook_kim_e31e7291ad98/fine-tune-llms-with-lora-and-qlora-2026-guide-33lf)
- [Unsloth Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Unsloth Qwen3 How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)
- [HuggingFace Qwen3-8B LoRA tutorial (optimum-neuron)](https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3)
- [NVIDIA NeMo 25.07 Qwen3 User Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.07/llms/qwen3.html)
- [DataCamp Fine-Tuning Qwen3 Tutorial](https://www.datacamp.com/tutorial/fine-tuning-qwen3)
- [n1n.ai Qwen3.5 Model Series 2026 Guide](https://explore.n1n.ai/blog/qwen3-5-model-series-2026-guide-2026-02-25)
- [Qwen3.5 Blog: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3-Coder-Next (qwen.ai)](https://qwen.ai/blog?id=qwen3-coder-next)
- [PEFT-Factory arXiv 2512.02764](https://arxiv.org/abs/2512.02764)
- [PEFT-Bench arXiv 2511.21285](https://arxiv.org/abs/2511.21285)
- [TLoRA+ arXiv 2604.13368](https://arxiv.org/abs/2604.13368)
- [NEAT arXiv 2410.01870](https://arxiv.org/html/2410.01870v1)
- [1LoRA arXiv 2503.08333](https://arxiv.org/abs/2503.08333)
- [S2FT arXiv 2412.06289](https://arxiv.org/html/2412.06289v2)
- [MoRA arXiv 2405.12130](https://arxiv.org/abs/2405.12130)
- [DAPO arXiv 2503.14476](https://arxiv.org/pdf/2503.14476)
- [MathForge / DGPO (ICLR 2026) arXiv 2601.20614](https://arxiv.org/abs/2601.20614)
- [MathForge GitHub](https://github.com/AMAP-ML/MathForge)
- [GRPO-LEAD arXiv 2504.09696](https://arxiv.org/abs/2504.09696)
- [GTPO / GRPO-S arXiv 2508.04349](https://arxiv.org/abs/2508.04349)
- [TROLL trust-region GRPO (ICLR 2026) arXiv 2510.03817](https://arxiv.org/pdf/2510.03817)
- [f-GRPO arXiv 2602.05946](https://arxiv.org/pdf/2602.05946)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond (llm-stats)](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [NVIDIA NeMo RL DAPO Guide](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)
- [OpenMathReasoning dataset (nvidia, HF)](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [SYNTHETIC-1 release (Prime Intellect)](https://www.primeintellect.ai/blog/synthetic-1-release)
- [Mechanistic Analysis of Catastrophic Forgetting arXiv 2601.18699](https://arxiv.org/html/2601.18699v1)
- [Hierarchical Layer-Wise Regularisation arXiv 2501.13669](https://arxiv.org/html/2501.13669v2)
- [VentureBeat: MIT SDFT](https://venturebeat.com/orchestration/mits-new-fine-tuning-method-lets-llms-learn-new-skills-without-losing-old)
- [AIToolly: MIT SDFT summary](https://aitoolly.com/ai-news/article/2026-02-12-mit-eth-zurich-develop-self-distillation-fine-tuning-sdft-to-enable-llms-to-learn-new-skills-without)
- [InfoWorld: Self-distillation fix for catastrophic forgetting](https://www.infoworld.com/article/4131242/researchers-propose-a-self-distillation-fix-for-catastrophic-forgetting-in-llms.html)
- [NVIDIA NVFP4 Quantization-Aware Distillation Report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- [VRLA Tech LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)
- [BestAIWeb: BitNet, FP8, 1-bit frontier 2026](https://www.bestaiweb.ai/bitnet-fp8-native-and-the-1-bit-frontier-where-quantization-is-heading-in-2026/)
- [D-QReLO arXiv 2604.16940](https://arxiv.org/html/2604.16940)
- [LoRA Soups arXiv 2410.13025](https://arxiv.org/html/2410.13025v2)
- [FedEx-LoRA arXiv 2410.09432](https://arxiv.org/abs/2410.09432)
- [FedRPCA arXiv 2506.01194](https://www.arxiv.org/pdf/2506.01194)
- [Awesome Model Merging (ACM CSUR 2026)](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications)
- [tahamajs/Qwen3-4B-GSM8k-GRPO-Unsloth HF model card](https://huggingface.co/tahamajs/Qwen3-4B-GSM8k-GRPO-Unsloth)
