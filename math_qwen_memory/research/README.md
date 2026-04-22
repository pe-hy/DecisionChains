# Research Notes — Index

Deep-dive literature reviews for porting the memory-steering + LoRA baseline experiments from synthetic decision chains (30M GPT-NeoX) to **Qwen3-8B on natural-language math** (MATH competition + algebra__linear_1d) on a 40 GB A100.

Read these before starting Phase 1 (memory-module design) or Phase 2 (LoRA finetune). Each file is self-contained with Overview → Methods → Actionable takeaways → Sources.

| # | File | Theme | Headline finding |
|---|------|-------|-----------------|
| 01 | [`01_external_memory_methods.md`](01_external_memory_methods.md) | External memory, PKM, kNN-LM, RETRO, Memory Layers, soft prompts | Hook residual stream at Qwen3-8B layers **16–22** (mid-depth); budget 1M–20M trainable params; mirror GQA's 8 KV heads |
| 02 | [`02_activation_steering.md`](02_activation_steering.md) | ActAdd, CAA, RepE, ITI, SAE features, persona vectors, CAST | Must beat **bias-only adaptation (Goncharov 2025)** as strongest static-vector baseline; probe layers 14–24 |
| 03 | [`03_cot_parsing_and_answer_extraction.md`](03_cot_parsing_and_answer_extraction.md) | Answer extraction, LaTeX normalization, thinking-mode delimiters | Use `math-verify` as primary grader; `<think>` ends at token-id **151668**; Qwen3-Thinking-2507 pre-emits opening `<think>` |
| 04 | [`04_branching_decision_points.md`](04_branching_decision_points.md) | Identifying DP tokens — entropy, step structure, PRMs, critical tokens | Start with **top-20% entropy tokens gated by step boundary** (~2% of tokens); confirmed by Qwen3 RLVR work |
| 05 | [`05_process_reward_models.md`](05_process_reward_models.md) | PRM800K, Math-Shepherd, open PRMs, step-level supervision | **Qwen2.5-Math-PRM-7B** recommended; fits with Qwen3-8B on 40 GB (~36–38 GB, tight); split steps on `\n\n` |
| 06 | [`06_lora_peft_math.md`](06_lora_peft_math.md) | LoRA, DoRA, QLoRA for Qwen3-8B math training | All-linear LoRA (r=32–64, α=2r, lr 2e-4, 3 epochs, grad-ckpt on) matches full FT per Thinking Machines 2025; QLoRA not needed |
| 07 | [`07_qwen3_architecture_and_hooks.md`](07_qwen3_architecture_and_hooks.md) | Qwen3-8B internals, HF hook patterns, thinking-mode tokens | **Critical**: `Qwen3DecoderLayer.forward` returns a bare Tensor not a tuple — hook must handle both; Q/K-norm quirk |
| 08 | [`08_evaluation_methodology.md`](08_evaluation_methodology.md) | MATH/GSM8K baselines, maj@k, lm-eval-harness, Wilson CI | Model card **forbids greedy** for thinking; use T=0.6/top_p=0.95/top_k=20; report Wilson 95% CI and McNemar for A/B |

## 2026 deltas (Jan – Apr 2026)

Newer papers that update or override the 2023-2025 recommendations above.

| # | File | Theme | Headline finding |
|---|------|-------|-----------------|
| 09 | [`09_2026_memory_and_steering_advances.md`](09_2026_memory_and_steering_advances.md) | 2026 steering & memory papers | **Router-over-vectors is default** (RISER/DIRECTER/CRL-Token); TinyLoRA-RL 13 params → 91% GSM8K sets new min-param bar; token-conditional sparse fires (0.2–2.5%) expected |
| 10 | [`10_2026_prm_and_critical_tokens.md`](10_2026_prm_and_critical_tokens.md) | 2026 PRMs & critical tokens | GenPRM/GRPO-as-PRM; **GTPO entropy-weighted GRPO** and **RISER vector-router** map directly onto our memory module |
| 11 | [`11_2026_peft_and_qwen_recipes.md`](11_2026_peft_and_qwen_recipes.md) | 2026 PEFT & Qwen recipes | Qwen3.5/3.6 released Feb/Apr 2026; consensus r=α=32 all-linear DoRA+NF4 LR 2e-4; **DGPO/MathForge +2.18pt**, TROLL, GRPO-LEAD; OpenMathReasoning + SYNTHETIC-1 |
| 12 | [`12_2026_reasoning_and_evaluation.md`](12_2026_reasoning_and_evaluation.md) | 2026 reasoning & eval | **math-verify upgrade ≈ +4.66pt** — must adopt; AIME/HMMT 2026 via MathArena; CGES replaces maj@16 (69% fewer calls); skip latent-CoT as a method |

## Cross-doc themes

- **Where to hook the residual stream**: 01 says layers 16–22; 02 says probe 14–24; 07 explains the Qwen3 block mechanics and the tuple-vs-Tensor gotcha. Starting sweep: **L ∈ {14, 18, 22}**.
- **What counts as a "decision point"**: 04 gives five candidate definitions; 05 argues PRM step scores can label them; 02 notes static steering ignores position entirely.
- **Trainable-param budget**: Qwen3-8B hidden=4096 means a naive port of our 270K-param memory blows up. Docs 01 and 07 converge on bottleneck design (down→up project) targeting 1–20M params.
- **Eval hygiene**: 08 + 03 must both be followed to avoid false-positive/negative accuracy deltas that masquerade as "method works".

## Open questions (cut across docs)

1. Use a PRM as DP-labeler or as training reward? (05 vs 04 vs 10)
2. Steer only during thinking-mode segment, only outside, or both? (02 + 07 + 09)
3. Can we evaluate both "MATH final-answer accuracy" (outcome) AND "step-level correctness" (process) in one pipeline? (05 + 08 + 10)
4. Qwen3-8B-Thinking-2507 vs base Qwen3-8B vs Qwen3-8B-Instruct-2507 vs new Qwen3.5/3.6 variants as primary? (07 + 11)
5. Adopt router-over-vectors as memory-module default (per 09 RISER/DIRECTER) or stay with single learned KV bank? (01 + 09)
6. Upgrade eval: math-verify latest + AIME/HMMT 2026 + CGES replacing maj@16? (12)

## Where to go next

Phase 1 design doc (not yet written) should consolidate these into a concrete architecture + training loop spec and resolve the open questions above. Expected baselines to beat (or match at lower param count):
- Bias-only adaptation (Goncharov 2025) — strongest static-vector baseline
- TinyLoRA-RL 13-params / 91% GSM8K — new minimum-param bar (09, 2026)
- Vanilla LoRA all-linear r=32 DoRA (11, 2026 consensus recipe)
