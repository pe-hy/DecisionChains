# 2026 Advances — PRMs, Critical Tokens, Step-level Reasoning

## Time horizon
January – April 2026. Earlier work (2023-2025 PRM canon, Wang "Beyond 80/20" 2025, Dhuliawala 2023, cDPO Lin 2024) is covered in `research/04_branching_decision_points.md` and `research/05_process_reward_models.md`; this file is strictly follow-ups from Jan–Apr 2026.

## New PRM checkpoints and methods

| Name / arXiv | Date | Base model | Training data / method | Key score |
|---|---|---|---|---|
| **PROMISE** (2601.04674) | Jan 2026 | PRM-style step scorer | Dense step-by-step verification for generative recommenders | Unlocks test-time scaling in rec |
| **PRL — Process Reward Learning** (2601.10201) | Jan 2026 | 7B math policy | Derives PRM from entropy-regularized RL objective; optimal process reward = decomposition of outcome objective | "Broadens reasoning boundary" vs PPO/GRPO baselines |
| **Trade-R1 / VPRM** (2601.03948, 2601.17223) | Jan 2026 | DeepSeek-Math, Qwen2.5 | Verifiable PRMs: rule-based deterministic verifiers checking each step | Extends RLVR beyond binary final-answer reward |
| **Noise-aware PRM** (2601.12748) | Jan 19, 2026 | Qwen2.5-7B-Math | Noise-aware learning on Monte-Carlo estimated labels; robust to MCE noise | +F1 on ProcessBench; reduces MCE label noise damage |
| **ToolPRMBench** (2601.12294) | Jan 2026 | — (benchmark) | Tool-agent trajectories → step-level test cases | First PRM benchmark for tool-using agents |
| **Agent-RRM** (2601.22154) | Jan 29, 2026 | — | Explicit-reasoning reward model producing decomposed judgment (trace + critique + score) | Beats classifier RMs on agent tasks |
| **FunPRM** (2601.22249) | Jan 2026 | Code-LLM | Function-as-step PRM with meta-reward correction | Step-level code rewards |
| **EDU-PRM** (2503.22233, v-rev Mar 9 2026) | Mar 2026 | 7B | Entropy-driven uncertainty segmentation; no manual step boundaries | Matches Qwen2.5-Math-PRM-72B with ~1% of annotation cost |
| **PRMs Meet Planning / PDDL-PRM** (2604.17957) | Apr 20, 2026 | Qwen2.5-Math-7B | ~1M reasoning steps auto-generated from PDDL planning problems | +large gains on ProcessBench; transfers to math |
| **MedPRMBench** (2604.17282) | Apr 2026 | — (benchmark) | 14 medical-specific error types, extending PRMBench's 9 | First medical PRM bench |
| **PRA — Process Reward Agents** (2604.09482) | Apr 2026 | Frozen policy + PRA | Test-time domain-grounded online step-wise reward; search-based pruning per token | Ranks + prunes trajectories at every generation step |
| **GRPO ≡ PRM** (2509.21154, v-rev Feb 20 2026) | Feb 2026 | Theoretical | Proves GRPO with outcome RM is equivalent to PRM-aware RL with MC-based PRM | Justifies outcome-only RL as implicit PRM |

**Base for our work**: Qwen2.5-Math-PRM-7B / -72B (Jan 2025) remain the default open checkpoints; GenPRM-7B (Apr 2025) still SOTA on ProcessBench (surpasses Qwen2.5-Math-PRM-72B using generative CoT-with-code judgments); R-PRM (Mar 2025) reports +11.9 F1 on ProcessBench, +8.5 on PRMBench over strongest baseline.

## Critical / fork token analysis 2026

**Do LLMs Encode Functional Importance of Reasoning Tokens?** (2601.03066, Jan 2026). Greedy pruning experiment: models can systematically compress CoT traces while retaining answer-critical info — evidence that token importance is *linearly decodable* from hidden states. Directly relevant: supports the hypothesis that a probe at decision points can route to per-step memory.

**Multiplex Thinking** (2601.08808, Jan 2026). At each step, samples K candidate tokens and aggregates their *embeddings* into one continuous "multiplex token". Outperforms discrete CoT from Pass@1 to Pass@1024 on AIME/Olympiad; shorter sequences. Concretely: they replace the hard argmax at forks with a soft mixture — a direct cousin of "steering at decision points".

**Dynamic Thinking-Token Selection (DynTS)** (2601.18383, Jan 2026). Observation: only a few decision-critical tokens actually steer the answer; the rest contribute negligibly. Keeps only critical-token KV entries at inference. Massive KV-memory savings with minimal accuracy loss.

**Selective Critical Token Fine-Tuning (CFT)** (2510.10974, rev 2026). Critical = counterfactual perturbation flips the final answer. Only updates those tokens during SFT (under 12% of tokens). Consistently beats standard SFT across 11 math benchmarks (GSM8K, MATH, SVAMP, ASDiv, MAWPS, CARP-En, TabMWP, Minerva-Math, Gaokao2023-En, OlympiadBench, College-Math). 25x speedup via parallel counterfactual decoding on Qwen2.5-7B.

**SED-SFT — Selectively Encouraging Diversity** (2602.07464, Feb 2026). Inverse framing: *encourage* diversity on non-critical tokens to preserve exploration. Directly confirms the "update-critical / free-non-critical" dichotomy.

**Token Priority in SFT** (2602.01227, Feb 2026). Argues SFT itself is bottlenecked by uniform token weighting; proposes priority-weighted loss keyed to counterfactual importance.

## Step-level RL and preference optimization 2026

**GTPO / GRPO-S** (2508.04349, v4 in 2026). Entropy-weighted per-token advantage inside GRPO. Reaches higher reward plateau than DAPO across datasets/sizes; GRPO-S more stable on long CoT. Core move: multiply each token's advantage by its entropy at emission time — exactly the "fork-token up-weighting" recipe applied to the *advantage*, not the loss mask.

**TLPO — Token-Level Policy Optimization** (2604.12736, Apr 2026). Links group-level rewards to token aggregation via sequence likelihood; gives per-token advantages without a learned PRM.

**Segment Policy Optimization (SPO)**. Segment-level credit between full-token and full-trajectory granularity; ICLR 2026.

**GRPO-λ** (ICLR 2026). Generalized advantage estimation for GRPO; +3 points avg across math benches, +4.5 at 7B.

**InT — Intervention Training** (2601.14209, Jan 2026). Model proposes its *own* targeted corrections to its trace → self-supervised credit. Interesting for our project because the "intervention" is applied at decision points.

**StepPO** (2604.18401, Apr 2026). Step-aligned policy optimization for agentic RL; aligns reward propagation with decision-making unit.

**IG-Search** (2604.15148, Apr 2026). Step-reward = information gain: confidence-in-gold-answer delta vs random-doc counterfactual. Beautiful formulation; applicable to our setting by defining IG over the next correct letter.

**T-SPMO — Token-Specific Prefix-Matching Opt.** (2504.20834, 2026 rev). Qwen2-1.5B on SVAMP: 46% → 70%+ via per-token credit. Among the largest reported gains from pure credit-reweighting.

## Entropy-based intervention 2026

**Efficient RL with Semantic and Token Entropy** (2512.04359, v2). Non-uniform treatment: KL regularization on *low*-entropy tokens (they control exploration), stronger constraints on high-covariance low-entropy subset.

**STEER — Rethinking Entropy Interventions** (2510.10150, rev 2026). Entropy-change perspective: adaptive per-step reweighting that keeps per-step entropy delta in a moderate band; suppresses destabilizing updates.

**Think Twice Before You Write** (2604.00018, Apr 2026). Pure inference-time entropy-guided decoding: at each step compute next-token entropy, identify high-uncertainty forks, *selectively branch only there*, maintain a pool of partial rollouts. No training. This is essentially a training-free version of Wang-2025 applied to search.

**Entropy-Aware Speculative Decoding** (2512.23765, late 2026). When both draft and target show high entropy with top-N overlap → reject and resample from target. Uses entropy to localize where speculative decoding is risky.

**TURN** (2502.05234v1, 2026). Uses the entropy turning point (EntP) to pick temperature adaptively per step.

**RL-based decoding controller**. PPO policy observes prompt + prefix + logits + entropy and selects (T, top_p) per token — learned extension of TURN.

## Thinking / budget tokens 2026 (post-s1)

**Remove "Wait" tokens** (2506.08343, v2 2026). Ablates s1's "Wait" insertion: removing thinking tokens can *improve* efficiency without hurting accuracy. Undermines naive budget-forcing.

**BudgetThinker** (OpenReview, ICLR 2026). Trained budget-aware control tokens instead of forced insertion.

**Steering LLM Thinking with Budget Guidance** (2506.13752). Learns a small adapter that nudges the model toward completing within a budget, rather than truncating.

**Dynamic Thinking-Token Selection** (same as above, 2601.18383). Also a budget technique — prune KV of non-critical thinking tokens rather than limit length.

**Multiplex Thinking** (2601.08808). Yields shorter sequences at higher Pass@K — orthogonal budget improvement via soft forks.

Take-away: the *naive* s1 recipe (append "Wait", truncate budget) is essentially dead in 2026. New direction is *token-selective* budget management guided by entropy / criticality signals.

## Manufacturing a PRM cheaply in 2026

- **Monte-Carlo Net Information Gain** (2603.17815): auto-label step quality as Δ(prob of correct final answer). More reliable than raw IG.
- **PDDL-PRM** (2604.17957): 1M auto-generated planning steps → transferable PRM features.
- **Noise-aware PRM** (2601.12748): explicitly handle MCE label noise to make cheap MC labels usable.
- **EDU-PRM** (2503.22233): entropy-driven segmentation removes the need for manual step boundaries.
- **GRPO ≡ PRM** (2509.21154 rev 2026): if proof holds, one doesn't *need* an explicit PRM at training time — outcome-only GRPO already implements one.
- **InT** (2601.14209): model self-proposes interventions; label is whether the intervention fixes the trace. Free data.

The practical 2026 recipe: MC-rollout labels + noise-aware training + entropy-based step boundaries → ~1% of the human-annotation cost of the original PRM800K/Math-Shepherd pipeline.

## Implications for our project (math_qwen_memory, decision-chain + memory)

1. **Decision-point labeling (Phase 1)**. Our decision functions `func_f` / `func_g` produce *provably* balanced forks. 2026 work (Wang 2025's "20% rule" confirmed by Multiplex Thinking, DynTS, CFT, GTPO) says only a few tokens actually branch — ours are explicitly those. We do not need MC-labeling or MCNIG; our DP labels are *exact* counterfactual critical tokens in the CFT sense. This is a major cheat.

2. **Memory-steering at decision points**. Maps directly onto Multiplex Thinking's soft-fork token aggregation (2601.08808) and RISER's Router-based intervention (2601.09269). Our "memory module at a DP" is exactly RISER's "reasoning vector library selected per input". Consider framing our Phase-2 writeup using RISER's vocabulary.

3. **Training signal**. Instead of loss-masking only the [TRACE] segment, consider GTPO-style entropy-weighted advantage (2508.04349) during GRPO fine-tune. The natural DP tokens already have the highest entropy in our chains, so this becomes near-free.

4. **PRM not needed for our task**. Because the correctness of each DP is deterministic (either f or g was applied), we have a *perfect free PRM*. The 2026 PRM bench race (ProcessBench, PRMBench, MedPRMBench, ToolPRMBench) is not our evaluation target — we should report our own DP-accuracy and P(STOP) metrics and reference PRMBench only as context.

5. **Replace or augment memory-steering?** Token-level alternatives that could replace memory-steering:
   - GTPO / GRPO-S entropy-weighted GRPO — needs zero new architecture.
   - CFT / SCFT selective fine-tuning on DP tokens only — zero architecture change.
   - Multiplex Thinking soft-fork aggregation — *architectural* change but parameter-light.
   - RISER Router over a library of DP-specific vectors — closest to our memory, more interpretable.
   
   Verdict: augment. Keep memory at DPs (gives interpretable per-DP slots), but add entropy-weighted GRPO as the default RL recipe, and benchmark memory vs. RISER-style vector router as two instantiations of the same abstract idea. If memory loses to RISER the project still has a story.

6. **Evaluation update**. Add: (a) Pass@k under budget forcing vs. natural length (following BudgetThinker), (b) DP-entropy histogram at train/eval (to confirm DPs are the 20% high-entropy forks), (c) CFT-style counterfactual sensitivity (flip the DP prediction, measure chain validity).

7. **Multi-hop / planning generalization**. PDDL-PRM (2604.17957) result that 1M planning steps transfer to math is encouraging: our synthetic DP chains should transfer to real math tasks similarly. Consider a downstream eval on GSM8K or MATH with a policy frozen after our DP training.

## Sources

- [Beyond 80/20 (Wang 2025) on fork tokens](https://arxiv.org/abs/2506.01939)
- [Critical Tokens Matter / cDPO (Lin 2024, ICML 2025)](https://arxiv.org/abs/2411.19943)
- [Selective Critical Token Fine-Tuning (CFT)](https://arxiv.org/abs/2510.10974)
- [Do LLMs Encode Functional Importance of Reasoning Tokens? (2601.03066)](https://arxiv.org/html/2601.03066)
- [Multiplex Thinking (2601.08808)](https://arxiv.org/abs/2601.08808v1)
- [Dynamic Thinking-Token Selection (2601.18383)](https://arxiv.org/html/2601.18383)
- [PRL: Process Reward Learning (2601.10201)](https://arxiv.org/html/2601.10201v1)
- [Noise-aware PRM (2601.12748)](https://arxiv.org/abs/2601.12748)
- [Verifiable PRM / VPRM (2601.17223)](https://arxiv.org/abs/2601.17223)
- [Trade-R1 (2601.03948)](https://arxiv.org/abs/2601.03948v1)
- [ToolPRMBench (2601.12294)](https://arxiv.org/abs/2601.12294)
- [PROMISE — PRM for recommendations (2601.04674)](https://arxiv.org/pdf/2601.04674)
- [Agent-RRM (2601.22154)](https://arxiv.org/pdf/2601.22154)
- [FunPRM (2601.22249)](https://arxiv.org/html/2601.22249v1)
- [InT: Self-Proposed Interventions (2601.14209)](https://arxiv.org/abs/2601.14209)
- [RISER — Adaptive activation steering (2601.09269)](https://arxiv.org/abs/2601.09269)
- [EDU-PRM (2503.22233)](https://arxiv.org/abs/2503.22233)
- [PRMs Meet Planning / PDDL-PRM (2604.17957)](https://arxiv.org/abs/2604.17957)
- [Process Reward Agents / PRA (2604.09482)](https://arxiv.org/abs/2604.09482)
- [MedPRMBench (2604.17282)](https://arxiv.org/html/2604.17282)
- [StepPO (2604.18401)](https://arxiv.org/html/2604.18401)
- [IG-Search (2604.15148)](https://arxiv.org/html/2604.15148)
- [Think Twice Before You Write (2604.00018)](https://arxiv.org/html/2604.00018)
- [Token-Level Policy Optimization / TLPO (2604.12736)](https://arxiv.org/abs/2604.12736)
- [GRPO is Secretly a PRM (2509.21154, Feb 2026 rev)](https://arxiv.org/abs/2509.21154)
- [GTPO / GRPO-S entropy-weighted tokens (2508.04349)](https://arxiv.org/abs/2508.04349)
- [STEER — Rethinking Entropy Interventions (2510.10150)](https://arxiv.org/pdf/2510.10150)
- [Token-Efficient RL / T-SPMO (2504.20834)](https://arxiv.org/abs/2504.20834)
- [GenPRM (Apr 2025, AAAI 2026)](https://huggingface.co/papers/2504.00891)
- [R-PRM (2503.21295)](https://arxiv.org/abs/2503.21295)
- [Qwen2.5-Math-PRM-72B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)
- [PRMBench](https://arxiv.org/abs/2501.03124)
- [s1 + budget forcing](https://arxiv.org/abs/2501.19393)
- [Remove "Wait" tokens (2506.08343)](https://arxiv.org/html/2506.08343v2)
- [BudgetThinker (ICLR 2026)](https://openreview.net/forum?id=ahatk5qrmB)
- [Steering LLM Thinking with Budget Guidance](https://arxiv.org/html/2506.13752v1)
- [Step-DPO (Lai 2024)](https://arxiv.org/html/2406.18629v1)
- [TIS-DPO (ICLR 2025)](https://arxiv.org/abs/2410.04350)
- [Activation Steering Field Guide 2026](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [Post-Training Techniques 2026 overview](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [Carnegie Mellon at ICLR 2026](https://blog.ml.cmu.edu/2026/04/20/carnegie-mellon-at-iclr-2026/)
