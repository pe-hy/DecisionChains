# Identifying Decision / Branching Tokens in CoT — Literature Review

## Why this matters for our project

In our prior synthetic decision-chain work, "decision points" (DPs) were crisp: a single letter token whose distribution over {f, g} choices we could inspect and supervise directly. The ground truth branching structure was built into the data generator.

Porting to natural-language MATH with Qwen3-8B, DPs become fuzzy. A CoT of several hundred tokens contains a handful of *load-bearing* tokens (where reasoning could fork and outcome changes) buried in a sea of *bureaucratic* tokens (grammar, LaTeX scaffolding, "we", "have", "="). If our memory-steering signal is diffuse across every token, we waste gradient on the 80% that carry no branching information. Worse, steering the wrong tokens can destabilize the CoT. We need a principled and *computable* way to pick DP positions before Phase-1 training starts.

## Candidate definitions

### 1. Entropy-based (most tractable)

Per-token predictive entropy H(p_t) from the model's own logits. Recent work (Wang et al. 2025, "Beyond the 80/20 Rule") shows ~20% of tokens in Qwen3 CoT carry high entropy and *act as forking tokens that steer reasoning trajectory*. Training RLVR only on those top-20% entropy tokens on Qwen3-8B matches or beats full-gradient updates; on Qwen3-14B/32B it beats full gradient by +5–11 points on AIME. Low-entropy tokens are grammatical fillers. Thresholds used: top-20% per trace, or absolute H > ~0.5 nats.

EDU-PRM (a follow-on to CoT-without-prompting, Wang & Zhou 2024) *explicitly uses high-uncertainty tokens as branching points* for a tree search — this is the same construction we want. Greedy decoding between branches, fork only at high-H steps. DeepConf (2025) uses lowest-group-confidence (equivalently highest-entropy window) to flag reasoning-breakdown regions.

### 2. Structural (cheapest)

A heuristic: a DP starts at newline / "Step N:" / end-of-equation / "=" / sentence boundary. This is how PRM800K and Math-Shepherd tokenize "steps" — they literally split on `\n\n` or sentence boundary markers. Prevalence in MATH CoT: ~8–15 steps per problem, so ~1 DP per 30–50 tokens. Free to compute but coarse; misses the *within-step* fork (e.g. whether to substitute vs. factor).

### 3. Semantic (discourse markers)

Regex on conjunctive / inferential tokens: "therefore", "so", "thus", "hence", "which means", "we get", "let", and mid-equation "=". These mark where the model commits to a step's conclusion. Cheap, but low-precision — many are filler. Works best combined with (1): entropy-gate the semantic candidates.

### 4. Gradient / attribution-based (most expensive)

Integrated gradients / attention rollout / activation patching to find tokens whose intervention flips the final answer (Chen et al. 2025, SAE+patching on GSM8K). These are the true *load-bearing* tokens by a causal definition. In practice: run the CoT once, patch each token's residual stream from a counterfactual run, measure answer log-prob delta. Cost: O(T) forward passes per example — prohibitive at training scale. Useful only as a one-off ground truth to validate a cheaper signal.

### 5. Learned (train a classifier or use a PRM)

Use an off-the-shelf PRM (Skywork-o1-PRM-7B, Math-Shepherd, or the PRM800K-trained verifier) to score each step; the steps where the PRM's score is most *sensitive* to a perturbed rollout are the DPs. Math-Shepherd's construction is directly applicable: for each step, do K MCTS rollouts from that step; steps with high reward variance across rollouts are DPs. rStar-Math uses exactly this as its "process preference" signal.

## Key papers

### Process reward models as step labelers
- **PRM800K / Let's Verify Step by Step** (Lightman et al. 2023, arXiv:2305.20050): 800K human step-level labels on MATH solutions; steps are defined by `\n\n` / sentence boundaries. Demonstrates 78% solve rate with process supervision vs. much lower with outcome-only.
- **Math-Shepherd** (Wang et al. 2024, arXiv:2312.08935): automatic step labeling via MCTS-style rollouts — a step is *good* if many rollouts from it reach correct answer. Converts any trace into step-level rewards *without human annotation*. Lifted Mistral-7B from 77.9 → 84.1 on GSM8K.
- **Skywork-o1-Open-PRM-Qwen-2.5-7B** (HF): drop-in open-source Qwen-2.5-based PRM. Directly compatible with Qwen3-8B tokenizer family — usable off-the-shelf in our pipeline to label steps.
- **rStar-Math** (Guan et al. 2025, arXiv:2501.04519): takes this to extreme — self-evolved policy + PPM with MCTS, lifts Qwen2.5-Math-7B MATH from 58.8 → 90.0.

### Tree search methods
- **Tree of Thoughts** (Yao et al. 2023, arXiv:2305.10601): reasoning as tree search; each node is a "thought" (few-sentence chunk). Branch points are manually defined per task (enumerated candidate next-thoughts).
- **rStar** (Qi et al. 2024, arXiv:2408.06195): MCTS over a SLM's reasoning; branch = sampled next-step, scored by a discriminator SLM.
- **Self-Consistency** (Wang et al. 2022, arXiv:2203.11171): branching only at the very first token (temperature-sampled full rollouts), majority vote. Implicit DP = root.

### Critical-token / entropy analysis
- **Critical Tokens Matter (cDPO)** (Lin et al. 2024, arXiv:2411.19943): identifies *critical tokens* in CoT by rollout-based contrastive estimation — a token is critical if forcing it shifts rollout correctness distribution. Uses these for token-level DPO rewards. Close to what we want: a principled "which tokens disproportionately cause errors" definition. ~3–5% of tokens per trace flagged critical.
- **Beyond the 80/20 Rule** (Wang et al. 2025, arXiv:2506.01939): demonstrates on Qwen3 that top-20% entropy tokens are the RL-active tokens. Most important paper for us — the model is exactly the one we use.
- **Chain-of-Thought Reasoning Without Prompting** (Wang & Zhou 2024, arXiv:2402.10200): entropy-based top-k decoding reveals latent CoT paths; high-confidence final answer correlates with presence of CoT along the decoded path. Introduces the "branching at high-uncertainty tokens" pattern.
- **DeepConf** (Fu et al. 2025, arXiv:2508.15260): uses *lowest group confidence* (highest-entropy window) as a trace-quality signal; early-exits low-confidence traces. 99.9% on AIME'25 with GPT-OSS-120B.

### Thinking / pause tokens
- **s1: Simple test-time scaling** (Muennighoff et al. 2025, arXiv:2501.19393): "Wait" appended at natural termination forces the model to re-check — implicitly creating a *new DP* where there was none. Budget-forcing knob.
- **Think before you speak / pause tokens** (Goyal et al. 2024, arXiv:2310.02226): trained `<pause>` tokens allocate extra hidden-state compute before commit. DP = pause location. +18% SQuAD, modest GSM8K gain.

### Mechanistic interpretability
- **SAE + activation patching on CoT** (arXiv:2507.22928): feature-level causal analysis of GSM8K CoT — CoT features are *broadly distributed*, not concentrated at top-K activations. Implies DP signal is distributed, not rank-1. Cautions against over-narrow DP selection.

## Practical recommendations for our project

Ranked by implementation cost × expected signal density:

**Option A (ship first): Entropy-gated structural.** Compute H(p_t) for every token in the Qwen3-8B teacher's CoT during our data prep. Mark a token as DP if it is in the top-20% entropy AND falls at a "step" boundary (newline, end-of-equation, after ";"). Prevalence: ~8–12 DPs per ~500-token MATH trace (~2% of tokens). Training gradient density: concentrated. Cost: one extra forward pass per training example (free if we already generate the CoT). Direct port of the Wang 2025 finding.

**Option B (safer fallback): Top-20% entropy, no structural gate.** Same computation, drop the boundary condition. Prevalence: ~100 DPs per 500-token trace. Higher recall, lower precision. Use if Option A under-fires. This is what "Beyond the 80/20" does for RLVR.

**Option C (stronger signal, more infra): PRM-labeled.** Run Skywork-o1-PRM-Qwen-2.5-7B on every CoT, take steps where PRM score gradient across consecutive steps is high (i.e. steps where reasoning quality is changing fast). Prevalence: ~5–8 per trace. Higher quality labels but adds a 7B forward-pass per example. Worth it if A+B plateau.

Recommend: **implement A for Phase-1, A/B/C bake-off in Phase-2.** A is cheapest, has strongest Qwen3-specific evidence, and aligns naturally with our prior synthetic DP framing.

## Sketch: how to label DP tokens in Qwen3 MATH output

Given problem `P` and model-generated CoT `y = y_1 ... y_T` ending in `####<answer>`:

```
1. Teacher-force the full CoT y through Qwen3-8B; collect logits l_t for each t.
2. H_t = entropy(softmax(l_t)).          # per-token entropy
3. q80 = quantile(H, 0.80).              # per-trace threshold
4. is_boundary[t] = y_{t-1} in {"\n", "=", ";", "."} or t is after "Step"
5. dp[t] = (H_t >= q80) AND is_boundary[t]    # Option A
         OR (H_t >= q80)                      # Option B
```

Worked example. Qwen3-8B on "If x + 3 = 7, what is x?":
```
  tok:  "To"  " solve"  ","  " I"  " subtract"  " 3"  " from"  " both"  ...
  H:     0.1    0.2     0.3   0.4    1.8         0.2    0.1      0.1
  bdy:    F      F       F     F      F           F      F        F
  dp:     F      F       F     F      F           F      F        F
  ...
  tok:  "\n"  "So"  " x"  " ="  " 4"  "."  " ####"  "4"
  H:     0.1   0.4   0.2   0.1   1.5   0.1   0.1     0.1
  bdy:    T    T      F     F     F     F     T       F
  dp:     F    F      F     F     F     F     F       F
  # Option A flags the " subtract" and " 4" tokens only if they land on a boundary;
  # in practice high-H math tokens often follow "=" so gate with prev-token being "=".
```
Real MATH traces, the two most common DP patterns: first token of a new `\n\n` step (discourse pivot), and the token immediately after "=" in an equation (the committed RHS value).

## Open questions

1. Does entropy on a *base* Qwen3-8B correlate with our *fine-tuned* memory-model's DPs? Fine-tuning sharpens distributions — we may need to re-compute entropy after each training epoch.
2. Top-20% per-trace vs. per-batch vs. absolute threshold — which produces most stable gradient?
3. Should DPs include the *committed-answer* token (final numeric) or only *mid-reasoning* forks? Wang 2025 excludes answer tokens; our prior DP framing included them.
4. Does Option A's boundary gate break on problems where the fork is *within* an equation (e.g. choosing sign in quadratic)? May need to add equation-internal high-H tokens as a third bucket.
5. Compatibility with s1-style "Wait" injection — does appending "Wait" at a detected DP produce stronger correction than appending at arbitrary token?
6. Does a PRM-labeled DP set (Option C) overlap meaningfully with entropy-labeled (Option A)? If yes, A is sufficient; if not, they carry complementary signal and we should union them.

## Sources

- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning (Wang et al. 2025)](https://arxiv.org/abs/2506.01939)
- [Chain-of-Thought Reasoning Without Prompting (Wang & Zhou 2024)](https://arxiv.org/abs/2402.10200)
- [Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability (Lin et al. 2024)](https://arxiv.org/abs/2411.19943)
- [Let's Verify Step by Step / PRM800K (Lightman et al. 2023)](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (Wang et al. 2024)](https://arxiv.org/abs/2312.08935)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al. 2023)](https://arxiv.org/abs/2305.10601)
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (Guan et al. 2025)](https://arxiv.org/abs/2501.04519)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers / rStar (Qi et al. 2024)](https://arxiv.org/abs/2408.06195)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- [s1: Simple test-time scaling (Muennighoff et al. 2025)](https://arxiv.org/abs/2501.19393)
- [Think before you speak: Training Language Models With Pause Tokens (Goyal et al. 2024)](https://arxiv.org/abs/2310.02226)
- [Deep Think with Confidence / DeepConf (Fu et al. 2025)](https://arxiv.org/abs/2508.15260)
- [How does Chain of Thought Think? Mechanistic Interpretability with Sparse Autoencoding (Chen et al. 2025)](https://arxiv.org/abs/2507.22928)
- [Skywork-o1-Open-PRM-Qwen-2.5-7B model card](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)
- [PRM800K dataset repository](https://github.com/openai/prm800k)
