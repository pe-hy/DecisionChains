# Process Reward Models for Math — Review

## Why PRMs matter here

We are porting a memory-steering experiment to Qwen3-8B on MATH/algebra. The
central unknown is: *which* CoT steps deserve steering or supervision. A Process
Reward Model (PRM) is precisely a per-step scalar correctness estimator trained
on (problem, prefix_of_steps → in {0,1}) — exactly the signal we need to
(a) identify **decision points** where the model becomes confused (PRM score
drops or inverts sign), (b) provide **cheap labels** at scale for our memory
module instead of rolling out completions ourselves, and (c) act as an
**evaluation oracle** on the validation set that is more informative than a
binary answer-correct/incorrect judgement. Several strong open PRMs now fit
into a 40 GB A100 budget alongside Qwen3-8B, making them immediately usable.

## Landmark papers

### Let's Verify Step by Step (Lightman et al., 2023, ICLR 2024)

OpenAI's foundational paper compared **outcome-supervised reward models (ORMs)**
against **process-supervised reward models (PRMs)** on MATH. A PRM trained on
human step-level labels solved 78% of a representative MATH test subset,
significantly outperforming ORM. The release included **PRM800K**: 800k
step-level human correctness labels over ~75k GPT-4 solutions. This is still
the "gold standard" supervised PRM training set — most later work either
trains on PRM800K directly or uses it as a test bed.

### Math-Shepherd (Wang et al., 2024, ACL)

Killed the human-annotation bottleneck. Given an intermediate step, it
performs **Monte-Carlo rollouts**: sample K completions from that prefix; if
any completion reaches the correct final answer, the step is labeled positive,
otherwise negative. This gave 273k annotated trajectories (per RLHFlow's
recipe), drove Mistral-7B from 28.6% → 33.0% on MATH via step-level PPO, and
to 43.5% with PRM verification. It remains the template for "automatic
process supervision" and underlies most open PRM datasets.

### The Lessons of Developing PRMs (Qwen team, 2025, arXiv 2501.07301)

This paper is the single most useful read for us. Key findings:

- Pure MC-style labels (Math-Shepherd) are **noisy**: completion models
  misjudge step correctness ~60% of the time.
- **Consensus filtering** — keep a step label only when MC and an LLM-as-judge
  agree — removes ~60% of noisy data and produces much stronger PRMs.
- Qwen2.5-Math-PRM-7B and PRM-72B result, with ProcessBench F1 of **63.6**
  and **78.3** respectively — state of the art for open PRMs.
- Warning on Best-of-N evaluation: many PRMs concentrate minimum scores on the
  *final-answer* step rather than on intermediate reasoning, so high BoN
  accuracy does not imply good process understanding.

### Later 2025 work worth knowing

- **R-PRM** (Mar 2025, NJUNLP): reasoning-driven *generative* PRM. ProcessBench
  F1 = 65.2, PRMBench 64.9, outperforming Qwen2.5-Math-PRM-7B by ~6 points.
- **ThinkPRM** (Apr 2025, Khalifa et al., arXiv 2504.16828): long-CoT generative
  PRM fine-tuned from R1-Distill-Qwen, 1K synthetic CoT labels (!). Releases
  1.5B / 7B / 14B sizes. Beats discriminative PRMs on ProcessBench and on
  OOD GPQA-Diamond / LiveCodeBench. Very data-efficient.
- **PRMBench** (Jan 2025, arXiv 2501.03124): fine-grained benchmark with 6,216
  problems, 83k step labels, testing Simplicity / Soundness / Sensitivity.
  Best model (Gemini-2-Thinking) 68.8 vs. humans 83.8, i.e. PRMs are still
  weak.
- **GRPO is Secretly a PRM** (Sep 2025, arXiv 2509.21154): shows GRPO's
  group-normalised advantage is mathematically equivalent to a PRM-aware RL
  objective with an MC-estimated step reward — relevant if we move to
  step-level RL on top of our memory module.

## Open PRM checkpoints

| Model | Size | Training data / method | ProcessBench F1 (avg) | Notes |
|-------|------|------------------------|-----------------------|-------|
| **Qwen2.5-Math-PRM-7B** | 7B (BF16, ~16 GB) | Consensus-filtered MC + LLM-as-judge, on Qwen2.5-Math-7B base | **~63.6** | Current open-weight SOTA at 7B; `<extra_0>` step token |
| **Qwen2.5-Math-PRM-72B** | 72B | Same, scaled | **78.3** | SOTA overall; too large for 40 GB |
| **Qwen2.5-Math-7B-PRM800K** | 7B | Pure PRM800K human labels | 56.5 | Clean baseline; no MC noise |
| **Skywork-o1-Open-PRM-Qwen-2.5-7B** | 7B | Qwen2.5-Math-7B-Inst base, automatic MC on GSM8K/MATH/GaoKao/Olympiad + code | ~42 (reported elsewhere) | Covers math+code; newline-step |
| **Skywork-o1-Open-PRM-Qwen-2.5-1.5B** | 1.5B | Same recipe, small | lower | Useful for cheap first pass |
| **RLHFlow/Llama3.1-8B-PRM-Mistral-Data** | 8B | Llama-3.1-8B on 273k Math-Shepherd-style labels from Mistral | ~48–50 | BoN@1024: 92.4% GSM8K, 46.3% MATH |
| **RLHFlow/Llama3.1-8B-PRM-Deepseek-Data** | 8B | Same recipe, Deepseek-generated | ~51 | Better OOD |
| **GAIR/ReasonEval-7B** | 7B | WizardMath-7B-v1.1 FT on PRM800K, per-step classifier | — | Labels each step with {valid, redundant, invalid}; unique taxonomy |
| **launch/ThinkPRM-1.5B** | 1.5B | R1-Distill-Qwen, 1K synthetic verification CoTs | beats discriminative on ProcessBench @ same budget | Very cheap; generative |
| **launch/ThinkPRM-14B** | 14B | Same, larger | best in family | Fits in 40 GB with Qwen3-8B only if sharded |

**Winner for MATH/algebra at 7–8B:** Qwen2.5-Math-PRM-7B. It is the best
open-weight PRM at this size on ProcessBench and on BoN. Skywork is its nearest
drop-in replacement and wins if we also need to score *code*. ThinkPRM-1.5B is
the right choice if we need a *very* small scorer running in parallel with a
large generator.

## How to use a PRM

### At inference

- **Best-of-N + PRM.** Sample N candidate solutions from Qwen3-8B, score each
  with the PRM, pick the one with the highest aggregate score. Aggregation
  matters: Qwen2.5-Math-PRM-7B uses *product of step probabilities*; Skywork
  uses *mean*; "min" is a common hedge against one bad step. This is the
  cheapest and most reliable use of a PRM and typically gives ~+10–15 pp over
  majority voting at MATH difficulty.
- **Weighted majority voting.** Aggregate by answer, with each answer's weight
  = its PRM score. Often beats BoN when there are many correct-ish candidates
  that differ only at the last step. As long as the PRM is "better than
  random", this strictly dominates unweighted majority voting.
- **PRM-guided beam / step-level search (e.g. REBASE, ReST-MCTS\*).** At every
  step boundary, expand the top-B prefixes by PRM score. Gets more of the PRM's
  value per sample but is implementation-heavy and interacts poorly with KV
  caching. Probably *not* the first thing to try here.

### At training

- **Rejection-sampling fine-tuning (RFT / RAFT).** Sample K completions per
  prompt, keep those whose PRM aggregate score exceeds a threshold (or whose
  *min* step score is positive), SFT on the survivors. Simple and compatible
  with our current pipeline; an easy first step to inject process signal into
  Qwen3-8B.
- **Step-level DPO.** Build (preferred, rejected) step pairs — same prefix,
  divergent next step, PRM disagrees — then apply DPO or IPO on that pair
  boundary. Much stronger process signal than response-level DPO but requires
  careful prefix matching.
- **GRPO with step-level reward.** Instead of a terminal 0/1 reward, the PRM
  provides a dense per-step reward. Equivalent to swapping the MC-estimated
  advantage in standard GRPO for PRM-derived advantages. "GRPO is Secretly a
  PRM" (Sep 2025) argues the two are equivalent up to the MC-estimation
  accuracy — so the PRM just removes variance. Use min-form credit assignment
  (arXiv 2501.xxxxx) to avoid reward-hacking of the high-reward step.

### As a labeler for our memory module

Most directly useful use case for us. Run Qwen3-8B on MATH/algebra prompts,
split the generation on `\n\n`, feed `(problem, step_1, …, step_i)` into the
PRM, collect per-step score trajectories s_1..s_N. Then:

- A **critical step** = step where s_{i+1} − s_i is strongly negative, i.e.
  where the model "broke" the trace. Candidate for memory steering.
- A **decision point** = step where s_i has high *variance* over many
  sampled continuations at that prefix — the PRM is uncertain. Equivalent to
  the disagreement signal we used in the DecisionChains variant, but now
  driven by a real math model.
- **Soft labels for memory writes.** Write to memory only at steps with high
  |Δs|, skipping routine algebraic manipulation where PRM is flat.

This maps cleanly onto the `[TRACE]` / decision-point machinery we already
have in the DecisionChains experiment: replace `func_f / func_g` coin-flip
oracle with a PRM-derived "here is where reasoning forks" oracle.

## Step boundary conventions

**No universal convention — this is the single biggest integration footgun.**

- **Qwen2.5-Math-PRM-7B / 72B**: split on `\n\n`, then join the pieces with
  the special `<extra_0>` token, which the model uses as the per-step
  scoring position. Needs `trust_remote_code=True` and the custom
  `modeling_qwen2_rm.py`.
- **Skywork-o1-Open-PRM**: newline `\n` as `step_token` argument. Rewards
  returned as a list per step.
- **Math-Shepherd / RLHFlow**: typically "Step N:" text prefix, scored at the
  position of a `ки` (Cyrillic) step-end token in the original Math-Shepherd
  paper — check the model card before use.
- **ReasonEval**: per-step multi-class head (valid / redundant / invalid); no
  in-line special token, feeds steps one at a time.
- **ThinkPRM**: generative — emits a verification CoT in free text; parse the
  per-step verdict from the output.

For Qwen3-8B output, reliable practice is to **force the generator's step
delimiter to match the PRM's**. Prompt Qwen3-8B with a "Separate steps with
`\n\n`" instruction, then use Qwen2.5-Math-PRM-7B — both agree on `\n\n`. If
Qwen3 uses its `<think>…</think>` block in thinking mode, strip it first and
score only the post-think solution, otherwise the PRM will be badly
out-of-distribution.

## Limitations and known failure modes

- **Reward hacking.** PRMs latch onto shallow cues — fluency, formatting,
  hedging phrases — rather than semantic correctness. The "Reward Under
  Attack" OpenReview paper shows reward deltas <0.1 under logic-breaking
  edits but style-only perturbations produce similar invariance, i.e. the
  signal is not orthogonal to surface form.
- **Final-step bias.** Qwen's own lessons paper flags that many PRMs assign
  their minimum score to the last step, effectively becoming ORMs. Check
  the step-score histogram before trusting an aggregate.
- **Calibration.** Raw PRM scores are not calibrated across problems.
  Per-problem z-scoring is usually necessary before thresholding.
- **BoN ceiling.** On ProcessBench Best-of-8, no open PRM beat simple
  majority voting on average — the gain only appears at larger N.
- **Granularity mismatch.** PRMs trained on 1–2 sentence "atomic" steps do
  not handle long chains-of-thought or multi-sentence steps well; recent
  ReasonFlux-PRM and PRINTS papers explicitly target this gap.
- **Step-split brittleness.** If the generator emits a blank line inside a
  LaTeX environment, `\n\n` splitting fragments the step and the PRM score
  collapses. Worth adding a LaTeX-aware splitter.

## Practical recommendation for our project

**First PRM to wire in: Qwen2.5-Math-PRM-7B.**

- Memory budget on one 40 GB A100: Qwen3-8B in BF16 ≈ 16 GB + KV cache
  (~4–6 GB at reasonable seq length) + Qwen2.5-Math-PRM-7B in BF16 ≈ 16 GB +
  small scoring KV. Total ≈ 36–38 GB — fits, but tight; keep batch size
  small on the PRM side or quantise the PRM to 8-bit (≈ 8 GB).
- Step delimiter: `\n\n`, no special chat-template gymnastics needed.
- Highest ProcessBench F1 among 7B-class open PRMs, same tokenizer family as
  our generator, best documented usage code on HuggingFace.
- If VRAM becomes a problem or we want parallel batched scoring: swap in
  **Skywork-o1-Open-PRM-Qwen-2.5-1.5B** (≈ 3 GB, newline split) or
  **ThinkPRM-1.5B** for a generative option.
- Expected utility: (i) Decision-point labeling at scale — we can label the
  entire MATH/algebra train set in a few hours. (ii) Evaluation oracle
  richer than final-answer accuracy. (iii) Optional reward signal for a
  later GRPO-on-memory phase.

Do **not** start with Math-Shepherd-PRM-7B or Skywork-7B for the labeling
task — Qwen's 2025 lessons paper shows their consensus-filtered model is
cleaner, and "clean labels" is exactly what we need if downstream modules
will take the PRM score as ground truth.

## Sources

- Lightman et al., "Let's Verify Step by Step", arXiv:2305.20050, https://arxiv.org/abs/2305.20050
- PRM800K dataset, https://github.com/openai/prm800k
- Wang et al., "Math-Shepherd", arXiv:2312.08935, https://arxiv.org/abs/2312.08935
- Qwen team, "The Lessons of Developing PRMs in Mathematical Reasoning", arXiv:2501.07301, https://arxiv.org/abs/2501.07301
- Qwen2.5-Math-PRM-7B model card, https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
- Qwen2.5-Math-PRM-72B model card, https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B
- Qwen blog on PRMs, https://qwenlm.github.io/blog/qwen2.5-math-prm/
- Skywork-o1-Open-PRM-Qwen-2.5-7B, https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
- Skywork-o1-Open-PRM-Qwen-2.5-1.5B, https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
- Skywork inference code, https://github.com/SkyworkAI/skywork-o1-prm-inference
- RLHFlow/Llama3.1-8B-PRM-Mistral-Data, https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data
- RLHFlow/Llama3.1-8B-PRM-Deepseek-Data, https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data
- RLHFlow math-rm code, https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm
- GAIR/ReasonEval-7B, https://huggingface.co/GAIR/ReasonEval-7B
- ReasonEval paper, https://arxiv.org/html/2404.05692v1
- ProcessBench paper, arXiv:2412.06559, https://arxiv.org/abs/2412.06559
- PRMBench paper, arXiv:2501.03124, https://arxiv.org/abs/2501.03124
- PRMBench project page, https://prmbench.github.io/
- R-PRM, arXiv:2503.21295, https://arxiv.org/abs/2503.21295
- ThinkPRM ("Process Reward Models That Think"), arXiv:2504.16828, https://arxiv.org/abs/2504.16828
- launch/ThinkPRM-1.5B, https://huggingface.co/launch/ThinkPRM-1.5B
- "GRPO is Secretly a PRM", arXiv:2509.21154, https://arxiv.org/html/2509.21154
- "Reward Under Attack", OpenReview, https://openreview.net/pdf?id=Hw24VOppus
- Inference scaling laws (Wu et al., ICLR 2025), https://arxiv.org/html/2408.00724v1
- HF test-time scaling blog, https://venturebeat.com/ai/hugging-face-shows-how-test-time-scaling-helps-small-language-models-punch-above-their-weight
- Survey: "A Survey of Process Reward Models", arXiv:2510.08049, https://arxiv.org/pdf/2510.08049
