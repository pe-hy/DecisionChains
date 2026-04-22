# 2026 Advances — Reasoning Methods & Evaluation

## Time horizon
January – April 2026. Only papers and releases landing inside this window are in
scope; everything older already covered in `research/03_cot_parsing_and_answer_extraction.md`
and `research/08_evaluation_methodology.md`.

## New reasoning methods 2026

**Latent / continuous CoT — post-Coconut wave.**
- **Latent Thoughts Tuning (LT-Tuning)**, arXiv:2602.10229, Feb 2026. Direct critique
  of Coconut at scale: the original recycled-hidden-state scheme degrades sharply
  at 8B (authors report Coconut-8B loses most of its 1B-scale gain on GSM8K). LT-Tuning
  decouples the latent state from the embedding matrix so untied-embedding models
  scale; 8B LT-Tuning roughly doubles Coconut-8B accuracy on their multi-hop suite.
  This is the cleanest signal yet that "just feed the hidden state back in" does
  not survive scale.
- **Chain of Latent Tool Calls (CoLT)**, arXiv:2602.04246. Replaces discrete tool
  tokens with continuous latent "tool-call" vectors; keeps the explicit answer
  token but hides the intermediate planning.
- **When Shallow Wins: the Depth–Accuracy Paradox in Latent Reasoning**,
  arXiv:2603.03475, Mar 2026. Shows that latent-CoT models with *fewer* layers can
  beat deeper ones on some multi-hop tasks because deeper models silently fail
  mid-chain; argues latent CoT is strictly bounded by depth (k-hop requires at
  least k layers), matching what implicit-reasoning theory predicted.
- **Loop, Think, & Generalize** (Recurrent-Depth Transformers), arXiv:2604.07822,
  Apr 2026. Looped transformers act like latent CoT at fixed parameter count; the
  paper shows they outperform parameter-matched explicit-CoT finetunes on
  arithmetic and logic.

**Parallel / diffusion CoT.**
- **Continuous CoT Enables Parallel Exploration (CoT2)**, arXiv:2505.23648, Mar 2026
  version. Extends Coconut with "superposition" continuous tokens — one embedding
  can carry a distribution over multiple discrete next tokens, tracking several
  reasoning branches per step.
- **Why Diffusion LMs Struggle with Truly Parallel Decoding**, arXiv:2602.23225,
  Feb 2026. Empirical: "parallel" diffusion LMs collapse to AR-like left-to-right
  generation on CoT data because the training traces are themselves sequential.
  Proposes NAP: curate *independent* reasoning trajectories and force
  multi-token parallel updates.
- **On the Reasoning Abilities of Masked Diffusion Language Models**, arXiv:2510.13117
  (Mar 2026 revision). Theoretical equivalence result: masked diffusion models can
  simulate any CoT-augmented transformer, and are *strictly faster* for problems
  admitting parallel decomposition. First clean separation of DLM from AR CoT in
  expressivity terms.

**Implicit CoT mechanistics.**
- **Layer Specialization** framework consolidated in the Implicit-Reasoning survey
  (arXiv:2509.02350, extended 2026): shallow / middle / deep layers map to
  retrieval, composition, answer-decoding. Relevant because our memory project
  conditions on middle-layer residuals; 2026 work says that is the "composition"
  regime where implicit multi-hop lives.

## Test-time compute 2026

- **When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling**,
  arXiv:2604.10739, Apr 2026. Quantifies the regression: beyond an
  instance-dependent optimum (often ~2k tokens for MATH), longer CoT causes models
  to *abandon correct answers*. Gives a per-task optimal-length distribution; the
  optimum for GSM8K-class problems is markedly shorter than for AIME.
- **FastTTS: Accelerating Test-Time Scaling for Edge LLM Reasoning**,
  arXiv:2509.00195v2 (2026 revision). Speculative-decoding + draft-verify pipeline
  tuned for TTS: same pass@k with 2-3x wall-clock on 7-8B models. Directly
  applicable to our Qwen3-8B runs if we move off vLLM.
- **S^3: Stratified Scaling Search for Test-Time in Diffusion LMs**,
  arXiv:2604.06260, Apr 2026. First TTS procedure native to diffusion LMs;
  probably not relevant to our Qwen AR setup but confirms that diffusion CoT
  needs its own budget-control primitives.
- **s1 follow-ups.** No major s1-branded replacement in Q1 2026, but budget-forcing
  ("Wait"-append) has been absorbed as a baseline. s1-32B still cited as the
  small-data SFT + budget-forcing proof-of-concept; newer work (e.g. CtrlCoT,
  below) compresses rather than lengthens.

## Self-consistency variants 2026

- **CGES: Confidence-Guided Early Stopping**, arXiv:2511.02603 (2026 ICLR).
  Bayesian posterior over candidate answers from token-prob or reward-model
  confidence; halts when posterior mass crosses a threshold. Reported **69.4%
  reduction in LLM calls** (16.0 → 4.9 average) with <0.1 pt accuracy loss vs.
  maj@16. This is the cheapest self-consistency variant we have seen.
- **ReASC (Reliability-Aware ASC)**, Kim et al., Jan 2026. Response-level
  confidence + weighted aggregation; 70–80% cost saving vs. naive SC on
  MATH/GSM8K.
- **Certified Self-Consistency**, arXiv:2510.17472 (2026). Gives formal
  statistical guarantees: PAC-style bounds on the probability that maj@k agrees
  with maj@∞. Useful when we want to claim statistical significance in a paper
  rather than just "maj@16 is higher".
- **Deep Think with Confidence** (DeepConf). Combines CoT branching with confidence
  filtering; sits between maj-vote and beam-search. Competitive with maj@32 at
  maj@8 cost in the paper's reporting.

## New math benchmarks 2026

- **AIME 2026** (released Feb 2026) and **HMMT Feb 2026**. Published through
  MathArena (`huggingface.co/datasets/MathArena/aime_2026`,
  `MathArena/hmmt_feb_2026`). 30 problems (AIME I+II), integer answers 0-999,
  evaluated 4 runs/problem with cost tracking. Uncontaminated by construction —
  models released before Feb 2026 cannot have seen them.
- **Saturation alert.** MathArena's own aggregate dropped AIME/HMMT from the
  weighted score in early 2026 because top models score 95-99%. GPT-5 reported at
  100% on AIME 2026; Qwen3.5-plus at 91.3%. **AIME as a frontier signal is dead
  for any model ≥ ~30B; it still discriminates at our 8B scale, where Qwen3-8B
  thinking-mode reports ~23% on AIME 2024.**
- **MathArena proof track.** First proof-writing benchmark with rubric-based LLM
  judges. On IMO 2025 top models land just under 40% — still enormous headroom.
- **Putnam-AXIOM** (ICML 2025, actively used through 2026). 522 undergraduate
  problems with a *variation protocol* that generates unlimited structurally
  equivalent instances, giving a contamination-resilient test bed. This is
  currently the gold standard for "unseen" math evaluation outside competition
  releases.
- **GAUSS Eval** (2026). Consistency study of 14 frontier models vs. human judges
  on MathArena USAMO 2025; documents systematic LLM leniency bias on proof
  grading. Not a pass/fail benchmark — use to calibrate any LLM-judge we deploy.

## Benchmark contamination & trust 2026

- **AIME 2024 is contaminated.** Multiple 2026 analyses (MathArena paper update,
  and independent contamination surveys) show most frontier models sit 10-20
  points above the human line on AIME 2024 — a signature of memorisation given
  human solvers' baseline. AIME 2025 shows the same curve shape, softer.
- **Qwen3 MATH-training decontamination.** The Qwen3 tech report (and Qwen2.5-Math
  before it) documents 13-gram matching plus structural similarity filtering
  against MATH/GSM8K test sets, and explicitly excludes variations of test items.
  No 2026 paper has published a contrary contamination claim *specific to Qwen3*
  on MATH, but absence of evidence is not evidence of absence. Closest signal:
  DICE (arXiv:2402, extended 2026) flags internal-state contamination signatures;
  public results do not single out Qwen3 on MATH.
- **CoDeC** (Contamination Detection via Context, 2026 ICLR). Probes whether the
  model responds differently to test items vs. near-paraphrase controls. Cheap
  enough to run on our Qwen3-8B baseline if we want a paper-ready contamination
  statement.
- **Dynamic-eval argument.** The 2026 meta-survey (arXiv:2502.17521v2) makes the
  now-consensus case that any static leaderboard ≥ 12 months old should be
  assumed contaminated. Implication for us: MATH-500 and GSM8K numbers should be
  reported with a disclaimer, not as the headline.

## Tooling updates 2026

- **math-verify (HuggingFace).** Latest releases swap sympy's `FiniteSet` for
  `latex2sympy2_extended`'s implementation, fix tuple/set comparison edge cases,
  and harden sort logic against `TimeoutError`. The Open LLM Leaderboard
  re-evaluation with math-verify raised average math scores by ~4.66 points
  across 3751 models; Qwen-family scores more than doubled, DeepSeek tripled
  — driven by correct handling of `\boxed{…}` outputs. **Upgrading math-verify
  is the single biggest free accuracy delta available to us.**
- **lm-evaluation-harness.** Dec 2025 / Q1 2026: CLI refactor (`run`, `ls`,
  `validate` subcommands), YAML config files first-class, model backends split
  out of base install (transformers/torch no longer required for the core package).
  New `gsm8k_platinum` task added. `leaderboard_math_hard` remains the canonical
  hard-math subset.
- **MathArena tooling** (`eth-sri/matharena` GitHub). Drop-in runner for AIME/HMMT
  2024-2026, USAMO, IMO, Putnam; already cited by Phi-4-Reasoning, Gemini-2.5-Pro,
  Grok-3 release notes. Standardises cost-per-correct reporting.
- **LLM-as-judge grading scale.** Jan 2026 study (arXiv:2601.03444) finds 0–5
  integer scales give the strongest human-LLM alignment for math proof grading;
  binary and 0–100 both underperform. Matters if we ever score proofs.

## Implications for our project

Updates to `research/08_evaluation_methodology.md`:

1. **Add AIME 2026 + HMMT Feb 2026 as our "clean" contamination-controlled
   benchmarks.** At 8B, Qwen3-8B thinking-mode should score in the 15-25%
   range on AIME 2026 — low enough that our memory/LoRA variants have real
   headroom to move the number. This replaces AIME 2024 as the headline
   "hard math" slot; keep AIME 2024 only as a secondary reference.
2. **Upgrade math-verify to the latest release before publishing any table.**
   Document the version in the harness. A 4-5 point free gain across the
   board would silently invalidate earlier comparisons.
3. **Replace `maj@16` with CGES** for self-consistency reporting. Same
   accuracy ceiling at roughly a third of the token cost; stronger selling
   point when readers ask about compute cost. Keep `maj@16` only for the
   head-to-head final table for parity with s1-era papers.
4. **Add a short-CoT cost metric.** Given the overthinking result, report
   accuracy at a capped 1k-token budget in addition to the 32k budget.
   This is the regime where CoT-compression methods (CtrlCoT, TokenSkip)
   could be relevant baselines.
5. **Contamination disclosure paragraph.** In the results section state
   explicitly that MATH-500 and GSM8K are legacy static sets and the
   "trust" claim rests on AIME/HMMT 2026. Cite CoDeC or dynamic-eval
   survey once.
6. **Do NOT pursue latent-CoT as a method swap.** The LT-Tuning result at
   8B and the Depth-Accuracy paradox paper both warn that latent CoT degrades
   at our scale; our memory-conditioning intervention is closer to
   implicit-reasoning layer specialization, not to Coconut-style
   hidden-state recycling.

## Sources

- LT-Tuning: https://arxiv.org/pdf/2602.10229
- CoLT: https://arxiv.org/pdf/2602.04246
- Depth-Accuracy paradox: https://arxiv.org/html/2603.03475
- Recurrent-Depth (Loop, Think, Generalize): https://arxiv.org/html/2604.07822
- CoT2 (continuous superposition): https://arxiv.org/html/2505.23648
- Diffusion parallelism: https://arxiv.org/abs/2602.23225
- Masked-diffusion reasoning: https://arxiv.org/abs/2510.13117
- Implicit reasoning survey: https://arxiv.org/html/2509.02350v1
- Overthinking: https://arxiv.org/html/2604.10739v1
- FastTTS: https://arxiv.org/html/2509.00195v2
- S^3 stratified TTS: https://arxiv.org/html/2604.06260
- s1 (reference): https://arxiv.org/abs/2501.19393
- CGES: https://arxiv.org/abs/2511.02603
- Certified self-consistency: https://arxiv.org/pdf/2510.17472
- DeepConf: https://jiaweizzhao.github.io/deepconf/static/pdfs/deepconf_arxiv.pdf
- MathArena paper: https://arxiv.org/html/2505.23281v2
- MathArena AIME 2026: https://huggingface.co/datasets/MathArena/aime_2026
- MathArena HMMT Feb 2026: https://huggingface.co/datasets/MathArena/hmmt_feb_2026
- MathArena platform: https://matharena.ai/
- Putnam-AXIOM: https://arxiv.org/html/2508.08292v1
- GAUSS Eval: https://gaussmath.ai/eval.html
- CoDeC / contamination surveys: https://arxiv.org/html/2502.17521v2
- Contamination via ICL: https://arxiv.org/html/2510.27055v1
- math-verify (HF): https://huggingface.co/blog/math_verify_leaderboard
- math-verify releases: https://github.com/huggingface/Math-Verify/releases
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness/releases
- MathArena GitHub: https://github.com/eth-sri/matharena
- LLM-judge grading scale: https://arxiv.org/html/2601.03444v1
- Qwen3 tech report (contamination section): https://arxiv.org/pdf/2505.09388
