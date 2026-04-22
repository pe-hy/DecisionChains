# Math Evaluation Methodology — Review & Recommendations

## Why this matters

We will compare a frozen Qwen3-8B baseline against LoRA- and memory-steered variants on
MATH-500, algebra__linear_1d (1k test), and possibly GSM8K / GSM-Symbolic. Without a
nailed-down eval protocol the comparison becomes noise: published numbers disagree by
5–10 points across papers on the same model/benchmark, and a 2–4 point difference
between our own runs can be entirely explained by (a) sampling variance, (b) answer-
extraction rules, or (c) prompt template drift. This doc fixes the protocol *before*
any results land in a table.

## Published baselines we need to match

Qwen3-8B has very uneven public reporting. The tech report (arXiv 2505.09388,
Table 6) only reports the **base model** on standard few-shot benchmarks; instruct
/ thinking-mode numbers are only published for the flagship 235B-A22B. Third-party
leaderboards fill the gap but use different protocols.

| Model | Bench | Score | Protocol | Source |
| --- | --- | --- | --- | --- |
| Qwen3-8B-Base | MATH | 60.80 | 4-shot CoT | Qwen3 tech report, Table 6 |
| Qwen3-8B-Base | GSM8K | 89.84 | 4-shot CoT | Qwen3 tech report, Table 6 |
| Qwen3-8B-Base | GPQA | 44.44 | 5-shot CoT | Qwen3 tech report, Table 6 |
| Qwen3-8B (non-thinking) | MATH-500 | ~43.55 | EvalScope pass@1 | EvalScope Qwen3 guide |
| Qwen3-8B (thinking) | AIME24 | 23.3 | avg pass@1 | EvalScope Qwen3 guide |
| Qwen3-8B (thinking) | AIME25 | 13.3 | avg pass@1 | EvalScope Qwen3 guide |
| Qwen2.5-Math-7B-Instruct | MATH | 83.6 (CoT) / 85.3 (TIR) | greedy | Qwen2.5-Math report |

Key facts:
- **Greedy decoding is explicitly warned against on the model card** ("can lead to
  performance degradation and endless repetitions"). This alone forces us away from
  greedy as the *primary* metric and toward low-temperature sampling.
- Recommended sampling params: thinking `T=0.6, top_p=0.95, top_k=20`; non-thinking
  `T=0.7, top_p=0.8, top_k=20`; output budget 32k (38k for competition math).
- The EvalScope 43.55% on MATH-500 is non-thinking mode. With thinking, Qwen3-8B
  should comfortably exceed 80% on MATH-500 given the 60.8% base score. We should
  **reproduce this before attempting any training comparison** — if we can't hit
  within 2 pts of a published reference, our harness is broken and every downstream
  number is suspect.

## Metrics we should report

1. **Pass@1 at recommended sampling temp** (primary): median over 8 seeds to dampen
   variance. For thinking mode `T=0.6, top_p=0.95, top_k=20`.
2. **maj@16** (secondary): self-consistency vote; 2-3x more reliable than pass@1
   at ~16x cost. Worth running only for the final head-to-head, not during sweeps.
3. **Pass@1 greedy** (control): documented as degraded but useful as a *lower-bound
   determinism check* — if greedy and T=0.6 diverge wildly, something is wrong.
4. **Avg generated tokens / problem** (cost axis): reasoning models spend 2-10x
   more tokens than instruct. Memory-steering is claimed to reduce reasoning
   length — we need to measure this.
5. **Tokens-per-correct-answer** = (avg tokens) / accuracy. A model that scores
   60% at 2k tokens is strictly better than 62% at 8k tokens in a serving budget.
6. **Error bars**: Wilson score 95% CI on every accuracy number. For n=500, the
   half-width at p=0.6 is about ±4.3 pp. Don't report a point estimate alone.

## Normalization = the invisible 5% of accuracy

The Open-LLM-Leaderboard post-mortem found that swapping their Minerva-style grader
for `math-verify` recovered on average **+4.66 MATH points per model** (up to +40
on some). Almost every "grader" failure is a false negative — the model had the
answer, the parser lost it. Because the gap is that large, **a parser bug can
entirely explain or mask a training effect**. See 03_cot_parsing_and_answer_extraction.md
for the detailed review; the one-line takeaway for this doc: use `math-verify` (HF)
as the grader for both baseline and trained variants, identical version, identical
config. Never swap graders between conditions.

## Recommended protocol (concrete)

```
for each (model, benchmark):
    1. Strip <think>...</think> blocks only for display; grader sees full text
    2. Generate with thinking=True, T=0.6, top_p=0.95, top_k=20, max_new=32768
       - 1 sample for pass@1
       - 16 samples for maj@16 (stored; reuse across metrics)
    3. Answer extraction via math-verify with LatexExtractionConfig + ExprExtractionConfig
       - For GSM8K: also accept the #### marker
       - For algebra__linear_1d: exact-match on normalized decimal/fraction
    4. Correctness = math-verify's symbolic-equivalence check
    5. Aggregate:
       - Pass@1 = mean correctness over N problems × 1 sample × 8 seeds
       - maj@16 = majority-vote answer per problem, then correctness; average over problems
       - Report Wilson 95% CI on every number
    6. Save raw outputs to JSONL (problem_id, seed, sample_idx, output, extracted, correct)
       for post-hoc re-grading when the grader inevitably changes
```

Determinism note: with flash-attention-2 + bf16, pass@1 greedy is **not bitwise
reproducible** across hardware even at T=0 — reductions are non-deterministic. We
accept this and report seed-median, not single-run.

## Using lm-evaluation-harness

Harness tasks relevant to us:
- `hendrycks_math500` — the 500-problem OpenAI Verify-Step-by-Step subset.
- `hendrycks_math_*` (7 subjects) — full 5k split.
- `minerva_math` — Minerva's 4-shot prompt + SymPy grader (stricter).
- `gsm8k` — 8-shot maj@1.
- `gsm8k_cot` — 0-shot CoT, matches how Qwen3 reports.
- `gsm8k_cot_llama` — variant with chat template (`--fewshot_as_multiturn
  --apply_chat_template`).

**Pros of using harness**: prompt / parsing already aligned to published numbers;
`--apply_chat_template` + `--system_instruction` handle the Qwen3 thinking
template; integrates with vLLM (`--model vllm`) for speed; `enable_thinking=True`
is a supported model arg for Qwen.

**Cons for our setup**: our memory-steered model needs a forward-hook to be
registered on the loaded HF model. The harness loads via its own path, so we'd
need to monkey-patch the `AutoModelForCausalLM.from_pretrained` or use
`--model hf --model_args pretrained=...` and attach hooks after load (the
harness exposes the model as `.model`).

**Recommendation**: use the harness for the baseline Qwen3-8B reference run
(one-time, to reproduce within 2pt of the tech report on GSM8K), then use our
own lightweight eval loop for all memory/LoRA runs so hook-attachment is
first-class rather than bolted-on. Keep the prompt and grader byte-identical
between the two paths. A snapshot test that runs both paths on 50 problems and
checks equal per-problem scores catches drift early.

## Benchmark splits and comparability

- **MATH-500**: the OpenAI subset used by DeepSeek / s1 / most recent papers.
  Our primary. Reporting "MATH" without specifying 500 vs 5k invites confusion —
  **always say "MATH-500"**.
- **MATH 5k**: if we want to reproduce the Qwen3 tech-report 60.8 number, use this
  and 4-shot CoT, not the 500 subset. The two are not interchangeable: MATH-500
  is skewed slightly easier than the full 5k, so scores run ~3-5 pp higher.
- **GSM8K**: 1319 test problems, standard. Qwen3-8B-Base reports 89.84 @ 4-shot
  CoT. Zero-shot CoT with chat template typically reads 1-2 pp higher.
- **algebra__linear_1d** (DeepMind mathematics): generated from
  `python -m mathematics_dataset.generate --filter=linear_1d`. The canonical
  protocol is exact-match on the single-token integer/fraction answer, sampled
  from the `interpolate` split (harder than train-easy). No few-shot.
  Flan-T5-Large trained *on the 2M train set* reaches 90.8% — an 8B LM without
  task-specific training will score lower than that ceiling but is the natural
  comparison for memory steering because the task is narrow and ruled-based.

## Stratification by difficulty

MATH has levels 1-5 (1 easiest) and 7 subjects. Averaged accuracy hides where a
technique helps. We will report:

- **Per-level accuracy**: 5 rows × each model × benchmark
- **Per-subject accuracy**: 7 rows × each model (for MATH only)
- Level 5 only (the MATH-Level-5 benchmark, 1324 problems) as a "hard-tail"
  secondary metric — this is where reasoning-length tradeoffs show up most.

A 2-point average gain that is +6 on level 5 and 0 on levels 1-2 tells a very
different story than +2 uniform. For memory-steering in particular, the
hypothesis is that the gain concentrates on problems the model *almost* gets
right — levels 3-4.

## Statistical noise

For n=500 and p=0.6, the Wilson 95% CI half-width is ±4.3 pp. For n=1000 (algebra
1d) at p=0.7, it's ±2.8 pp. Delta thresholds:

| Compare | n | Minimum detectable delta (95% power, 5% α) |
| --- | --- | --- |
| MATH-500 A vs B, both ~0.60 | 500 | ~6 pp |
| MATH-500 paired (same problems) | 500 | ~3 pp (McNemar) |
| algebra__linear_1d | 1000 | ~4 pp |
| GSM8K | 1319 | ~3.5 pp |

**Always compute McNemar on paired per-problem correctness**, not the two
marginal accuracies — we evaluate both models on the same items so the paired
test is strictly more powerful. A 2-point gap on MATH-500 in the marginal
comparison is not significant; the same gap in the paired test often is.

## Cost estimates on 40 GB A100

Qwen3-8B in bf16 is ~16 GB weights + KV cache. On A100 40GB with vLLM:
- Throughput (decode-bound, batch>=16): **~2000-3000 tok/s aggregate**.
- Single-stream decode: ~60-80 tok/s.
- Thinking traces on MATH average 2-6k tokens; budget 4k per-problem.

Wall time estimates (500 problems × 4k tokens × K samples, batched):
- K=1 (pass@1): 500 × 4k = 2M tokens → ~15 min at 2.5k tok/s
- K=16 (maj@16): 32M tokens → ~3.5 hours
- K=64 (ablation): 128M tokens → ~14 hours

These are aggressive estimates assuming vLLM + prefix cache + batched generation.
HuggingFace native `generate()` is 3-5x slower. **Use vLLM for eval**; we can
attach the memory hook via vLLM's `--override-generation-config` + a custom
`LogitsProcessor`, or by saving a full HF checkpoint with the hook's effect
baked into the weights. If the hook is non-trivially dynamic, fall back to HF
`generate()` and budget 4x the above.

Memory: 40 GB fits Qwen3-8B + 32k context + batch 4 comfortably in bf16.
Batch 8 at 32k is the practical ceiling; drop to 16k context for batch 16.

## Cross-experiment comparison schema

Mirror the naive_ft JSON format so comparison plots stay trivial. One file per
run:

```json
{
  "run_id": "mem_v3_lr1e4_seed42",
  "model": "Qwen3-8B",
  "mode": "thinking",
  "intervention": {"type": "memory_hook", "layer": 16, "rank": 64, "params": 4.2e6},
  "eval": {
    "benchmark": "math500",
    "protocol": "thinking_t0.6_topp0.95_topk20_max32k",
    "grader": "math-verify-0.8.0",
    "n_samples_per_problem": 16,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7]
  },
  "results": {
    "pass@1_mean": 0.612, "pass@1_ci95": [0.568, 0.654],
    "maj@16": 0.684, "maj@16_ci95": [0.641, 0.724],
    "avg_tokens": 3421, "tokens_per_correct": 5589,
    "per_level": {"1": 0.89, "2": 0.81, "3": 0.72, "4": 0.51, "5": 0.33},
    "per_subject": {"algebra": 0.74, "geometry": 0.48, ...}
  },
  "cost": {"gpu_hours": 3.2, "total_tokens": 32_000_000},
  "ts": "2026-04-22T12:34:56Z"
}
```

The comparison table is then generated from all per-run JSONs in a single pass:

| Variant | Train params | Train h | MATH-500 pass@1 | MATH-500 maj@16 | alg_1d | Avg tok | Tok/correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B baseline | 0 | 0 | 60.8 ± 4.3 | 68.4 ± 4.0 | ?? | 3.4k | 5.6k |
| LoRA-A (r=16) | ... | ... | ... | ... | ... | ... | ... |
| Memory v3 | ... | ... | ... | ... | ... | ... | ... |

## Pitfalls

1. **Flash-attn-2 + bf16 is not bitwise deterministic**. Fix seeds but report
   seed-median, not single-run. Don't compare raw log probabilities across runs.
2. **Greedy on Qwen3 repeats**. Never report greedy pass@1 as the only metric.
   If greedy is shown, pair it with a T=0.6 sample.
3. **Chat template drift**. The Qwen3 template has distinct tokens for
   thinking-on vs thinking-off. Mismatching it silently tanks scores 5-15 pp.
   Always use `tokenizer.apply_chat_template(messages, enable_thinking=True)`.
4. **`<think>` leaking into answer extraction**. If the boxed regex finds the
   first `\boxed{...}` inside `<think>`, you score the thought, not the answer.
   Strip or look for the **last** `\boxed{}` *after* the `</think>` token.
5. **Non-stripped EOS / padding tokens** in the generated string cause the
   grader's parser to fail on otherwise-correct outputs. Always strip specials
   before the grader.
6. **Train-test contamination**: Qwen3 was trained on MATH and GSM8K. The
   benchmarks are *not* held-out for the base model. For our purposes this
   is acceptable because all conditions (baseline, LoRA, memory) share the
   same contamination — we measure **the training intervention's delta**, not
   raw capability. But we should note this in any external writeup.
7. **KV-cache position bugs** surface when the hook injects tokens mid-generation
   or alters attention. Smoke-test: generate a fixed prompt twice in a batch
   alongside a third throw-away prompt; the first two outputs must be identical.
8. **Different max_new_tokens between baseline and trained variant**. If the
   memory model terminates at 2k and baseline runs to 32k, you're partly
   measuring truncation. Fix the budget identically.
9. **Partial-score averaging**: never average accuracy *across* benchmarks
   into a "mean score" — the benchmarks have different ns and difficulties.
   Report every benchmark separately.

## Sources

- [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 blog — Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [EvalScope Qwen3 eval guide](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- [Qwen2.5-Math blog](https://qwenlm.github.io/blog/qwen2.5-math/)
- [Qwen2.5-Math Technical Report (arXiv 2409.12122)](https://arxiv.org/html/2409.12122v1)
- [Self-Consistency (Wang et al., ICLR 2023, arXiv 2203.11171)](https://arxiv.org/abs/2203.11171)
- [lm-evaluation-harness — hendrycks_math README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/README.md)
- [lm-evaluation-harness — gsm8k README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md)
- [HuggingFace Math-Verify](https://github.com/huggingface/Math-Verify)
- [MATH dataset (Hendrycks et al., NeurIPS 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)
- [MATH-500 dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [DeepMind mathematics_dataset repo](https://github.com/google-deepmind/mathematics_dataset)
- [Wilson score interval — Wikipedia](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- [Artificial Analysis MATH-500 leaderboard](https://artificialanalysis.ai/evaluations/math-500)
- [MATH Level 5 benchmark (Epoch AI)](https://epoch.ai/benchmarks/math-level-5)
