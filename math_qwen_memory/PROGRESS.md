# Progress

Snapshot as of 2026-04-23. Phase 0 complete: model loading, data, baseline eval, trace capture with per-token entropies, robust grading. No memory module or LoRA yet.

## Model

**Qwen/Qwen3-8B** (stock HuggingFace). Loaded bf16, `attn_implementation="sdpa"`, `device_map="cuda:0"`. Downloaded weights to local `hf_cache/` (26 GB, gitignored via root `.gitignore: hf_cache/`). **Not** the `Qwen3-8B-Thinking-2507` variant — base Qwen3-8B exposes the same thinking toggle via the chat template's `enable_thinking=True` and is broadly supported.

VRAM at load: **16.4 GB** on a single A100-40GB. Inference peaks well under 25 GB at 16K context.

Hardware note: GPU 1 used (others on the node contested). `CUDA_VISIBLE_DEVICES=1 python generate_traces.py ...`.

## Datasets

| Dataset | Source | Path | Split used | N |
|---------|--------|------|-----------|---|
| MATH (competition) | `HuggingFaceH4/MATH-500` subset (local copy of `lilave_puvodni`) | `data/MATH/test_500.jsonl` | test | 20 |
| algebra__linear_1d | DeepMind mathematics dataset (local copy) | `data/algebra__linear_1d/test.jsonl` | test | 20 |
| AIME 2026 | `MathArena/aime_2026` (HF, downloaded via `datasets.load_dataset`) | `data/AIME_2026/test.jsonl` | all 30 | 30 (running) |

All three normalize to `{problem, gold}` internally. Data dir `math_qwen_memory/data/` is **not tracked in git** (root `.gitignore`). Re-copy instructions in `CLAUDE.md`.

**Contamination status** (see `CLAUDE.md` + `research/12`): MATH and algebra were almost certainly in Qwen3 pretraining. AIME 2026 is post-cutoff per MathArena, so it is the contamination-resistant eval.

## Prompt (unified across datasets)

```
System: Please reason step by step, and put your final answer within \boxed{}.
User:   {problem}
```

Canonical Minerva/OpenAI prompt; recommended in the Qwen3-8B model card for math tasks. Applied via `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, return_tensors="pt")`. The `enable_thinking=True` flag causes the model to emit a `<think>...</think>` block before the final answer.

Reason for unifying: single extraction path (`\boxed{...}`) across MATH, algebra, AIME. Simpler, fewer code paths, fewer regex fallbacks.

## Generation config

```
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
output_scores=True       # needed for per-token entropy
return_dict_in_generate=True
pad_token_id=tok.pad_token_id or tok.eos_token_id
```

Sampling follows the Qwen3 model-card recommendation; **greedy is explicitly discouraged for thinking mode** (per `research/08`).

Token budgets (choose per dataset):
| Dataset | max_new_tokens | Rationale |
|---------|----------------|-----------|
| algebra | 2048 | Avg 1661 tokens; 2 of 20 hit cap |
| MATH | 8192 | Avg 4354 tokens; 2 of 20 hit cap |
| AIME 2026 | 16384 | Harder problems; thinking mode routinely uses 10K+ |

## Parsing & grading

Pipeline steps in `generate_traces.py`:

1. `strip_thinking(text)` — regex strip of `<think>...</think>` block (if present) before answer extraction.
2. `extract_boxed(text)` — find the **last** `\boxed{...}` with brace-matching (handles nested `\frac{a}{b}` inside). Returns `None` if no boxed answer.
3. `grade(pred, gold)` — primary path: `math_verify.verify(parse(f"${gold}$"), parse(f"${pred}$"))` — handles LaTeX equivalence (`\dfrac↔\frac`, `^\circ`, fractions vs decimals, sets, intervals). Fallback: string-normalized equality after stripping `\left`, `\right`, `\dfrac→\frac`, `\text{}`, `\mathrm{}`, whitespace, `$`.

Per-token entropy:
```python
for logits in out.scores:
    probs = F.softmax(logits[0].float(), dim=-1)
    h = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    entropies.append(h)
```
Saved alongside tokens and token IDs so later DP-analysis scripts can operate offline.

Each record in the output JSONL:
```
{idx, problem, gold, generation, answer_region, pred, correct,
 tokens, token_ids, entropies}
```

## Pilot results so far

| Dataset | N | Accuracy | Notes |
|---------|---|----------|-------|
| algebra | 20 | **90%** (18/20) | 2 misses = 2048-token truncation; no reasoning errors |
| MATH | 20 | **90%** (18/20) after math-verify rescore (70% raw → 90%) | 2 misses = 8192-token truncation; 4 false-negatives recovered |
| AIME 2026 | 30 | running | GPU 1, 16K-token budget, ETA ~1.5-3h |

Entropy distribution (aggregated over all tokens in all traces):
- algebra: p50=0.000, p95=0.61, p99=0.93
- MATH: p50=0.000, p80=0.20, p95=0.69, p99=1.11

Median entropy ≈ 0 on both datasets — model is extremely confident most of the time, only **top 5-20% of tokens** carry entropy signal. Per-generation percentile thresholding is required for any DP-gating method; absolute thresholds will not generalize.

## Files

- `generate_traces.py` — single-file pipeline. ~240 LOC. CLI: `--dataset {math,algebra,aime2026} --n N --max_new_tokens N` plus `--rescore PATH` for re-grading existing JSONL in place.
- `outputs/pilot/*.jsonl` — trace artifacts (gitignored via `**/outputs/pilot/`).
- `hf_cache/` — HF download cache (gitignored).
- `data/` — dataset copies (gitignored; recreated from `lilave_puvodni` + HF AIME).

## What's next

Per `research/15_pilot_claude_advice_validation.md`:

1. Wait for AIME 2026 run, compute accuracy + stratify by entropy headroom.
2. Send a handful of Qwen3 traces to Claude → ask where to give advice → compare Claude-marks to top-20% entropy → decide whether entropy is a good proxy or we need Claude as labeler.
3. Based on outcome, write Phase-1 memory-module code per `research/14`.

## Research references

- `PLAN.md` — original Phase-0 plan (superseded by this doc for status).
- `CLAUDE.md` — full context for future sessions: research question, hypothesis, engineering gotchas, baselines to beat.
- `research/README.md` — index of 15 literature reviews.
- `research/14_entropy_teacher_memory_design.md` — Phase-1 design: entropy-gated teacher distillation into KV memory.
- `research/15_pilot_claude_advice_validation.md` — the pilot whose prerequisite was this Phase 0.
