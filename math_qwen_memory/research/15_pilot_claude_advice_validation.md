# Pilot: Can Claude Identify Advice Points in Qwen3 Traces?

## Purpose

Before committing to the full entropy-gated teacher-distillation pipeline (`research/14`), run a **cheap validation pilot** to answer three concrete questions:

1. **Does "advice point" mean anything to a strong model looking at actual Qwen3 math CoT traces?** If Claude can't find coherent intervention points in natural-language reasoning, the whole premise is shaky.
2. **How well do Claude's semantically-chosen advice points align with token-level entropy peaks from Qwen3?** If they align, entropy is a cheap proxy for our pipeline. If they diverge, we need a smarter selector (or accept that Claude is our labeler, not entropy).
3. **What does Claude's "advice" actually look like?** Short hints? Full rewrites? Corrections? This constrains the design of C1 in `research/14` (logit-vs-text advice).

Pilot is ~20–50 problems per dataset, one afternoon of work, no training.

## Procedure

### Step 1 — Generate Qwen3 traces

Pick **20 MATH problems** (5 per level × 4 levels) + **20 algebra__linear_1d problems**. Run Qwen3-8B-Thinking-2507 with:
- Sampling: T=0.6, top_p=0.95 (per model card — `research/08`).
- `output_scores=True`, `return_dict_in_generate=True` — save full logits per token.
- `max_new_tokens=2048`.
- Save per-problem JSON: `{problem, gold_answer, generation_tokens, generation_text, token_logprobs, token_entropies, is_step_boundary[], pred, correct}`.

Sample at least one correct and one incorrect generation per level where possible. Errors are more interesting — that's where advice would actually help.

### Step 2 — Ask Claude to mark advice points

Send Claude the **full trace** (problem + Qwen3's generation as rendered text). Use this prompt:

```
You are reviewing a smaller model's math reasoning. The problem and the model's
step-by-step solution follow. Your job: identify the SPECIFIC POSITIONS in the
solution where, if you could give one short piece of advice, you would change
the model's direction.

Rules:
- Only mark positions where the model is about to make a choice that matters
  for correctness (pick a method, substitute vs eliminate, case split,
  algebraic simplification direction, etc.).
- Do NOT mark routine arithmetic unless the model is about to make an error.
- Zero marks is a valid answer if the solution is clean.
- For each mark, quote the 3-5 word span you'd intervene BEFORE, and write one
  sentence of advice you would give.

Return JSON: [{"quote": "...", "advice": "..."}]

Problem: {problem}

Model's solution:
{generation_text}
```

Collect Claude's response. Repeat for all 40 problems.

### Step 3 — Locate Claude's marks in the token stream

For each `quote` Claude returned, find its token position in the original generation (string search → map to token index). This yields a list of **Claude-marks** `c_j` per problem.

### Step 4 — Compare Claude-marks to entropy

For each problem, compute:
- **Entropy-marks** `e_i`: top-20% of token positions by entropy (also try top-5%, top-10% for threshold sensitivity).
- **Step-boundary-gated entropy-marks** `e'_i`: intersection of top-20% entropy with step boundaries (newline, after `=`, after "therefore" — see `research/04` gate).

Metrics:
- **Recall**: fraction of Claude-marks that fall within ±3 tokens of an entropy-mark.
- **Precision**: fraction of entropy-marks that have a Claude-mark nearby.
- **Rank correlation**: for each Claude-mark, what is its entropy percentile? Expected ≥50% (above median) if the premise holds.
- **Dataset split**: report MATH and algebra separately; they likely differ.

### Step 5 — Qualitative inspection

Pick 5 problems with highest Claude-mark density and 5 with zero marks. Visualize the trace with entropy bars and Claude-marks overlaid (HTML like `../visualization/visualize_superposition.py` but for natural language).

Questions to answer by eyeball:
- Where Claude marks but entropy is low — what's Claude seeing? Is it predicting an error before the model gets uncertain?
- Where entropy is high but Claude doesn't mark — is the model uncertain on something that doesn't matter (formatting, tokenization artifacts)?
- Are Claude's advice sentences consistently coherent and specific, or generic? Generic advice ("check your work") is a red flag.

## Decisions the pilot informs

| Outcome | What it means | Next step |
|---------|---------------|-----------|
| High precision + high recall (Claude-marks ≈ entropy-marks) | Entropy is a good free proxy for semantic branching | Proceed to `research/14` design as-is with entropy gating |
| High recall, low precision (entropy marks many tokens Claude doesn't care about) | Entropy is noisy but catches real points | Keep entropy + step-boundary gate; tune threshold down to top-5% |
| Low recall (Claude marks tokens entropy missed) | Entropy misses semantic decisions | Use Claude directly as labeler in Phase 1b; skip entropy gate, accept higher teacher-query cost |
| Claude returns zero marks on correct solutions | Advice is only needed on errors; model is competent enough already | Shift framing: collect advice only on failure cases (rejection-sample to find errors first) |
| Claude's advice is all generic ("recheck arithmetic") | Advice too vague to distill; memory can't learn from this signal | Switch teacher to a math-specific model (Qwen3-Math-72B, DeepSeek-Math) or reformulate advice as rewrites (C1 option C in `research/14`) |
| Claude marks look semantically great but don't cluster at entropy peaks | The "branching token" concept from `research/04` is specific to deterministic CoTs; natural-language reasoning has distributed decisions | Pivot to step-level rather than token-level interventions; revise memory hook to fire once per step |

## Budget

- Qwen3-8B traces for 40 problems: ~3 minutes on A100.
- Claude API: 40 problems × ~2K tokens in + 1K out ≈ 120K tokens → **~$1.50 on Claude Sonnet**.
- Analysis + visualization: one afternoon of scripting.

Essentially free. Run before any Phase-1 coding.

## Deliverables

- `outputs/pilot/traces.jsonl` — Qwen3 generations with logprobs/entropy.
- `outputs/pilot/claude_marks.jsonl` — Claude's advice points per problem.
- `outputs/pilot/alignment_report.md` — the 5 metrics + qualitative observations.
- `outputs/pilot/inspect.html` — interactive viewer with entropy bars + Claude-marks overlay.
- Updated `research/14` with concrete choices based on pilot outcome.

## Open sub-questions

1. Should Claude see the **correct answer** when marking? Arguments both ways:
   - **Yes** → Claude can pinpoint where the model went wrong (strong label).
   - **No** → mirrors inference-time conditions where no answer is known (honest signal).
   - **Recommendation**: run both; compare. Cheap.
2. Use a different strong teacher (GPT-4o, Gemini 2.5 Pro, Qwen3-Max) as a second annotator — is advice consistent across teachers? Cross-teacher agreement is a rough ceiling on what any single teacher can offer.
3. Does thinking-mode content (inside `<think>...</think>`) need separate handling? Claude should probably only see the **final answer section** (outside thinking), or all of it — decide per dataset.

## Connection to later phases

If this pilot succeeds, Phase 1a/1b (`research/14`) uses **Claude or an equivalent teacher** on ~1000 problems with the same prompt format, distilled into the KV memory. The pilot's prompt template becomes production. The pilot's JSON format becomes the training data schema.

If it fails, we save weeks of engineering by pivoting early.
