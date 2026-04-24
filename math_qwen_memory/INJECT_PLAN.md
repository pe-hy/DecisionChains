# Plan: Qwen3 Thinking-Block Prefill with Algorithm Injection

## Context

`outputs/pilot/aime2026_test_n30.aligned.jsonl` — 30 AIME-2026 records where a Claude annotator tagged reasoning **algorithms** (named canonical routines with pseudo-code) and for each algorithm listed its **spans** (locations in the generation where that routine appeared). Counts: 30 records, 197 algorithms, **187 in-think spans**, 96 out-of-think spans.

New task: for each (example, algorithm, in-think-span) triple, **prefill Qwen3's `<think>` block up to the span's start, inject the algorithm's pseudo-code (code-fenced), let the model continue**. One prompt per (algo, span) — if an algorithm has 2 spans we emit 2 prompts, each replacing only that one span. Output: intervention dataset. Does injecting a canonical routine change Qwen3's downstream reasoning or final answer?

Upstream step before memory-steering experiments — grounds whether "advice at a decision point" is a coherent intervention in natural-language reasoning.

## Decisions (resolved)

| Choice | Value |
|--------|-------|
| Injection text | **pseudo-code only**, wrapped in triple-backtick code fence |
| Span scope | **in-think only** (187 spans, `start_think` not null) |
| `max_new_tokens` | **16384** (match baseline AIME budget) |
| Output | new JSONL **`outputs/pilot/aime2026_injections.jsonl`**, one record per (ex, algo, span), incremental save |

## Prefill construction

**Validated** by inspecting Qwen3-8B's chat template: `apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=False)` ends at `"<|im_start|>assistant\n"` — it does **NOT** pre-emit `<think>`. Model normally opens the tag itself at gen-time. For prefill we add `<think>\n` explicitly.

**Do NOT use `continue_final_message=True`** — with Qwen3 it inserts a broken double-think artifact (`<think>\n\n</think>\n\n<think>\n...`). Manual concatenation is the right path.

Final construction per span:

```python
header = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user",   "content": problem}],
    add_generation_prompt=True,
    enable_thinking=True,
    tokenize=False,
)
# header ends at "<|im_start|>assistant\n"
injection = f"\n```\n{algo['pseudo-code'].strip()}\n```\n"
prefill_text = (
    header
    + "<think>\n"
    + think_text[:span["start_think"]]
    + injection
)
input_ids = tok(prefill_text, add_special_tokens=False, return_tensors="pt").input_ids
```

Model continues from the closing code fence. Expected continuation: more thought, `</think>`, final answer with `\boxed{...}`.

## Output record schema

One record per (ex_idx, algo_idx, span_idx):

```json
{
  "ex_idx": 0,
  "algo_idx": 0,
  "algo_name": "Linear System Solver (Substitution / Elimination)",
  "span_idx": 0,
  "span_start_think": 2117,
  "span_end_think": 3161,
  "original_span_text": "...",
  "injection_text": "\n```\n{pseudo_code}\n```\n",
  "prefill_text": "<full chat template + think[:start] + injection>",
  "continuation": "<what the model generated>",
  "full_generation": "<prefill + continuation>",
  "answer_region": "<continuation with <think>...</think> stripped>",
  "pred": "...",
  "gold": "...",
  "correct": true,
  "tokens": [...],
  "token_ids": [...],
  "entropies": [...]
}
```

Entropies are for **the continuation only**. Prefill has no entropy — it was forced.

## Files to create

**1. `math_qwen_memory/inject.py`** — new, ~150 LOC. Reuses helpers from `generate_traces.py` via import:

- `MODEL_ID`, `HF_CACHE`, `SYSTEM_PROMPT`
- `compute_token_entropy`
- `decode_with_entropy` (OOM-safe manual decode, keeps only scalars)
- `strip_thinking`, `extract_boxed`, `grade`

No changes needed to `generate_traces.py`.

CLI:
```
python inject.py \
    --aligned outputs/pilot/aime2026_test_n30.aligned.jsonl \
    --out     outputs/pilot/aime2026_injections.jsonl \
    --max-examples 1           # optional, for pilot subset
    --max-new-tokens 16384
```

## Resume / idempotency

Open output JSONL in **append mode**. At startup, scan existing records, build set of processed `(ex_idx, algo_idx, span_idx)` tuples. Skip those. Flush + `fsync` after each completed record.

Kill/relaunch safe. Same pattern as `generate_traces.py`'s incremental save.

## Budget estimate

187 prompts × ~10K continuation tokens × custom-decode ~25 tok/s ≈ **~11 min each**, **~34 hrs total** on a dedicated A100. Practical approach:

1. **Pilot first**: `--max-examples 1` → ~5 spans for example 0 → ~1 hr. Verify schema + quality.
2. Ramp to `--max-examples 5` → ~30-40 spans → ~6 hrs overnight.
3. Full run (~34 hrs) only if pilot outputs look usable.

## Critical files

- `math_qwen_memory/outputs/pilot/aime2026_test_n30.aligned.jsonl` (input, read-only) — provided by user
- `math_qwen_memory/generate_traces.py` (read-only, imported) — `decode_with_entropy`, `SYSTEM_PROMPT`, `grade`, `extract_boxed`, `strip_thinking`
- `math_qwen_memory/inject.py` (new)
- `math_qwen_memory/outputs/pilot/aime2026_injections.jsonl` (new output, gitignored via `**/outputs/pilot/`)

## Verification

1. **Smoke**: `python inject.py --aligned ... --max-examples 1 --max-new-tokens 2048` on free GPU. Expect ~5 records in output JSONL for ex 0 in-think spans; runtime ~15-30 min.
2. **Schema check**: one record read back via `json.loads` has all expected keys; `len(tokens) == len(entropies)`; continuation contains `</think>` and a `\boxed{...}`.
3. **Prefill correctness**: `prefill_text` ends with a closed code fence + trailing newline. Continuation starts immediately after (no re-opened `<think>`).
4. **Grader sanity**: at least one of the 5 records has `correct=True` (baseline was 277 correct on ex 0) — demonstrates prefill didn't break model.
5. **Resume**: kill smoke mid-run, relaunch — new records append, completed triples skipped.
6. **Full run (optional)**: background on free GPU with `--max-examples 30`, ~30 hrs, monitor via `tail -f`.

## Analysis / results table (second script)

Second file: **`math_qwen_memory/compare_injections.py`** — ~80 LOC. Reads baseline `aime2026_test_n30.jsonl` (or the aligned file for `correct`/`pred`) and `aime2026_injections.jsonl`. Produces side-by-side table:

| ex_idx | algo | span | baseline_correct | injected_correct | baseline_pred | injected_pred | gold | verdict |
|--------|------|------|------------------|------------------|---------------|---------------|------|---------|
| 0 | Linear System Solver | 0 | ✓ (277) | ✓ (277) | 277 | 277 | 277 | no change |
| 0 | Euclidean GCD | 0 | ✓ | ✗ (250) | 277 | 250 | 277 | injection broke |
| 0 | Arrival-Time Equality | 0 | ✗ (truncated) | ✓ (277) | None | 277 | 277 | injection fixed |

Aggregate rows:
- overall: N prompts, baseline acc %, injected acc %, flip-rates (→correct, →incorrect, no-change)
- per-algorithm: same breakdown grouped by `algo_name`
- "unresolved": split between truncated (no `\boxed`) and wrong-but-finished
- per-example: how many of its spans' injections preserved correctness

Output:
- `outputs/pilot/aime2026_injections_table.md` — human-readable markdown (mirrors layout of prior `comparison_table.tex`)
- `outputs/pilot/aime2026_injections_summary.json` — aggregates for programmatic use

## Out of scope

- Out-of-think spans (96) — different intervention pattern, later if needed.
- Multi-span injection per prompt (user explicit: one span changed per prompt).
- Per-span entropy percentile computation — separate script.
