# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The research question

Can a small trainable **external memory module** hooked into a frozen Qwen3-8B's residual stream match or beat LoRA fine-tuning at comparable parameter budgets on **natural-language math reasoning**, while preserving the base model's general capabilities? Prior work (`../memory_experiment/`) showed memory beats LoRA on a synthetic 30M-model decision-chain task at matched params (+3.7 pp alignment, +4.2 pp op-accuracy). This project tests whether that carries over to 8B-scale natural language.

## Hypothesis & why it's non-obvious

Prior work ran on a 30M toy model with a perfectly-defined "decision point" (one of two letter tokens). For natural language math:
- The "decision point" analog is unclear (token entropy? step boundary? PRM-scored critical step?). See `research/04_branching_decision_points.md`.
- The hidden-dim jumps 512 → 4096, so naive porting blows up parameter counts. Need a bottleneck design. See `research/01_external_memory_methods.md`.
- The base model is pretrained on the eval data (contamination). Baseline saturation limits measurable headroom. See `research/12_2026_reasoning_and_evaluation.md`.

These three uncertainties are the project's core unknowns. Everything else is engineering.

## Hard constraints

| Constraint | Value | Why it matters |
|------|-------|----------------|
| Hardware | Single **40 GB A100** | Rules out full FT of 8B (needs >80 GB); LoRA + bf16 + `flash_attention_2` fits comfortably |
| Base model | **Qwen3-8B** (likely Qwen3-8B-Thinking-2507) | Newest dense Qwen3 that fits; thinking mode natively produces CoT |
| Precision | bf16 | Standard for A100 + Qwen3 |
| Thinking mode | Uses sampling, NOT greedy | Qwen3 model card explicitly forbids greedy for thinking variants. Default gen: T=0.6, top_p=0.95, top_k=20. See `research/08_evaluation_methodology.md`. |
| Data | MATH (competition) + algebra__linear_1d | Small (~1.5 MB). Almost certainly in Qwen3 pretraining — expect near-ceiling baselines |

## Key engineering gotchas (easy to miss)

1. **`Qwen3DecoderLayer.forward` returns a bare `torch.Tensor`, not a tuple**. Hooks that return `(tensor,)` break. Hook must handle both forms defensively. See `research/07_qwen3_architecture_and_hooks.md`.
2. **Residual-memory design rule**: ADD memory output to residual, never replace an MLP block. Initialize so initial output ≈ 0 to avoid shocking the frozen model. Rationale: Kim 2020 PKM+ResM paper; catastrophic drift otherwise. See `research/13_local_memory_papers_review.md` entry for `2010.03881`.
3. **Answer grading**: use `math-verify` as primary; the late-2025/early-2026 upgrade fixed ≈ +4.66 pt of spurious false-negatives. Fallback: `lm-eval-harness`'s `minerva_math` parse. See `research/03_cot_parsing_and_answer_extraction.md`.
4. **Thinking-mode delimiter**: `<think>` ends at token id **151668** for Qwen3; Qwen3-Thinking-2507 pre-emits the opening `<think>` so only split on closer. Strip thinking from answer-extraction surface.
5. **Flash-attn + bf16 generation is non-deterministic** even with temp=0. Report Wilson 95% CI on every number; use paired McNemar for A/B comparisons.

## Design principles (distilled from research corpus)

- **Frozen backbone + small trainable side-module** with residual write-back — LongMem / PKM-ResM pattern. Do NOT modify backbone weights.
- **Parameter-efficient memory at 4096 hidden dim**: use a bottleneck (down-project → attention → up-project) to keep memory module in the 1–20M param range. See `research/01`.
- **Hook layer L**: start sweep at L ∈ {14, 18, 22} (mid-depth of 36 layers). Multiple research docs converge on this range.
- **Sparse slot activation is normal and desired** (Meta sparse memory FT 2025: ~0.01% of slots fire per token). Track per-slot utilization; a "dense" firing pattern indicates a bug.
- **Cross-experiment JSON result schema** mirrors `../naive_ft/ft.py:266-282` so comparison scripts work across projects.
- **Baseline caching**: compute frozen-model eval once per `n_eval`, cache to `_baseline_cache_n{N}.json`, reuse across ablations. Pattern from `../naive_ft/ft.py:221-244`.

## Baselines to beat (at comparable param count)

| Baseline | Origin | Why it's the bar |
|----------|--------|------------------|
| Bias-only adaptation | Goncharov 2025 (`research/02`) | Strongest static-vector steering baseline at tiny params |
| TinyLoRA-RL 13-param | 2026 (`research/09`) | Current minimum-param bar: 13 params → 91% GSM8K |
| LoRA all-linear r=32 DoRA | 2026 consensus (`research/06`, `research/11`) | Standard PEFT recipe; matches full FT per Thinking Machines 2025 |
| Engram (DeepSeek, Jan 2026) | `research/13` | Conditional-memory module hits MATH +2.4 / GSM8K +2.2 at 27B. If our 8B variant much smaller gain, method doesn't scale down |
| CAMELoT (training-free) | `research/13` | Zero-gradient associative memory. If our trained version doesn't beat this, learning isn't earning its keep |

## Phase roadmap

- **Phase 0** (current): model + data loaders, baseline eval, repo hygiene. See `PLAN.md`.
- **Phase 1**: memory-module skeleton hooked into Qwen3 residual stream. Design doc not yet written; must resolve DP-labeling, hook layer, param budget, training loss — see open questions in `research/README.md`.
- **Phase 2**: LoRA + full-FT baselines matching `../naive_ft/` methodology on Qwen3-8B.
- **Phase 3**: Ablations — hook layer sweep, DP-label variants, sparse updates, TTT-style per-problem writes (FwPKM idea), training-free control (CAMELoT-style).

## Where to read next

- `PLAN.md` — Phase 0 concrete setup plan.
- `research/README.md` — index of 13 literature reviews (~24K words). Start there for any conceptual question.
- `research/13_local_memory_papers_review.md` — triage of the 10 PDFs in `../memory_papers/` with H/M/L relevance scoring and suggested reading order.
- `../memory_experiment/exp.py` (lines 53–88) — the architecture being ported.
- `../memory_experiment/metrics.py` (lines 377–487) — scoring structure template.
- `../naive_ft/ft.py` (lines 50–68, 73–125, 197–282) — LoRA config + training loop + result JSON schema to mirror.

## Repo conventions

- **`.gitignore` at the repo root only** (`../.gitignore`). Do not create per-project gitignores — extend the root. Patterns scoped to this project use `math_qwen_memory/` prefix.
- **Data not tracked**: `math_qwen_memory/data/` is gitignored. Re-copy after clone:
  ```bash
  cp -r /mnt/raid/data/Hyner_Petr/lilave_puvodni/data/MATH math_qwen_memory/data/
  cp -r /mnt/raid/data/Hyner_Petr/lilave_puvodni/data/algebra__linear_1d math_qwen_memory/data/
  ```
- **Outputs not tracked**: `outputs/baseline/`, checkpoints, `hf_cache/`, `wandb/` all ignored via root `.gitignore`. Exception: markdown / tex / pdf writeups inside `outputs/` are tracked (root `.gitignore` convention).

## Parent-repo context that does NOT apply here

See `../CLAUDE.md` for the broader DecisionChains repo overview. Key differences:
- Sibling projects use **LitGPT** + a custom word-level tokenizer. This project uses **stock HuggingFace `transformers`** + Qwen3 tokenizer via `AutoTokenizer`.
- Sibling evaluation parses letter traces with regex. This project uses `math-verify` on LaTeX / numeric answers.
- Sibling `config/*.yaml` (Hydra) is not used here. Config lives in plain Python (`config.py`) for simplicity.
