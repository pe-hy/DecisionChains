# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Phase 0 project. No source code yet — see `PLAN.md` for the full setup plan (model choice, datasets, file layout, verification steps). Read `PLAN.md` before starting any work here.

## Purpose

Port the external-memory steering + LoRA/full-FT baselines from sibling folders to a Qwen3 base model on natural-language math reasoning. Prior work used a custom 30M GPT-NeoX on synthetic decision chains; this project swaps in Qwen3-8B and real math datasets.

- Sibling `../memory_experiment/` — the external-memory method being ported. Key files: `exp.py` (MemoryAttention module + training loop), `metrics.py` (trace scoring + aggregation), `README.md`, `DIAGRAM.txt`.
- Sibling `../naive_ft/` — full-FT and LoRA baselines. Key file: `ft.py` (LoRA wrapping, training loop, evaluation schema).
- Both siblings write results to per-run JSON with a shared schema so comparison tables work across experiments — preserve that schema here too.

## Hardware constraint

Single **40 GB A100**. Planned base model: **Qwen3-8B** (bf16, ~16 GB weights). Fallback: Qwen3-14B for inference-only comparison. Any new training config must fit in 40 GB; LoRA + `flash_attention_2` + bf16 is the default path. Full FT of 8B does NOT fit without 8-bit optimizer + gradient checkpointing.

## Datasets

- **MATH (competition)**: `data/MATH/{train_1k.jsonl,test_500.jsonl}` — has LaTeX CoT `solution` field. Final answer extracted from `\boxed{}`.
- **algebra__linear_1d**: `data/algebra__linear_1d/{train_1k.jsonl,test.jsonl}` — symbolic equations, no GT CoT. Model is prompted to produce reasoning; eval on final numeric answer only.

Data is **not tracked in git** (ignored via root `.gitignore: math_qwen_memory/data/`). After cloning, re-copy from source:
```bash
cp -r /mnt/raid/data/Hyner_Petr/lilave_puvodni/data/MATH math_qwen_memory/data/
cp -r /mnt/raid/data/Hyner_Petr/lilave_puvodni/data/algebra__linear_1d math_qwen_memory/data/
```
Do not modify the jsonl files.

**Contamination note**: Both datasets are almost certainly in Qwen3's pretraining (MATH is on GitHub/HF, algebra__linear_1d from DeepMind mathematics dataset is on HF). Expect near-ceiling baselines. Plan to add contamination-resistant evals (AIME/HMMT 2026, Putnam-AXIOM) in later phase — see `research/12_2026_reasoning_and_evaluation.md`.

## Conventions

- Keep `outputs/`, `hf_cache/`, `wandb/`, checkpoints, and weight files out of git. See `.gitignore` once created. The sibling repos leaked logs and outputs into git history — do not repeat.
- Mirror the JSON result schema from `../naive_ft/ft.py` (lines 266–282) when writing run results, so comparison scripts can consume runs from this folder alongside sibling results.
- Baseline caching pattern (sibling `../naive_ft/ft.py` lines 221–244): compute frozen-model eval once per n_eval, cache to `_baseline_cache_n{N}.json`, reuse across ablations.

## Parent repo context

See `../CLAUDE.md` for the broader DecisionChains repo overview (custom tokenizer, LitGPT usage, GRPO pipeline). That context does NOT apply directly here — this project uses stock HuggingFace Qwen3 + its tokenizer, not LitGPT or the custom word-level tokenizer.
