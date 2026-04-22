# Plan: Qwen Math Memory/FT Experiments — Phase 0 (Setup + Baseline)

## Context

We want to port the methodology from `memory_experiment/` (external key-value memory hook on frozen transformer) and `naive_ft/` (full-FT + LoRA baselines) from the synthetic decision-chains task to **natural-language math reasoning** with a Qwen model. Working directory is `math_qwen_memory/` (sibling to those folders).

Key differences vs. prior work:
- Base model: Qwen3 (natural language, tokenizer, thinking mode) instead of custom 30M GPT-NeoX.
- Task: math word problems / symbolic algebra with free-form reasoning traces instead of deterministic letter traces.
- Hardware: single **40 GB A100**, which constrains model size and training config.
- "Decision point" analog in reasoning is not obvious — defer until we see model outputs.

**This plan covers Phase 0 only**: model download + load, dataset loaders, zero-shot baseline eval, `.gitignore`. No training, no memory module yet. Intent: establish a working substrate and characterize baseline behavior before committing to any particular steering mechanism.

## Decisions

| Topic | Choice | Rationale |
|------|--------|-----------|
| Base model | **Qwen3-8B (Thinking-2507 variant if available, else base Qwen3-8B)** | Newest Qwen3 dense <14B. ~16 GB bf16 weights leaves ~24 GB headroom on 40 GB A100 for activations, LoRA adapters, optimizer state. Built-in thinking mode matches CoT study. |
| Fallback | Qwen3-14B (inference only) | ~30 GB bf16; fits inference with room to spare but LoRA training tight (needs grad checkpoint + small batch). Keep as optional eval comparator, not primary. |
| Precision | bf16 | Standard for A100 + Qwen3. Loading in 4-bit NOT used in Phase 0 (we want clean baseline). |
| Datasets | **MATH** (competition, with LaTeX CoT solutions) + **algebra__linear_1d** (symbolic, no GT CoT — we prompt model to reason) | User-selected. MATH provides GT reasoning traces for supervised FT later. algebra tests model's self-generated CoT against a numeric ground truth. |
| Dataset paths | `data/MATH/{train_1k.jsonl,test_500.jsonl}` and `data/algebra__linear_1d/{train_1k.jsonl,test.jsonl}` (copied locally from `/mnt/raid/data/Hyner_Petr/lilave_puvodni/data/`) | Small (~1.5 MB total), safe to track in git. |
| DP analog | **Deferred** | Will inspect zero-shot outputs to decide: final-answer tokens (`\boxed{}` for MATH, last number for algebra), step-boundary tokens, or full trace. |
| Repo hygiene | New `.gitignore` covering `outputs/`, `checkpoints/`, `hf_cache/`, `wandb/`, `*.pt`, `__pycache__/`, `.venv/`, large JSON results | Prior folders leaked logs/ and outputs/ into git (see `git status`); avoid repeat. |

## Layout to create in `math_qwen_memory/`

```
math_qwen_memory/
├── .gitignore
├── CLAUDE.md                     # guidance for future Claude sessions
├── PLAN.md                       # this file
├── config.py                     # model id, dataset paths, eval knobs
├── data/                         # local copies of datasets (tracked in git)
│   ├── MATH/{train_1k.jsonl,test_500.jsonl}
│   └── algebra__linear_1d/{train_1k.jsonl,test.jsonl}
├── data_loaders.py               # JSONL loaders, prompt formatting, tokenization for each dataset
├── load_model.py                 # HF AutoModelForCausalLM + tokenizer loader, mem probe
├── eval_baseline.py              # zero-shot generation + answer extraction + score
├── metrics.py                    # extract_final_answer(dataset, text) + accuracy
├── prompts.py                    # system / user prompt templates per dataset (MATH vs algebra)
├── outputs/                      # gitignored — per-run JSON results, sample generations
│   └── baseline/
├── hf_cache/                     # gitignored — HF download cache
└── README.md                     # how to run Phase 0 (brief)
```

## File-by-file detail

### `.gitignore`
```
outputs/
hf_cache/
wandb/
checkpoints/
*.pt
*.bin
*.safetensors
__pycache__/
.venv/
.ipynb_checkpoints/
*.log
.DS_Store
```

### `config.py`
- `MODEL_ID = "Qwen/Qwen3-8B-Thinking-2507"` (confirm the id exists; else `"Qwen/Qwen3-8B"`).
- `HF_CACHE_DIR = "./hf_cache"` — keep downloads inside project dir (disk has space on `/mnt/raid`).
- `DATA_ROOT = "./data"` (local copy under repo).
- Per-dataset section with train/test paths.
- Generation defaults: `max_new_tokens=1024` (MATH needs length for CoT), `temperature=0.0` for baseline determinism, `do_sample=False`.
- `DEVICE = "cuda"`, `DTYPE = torch.bfloat16`.

### `data.py`
- `load_math(split)` → list of `{"problem", "solution", "answer", "level", "subject"}`.
- `load_algebra(split)` → list of `{"question", "answer"}`.
- `format_prompt(dataset_name, example)` — delegates to `prompts.py`.
- No tokenization yet (baseline eval uses HF `tokenizer.apply_chat_template`).

### `prompts.py`
- MATH system prompt: ask for reasoning in LaTeX then `\boxed{ANSWER}`.
- algebra system prompt: "Solve for the variable. Show reasoning step by step, then state the final numeric answer on a line starting with `Answer:`."
- Use Qwen3 chat template (`tokenizer.apply_chat_template(..., enable_thinking=True)` where applicable — thinking mode gives CoT natively).

### `load_model.py`
- `load_qwen()` returns `(model, tokenizer)` with `torch_dtype=bfloat16`, `device_map="cuda:0"`, `attn_implementation="flash_attention_2"` (fallback to `sdpa` if flash-attn missing), `cache_dir=HF_CACHE_DIR`.
- After load, print `torch.cuda.memory_allocated()` and trainable/total param counts.
- Smoke test: generate 50 tokens on a fixed prompt, print.

### `metrics.py`
- `extract_math_answer(text)` — regex for `\boxed{(.+?)}` (handle nested braces with a small stack), fallback to last number.
- `extract_algebra_answer(text)` — regex for `Answer:\s*(-?\d+(?:\.\d+)?)`, fallback to last integer in output.
- `normalize_answer(s)` — strip whitespace, lower, remove `$`, handle fractions `\frac{a}{b}` → `a/b` for MATH comparison.
- `score(pred, gold, dataset)` → bool + normalized forms (for logging).

### `eval_baseline.py`
- CLI: `--dataset {math,algebra}`, `--split {train,test}`, `--n 200`, `--out outputs/baseline/<name>.json`.
- Load model once, iterate examples, generate, extract answer, score.
- Progress bar (tqdm).
- Save JSON: list of `{idx, prompt, generation, pred, gold, correct}` + summary `{accuracy, n, model_id, gen_config}`.
- Also save 5 full verbatim samples to `outputs/baseline/<name>_samples.txt` for manual inspection — this is the input for the later DP-analog decision.

### `README.md` (short)
- How to install deps (`transformers`, `accelerate`, `flash-attn` optional, `peft` for later phase).
- `python eval_baseline.py --dataset math --split test --n 200`
- `python eval_baseline.py --dataset algebra --split test --n 200`

## Critical files from sibling repos to reference (read-only)

- `../memory_experiment/exp.py` (lines 53–88) — memory architecture we'll port in Phase 1+.
- `../memory_experiment/metrics.py` (lines 377–487) — scoring structure template.
- `../naive_ft/ft.py` (lines 50–68, 73–125) — LoRA config + training loop template.
- `../naive_ft/ft.py` (lines 197–282) — data pipeline + JSON result schema to mirror for cross-experiment comparison.

## VRAM sanity check (Phase 0)

Qwen3-8B bf16:
- Weights: ~16.4 GB
- KV cache at 2k ctx, bs=1: ~1 GB
- Activations for gen: <2 GB
- **Total inference: ~20 GB on 40 GB A100 — comfortable.**

Qwen3-14B as fallback:
- Weights: ~29.6 GB
- Inference only: ~32–34 GB. OK.
- LoRA training: would need grad checkpoint + bs=1 + maybe 8-bit optimizer. Phase 1 concern, not Phase 0.

## Open questions deferred to later phases (not part of Phase 0)

- **DP analog**: pick after looking at baseline generations.
- **Memory-module hook layer**: need to map prior L=1–2 (12-layer net) to Qwen3-8B's 36 layers — likely try L∈{2,4,8} as starting sweep.
- **Memory hidden_dim**: Qwen3-8B hidden = 4096 → `W_q` alone = 16M params vs prior 262K. Scale of "external memory" changes meaningfully; may need to use a bottleneck (down-project to e.g. 256 before attention, up-project after) to keep it parameter-efficient.
- **Supervision mode for algebra dataset**: it has no GT CoT. Options: (a) SFT on final-answer only with model's own generated reasoning frozen; (b) distill CoT from a stronger model; (c) RL/GRPO on final-answer correctness. Decide in Phase 2.
- **Evaluation of reasoning-trace quality**: beyond final-answer accuracy, do we want to track trace length, step count, self-consistency@k? Defer.

## Verification for Phase 0

1. `python load_model.py` — model loads, prints param count and VRAM, generates coherent text for a test prompt.
2. `python eval_baseline.py --dataset math --split test --n 50` completes without OOM, produces JSON with non-zero accuracy (Qwen3-8B-Thinking should score >50% on MATH test).
3. `python eval_baseline.py --dataset algebra --split test --n 50` completes, accuracy should be high (>80%) since linear_1d is easy.
4. Manual inspection of `outputs/baseline/*_samples.txt` — reasoning traces present, `\boxed{}` / `Answer:` extractable.
5. `git status` — only tracked source files show as new; `hf_cache/`, `outputs/`, `wandb/` correctly ignored.
