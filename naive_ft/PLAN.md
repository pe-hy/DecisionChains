# Naive Finetuning Baseline — Plan (minimal start)

## Why

We have memory-based alignment in `../memory_experiment/`: ~330K trainable
params, ~3 min training, full-val `f_selection` 0.53 → 0.97. Before
claiming memory "works well", we need a FT baseline.

## Minimal first pass — 4 runs

One run per variant at the memory winner's compute budget (`n_train=3000,
epochs=10`). Enough to see if the approach works at all and where it
lands on the metric table.

| Variant | Method | Mode | data | dp_target | lr | extra |
|---|---|---|---|---|---|---|
| **FT-A** | full FT | A | ffff | gt | 5e-5 | "most naive" |
| **FT-B** | full FT | B | all | f (override) | 5e-5 | matches memory's supervision |
| **LoRA-A** | LoRA (r=8) | A | ffff | gt | 5e-4 | parameter-efficient |
| **LoRA-B** | LoRA (r=8) | B | all | f (override) | 5e-4 | parameter-efficient + override |

All share: `ce_mode=full_seq, batch_size=4, epochs=10, n_train=3000,
n_eval=200, seed=0, cosine schedule with 10% warmup`.

Memory baseline for comparison: `P3_compound_n3000_e10` (layer 2, mem=64,
lr=3e-3, bs=4), already in `../memory_experiment/outputs/experiments/`.

**Budget**: 4 runs × ~10 min each = ~40 min.

## Expand only if minimum pass leaves open questions

Possible follow-ups (only if warranted):
- Tighter LR sweep per variant (e.g. 1e-5, 5e-5, 1e-4 for full FT).
- Less data (`n_train=1000`) to see sample-efficiency.
- Higher LoRA rank (r=32) if r=8 underperforms full FT.
- Multi-seed confirmation for the winner.

## Files (minimal)

```
naive_ft/
├── README.md
├── PLAN.md
├── ft.py                # unified trainer: --method {full,lora} × --mode {A,B}
├── run_ft_quick.sh      # 4-run minimum pass
├── compare_vs_memory.py # headline table after runs complete
├── metrics.py           # symlink to ../memory_experiment/metrics.py
└── outputs/
    └── experiments/
```

No screening / phase1 / phase2 infra until we see the initial numbers.

## `ft.py` — compact design

Flags:

```
--name             run identifier
--method           full | lora             mechanism
--mode             A | B                   ffff+GT vs all+override
--n_train          int  (default 3000)
--epochs           int  (default 10)
--lr               float
--batch_size       int  (default 4)
--n_eval           int  (default 200)
--seed             int  (default 0)
--lora_rank        int  (LoRA only; default 8)
--lora_alpha       int  (LoRA only; default 16)
--save_model       flag (default off)
--wandb            flag
--wandb_project    str  (default "memory-experiment")
--wandb_group      str
```

Core logic:

1. Load base HF model + tokenizer from
   `../memory_experiment/checkpoint/12l-8h-512d-decision-chains-ext_6_2M/hf/`.
2. If `--method lora`: wrap via `peft.LoraConfig` targeting
   GPT-NeoX's `query_key_value` and `dense`. Everything else frozen.
3. Data prep: import `prepare_training_data` from
   `../memory_experiment/exp.py` with `ce_mode=full_seq` and
   `dp_target="f"` (B) or `dp_target="gt"` (A). Filter to ffff for mode A.
4. Training: AdamW + linear warmup + cosine decay. Log per-epoch CE.
5. Eval: reuse `evaluate()` from `../memory_experiment/exp.py` on both
   `val_fonly` and `val_full`.
6. Output JSON schema identical to memory_experiment's (so
   `compare.py` and `compare_vs_memory.py` can read both):
   `{name, config, trainable_params, train_time_sec, history,
     baseline_fonly, baseline_full, memory_fonly, memory_full}`
   (the last two keys keep the "memory" name to match schema; the
   README documents that for FT they mean "finetuned-model" eval).
7. W&B: project `memory-experiment`, group per variant.

Baseline cache shared with memory_experiment: if
`../memory_experiment/outputs/experiments/_baseline_cache_n200.json`
exists, reuse it (same model, same val data, same n_eval).

## `compare_vs_memory.py`

Loads:
- Best (or only) run per variant from `outputs/experiments/*.json`
- Memory winner from `../memory_experiment/outputs/experiments/P3_compound_n3000_e10_s{0,1,2}.json`

Prints a single table:

| Method | Trainable params | Train time | full f_sel Δ | full full_align Δ | full op_acc Δ |
|---|---|---|---|---|---|
| Memory (winner) | 330K | 170s | +0.445 | +0.794 | −0.012 |
| FT-full-A | 30M | ?s | ? | ? | ? |
| FT-full-B | 30M | ?s | ? | ? | ? |
| LoRA-r8-A | ~150K | ?s | ? | ? | ? |
| LoRA-r8-B | ~150K | ?s | ? | ? | ? |

That's the headline: **does naive FT or LoRA actually beat the memory
approach, and if yes, at what compute/param cost?**

## Risks

1. **Mode A might not align**: just seeing ffff data could fail to shift
   the model's 50/50 habit meaningfully. If so, it's a real finding —
   it shows the memory's win comes from the `dp_target=f` *supervision*,
   not the *mechanism*.
2. **Catastrophic forgetting**: too aggressive FT on restricted data
   could destroy arithmetic. Monitor `operation_accuracy` closely.
3. **LoRA target modules**: GPT-NeoX uses fused `query_key_value`. Need
   to verify PEFT targets it correctly; will check on first LoRA run.
4. **flash_attention_2 + PEFT compatibility**: should work but a known
   failure mode is deprecated API. Will catch on first run.
5. **Disk for checkpoints**: default `--save_model=False`. Only save
   winners (4 × 76MB = ~300MB if we save all four).

## Ready to implement

Files to create in order:
1. `metrics.py` → symlink
2. `ft.py` → full FT + mode A (simplest), then add mode B, then LoRA
3. Smoke test: 1 run at `--method full --mode A --n_train 100 --epochs 1`
4. If smoke test passes: `run_ft_quick.sh` with the 4 minimum runs
5. `compare_vs_memory.py` after runs complete
