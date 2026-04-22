# naive_ft — Finetuning Baselines

Standard finetuning baselines for the memory-experiment comparison in
`../memory_experiment/`. The question this folder answers: **does naive
finetuning match the memory approach, and at what cost?**

Two mechanisms × two supervision modes = four baselines, each compared
against the memory's 295K-trainable-param module.

---

## Methods

|  | **Mode A** (ffff-only, GT labels) | **Mode B** (all data, f-override) |
|---|---|---|
| **Full FT** | All 30M base-model params trainable | Same, with f-override supervision |
| **LoRA r=8** | 295K adapter params (same count as memory) | 295K adapters with f-override |

- **Mode A** is the "most naive" baseline: show the model only successful
  $f$-path traces and let it learn by imitation. Standard causal LM loss.
- **Mode B** uses the exact supervision signal the memory uses: all
  coin-flip data, but override labels at decision points to $f(v)$'s
  letter. Isolates **mechanism** from **supervision**.

---

## Results (earlier 200-example eval)

All at $n_\text{train}=3000$, $\text{epochs}=10$, $\text{batch\_size}=4$,
single seed. See `../memory_experiment/outputs/comparison_table.pdf` for
the full comparison including memory.

| Method | Params | Time | Δ full f_sel | Δ full align | Δ op_acc |
|--------|-------:|-----:|-------------:|-------------:|---------:|
| FT full, mode A | 37.9M | 335s | +42.3 | +71.0 | −3.9 |
| FT full, mode B | 37.9M | 355s | +40.0 | +63.5 | **−11.0** |
| FT LoRA, mode A | 295K | 297s | +28.4 | +26.0 | −0.8 |
| FT LoRA, mode B | 295K | 302s | **+43.9** | +75.5 | −5.7 |
| Memory L=2 N=32 (ref) | 295K | 220s | +44.9 | +79.2 | −1.5 |

**Key observations:**
- LoRA~B is the closest baseline. Same param count as memory, similar
  alignment, but ~4× worse op_accuracy cost.
- Mode B **hurts** full FT (breaks arithmetic) but **helps** LoRA. With
  30M free params the override signal confuses the model; with 295K
  adapters it's a focused update.
- Mode A LoRA fails on full val (+28 pp only) — trained on ffff-only,
  generalizes poorly to off-path hidden states.

---

## Usage

### Single run

```bash
cd naive_ft
CUDA_VISIBLE_DEVICES=1 python ft.py --name myrun \
    --method lora --mode B --lr 5e-4 --lora_rank 8 \
    --n_train 3000 --epochs 10 --batch_size 4 \
    --ffff_val_file ../outputs/data/decision_chains_extended/val_ffff.json \
    --n_eval 2000 \
    --wandb --wandb_project memory-experiment --wandb_group my_ft_sweep
```

### Flags

| Flag | Values | Notes |
|------|--------|-------|
| `--method` | `full` \| `lora` | Required. |
| `--mode` | `A` \| `B` | A = ffff+GT, B = all+f-override. |
| `--lr` | float | Required. Use ~5e-5 for full, ~5e-4 for LoRA. |
| `--lora_rank` | int (default 8) | LoRA rank. |
| `--lora_alpha` | int (default 16) | LoRA α. |
| `--lora_target` | str | Default `query_key_value,dense`. |
| `--ffff_val_file` | path | Optional dedicated ffff val. |
| `--n_train`, `--epochs`, `--lr`, `--batch_size`, `--seed` | | Usual. |

JSON schema matches memory_experiment's so cross-folder comparison works
out of the box.

### Quick 4-run pass

```bash
bash run_ft_quick.sh
```

Runs FT_full_A, FT_full_B, FT_lora_A, FT_lora_B at matched compute.

### Compare to memory winner

```bash
python compare_vs_memory.py
```

Pulls memory winner from `../memory_experiment/outputs/experiments/` and
prints headline comparison table.

---

## Files

| File | Role |
|------|------|
| `README.md` | This file. |
| `PLAN.md` | Original experiment design notes. |
| `ft.py` | Unified trainer (full / LoRA × A / B). |
| `run_ft_quick.sh` | 4-run minimum pass. |
| `compare_vs_memory.py` | Headline comparison with memory runs. |
| `metrics.py` | Symlink → `../memory_experiment/metrics.py`. |
| `outputs/experiments/` | Per-run JSONs (same schema as memory). |
| `logs/` | Stdout + W&B cache. |

---

## Implementation notes

- `ft.py` imports `prepare_training_data`, `evaluate`,
  `load_model_and_tokenizer`, `filter_data`, `build_letter_tid_map`,
  `convert_checkpoint_if_needed` directly from
  `../memory_experiment/exp.py` — no code duplication.
- AdamW optimizer, linear warmup (10%) + cosine decay, gradient clip 1.0.
- Baseline cache is shared with memory_experiment (same base model, same
  val slices). First run computes baseline; subsequent runs reuse.
- JSON schema keeps the `memory_fonly` / `memory_full` keys for
  compatibility with memory_experiment's tools — those fields actually
  hold the finetuned-model eval.
