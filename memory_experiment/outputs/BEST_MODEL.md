# Best Memory-Alignment Recipe

**What it does.** A tiny external key-value memory (328K trainable params) is
hooked into layer 2 of a frozen 12L-8H-512D GPT-NeoX decision-chain model. The
base model stays entirely frozen. At inference time, the memory adds a learned
correction to the residual stream at every token position. Trained for
~3 minutes on 3,000 traces, it reliably steers the model away from its
pretrained 50/50 f/g coin flip toward always picking f.

---

## The config (and what each flag does)

```
python exp.py --name best                  \
    --data_filter all                      \
    --ce_mode full_seq                     \
    --dp_target f                          \
    --gate                                 \
    --layers 2                             \
    --mem_entries 64                       \
    --lr 3e-3                              \
    --batch_size 4                         \
    --n_train 3000                         \
    --epochs 10
```

| Flag | Value | Role |
|---|---|---|
| `--data_filter` | `all` | Train on the full data (not only ffff traces). Lets the memory see hidden states produced by *all* coin-flip histories — the memory needs to recognize DPs even when the model is mid-generation on a non-f path. Using only ffff ("filtered") costs ~14 pts on full f_selection. |
| `--ce_mode` | `full_seq` | Compute cross-entropy on *every token* after `[TRACE]`, not just DP letter positions. That gives ~20× more gradient signal per example and, just as importantly, anchors the arithmetic tokens so the memory can't accidentally break them. `dp_only` at this scale breaks op_accuracy (0.95 → 0.49). |
| `--dp_target` | `f` | At decision points, override the ground-truth label in the CE loss to `f(current_intermediate_vec)` regardless of what coin flip the training example actually used. This is *the* alignment signal — without it, CE just reinforces the pretrained 50/50 behavior. |
| `--gate` | *(on)* | A learnable scalar γ multiplies the memory output: `h' = h + γ·(attn @ V)`. Lets the correction scale up/down during training. Small win (+6 pts f_sel) vs fixed γ=1. |
| `--layers` | `2` | Which residual stream to inject into. **This was the biggest single win.** Injecting at layer 2 instead of layer 10 more than doubles f_selection gain (+0.40 vs +0.17). Intuition: an early correction has 10 more layers of nonlinearity to propagate into the final logits, so it has much more leverage per unit of correction vector. |
| `--mem_entries` | `64` | Number of (key, value) rows. Narrowly beat mem=32. At the minimum-viable data point (n=1000) mem=8 was enough; with n=3000 the extra capacity helps. |
| `--lr` | `3e-3` | 3× the default 1e-3. Found via the `E_lr` sweep. The memory is a small add-on and can absorb larger updates; the default was inherited from a large-model setting. |
| `--batch_size` | `4` | Smaller batch ⇒ more gradient updates per epoch at fixed data. At 3000 examples × 10 epochs × b=4, we do 7,500 updates instead of 3,750 with b=8. |
| `--sparsity` | `none` | No penalty on memory-output magnitude at non-DP positions. Sparsity (L2 on non-DP `‖mem_out‖`) was tested at coefficients {0.001, 0.01, 0.1, 0.3, 1.0}: small values didn't change anything, large values zero'd the memory out. With layer-2 injection the memory already doesn't damage arithmetic (op_acc −1 pt), so no need. |
| `--n_train` | `3000` | Data-efficiency sweep showed: <1000 = broken, 1000 = minimum viable, 3000 = the 90%-of-E4 point, 4096 = diminishing returns. 3000 is the sweet spot. |
| `--epochs` | `10` | We tested 10 / 20 / 30 epochs — all gave identical results (+0.445, +0.444, +0.446). The memory converges well before epoch 10; more epochs don't help. |

**Cost:** ~170 seconds of training + two eval passes (~12 min total per run on an A100). **Trainable params:** 328K. **Frozen params:** ~30M (the base model, untouched).

---

## Results — best model vs second best vs original reference

All numbers averaged over 3 seeds. Percentages in parentheses are changes from baseline. Primary alignment metric is **full-val f_selection**; full-val `full_f_alignment` is the strict all-or-nothing per-example version.

### Full val (200–300 examples, all coin-flip patterns — the honest alignment test)

| Metric | BASELINE | **Winner (P3_compound)** | 2nd (P2_B_layer2) | Ref (E4) |
|---|---:|---:|---:|---:|
| f_selection | 0.525 | **0.973** (+0.445) | 0.960 (+0.435) | 0.910 (+0.385) |
| full_f_alignment | 0.077 | **0.871** (+0.794) | 0.807 (+0.727) | 0.560 (+0.480) |
| operation_accuracy | 0.945 | **0.933** (−0.012) | 0.930 (−0.022) | 0.897 (−0.056) |

### ffff val (f-only subset — where `complete_solution` is meaningful)

| Metric | BASELINE | **Winner (P3_compound)** | 2nd (P2_B_layer2) | Ref (E4) |
|---|---:|---:|---:|---:|
| f_selection | 0.734 | **0.997** (+0.264) | 0.988 (+0.259) | 0.986 (+0.257) |
| full_f_alignment | 0.593 | **0.990** (+0.397) | 0.960 (+0.370) | 0.950 (+0.360) |
| operation_accuracy | 0.946 | **0.998** (+0.053) | 0.996 (+0.048) | 0.996 (+0.048) |
| complete_solution | 0.723 | **0.956** (+0.233) | 0.937 (+0.217) | 0.925 (+0.205) |

### Compute comparison

| Config | n_train | epochs | grad steps | trainable params | train time |
|---|---:|---:|---:|---:|---:|
| **Winner** | 3,000 | 10 | 7,500 | 328K | ~170 s |
| 2nd | 3,000 | 10 | 3,750 | 295K | ~100 s |
| E4 (ref) | 4,096 | 20 | 10,240 | 295K | ~210 s |

---

## Why the winner beats the second-best

Both use **layer 2** — the biggest finding. The winner's three extra ingredients on top of that:

1. **lr=3e-3 vs 1e-3**: faster convergence, slightly higher final plateau (+0.013 full f_sel).
2. **mem_entries=64 vs 32**: a bit more capacity to specialize. Tiny win (+0.003 f_sel) but bigger on full_f_alignment (+0.064).
3. **batch_size=4 vs 8**: twice as many gradient steps per epoch. Combined with the higher LR, helps the memory exploit the data better.

The second-best is 89% of the winner at 67% of the compute — a reasonable fallback if you care more about simplicity than the last 6 pts of full_f_alignment.

## Why the winner beats E4

E4 was built before we knew:
- **Layer 2 >> Layer 10** (+0.22 full f_sel from this alone).
- More data does **not** help past n=3000 (E4 used n=4096).
- Default LR was too low (+0.13 full f_sel at lr=3e-3).

Stacking these insights gives the winner full_f_alignment at 87% vs E4's 56% on the same test set. And because the correction sits at layer 2 (where arithmetic representations haven't fully formed yet), it doesn't disrupt arithmetic computation — op_accuracy regression is 1.2 pts vs E4's 5.6 pts.

---

## What to remember for future models

1. **Pick the injection layer deliberately.** Early layers give the correction more layers of leverage. Don't default to "somewhere near the end".
2. **The alignment signal is `dp_target=f` with mixed training data.** Without the override, mixed training is just pretraining again. Without the mixed data, the memory never sees the hidden states it needs to correct.
3. **Data diversity matters more than epochs.** The curve saturates at n=3000; more epochs on any smaller dataset overfits.
4. **Small external modules can absorb higher LRs.** The LR bump from 1e-3 → 3e-3 added as much alignment as going from mem=8 to mem=64.

## Files

- `outputs/final_results.md` — per-group Phase 1 winners + Phase 2/3 seed-averaged results
- `outputs/experiments/P3_compound_n3000_e10_s{0,1,2}.json` — the raw winning run JSONs
- `outputs/E1-E4_results.{md,txt}` — pilot 2×2 comparison
- `outputs/data_efficiency_results.txt` — n_train × epochs sweep
- `logs/overnight_sweep.log` — full stdout from the 73-run sweep
- W&B project `memory-experiment` (group `p3_final`) — browse the winner's run history
