# Final Sweep Results

## Phase 1 — Top configuration per group

Ranked by `full_val f_selection` delta. `op_acc` is the cost on full val.

| Group | Winner | Δ full f_sel | Δ full full_align | Δ full op_acc |
|---|---|---|---|---|
| Ingredient (A1-A5) | `A1_ref_s1` | +0.178 | +0.120 | +0.005 |
| Layer (B_) | `B_layer2` | +0.397 | +0.600 | -0.025 |
| Memory size (C_) | `C_mem64_n3000` | +0.214 | +0.180 | -0.057 |
| Sparsity (D_) | `D_n3000_e10` | +0.198 | +0.107 | -0.039 |
| Learning rate (E_) | `E_lr3e-3` | +0.304 | +0.280 | -0.007 |
| Batch size (F_) | `F_bs4` | +0.229 | +0.140 | -0.033 |
| Epochs @ n1000 (G_) | `G_n1000_e50` | +0.255 | +0.180 | -0.064 |
| Epochs @ n3000 (H_) | `H_n3000_e30` | +0.360 | +0.420 | -0.005 |
| Compound (I_) | `I_mem4_sp0p01_l10` | +0.054 | +0.040 | -0.006 |

## Phase 2 — Top-5 configs, 3-seed confirmation

At `n_eval=200`, each config × seeds {0,1,2}.

| Config | Δ full f_sel | Δ full full_align | Δ full op_acc |
|---|---|---|---|
| `P2_B_layer2` (3 seeds) | +0.435 ±0.007 | +0.727 ±0.018 | -0.022 ±0.004 |
| `P2_B_layer4` (3 seeds) | +0.413 ±0.004 | +0.615 ±0.017 | -0.018 ±0.008 |
| `P2_B_layers_6_10` (3 seeds) | +0.408 ±0.008 | +0.598 ±0.038 | -0.038 ±0.018 |
| `P2_B_layer6` (3 seeds) | +0.400 ±0.015 | +0.578 ±0.058 | -0.039 ±0.013 |
| `P2_H_n3000_e30` (3 seeds) | +0.380 ±0.016 | +0.508 ±0.020 | -0.052 ±0.011 |

## Phase 3 — Compound best ingredients

At `n_eval=300`. `compound` config is (best layer, sparsity, mem, lr, bs) from Phase 1 ingredient ablations, varying epoch budget.

| Config | Δ full f_sel | Δ full full_align | Δ full op_acc |
|---|---|---|---|
| `P3_compound_n3000_e30` (3 seeds) | +0.446 ±0.008 | +0.794 ±0.035 | -0.006 ±0.010 |
| `P3_compound_n3000_e10` (3 seeds) | +0.445 ±0.003 | +0.794 ±0.008 | -0.012 ±0.007 |
| `P3_compound_n3000_e20` (3 seeds) | +0.444 ±0.008 | +0.790 ±0.028 | -0.008 ±0.010 |
| `P3_bonus_I_mem4_sp0p01_l10` (3 seeds) | +0.041 ±0.071 | +0.019 ±0.033 | -0.009 ±0.016 |

## Anchors

| Reference | Δ full f_sel | Δ full full_align | Δ full op_acc |
|---|---|---|---|
| `E4_all_fullseq_f` | +0.385 | +0.480 | -0.056 |
| `D_n1000_e10` | +0.114 | +0.047 | -0.066 |
| `D_n3000_e10` | +0.198 | +0.107 | -0.039 |