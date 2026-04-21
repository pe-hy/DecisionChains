# Pilot Experiments: E1–E4

Base model: frozen **12L-8H-512D GPT-NeoX** pretrained on decision chains.
Memory: external K/V attention, hooked on layer 10 residual stream. **Only the memory is trained** (~270K params).

## Configurations

| Run | Train data | CE supervision | DP target | Mem N | Gate | Sparsity | Layer | n_train | n_eval | Epochs | LR | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1_ffff_dponly** | ffff | dp_only | f | 16 | ✗ | l2_nondp (λ=0.1) | 10 | 4096 | 200 | 20 | 0.001 | 278,528 |
| **E2_ffff_fullseq** | ffff | full_seq | gt | 32 | ✓ | none | 10 | 4096 | 200 | 20 | 0.001 | 294,913 |
| **E3_all_dponly_f** | all | dp_only | f | 16 | ✗ | l2_nondp (λ=0.1) | 10 | 4096 | 200 | 20 | 0.001 | 278,528 |
| **E4_all_fullseq_f** | all | full_seq | f | 32 | ✓ | none | 10 | 4096 | 200 | 20 | 0.001 | 294,913 |

## Results — ffff val (200 examples)

All four metrics are meaningful on this slice (GT OUTPUT = f-path output ⇒ `complete_solution` interpretable).

| Run | Op acc | F-or-G valid sel | **f_selection** | **full_f_alignment** | chain=out | **complete_solution** |
|---|---|---|---|---|---|---|
| **E1_ffff_dponly** | 0.948 → **0.958** (+0.011) | 0.969 → **0.976** (+0.007) | 0.729 → **0.791** (+0.062) | 0.590 → **0.695** (+0.105) | 0.720 → **0.775** (+0.055) | 0.720 → **0.775** (+0.055) |
| **E2_ffff_fullseq** | 0.948 → **0.987** (+0.039) | 0.969 → **0.986** (+0.017) | 0.729 → **0.903** (+0.173) | 0.590 → **0.855** (+0.265) | 0.720 → **0.865** (+0.145) | 0.720 → **0.865** (+0.145) |
| **E3_all_dponly_f** | 0.948 → **0.970** (+0.022) | 0.969 → **0.967** (-0.001) | 0.729 → **0.862** (+0.133) | 0.590 → **0.740** (+0.150) | 0.720 → **0.790** (+0.070) | 0.720 → **0.790** (+0.070) |
| **E4_all_fullseq_f** | 0.948 → **0.996** (+0.048) | 0.969 → **0.994** (+0.025) | 0.729 → **0.986** (+0.257) | 0.590 → **0.950** (+0.360) | 0.720 → **0.925** (+0.205) | 0.720 → **0.925** (+0.205) |

## Results — full val (200 examples, all coin-flip patterns)

Honest test of inference-time alignment. `complete_solution` near zero by design — GT OUTPUT ≠ f-path output on non-ffff examples.

| Run | Op acc | F-or-G valid sel | **f_selection** | **full_f_alignment** | chain=out | complete_solution |
|---|---|---|---|---|---|---|
| **E1_ffff_dponly** | 0.953 → **0.955** (+0.003) | 0.971 → **0.971** (+0.000) | 0.525 → **0.556** (+0.032) | 0.080 → **0.090** (+0.010) | 0.650 → **0.640** (-0.010) | 0.650 → **0.640** (-0.010) |
| **E2_ffff_fullseq** | 0.953 → **0.952** (-0.001) | 0.971 → **0.962** (-0.008) | 0.525 → **0.613** (+0.088) | 0.080 → **0.135** (+0.055) | 0.650 → **0.560** (-0.090) | 0.650 → **0.560** (-0.090) |
| **E3_all_dponly_f** | 0.953 → **0.879** (-0.073) | 0.971 → **0.889** (-0.082) | 0.525 → **0.658** (+0.133) | 0.080 → **0.150** (+0.070) | 0.650 → **0.305** (-0.345) | 0.650 → **0.295** (-0.355) |
| **E4_all_fullseq_f** | 0.953 → **0.897** (-0.056) | 0.971 → **0.938** (-0.033) | 0.525 → **0.910** (+0.385) | 0.080 → **0.560** (+0.480) | 0.650 → **0.115** (-0.535) | 0.650 → **0.115** (-0.535) |

## Overall ranking  (by f_selection on full val)

| Rank | Run | full_val f_selection | full_val full_f_alignment | full_val op_accuracy |
|---|---|---|---|---|
| 1 | **E4_all_fullseq_f** | 0.910 | 0.560 | 0.897 |
| 2 | **E3_all_dponly_f** | 0.658 | 0.150 | 0.879 |
| 3 | **E2_ffff_fullseq** | 0.613 | 0.135 | 0.952 |
| 4 | **E1_ffff_dponly** | 0.556 | 0.090 | 0.955 |

## Key takeaways

- **E4 wins decisively**: full-val f_selection 0.525 → 0.910 (+0.385); full_f_alignment 0.080 → 0.560 (+0.480).
- **Two independent gains compound**: `all` training data (E3 vs E1) and `full_seq` supervision (E2 vs E1) each help, and together (E4) yield more than additive improvement.
- **Cost**: op_accuracy on full val regresses ~5–7 pts for E3/E4 — expected cascading-arithmetic cost when the memory steers to the f-path from off-path intermediate vectors.
- **`complete_solution` crash on full val is methodological, not a regression**: when alignment succeeds, the final vector no longer matches the GT mixed-coin-flip OUTPUT. Only meaningful on ffff val.