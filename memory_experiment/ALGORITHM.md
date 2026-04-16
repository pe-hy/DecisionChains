# Surgical Memory Injection

## Goal

Steer the model's letter choices toward f-branch at decision points, without touching arithmetic. The base model (12L-8H-512D GPT-NeoX) is completely frozen. Only a small external memory (~8K params) is trained.

## Architecture

```
frozen layers 0..L-1
    ↓
layer L output: h              (B, T, 512)
    ↓
query = W_q @ h                project to query space
attn  = softmax(q @ K^T / √d) attend over N memory entries
out   = attn @ V               weighted sum of value vectors
    ↓
h' = h + out                   inject into residual stream
    ↓
frozen layers L+1..11 → logits
```

Trainable parameters: K (N×512 keys), V (N×512 values), W_q (512×512 query projection). With N=8 entries: 8×512 + 8×512 + 512×512 = 270K params. Everything else is frozen.

## Training

**Loss = CE_decision + λ · L1_sparsity**

- **CE_decision**: Cross-entropy at decision-point positions only ([TRACE] and each `;`). Target = f(intermediate_vector)'s letter. Not computed at arithmetic/result/prompt positions.

- **L1_sparsity**: Mean |attention weight| across all positions. Pushes memory to output zeros everywhere except where CE demands a correction.

The tension: CE wants high attention at decision points (to shift logits toward f's letter). Sparsity wants low attention everywhere. At decision points, CE dominates. At arithmetic positions, sparsity wins and the memory adds nothing to the residual stream.

## Metrics

| Metric | Measures | Expected behavior |
|--------|----------|-------------------|
| **f_selection** | Fraction of steps using f's letter | Goes UP (main goal) |
| **operation_accuracy** | Arithmetic correctness | Stays FLAT (must not break) |
| **operation_selection** | Letter is valid f or g | Stays high or goes up |
| **complete_solution** | Full trace correct + output matches | May change (chain path differs) |

The key check: f_selection improves while operation_accuracy holds.

## One-example walkthrough

The script prints a detailed before/after showing:

1. **Both traces** — baseline (memory off) and memory (memory on), with per-step op/sel/f stats
2. **Decision-point comparison** — at each step, which letter f and g select, what baseline picked, what memory picked, whether it FLIPPED
3. **Attention weights** — max attention across memory entries at each position type:
   - Prompt positions: should be near zero (~0.001)
   - Decision points ([TRACE], `;`): should be high (~0.5-0.9)
   - Arithmetic positions: should be near zero (~0.001)

This directly shows the memory is surgical: active at decisions, silent at arithmetic.

## Running

```bash
# Default: 500 train, 200 eval, layer 6, 8 entries, 5 epochs
python main.py

# More training data, more entries
python main.py --n_train 2000 --mem_entries 16 --epochs 10

# Try different layers
python main.py --layer 3
python main.py --layer 9

# Show a specific example in detail
python main.py --show_idx 5

# Tune sparsity (higher = more silent, lower = more active)
python main.py --sparsity 0.01
python main.py --sparsity 1.0
```

## Design notes

**Why f-only targeting?** The pretrained model sees f and g as equally valid (50-50 in training data). By defining "correct = f's letter", we give the memory a single ground truth per decision point. This is analogous to fine-tuning on a restricted domain where the correct behavior is unambiguous.

**Why not detect-then-fix?** The old pipeline (`memory_inject.py`) generates a trace, finds the first error, learns a per-example correction vector, re-generates. This is a two-step process where detection and correction are separate. The new approach integrates correction into generation — the memory fires automatically at decision points via learned key matching. No explicit detection step.

**The cascading issue.** Changing the letter at step k changes the intermediate vector. Steps k+1, k+2, ... now operate on a different vector, so f and g select different letters. This is correct — the model runs its program on the new path. The right metric is operation_accuracy (does the arithmetic stay correct?), not "are the downstream letters the same?" (they won't be).
