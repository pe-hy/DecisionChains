# CLAUDE.md

Pointer file. **Open `DIAGRAM.txt` for the one-page visual**, then
**read `README.md` for the full picture** — the mechanism, loss,
train/val split, metrics, files, and how to run experiments.

## TL;DR

- We're studying **inference-time alignment** of a frozen pretrained
  decision-chain transformer via a small external **key-value memory**
  hooked into the residual stream at one layer.
- The memory is the only thing trained (~270K params); the 30M-param base
  model stays frozen.
- Goal: bias the model from a 50/50 `f`/`g` choice at decision points
  toward always picking `f`, without retraining the base.
- All experiments go through `exp.py` (one config = one JSON in
  `outputs/experiments/`). `compare.py` ranks them.

## Don'ts

- Don't reintroduce L1-on-softmax-attention as a sparsity loss — it's a
  no-op (softmax sums to 1; abs.mean = 1/N regardless of params). Use
  `--sparsity l2_nondp` instead.
- Don't conclude alignment from `ffff` val alone — the off-path
  generalization test lives on `val_full`.
- Don't filter to `ffff` for training without realizing it's a coin-flip
  selection (the input/letter-tuple distribution is unaffected, but the
  hidden-state distribution the memory sees is restricted to f-paths).

See README.md sections 2–6 for the rationale behind each.
