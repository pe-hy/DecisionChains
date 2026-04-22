# CLAUDE.md

Pointer file. **Read `README.md` for the full picture.**
Quick visual: `DIAGRAM.txt`. Result tables: `make_tables.py`.

## TL;DR

Inference-time alignment of a frozen 30M-param GPT-NeoX on
decision-chain tasks. A small **external key-value memory** (~295K
trainable params) is hooked into an early residual-stream layer (L=1 or
L=2, tied). Base model stays frozen.

The pretrained model picks decision function `f` or `g` roughly 50/50 at
each step. Goal: steer it to always pick `f`.

Best result: f-selection 52.5% → 97% (+45 pp), full alignment 8.8% → 88%
(+79 pp), op.\ accuracy preserved (-1.5 pp).

## Workflow

```
exp.py --name RUN_NAME [flags]           # one config → one JSON
make_tables.py [--v2] [--prefix X]       # tabulate all runs
visualization/build_data.py && open      # HTML viewer
outputs/comparison_table.pdf             # academic writeup
```

All runs land as JSONs in `outputs/experiments/{name}.json`. The JSON
schema is the single source of truth; every downstream tool reads it.

## Key facts to remember

- **Layer 1 ≈ Layer 2** on alignment. Both much better than L≥4.
- **N=1 works.** A single memory entry gives +42 pp f_sel with
  *improved* op.\ accuracy. Evidence for linear-direction hypothesis.
- **2k ffff val** (`val_ffff.json`) is disjoint from train on both
  letter-tuples and input-vectors. Use it via `--ffff_val_file`.
- **Baseline cache** lives in `outputs/experiments/_baseline_cache_*.json`,
  keyed by `(n_eval, hash of ffff val file)`. Skips the 55-min baseline
  eval on subsequent runs.

## Don'ts

- Don't reintroduce L1-on-softmax-attention sparsity (no-op). Use
  `--sparsity l2_nondp`.
- Don't conclude alignment from `ffff` val alone — use `val_full` for
  the honest test.
- Don't delete `outputs/experiments/` — that's the primary data store.
