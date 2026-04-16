# Data Flow in classify_decision_points.py

## Step 1: Load examples

We load N examples (default 500) from `val.json`. Each example is a dict with `input` and `output` strings:
```
input:  "INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ] OUTPUT : [ 2 , 0 , 8 , 4 , 3 , 9 ]"
output: "n : 4 + 4 = 8 , ... R [ 8 , 7 , 2 , 1 , 1 , 8 ] ; r : 8 + 0 = 8 , ... R [ ... ] ; s : ... ; q : ..."
```

## Step 2: Tokenize

Each example is concatenated as `[BOS] {input} [TRACE] {output} [EOS]` and tokenized. All sequences are padded to the max length in the batch. With 100 examples, this gave 100 sequences of length 292.

## Step 3: Label decision points

For each token position `i`, we label it `1` (decision point) if:
- `token[i+1]` is a letter (a-t, IDs 67-86), AND
- position `i` is at or after the `[TRACE]` token

In practice, the labeled positions are always `[TRACE]` and `;` tokens — they precede the letter that selects the next transformation function.

With 100 examples: 405 decision points out of 21,108 non-padding tokens (1.9%).

## Step 4: Extract hidden states

The pretrained model runs a forward pass (teacher-forced, no generation) on all tokenized sequences with `output_hidden_states=True`. This returns hidden state vectors at every layer for every token position.

Result: 13 tensors (embedding layer + 12 transformer layers), each of shape `(100, 292, 512)`.

## Step 5: Prepare probe data

For each layer, we **flatten** all token positions across all examples into a single pool of vectors:

```
(100 examples × 292 positions) → filter out padding → ~21,108 vectors of dim 512
Each vector has a binary label: 1 = decision point, 0 = not
```

## Step 6: Train/val split (current — per token, NOT per example)

The ~21,108 vectors are **randomly shuffled** and split 80/20:
- Probe training set: ~16,886 vectors
- Probe validation set: ~4,222 vectors

**Problem**: Vectors from the SAME sequence can appear in both probe-train and probe-val. The probe might exploit within-sequence patterns rather than learning a general decision-point detector.

## Step 7: Train linear probe

For each layer independently:
- `nn.Linear(512, 2)` — 1,026 parameters (512×2 weights + 2 biases)
- Trained with Adam (lr=1e-3) + cross-entropy loss with class weights (to handle the 1.9% vs 98.1% imbalance)
- 50 epochs, mini-batches of 1024

## Step 8: Evaluate

On the held-out 20% of vectors: accuracy, precision, recall, F1.

## Result

All layers hit 100% or near-100%. But this is expected: the decision-point signal is **structurally trivial** — the token at a decision point is always `;` or `[TRACE]`, so even the embedding layer can distinguish it. The random per-token split makes this even easier since the probe sees other positions from the same sequences during training.
