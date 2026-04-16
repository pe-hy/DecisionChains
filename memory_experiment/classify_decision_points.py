"""
Decision-Point Linear Probe: classify which token positions are decision points
(positions where the next token is a letter a-t selecting a transformation function).

Tests whether the model's hidden states contain a linearly separable signal for
decision points — a prerequisite for external key-value memory.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / "12l-8h-512d-decision-chains-ext_6_2M"
HF_DIR = CHECKPOINT_DIR / "hf"
VAL_DATA_PATH = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended" / "val.json"

LETTER_TOKEN_IDS = set(range(67, 87))  # a=67 ... t=86
TRACE_TOKEN_ID = 87
BOS_TOKEN_ID = 88
EOS_TOKEN_ID = 92
PAD_TOKEN_ID = 89


def convert_checkpoint_if_needed():
    """Convert LitGPT checkpoint to HF format if not already done."""
    if (HF_DIR / "pytorch_model.bin").exists():
        return
    print("Converting LitGPT checkpoint to HF format...")
    from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
    from litgpt.utils import copy_config_files

    HF_DIR.mkdir(parents=True, exist_ok=True)
    copy_config_files(source_dir=CHECKPOINT_DIR, out_dir=HF_DIR)
    convert_lit_checkpoint(checkpoint_dir=CHECKPOINT_DIR, output_dir=HF_DIR)

    # Rename model.pth -> pytorch_model.bin for HF
    state_dict = torch.load(HF_DIR / "model.pth")
    torch.save(state_dict, HF_DIR / "pytorch_model.bin")

    # Copy config.json and tokenizer files
    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = CHECKPOINT_DIR / f
        if src.exists():
            shutil.copy2(src, HF_DIR / f)
    print(f"  Saved HF checkpoint to {HF_DIR}")


def load_model_and_tokenizer(device):
    convert_checkpoint_if_needed()

    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR,
        dtype=torch.float32,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(CHECKPOINT_DIR / "tokenizer.json")
    )
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"

    return model, tokenizer


def load_val_data(n_examples):
    with open(VAL_DATA_PATH) as f:
        data = json.load(f)
    if n_examples < len(data):
        data = data[:n_examples]
    return data


def tokenize_examples(examples, tokenizer):
    """Tokenize examples as [BOS] input [TRACE] output [EOS], return padded tensors."""
    all_ids = []
    for ex in examples:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.append(ids)

    max_len = max(len(ids) for ids in all_ids)
    input_ids = torch.full((len(all_ids), max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)

    for i, ids in enumerate(all_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1

    return input_ids, attention_mask


def label_decision_points(input_ids):
    """Label position i as True if token[i+1] is a letter AND i is after [TRACE]."""
    batch_size, seq_len = input_ids.shape
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for b in range(batch_size):
        after_trace = False
        for i in range(seq_len - 1):
            tok = input_ids[b, i].item()
            next_tok = input_ids[b, i + 1].item()
            if tok == TRACE_TOKEN_ID:
                after_trace = True
            if after_trace and next_tok in LETTER_TOKEN_IDS:
                labels[b, i] = 1

    return labels


def extract_hidden_states(model, input_ids, attention_mask, batch_size, device):
    """Run forward pass in batches, collect hidden states at all layers."""
    n_examples = input_ids.shape[0]
    all_hidden = None  # will be list of lists per layer

    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        batch_ids = input_ids[start:end].to(device)
        batch_mask = attention_mask[start:end].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (n_layers+1) tensors, each (bs, seq, hidden)
        if all_hidden is None:
            all_hidden = [[] for _ in range(len(outputs.hidden_states))]

        for layer_idx, hs in enumerate(outputs.hidden_states):
            all_hidden[layer_idx].append(hs.cpu())

    # Concatenate batches per layer
    return [torch.cat(layer_list, dim=0) for layer_list in all_hidden]


def prepare_probe_data(hidden_states_layer, labels, attention_mask):
    """Flatten batch×seq into individual vectors, filter out padding."""
    bs, seq_len, hidden_dim = hidden_states_layer.shape

    h_flat = hidden_states_layer.reshape(-1, hidden_dim)
    l_flat = labels.reshape(-1)
    m_flat = attention_mask.reshape(-1)

    # Keep only non-padding, non-last-position tokens
    valid = m_flat.bool()
    h_valid = h_flat[valid]
    l_valid = l_flat[valid]

    return h_valid, l_valid


def train_probe(X, y, epochs, lr, device):
    """Train nn.Linear(hidden_dim, 2) probe, return metrics on held-out split."""
    n = X.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Class weights for imbalance
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    if n_pos > 0 and n_neg > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], device=device)
    else:
        weight = None

    hidden_dim = X.shape[1]
    probe = nn.Linear(hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Mini-batch training
    dataset = TensorDataset(X_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(X_val.to(device))
        preds = logits.argmax(dim=1).cpu()
        y_v = y_val.cpu()

    tp = ((preds == 1) & (y_v == 1)).sum().item()
    fp = ((preds == 1) & (y_v == 0)).sum().item()
    fn = ((preds == 0) & (y_v == 1)).sum().item()
    tn = ((preds == 0) & (y_v == 0)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return acc, prec, rec, f1, probe


def main():
    parser = argparse.ArgumentParser(description="Decision-point linear probe")
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {CHECKPOINT_DIR}")
    model, tokenizer = load_model_and_tokenizer(args.device)

    print(f"Loading {args.n_examples} examples from {VAL_DATA_PATH}")
    examples = load_val_data(args.n_examples)

    print("Tokenizing...")
    input_ids, attention_mask = tokenize_examples(examples, tokenizer)
    print(f"  Sequences: {input_ids.shape[0]}, max length: {input_ids.shape[1]}")

    print("Labeling decision points...")
    labels = label_decision_points(input_ids)
    n_pos = labels.sum().item()
    n_total = attention_mask.sum().item()
    print(f"  Decision points: {n_pos} / {int(n_total)} tokens ({100*n_pos/n_total:.1f}%)")

    # Sanity check: print first example's decision points
    print("\n--- Sanity check (first example) ---")
    seq = input_ids[0]
    lab = labels[0]
    mask = attention_mask[0]
    tokens = tokenizer.convert_ids_to_tokens(seq[mask.bool()].tolist())
    dp_positions = lab[mask.bool()].nonzero(as_tuple=True)[0].tolist()
    for pos in dp_positions:
        ctx_start = max(0, pos - 1)
        ctx_end = min(len(tokens), pos + 3)
        context = tokens[ctx_start:ctx_end]
        print(f"  pos {pos}: ...{' '.join(context)}...  (next token = {tokens[pos+1] if pos+1 < len(tokens) else '?'})")
    print()

    print("Extracting hidden states...")
    all_hidden = extract_hidden_states(model, input_ids, attention_mask, args.batch_size, args.device)
    print(f"  Layers: {len(all_hidden)} (embedding + {len(all_hidden)-1} transformer)")

    print(f"\nTraining linear probes ({args.probe_epochs} epochs each)...")
    print(f"{'Layer':>5} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 48)

    for layer_idx in range(len(all_hidden)):
        X, y = prepare_probe_data(all_hidden[layer_idx], labels, attention_mask)
        acc, prec, rec, f1, probe = train_probe(X, y, args.probe_epochs, args.lr, args.device)
        label = "emb" if layer_idx == 0 else str(layer_idx)
        print(f"{label:>5} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
