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
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended"
TRAIN_DATA_PATH = DATA_DIR / "train.json"
VAL_DATA_PATH = DATA_DIR / "val.json"

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


def load_data(path, n_examples):
    with open(path) as f:
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


def train_probe(X_train, y_train, X_test, y_test, epochs, lr, device):
    """Train nn.Linear(hidden_dim, 2) on train data, evaluate on test data."""
    # Class weights for imbalance
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    if n_pos > 0 and n_neg > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], device=device)
    else:
        weight = None

    hidden_dim = X_train.shape[1]
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

    # Evaluate on test set
    probe.eval()
    with torch.no_grad():
        logits = probe(X_test.to(device))
        preds = logits.argmax(dim=1).cpu()
        y_v = y_test.cpu()

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
    parser.add_argument("--n_train", type=int, default=500, help="Examples from train.json for probe training")
    parser.add_argument("--n_test", type=int, default=500, help="Examples from val.json for probe testing")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {CHECKPOINT_DIR}")
    model, tokenizer = load_model_and_tokenizer(args.device)

    # Load train and test data from separate files
    print(f"Loading {args.n_train} train examples from {TRAIN_DATA_PATH}")
    train_examples = load_data(TRAIN_DATA_PATH, args.n_train)
    print(f"Loading {args.n_test} test examples from {VAL_DATA_PATH}")
    test_examples = load_data(VAL_DATA_PATH, args.n_test)

    # Tokenize and label — train
    print("Tokenizing train...")
    train_ids, train_mask = tokenize_examples(train_examples, tokenizer)
    train_labels = label_decision_points(train_ids)
    n_pos = train_labels.sum().item()
    n_total = train_mask.sum().item()
    print(f"  {train_ids.shape[0]} seqs, max len {train_ids.shape[1]}, "
          f"decision points: {n_pos}/{int(n_total)} ({100*n_pos/n_total:.1f}%)")

    # Tokenize and label — test
    print("Tokenizing test...")
    test_ids, test_mask = tokenize_examples(test_examples, tokenizer)
    test_labels = label_decision_points(test_ids)
    n_pos = test_labels.sum().item()
    n_total = test_mask.sum().item()
    print(f"  {test_ids.shape[0]} seqs, max len {test_ids.shape[1]}, "
          f"decision points: {n_pos}/{int(n_total)} ({100*n_pos/n_total:.1f}%)")

    # Sanity check on first test example
    print("\n--- Sanity check (first test example) ---")
    seq = test_ids[0]
    lab = test_labels[0]
    mask = test_mask[0]
    tokens = tokenizer.convert_ids_to_tokens(seq[mask.bool()].tolist())
    dp_positions = lab[mask.bool()].nonzero(as_tuple=True)[0].tolist()
    for pos in dp_positions:
        ctx_start = max(0, pos - 1)
        ctx_end = min(len(tokens), pos + 3)
        context = tokens[ctx_start:ctx_end]
        print(f"  pos {pos}: ...{' '.join(context)}...  (next token = {tokens[pos+1] if pos+1 < len(tokens) else '?'})")
    print()

    # Extract hidden states for train and test
    print("Extracting hidden states (train)...")
    train_hidden = extract_hidden_states(model, train_ids, train_mask, args.batch_size, args.device)
    print(f"  Layers: {len(train_hidden)} (embedding + {len(train_hidden)-1} transformer)")

    print("Extracting hidden states (test)...")
    test_hidden = extract_hidden_states(model, test_ids, test_mask, args.batch_size, args.device)

    # Train probes on train data, evaluate on test data
    print(f"\nTraining linear probes ({args.probe_epochs} epochs each)...")
    print(f"  Train: {args.n_train} examples from train.json")
    print(f"  Test:  {args.n_test} examples from val.json")
    print(f"{'Layer':>5} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 48)

    for layer_idx in range(len(train_hidden)):
        X_train, y_train = prepare_probe_data(train_hidden[layer_idx], train_labels, train_mask)
        X_test, y_test = prepare_probe_data(test_hidden[layer_idx], test_labels, test_mask)
        acc, prec, rec, f1, probe = train_probe(
            X_train, y_train, X_test, y_test, args.probe_epochs, args.lr, args.device
        )
        label = "emb" if layer_idx == 0 else str(layer_idx)
        print(f"{label:>5} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
