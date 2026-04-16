"""
Surgical memory for decision-chain transformers.

Trains a small key-value memory (~8K params) that hooks into one layer of a
frozen 12-layer GPT-NeoX. At every token position, the hidden state queries
the memory via dot-product attention. The memory learns to:

  - Fire at decision points → steer letter choice toward f's letter
  - Stay silent everywhere else → preserve arithmetic

Training loss:
  CE on decision-point positions only (target = f's letter)
  + λ * L1 on attention weights everywhere (sparsity)

Usage:
  python main.py --n_train 500 --n_eval 200 --layer 6
  python main.py --n_train 2000 --epochs 10 --mem_entries 16
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm

from metrics import (
    score_trace, aggregate_scores, extract_input_output,
    decision_letters, parse_trace, apply_letter, LETTERS, LETTER_TO_OP,
)

# ── Paths & constants ─────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / "12l-8h-512d-decision-chains-ext_6_2M"
HF_DIR = CHECKPOINT_DIR / "hf"
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended"

TRACE_ID = 87
BOS_ID   = 88
EOS_ID   = 92
PAD_ID   = 89


# ── Memory module ─────────────────────────────────────────────────────────────

class MemoryAttention(nn.Module):
    """External key-value memory with dot-product attention.

    At each position: query = W_q(h), attend over N entries, return weighted V.
    Added to residual stream: h' = h + memory_output.
    """
    def __init__(self, hidden_dim, n_entries):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.enabled = True
        self.last_attn = None

    def forward(self, h):
        if not self.enabled:
            B, T, D = h.shape
            return torch.zeros_like(h), torch.zeros(B, T, self.keys.shape[0], device=h.device)
        q = self.query_proj(h)
        attn = F.softmax(q @ self.keys.T * self.scale, dim=-1)
        out = attn @ self.values
        self.last_attn = attn
        return out, attn


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(device):
    """Load frozen GPT-NeoX and tokenizer from checkpoint."""
    if not (HF_DIR / "pytorch_model.bin").exists():
        from classify_decision_points import convert_checkpoint_if_needed
        convert_checkpoint_if_needed()

    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, dtype=torch.float32, local_files_only=True,
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    tok = PreTrainedTokenizerFast(tokenizer_file=str(CHECKPOINT_DIR / "tokenizer.json"))
    tok.bos_token, tok.eos_token = "[BOS]", "[EOS]"
    tok.pad_token, tok.unk_token = "[PAD]", "[UNK]"
    return model, tok


def build_letter_tid_map(tokenizer):
    return {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS}


def filter_f_only(examples):
    """Keep only examples where every decision step used function f.

    With f-only data, the GT trace IS the f-path. Teacher forcing follows
    exactly the path we want the model to take — no train/generation mismatch.
    """
    return [ex for ex in examples
            if all(d == "f" for d in ex.get("decision_funcs", []))]


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_training_data(examples, tokenizer, letter_tids):
    """Tokenize examples, find decision points, compute f-letter targets.

    At each decision point (position whose next token is a letter a-t),
    the target is f(current_intermediate_vector). Intermediate vectors
    are computed by walking the ground-truth trace.

    Returns (input_ids, attention_mask, decision_targets) as padded tensors.
    decision_targets[i, pos] = f-letter token ID at decision positions, -100 elsewhere.
    """
    semi_id = tokenizer.encode(";", add_special_tokens=False)[0]
    all_ids, all_tgts = [], []

    for ex in examples:
        vec, _ = extract_input_output(ex["input"])
        if vec is None:
            continue

        ids = tokenizer.encode(
            f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]",
            add_special_tokens=False,
        )
        trace_pos = ids.index(TRACE_ID)
        # Decision positions: [TRACE] for step 0, then each ; for steps 1,2,...
        dp_positions = [trace_pos] + [
            i for i, t in enumerate(ids) if t == semi_id and i > trace_pos
        ]

        # Walk GT trace to get intermediate vectors → compute f-letter targets
        blocks = parse_trace(ex["output"])
        current = list(vec)
        tgts = []
        for step, block in enumerate(blocks):
            if step >= len(dp_positions) or current is None or len(current) < 5:
                break
            letter_f, _ = decision_letters(current)
            tid = letter_tids.get(letter_f)
            if tid is not None:
                tgts.append((dp_positions[step], tid))
            # Advance using GT letter (follow the GT path for intermediate vectors)
            if block.letter in LETTER_TO_OP:
                current, _ = apply_letter(block.letter, current)
            elif block.vec is not None:
                current = block.vec
            else:
                break

        all_ids.append(ids)
        all_tgts.append(tgts)

    # Pad to uniform length
    N = len(all_ids)
    max_len = max(len(ids) for ids in all_ids)
    input_ids = torch.full((N, max_len), PAD_ID, dtype=torch.long)
    attn_mask = torch.zeros(N, max_len, dtype=torch.long)
    dec_tgts = torch.full((N, max_len), -100, dtype=torch.long)

    for i, (ids, tgts) in enumerate(zip(all_ids, all_tgts)):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn_mask[i, :len(ids)] = 1
        for pos, tid in tgts:
            dec_tgts[i, pos] = tid

    n_dp = (dec_tgts != -100).sum().item()
    print(f"  {N} examples, {n_dp} decision points ({n_dp / N:.1f} per example)")
    return input_ids, attn_mask, dec_tgts


# ── Training ──────────────────────────────────────────────────────────────────

def register_memory_hook(model, memory, layer_idx):
    """Hook that adds memory output to residual stream at layer_idx."""
    def hook_fn(module, inp, output):
        h = output[0]
        mem_out, _ = memory(h)
        return (h + mem_out,) + output[1:]
    return model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)


def train_memory(model, memory, layer_idx, input_ids, attn_mask, dec_tgts,
                 epochs, batch_size, lr, sparsity_coeff, device):
    """Train memory by backpropagating CE + sparsity through frozen layers."""
    optimizer = torch.optim.Adam(memory.parameters(), lr=lr)
    N = input_ids.shape[0]
    handle = register_memory_hook(model, memory, layer_idx)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        ce_sum, sp_sum, n_batches = 0.0, 0.0, 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            b_ids = input_ids[idx].to(device)
            b_mask = attn_mask[idx].to(device)
            b_tgt = dec_tgts[idx].to(device)

            logits = model(input_ids=b_ids, attention_mask=b_mask).logits

            # CE loss only at decision-point positions
            dp_mask = b_tgt != -100
            if dp_mask.any():
                ce = F.cross_entropy(logits[dp_mask], b_tgt[dp_mask])
            else:
                ce = torch.tensor(0.0, device=device)

            # L1 sparsity on all attention weights
            if memory.last_attn is not None:
                sp = memory.last_attn.abs().mean()
            else:
                sp = torch.tensor(0.0, device=device)

            loss = ce + sparsity_coeff * sp
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
            optimizer.step()

            ce_sum += ce.item()
            sp_sum += sp.item()
            n_batches += 1

        print(f"  epoch {epoch + 1}/{epochs}  "
              f"CE={ce_sum / n_batches:.4f}  "
              f"sparsity={sp_sum / n_batches:.6f}")

    return handle  # keep active for evaluation


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_trace(model, tokenizer, example, device):
    """Greedy-generate a trace from the prompt. Returns decoded text."""
    prompt = f"[BOS] {example['input']} [TRACE]"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    out = model.generate(
        torch.tensor([ids], device=device),
        max_length=512, do_sample=False,
        pad_token_id=PAD_ID, eos_token_id=EOS_ID,
    )
    gen = out[0, len(ids):].tolist()
    if EOS_ID in gen:
        gen = gen[:gen.index(EOS_ID)]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── Scoring ───────────────────────────────────────────────────────────────────

def f_selection_details(input_vec, generated_text):
    """Walk generated trace, check each step against f's letter.

    Returns (f_hits, n_steps, per_step_details).
    per_step_details: list of (letter_f, letter_g, chosen, is_f_hit).
    """
    blocks = parse_trace(generated_text)
    current = list(input_vec)
    hits, n = 0, 0
    details = []

    for block in blocks:
        if current is None or len(current) < 5:
            break
        lf, lg = decision_letters(current)
        n += 1
        is_f = block.letter == lf
        if is_f:
            hits += 1
        details.append((lf, lg, block.letter, is_f))

        if block.letter in LETTER_TO_OP:
            current, _ = apply_letter(block.letter, current)
        elif block.vec is not None:
            current = block.vec
        else:
            break

    return hits, n, details


def evaluate(model, tokenizer, examples, device, desc="Eval"):
    """Generate + score examples. Returns metrics dict."""
    all_scores = []
    f_hits_total, f_n_total = 0, 0

    for ex in tqdm(examples, desc=desc):
        vec, ovec = extract_input_output(ex["input"])
        if vec is None:
            continue
        text = generate_trace(model, tokenizer, ex, device)
        all_scores.append(score_trace(vec, ovec, text))
        fh, fn, _ = f_selection_details(vec, text)
        f_hits_total += fh
        f_n_total += fn

    agg = aggregate_scores(all_scores)
    agg["f_selection"] = f_hits_total / f_n_total if f_n_total else 0.0
    return agg


# ── One-example walkthrough ──────────────────────────────────────────────────

def show_example(model, memory, tokenizer, example, device):
    """Print detailed before/after for one example."""
    vec, ovec = extract_input_output(example["input"])
    if vec is None:
        print("  (could not parse example)")
        return

    print(f"\n  Input:  {vec}")
    print(f"  Output: {ovec}")

    # Baseline (memory disabled)
    memory.enabled = False
    bl_text = generate_trace(model, tokenizer, example, device)
    bl_score = score_trace(vec, ovec, bl_text)
    _, _, bl_steps = f_selection_details(vec, bl_text)
    memory.enabled = True

    # With memory
    mem_text = generate_trace(model, tokenizer, example, device)
    mem_score = score_trace(vec, ovec, mem_text)
    _, _, mem_steps = f_selection_details(vec, mem_text)

    # Traces
    bl_f = sum(s[3] for s in bl_steps)
    print(f"\n  BASELINE (op={bl_score.op_accuracy:.0%} "
          f"sel={bl_score.sel_accuracy:.0%} f={bl_f}/{len(bl_steps)}):")
    for block in bl_text.split(" ; "):
        print(f"    {block}")

    mem_f = sum(s[3] for s in mem_steps)
    print(f"\n  MEMORY (op={mem_score.op_accuracy:.0%} "
          f"sel={mem_score.sel_accuracy:.0%} f={mem_f}/{len(mem_steps)}):")
    for block in mem_text.split(" ; "):
        print(f"    {block}")

    # Decision-point comparison
    print(f"\n  Decision points:")
    for i in range(max(len(bl_steps), len(mem_steps))):
        bl = bl_steps[i] if i < len(bl_steps) else None
        me = mem_steps[i] if i < len(mem_steps) else None
        if bl and me:
            lf, lg = bl[0], bl[1]
            tag = ""
            if bl[2] != me[2]:
                tag = " FLIPPED"
            elif me[3]:
                tag = " (already f)"
            print(f"    step {i}: f->{lf} g->{lg}  "
                  f"baseline={bl[2]}({'f' if bl[3] else 'g'})  "
                  f"memory={me[2]}({'f' if me[3] else 'g'}){tag}")

    # Attention analysis via teacher-forced forward pass on the memory trace
    semi_id = tokenizer.encode(";", add_special_tokens=False)[0]
    full_ids = tokenizer.encode(
        f"[BOS] {example['input']} [TRACE] {mem_text} [EOS]",
        add_special_tokens=False,
    )
    with torch.no_grad():
        model(torch.tensor([full_ids], device=device))

    if memory.last_attn is not None:
        attn = memory.last_attn[0].cpu()
        max_attn = attn.max(dim=-1).values
        trace_pos = full_ids.index(TRACE_ID)
        dp_set = {trace_pos} | {
            i for i, t in enumerate(full_ids) if t == semi_id and i > trace_pos
        }
        non_dp = [i for i in range(trace_pos + 1, len(full_ids)) if i not in dp_set]

        n_entries = memory.keys.shape[0]
        print(f"\n  Memory attention (max across {n_entries} entries):")
        print(f"    prompt (0..{trace_pos - 1}):      "
              f"avg={max_attn[:trace_pos].mean():.4f}  "
              f"max={max_attn[:trace_pos].max():.4f}")
        for j, dp in enumerate(sorted(dp_set)):
            if dp < len(max_attn):
                label = "[TRACE]" if dp == trace_pos else f"; (step {j})"
                print(f"    {label:20s} (pos {dp:3d}):  {max_attn[dp]:.4f}")
        if non_dp:
            na = max_attn[non_dp]
            print(f"    arithmetic positions:  "
                  f"avg={na.mean():.4f}  max={na.max():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Surgical memory injection")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--mem_entries", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity", type=float, default=0.1,
                        help="L1 sparsity coefficient on memory attention")
    parser.add_argument("--show_idx", type=int, default=0,
                        help="Val example index for the detailed walkthrough")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.device)
    letter_tids = build_letter_tid_map(tokenizer)

    print(f"Loading f-only data from {DATA_DIR}")
    with open(DATA_DIR / "train.json") as f:
        train_all = json.load(f)
    with open(DATA_DIR / "val.json") as f:
        val_all = json.load(f)
    train_examples = filter_f_only(train_all)[:args.n_train]
    val_examples = filter_f_only(val_all)[:args.n_eval]
    print(f"  train: {len(train_examples)} f-only (from {len(train_all)} total)")
    print(f"  val:   {len(val_examples)} f-only (from {len(val_all)} total)")

    # Prepare
    print("Preparing training data...")
    train_ids, train_mask, train_tgts = prepare_training_data(
        train_examples, tokenizer, letter_tids,
    )

    # Baseline
    print("\n── Baseline (no memory) ──")
    baseline = evaluate(model, tokenizer, val_examples, args.device, "Baseline")
    for k in ["operation_accuracy", "operation_selection", "f_selection", "complete_solution"]:
        print(f"  {k:22s} {baseline.get(k, 0):.4f}")

    # Train memory
    memory = MemoryAttention(model.config.hidden_size, args.mem_entries).to(args.device)
    n_params = sum(p.numel() for p in memory.parameters())
    print(f"\n── Training ({args.mem_entries} entries, {n_params:,} params, layer {args.layer}) ──")

    t0 = time.time()
    handle = train_memory(
        model, memory, args.layer,
        train_ids, train_mask, train_tgts,
        args.epochs, args.batch_size, args.lr, args.sparsity,
        args.device,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # Evaluate with memory
    print("\n── With memory ──")
    with_mem = evaluate(model, tokenizer, val_examples, args.device, "Memory")
    for k in ["operation_accuracy", "operation_selection", "f_selection", "complete_solution"]:
        print(f"  {k:22s} {with_mem.get(k, 0):.4f}")

    # Comparison
    print(f"\n{'=' * 62}")
    print(f"  {'metric':22s}  {'baseline':>10s}  {'memory':>10s}  {'delta':>10s}")
    print(f"  {'-' * 57}")
    for k in ["operation_accuracy", "operation_selection", "f_selection", "complete_solution"]:
        b, m = baseline.get(k, 0), with_mem.get(k, 0)
        print(f"  {k:22s}  {b:>10.4f}  {m:>10.4f}  {m - b:>+10.4f}")

    # One-example walkthrough
    print(f"\n{'=' * 62}")
    print("  ONE EXAMPLE WALKTHROUGH")
    print(f"{'=' * 62}")
    show_idx = min(args.show_idx, len(val_examples) - 1)
    show_example(model, memory, tokenizer, val_examples[show_idx], args.device)

    # Save
    save_path = SCRIPT_DIR / "memory.pt"
    torch.save(memory.state_dict(), save_path)
    print(f"\nMemory saved to {save_path}")

    handle.remove()


if __name__ == "__main__":
    main()
