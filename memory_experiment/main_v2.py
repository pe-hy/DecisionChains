"""
main_v2.py — Integrated Memory Attention for Self-Correcting Chain Generation

Instead of the detect-then-fix pipeline (main.py), this trains a small external
key-value memory that the model attends to at every token position. The memory
learns to fire at decision points where the model would make errors, providing
correction vectors that steer the letter choice. At all other positions,
attention stays near-zero.

Dataset simplified: only F-branch examples (at every step, f() selects the letter).
This removes the F/G ambiguity — "correct selection" now means "picked F's letter"
which is a single ground-truth answer, not a choice between two valid options.

Architecture:
    input_ids → [frozen GPT-NeoX layers 0..L-1]
             → layer L
             → h = layer_L output           ← hidden state at every position
             → mem_out = attention(h, K, V)  ← memory fires if correction needed
             → h' = h + mem_out              ← corrected residual stream
             → [frozen GPT-NeoX layers L+1..11] → logits

    K, V: (N, 512) learnable memory entries
    query = Linear(h) → dot product with K → softmax → weighted sum of V
    N is small (16-32 entries), so this adds ~16K parameters.

Training:
    - Freeze all base model weights
    - Train only memory (keys, values, query projection)
    - CE loss on output tokens (after [TRACE]) + L1 sparsity on attention weights
    - The sparsity loss encourages the memory to stay silent unless needed

Usage:
    python main_v2.py --n_train 50000 --n_val 500 --layer 6 --mem_entries 16 --epochs 3
    python main_v2.py --layer 6 --mem_entries 32 --lr 1e-3 --sparsity 0.01
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm

from metrics import (
    extract_input_output, decision_letters, parse_trace, apply_letter,
    LETTERS, LETTER_TO_OP,
)

# ─── Paths and constants ──────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / "12l-8h-512d-decision-chains-ext_6_2M"
HF_DIR = CHECKPOINT_DIR / "hf"
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended"

TRACE_TOKEN_ID = 87
BOS_TOKEN_ID = 88
EOS_TOKEN_ID = 92
PAD_TOKEN_ID = 89


# ─── Memory Attention Module ──────────────────────────────────────────────────

class MemoryAttention(nn.Module):
    """External key-value memory with dot-product attention.

    At every token position, the hidden state h is projected to a query.
    The query attends over N memory entries via dot product. The weighted
    sum of value vectors is returned to be added to the residual stream.

    What it learns:
      - keys: patterns that match "this position needs a correction"
      - values: the correction vectors to apply at those positions
      - query_proj: how to extract "am I at a decision point?" from h
    """

    def __init__(self, hidden_dim=512, n_entries=16):
        super().__init__()
        self.n_entries = n_entries
        self.keys = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.enabled = True
        self._last_attn = None  # (B, T, N) — saved for sparsity loss

    def forward(self, hidden_states):
        """
        Args:   hidden_states (B, T, D)
        Returns: output (B, T, D), attn_weights (B, T, N)
        """
        if not self.enabled:
            B, T, D = hidden_states.shape
            return (torch.zeros_like(hidden_states),
                    torch.zeros(B, T, self.n_entries, device=hidden_states.device))

        queries = self.query_proj(hidden_states)              # (B, T, D)
        attn = F.softmax(queries @ self.keys.T * self.scale, dim=-1)  # (B, T, N)
        output = attn @ self.values                           # (B, T, D)
        self._last_attn = attn
        return output, attn


# ─── Model + Memory Integration ───────────────────────────────────────────────

class ModelWithMemory:
    """Wraps a frozen GPT-NeoX with memory attention at a chosen layer.

    A forward hook on model.gpt_neox.layers[layer_idx] adds the memory output
    to the residual stream. The hook is always active. Set memory.enabled=False
    to disable (e.g., for baseline evaluation).

    Gradients flow through the hook into the memory parameters, even though
    the base model is frozen. The frozen layers act as fixed nonlinear
    transformations that the gradient passes through.
    """

    def __init__(self, model, memory, layer_idx):
        self.model = model
        self.memory = memory
        self.layer_idx = layer_idx

        mem = self.memory
        def hook_fn(module, input, output):
            hidden = output[0]
            mem_out, _ = mem(hidden)
            return (hidden + mem_out,) + output[1:]

        self._handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)

    def __call__(self, input_ids, attention_mask=None):
        self.memory._last_attn = None
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    def sparsity_loss(self):
        """L1 penalty on attention weights — encourages sparse corrections."""
        if self.memory._last_attn is None:
            return torch.tensor(0.0, device=next(self.memory.parameters()).device)
        return self.memory._last_attn.abs().mean()

    def cleanup(self):
        if self._handle:
            self._handle.remove()


# ─── Data Preparation ─────────────────────────────────────────────────────────

def load_f_only(path, n=0):
    """Load examples where every step used decision function f.

    In the original data, each step picks f or g randomly (50-50).
    By keeping only all-f examples, we remove the ambiguity: at every
    step there is exactly one correct letter (f's letter), not two.
    """
    with open(path) as f:
        data = json.load(f)
    f_only = [ex for ex in data
              if all(d == "f" for d in ex.get("decision_funcs", []))]
    return f_only[:n] if n else f_only


def prepare_dataloader(examples, tokenizer, batch_size, max_len=512):
    """Tokenize examples, create masked labels, return DataLoader.

    Format:  [BOS] input [TRACE] output [EOS]
    Labels:  same as input_ids, with -100 for everything up to and
             including [TRACE] (we only train the model to predict the output).
    """
    all_ids = []
    for ex in examples:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
        all_ids.append(ids)

    seq_len = max(len(ids) for ids in all_ids)

    input_ids = torch.full((len(all_ids), seq_len), PAD_TOKEN_ID, dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids)

    attention_mask = (input_ids != PAD_TOKEN_ID).long()

    # Labels: input_ids with -100 mask for loss computation
    labels = input_ids.clone()
    labels[labels == PAD_TOKEN_ID] = -100
    for i in range(len(all_ids)):
        trace_pos = (input_ids[i] == TRACE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(trace_pos) > 0:
            labels[i, :trace_pos[0].item() + 1] = -100  # mask input portion

    dataset = TensorDataset(input_ids, labels, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ─── F-only Scoring ───────────────────────────────────────────────────────────

def score_f_only(input_vec, output_vec, generated_text):
    """Score a trace where "correct selection" = "picked F's letter".

    This is the key difference from metrics.score_trace which checks if
    the letter is F's OR G's. Here only F's letter counts as correct.
    """
    blocks = parse_trace(generated_text)
    current = list(input_vec)
    op_correct = 0
    sel_correct = 0
    n_steps = len(blocks)

    for block in blocks:
        letter_f, _ = decision_letters(current)  # only F matters now
        chosen = block.letter

        # Selection: must be F's letter specifically
        if chosen == letter_f:
            sel_correct += 1

        # Operation: re-execute and compare block text
        if current is not None and chosen in LETTER_TO_OP:
            expected_vec, expected_block = apply_letter(chosen, current)
            op_ok = (block.block == expected_block)
        else:
            expected_vec, op_ok = None, False
        if op_ok:
            op_correct += 1

        # Advance intermediate vector
        if op_ok and expected_vec is not None:
            current = expected_vec
        elif block.vec is not None:
            current = block.vec
        else:
            current = None

    final_vec = blocks[-1].vec if blocks else None
    matches_output = final_vec is not None and list(final_vec) == list(output_vec)
    full_correct = (n_steps > 0 and op_correct == n_steps
                    and sel_correct == n_steps and matches_output)

    return {
        "op_accuracy": op_correct / n_steps if n_steps else 0.0,
        "sel_accuracy": sel_correct / n_steps if n_steps else 0.0,
        "matches_output": matches_output,
        "full_correct": full_correct,
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_wrap, tokenizer, examples, device, desc="Evaluating"):
    """Generate traces for all examples, score with F-only metrics."""
    scores = []
    attn_all = []  # collect attention weights for analysis

    for ex in tqdm(examples, desc=desc):
        input_vec, output_vec = extract_input_output(ex["input"])
        if input_vec is None:
            continue

        prompt = f"[BOS] {ex['input']} [TRACE]"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model_wrap.generate(
                inp, max_length=512, do_sample=False,
                pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
            )

        gen_ids = out[0, len(prompt_ids):].tolist()
        if EOS_TOKEN_ID in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(EOS_TOKEN_ID)]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        scores.append(score_f_only(input_vec, output_vec, gen_text))

        # Capture attention weights from memory (if enabled)
        if model_wrap.memory.enabled and model_wrap.memory._last_attn is not None:
            attn_all.append(model_wrap.memory._last_attn[0].cpu().numpy())  # (T, N)

    n = len(scores)
    if n == 0:
        return {}, {}

    metrics = {
        "op_accuracy": sum(s["op_accuracy"] for s in scores) / n,
        "sel_accuracy": sum(s["sel_accuracy"] for s in scores) / n,
        "full_correct": sum(s["full_correct"] for s in scores) / n,
        "matches_output": sum(s["matches_output"] for s in scores) / n,
        "n": n,
    }

    # Attention analysis
    attn_stats = {}
    if attn_all:
        import numpy as np
        all_attn = np.concatenate(attn_all, axis=0)  # (total_positions, N)
        attn_stats = {
            "mean_attn": float(all_attn.mean()),
            "max_attn": float(all_attn.max()),
            "frac_above_0.01": float((all_attn.max(axis=-1) > 0.01).mean()),
            "frac_above_0.1": float((all_attn.max(axis=-1) > 0.1).mean()),
        }

    return metrics, attn_stats


# ─── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model_wrap, train_loader, optimizer, vocab_size, sparsity_coeff, device):
    """One epoch of training. Returns (avg_ce_loss, avg_sparsity_loss)."""
    model_wrap.model.eval()  # base model stays in eval mode (frozen)
    total_ce = 0
    total_sp = 0
    n_batches = 0

    for input_ids, labels, attention_mask in tqdm(train_loader, desc="Training"):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass (hook adds memory output automatically)
        logits = model_wrap(input_ids=input_ids, attention_mask=attention_mask).logits

        # Causal LM: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        sp_loss = model_wrap.sparsity_loss()
        loss = ce_loss + sparsity_coeff * sp_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_wrap.memory.parameters(), 1.0)
        optimizer.step()

        total_ce += ce_loss.item()
        total_sp += sp_loss.item()
        n_batches += 1

    return total_ce / max(n_batches, 1), total_sp / max(n_batches, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Integrated memory attention")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--layer", type=int, default=6, help="Layer to inject memory at")
    parser.add_argument("--mem_entries", type=int, default=16, help="Number of memory entries")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity", type=float, default=0.01, help="L1 sparsity coefficient")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── Load model ──
    print("Loading model...")
    if not (HF_DIR / "pytorch_model.bin").exists():
        from classify_decision_points import convert_checkpoint_if_needed
        convert_checkpoint_if_needed()

    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, dtype=torch.float32, local_files_only=True,
    )
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(CHECKPOINT_DIR / "tokenizer.json")
    )
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = " Padres"
    tokenizer.unk_token = "[UNK]"

    vocab_size = model.config.vocab_size

    # ── Load F-only data ──
    print("Loading F-only data...")
    train_data = load_f_only(DATA_DIR / "train.json", args.n_train)
    val_data = load_f_only(DATA_DIR / "val.json", args.n_val)
    print(f"  Train: {len(train_data)}   Val: {len(val_data)}")

    train_loader = prepare_dataloader(train_data, tokenizer, args.batch_size)

    # ── Create memory and wrap model ──
    memory = MemoryAttention(
        hidden_dim=model.config.hidden_size, n_entries=args.mem_entries,
    ).to(args.device)
    model_wrap = ModelWithMemory(model, memory, args.layer)

    n_params = sum(p.numel() for p in memory.parameters())
    print(f"  Memory: {args.mem_entries} entries × 512d = {n_params:,} trainable params")
    print(f"  Base model: frozen")

    optimizer = torch.optim.Adam(memory.parameters(), lr=args.lr)

    # ── Baseline (memory disabled) ──
    print("\n── Baseline (no memory) ──")
    memory.enabled = False
    bl_metrics, _ = evaluate(model_wrap, tokenizer, val_data, args.device, desc="Baseline")
    print(f"  op_accuracy:    {bl_metrics.get('op_accuracy', 0):.4f}")
    print(f"  sel_accuracy:   {bl_metrics.get('sel_accuracy', 0):.4f}  ← 'picked F's letter'")
    print(f"  full_correct:   {bl_metrics.get('full_correct', 0):.4f}")
    print(f"  matches_output: {bl_metrics.get('matches_output', 0):.4f}")

    # ── Train ──
    print(f"\n── Training ({args.epochs} epochs, lr={args.lr}, sparsity={args.sparsity}) ──")
    memory.enabled = True
    mem_metrics = bl_metrics  # will be overwritten after last epoch

    for epoch in range(args.epochs):
        ce, sp = train_epoch(
            model_wrap, train_loader, optimizer, vocab_size, args.sparsity, args.device,
        )

        memory.enabled = True
        mem_metrics, attn_stats = evaluate(
            model_wrap, tokenizer, val_data, args.device, desc=f"Eval epoch {epoch+1}",
        )
        print(f"  Epoch {epoch+1}: CE={ce:.4f}  sparsity={sp:.6f}  "
              f"sel={mem_metrics.get('sel_accuracy', 0):.4f}  "
              f"op={mem_metrics.get('op_accuracy', 0):.4f}  "
              f"full={mem_metrics.get('full_correct', 0):.4f}")
        if attn_stats:
            print(f"           attn: mean={attn_stats['mean_attn']:.4f}  "
                  f">0.01: {attn_stats['frac_above_0.01']:.1%}  "
                  f">0.1: {attn_stats['frac_above_0.1']:.1%}")

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(val_data)} F-only val examples)")
    print(f"{'='*60}")
    print(f"  {'Metric':20s}  {'Baseline':>10s}  {'Memory':>10s}  {'Delta':>10s}")
    print(f"  {'-'*55}")
    for key in ["op_accuracy", "sel_accuracy", "full_correct", "matches_output"]:
        bl = bl_metrics.get(key, 0)
        me = mem_metrics.get(key, 0)
        delta = me - bl
        print(f"  {key:20s}  {bl:>10.4f}  {me:>10.4f}  {delta:>+10.4f}")

    model_wrap.cleanup()


if __name__ == "__main__":
    main()
