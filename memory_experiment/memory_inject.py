"""
External Memory Injection Experiment.

For each test example, learns a value vector V that, when added to the residual
stream at the [TRACE] position in a chosen layer, steers the model's first
letter prediction from one valid choice to the other (f→g or g→f branch).

The model weights are frozen — only V (512 dims) is optimized.

Example:
    Prompt ends with [TRACE] at position 31.
    Hidden state h[31] at layer L produces logits → predicts letter "n".
    We learn V such that h[31] + V → predicts letter "h" instead.
    Then we generate with V injected and score the full trace.

Usage:
    python memory_inject.py --n_examples 10 --layers 6 --steps 200
    python memory_inject.py --n_examples 50 --layers 1,6,12 --steps 300
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from metrics import (
    score_trace, extract_input_output, aggregate_scores, format_report,
    decision_letters, LETTERS, INT_TO_LETTER,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / "12l-8h-512d-decision-chains-ext_6_2M"
HF_DIR = CHECKPOINT_DIR / "hf"
VAL_DATA_PATH = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended" / "val.json"

TRACE_TOKEN_ID = 87
BOS_TOKEN_ID = 88
EOS_TOKEN_ID = 92
PAD_TOKEN_ID = 89


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(device):
    # Convert LitGPT → HF if needed (reuse the converted checkpoint)
    if not (HF_DIR / "pytorch_model.bin").exists():
        from classify_decision_points import convert_checkpoint_if_needed
        convert_checkpoint_if_needed()

    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, dtype=torch.float32, local_files_only=True,
    )
    model.to(device)
    model.eval()

    # Freeze all model parameters — only V will be optimized
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(CHECKPOINT_DIR / "tokenizer.json")
    )
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"

    return model, tokenizer


# ─── Prompt / tokenization helpers ───────────────────────────────────────────

def build_prompt_ids(example, tokenizer):
    """Tokenize the prompt: [BOS] {input} [TRACE].
    Returns token ids and the position of [TRACE] (last token)."""
    text = f"[BOS] {example['input']} [TRACE]"
    ids = tokenizer.encode(text, add_special_tokens=False)
    trace_pos = ids.index(TRACE_TOKEN_ID)
    return ids, trace_pos


def build_full_ids(example, tokenizer):
    """Tokenize the full sequence: [BOS] {input} [TRACE] {output} [EOS].
    Used for teacher-forced learning of V."""
    text = f"[BOS] {example['input']} [TRACE] {example['output']} [EOS]"
    return tokenizer.encode(text, add_special_tokens=False)


def get_baseline_letter(example, tokenizer, model, prompt_ids, trace_pos, device):
    """Run baseline generation and return the first letter the model picks."""
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_tensor, max_length=512, do_sample=False,
            pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
        )
    gen_ids = out[0, len(prompt_ids):].tolist()
    if gen_ids:
        first_token = tokenizer.decode([gen_ids[0]]).strip()
        if first_token in LETTERS:
            return first_token
    return None


# ─── Core: learn the value vector V ──────────────────────────────────────────

def learn_value_vector(model, full_ids, trace_pos, target_letter_id, layer_idx, lr, steps, device):
    """Learn a 512-dim vector V that steers the prediction at trace_pos toward target_letter.

    Mechanism:
        At layer `layer_idx`, position `trace_pos`, the hidden state is modified:
            h[trace_pos] = h[trace_pos] + V
        This changes the logits at trace_pos to predict target_letter instead
        of whatever the model originally predicted.

    Returns:
        V: the learned value vector (detached, on CPU)
        final_loss: loss at the last optimization step
        target_prob: P(target_letter) at the last step
    """
    hidden_dim = model.config.hidden_size
    V = torch.zeros(hidden_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([V], lr=lr)
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    target = torch.tensor([target_letter_id], dtype=torch.long, device=device)

    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states_tensor,)
        # hidden_states_tensor shape: (batch=1, seq_len, hidden_dim=512)
        hidden = output[0]
        hidden[:, trace_pos, :] = hidden[:, trace_pos, :] + V
        return (hidden,) + output[1:]

    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(input_tensor).logits[0, trace_pos]  # (vocab_size,)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        loss.backward()
        optimizer.step()

    # Final stats
    with torch.no_grad():
        logits = model(input_tensor).logits[0, trace_pos]
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[target_letter_id].item()
        final_loss = F.cross_entropy(logits.unsqueeze(0), target).item()

    handle.remove()
    return V.detach().cpu(), final_loss, target_prob


# ─── Core: generate with V injected ──────────────────────────────────────────

@torch.no_grad()
def generate_with_injection(model, tokenizer, prompt_ids, V, trace_pos, layer_idx, device):
    """Generate a trace with V added to the residual stream at trace_pos.

    During generation, the first forward pass processes the full prompt.
    The hook adds V at trace_pos, changing the first letter prediction.
    Subsequent tokens are generated normally (hook still fires but trace_pos
    is in the KV cache, so V only affects the initial pass).

    Returns the decoded trace text (everything after [TRACE]).
    """
    V_device = V.to(device)

    def hook_fn(module, input, output):
        hidden = output[0]
        if hidden.shape[1] > trace_pos:
            hidden[:, trace_pos, :] = hidden[:, trace_pos, :] + V_device
        return (hidden,) + output[1:]

    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_tensor, max_length=512, do_sample=False,
        pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
    )

    handle.remove()

    # Decode only the generated part (after the prompt)
    gen_ids = out[0, len(prompt_ids):].tolist()
    if EOS_TOKEN_ID in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(EOS_TOKEN_ID)]
    trace_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return trace_text


# ─── Baseline generation (no injection) ──────────────────────────────────────

@torch.no_grad()
def generate_baseline(model, tokenizer, prompt_ids, device):
    """Generate a trace without any injection. Returns decoded trace text."""
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_tensor, max_length=512, do_sample=False,
        pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
    )
    gen_ids = out[0, len(prompt_ids):].tolist()
    if EOS_TOKEN_ID in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(EOS_TOKEN_ID)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="External memory injection experiment")
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--layers", type=str, default="1,6,12",
                        help="Comma-separated layer indices, or 'all' for 0-11")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.layers == "all":
        layers = list(range(12))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # ── Load ──
    print(f"Loading model from {HF_DIR}")
    model, tokenizer = load_model_and_tokenizer(args.device)

    print(f"Loading {args.n_examples} examples from {VAL_DATA_PATH}")
    with open(VAL_DATA_PATH) as f:
        all_data = json.load(f)
    examples = all_data[:args.n_examples]

    # Build letter → token id mapping
    letter_to_tid = {}
    for letter in LETTERS:
        tids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(tids) == 1
        letter_to_tid[letter] = tids[0]

    # ── Run experiment ──
    # For each example: baseline generation, then injection at each layer.
    # Target = the OTHER valid f/g letter (switch branches).

    print(f"\n{'='*80}")
    print(f"  External Memory Injection Experiment")
    print(f"  Layers: {layers}  |  LR: {args.lr}  |  Steps: {args.steps}")
    print(f"  Examples: {args.n_examples}")
    print(f"{'='*80}\n")

    baseline_scores = []
    # Per-layer results: layer_idx → list of dicts
    layer_results = {L: [] for L in layers}
    skipped = 0

    for ex_idx, example in enumerate(examples):
        input_vec, output_vec = extract_input_output(example["input"])
        if input_vec is None:
            skipped += 1
            continue

        prompt_ids, trace_pos = build_prompt_ids(example, tokenizer)
        full_ids = build_full_ids(example, tokenizer)

        # ── Baseline (no injection) ──
        baseline_text = generate_baseline(model, tokenizer, prompt_ids, args.device)
        baseline_score = score_trace(input_vec, output_vec, baseline_text)
        baseline_scores.append(baseline_score)

        # What letter did the model choose?
        baseline_letter = baseline_score.per_step_letter[0] if baseline_score.n_steps > 0 else None

        # What are the two valid letters from f and g?
        letter_f, letter_g = decision_letters(input_vec)

        # Pick target = the other valid letter. Skip if f==g or baseline isn't f/g.
        if letter_f == letter_g:
            # f and g agree — no alternative branch to switch to
            print(f"  [{ex_idx}] f=g={letter_f}, skipping (no alternative branch)")
            skipped += 1
            continue
        if baseline_letter == letter_f:
            target_letter = letter_g
        elif baseline_letter == letter_g:
            target_letter = letter_f
        else:
            # Model didn't pick either valid letter — still try to steer to letter_f
            target_letter = letter_f

        target_tid = letter_to_tid[target_letter]

        # Print example header
        print(f"  [{ex_idx}] vec={input_vec}  f→{letter_f}  g→{letter_g}  "
              f"baseline={baseline_letter}  target={target_letter}")

        # ── Injection at each layer ──
        for layer_idx in layers:
            V, final_loss, target_prob = learn_value_vector(
                model, full_ids, trace_pos, target_tid, layer_idx,
                args.lr, args.steps, args.device,
            )

            injected_text = generate_with_injection(
                model, tokenizer, prompt_ids, V, trace_pos, layer_idx, args.device,
            )

            injected_score = score_trace(input_vec, output_vec, injected_text)
            injected_letter = injected_score.per_step_letter[0] if injected_score.n_steps > 0 else None
            target_hit = (injected_letter == target_letter)

            layer_results[layer_idx].append({
                "target_hit": target_hit,
                "score": injected_score,
                "target_prob": target_prob,
                "final_loss": final_loss,
                "v_norm": V.norm().item(),
            })

            status = "HIT" if target_hit else "MISS"
            print(f"         layer {layer_idx:2d}: {status}  "
                  f"P(target)={target_prob:.3f}  "
                  f"letter={injected_letter}  "
                  f"op={injected_score.op_correct}/{injected_score.n_steps}  "
                  f"sel={injected_score.sel_correct}/{injected_score.n_steps}  "
                  f"|V|={V.norm().item():.2f}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  RESULTS  ({len(baseline_scores)} examples evaluated, {skipped} skipped)")
    print(f"{'='*80}\n")

    # Baseline
    baseline_agg = aggregate_scores(baseline_scores)
    print(format_report(baseline_agg, "Baseline (no injection)"))

    # Per layer
    print(f"\n{'Layer':>5} | {'Target hit':>10} | {'op_accuracy':>11} | {'sel_accuracy':>12} | {'complete':>8} | {'avg P(target)':>13} | {'avg |V|':>8}")
    print("-" * 88)

    for layer_idx in layers:
        results = layer_results[layer_idx]
        if not results:
            continue
        n = len(results)
        hit_rate = sum(r["target_hit"] for r in results) / n
        scores = [r["score"] for r in results]
        agg = aggregate_scores(scores)
        avg_p = sum(r["target_prob"] for r in results) / n
        avg_v = sum(r["v_norm"] for r in results) / n

        print(f"{layer_idx:>5} | {hit_rate:>10.3f} | "
              f"{agg['operation_accuracy']:>11.3f} | "
              f"{agg['operation_selection']:>12.3f} | "
              f"{agg['complete_solution']:>8.3f} | "
              f"{avg_p:>13.3f} | "
              f"{avg_v:>8.2f}")

    print()


if __name__ == "__main__":
    main()
