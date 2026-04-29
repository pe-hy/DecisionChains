"""
exp.py — Unified configurable memory-alignment experiment.

One invocation = one configuration. Results saved to
outputs/experiments/{name}.json for later comparison by compare.py.

Dimensions (all CLI flags):
  --data_filter    ffff | all              (training data slice)
  --ce_mode        dp_only | full_seq
  --dp_target      gt | f                  (override coin-flip choice?)
  --layers         "10" or "6,10"
  --mem_entries    int
  --gate           flag: learnable scalar on mem_out
  --sparsity       none | l2_nondp         (L2 penalty on ||mem_out|| off-DP)
  --sparsity_coeff float
  --n_train, --n_eval, --epochs, --lr, --batch_size, --seed
"""

import argparse, json, os, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from metrics import (
    score_trace, aggregate_scores, extract_input_output,
    decision_letters, parse_trace, apply_letter, LETTERS, LETTER_TO_OP,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / "12l-8h-512d-decision-chains-ext_6_2M"
HF_DIR = CHECKPOINT_DIR / "hf"
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "data" / "decision_chains_extended"
RESULTS_DIR = SCRIPT_DIR / "outputs" / "experiments"

TRACE_ID = 87
BOS_ID = 88
EOS_ID = 92
PAD_ID = 89


# ── Memory ───────────────────────────────────────────────────────────────────

class MemoryAttention(nn.Module):
    def __init__(self, hidden_dim, n_entries, use_gate):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_entries, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if use_gate:
            self.gate = nn.Parameter(torch.ones(1))
            self._gate_trainable = True
        else:
            self.register_buffer("gate", torch.ones(1))
            self._gate_trainable = False
        self.scale = hidden_dim ** -0.5
        self.enabled = True
        self.last_attn = None
        self.last_out = None

    def forward(self, h):
        if not self.enabled:
            self.last_attn = None
            self.last_out = None
            return torch.zeros_like(h)
        q = self.query_proj(h)
        attn = F.softmax(q @ self.keys.T * self.scale, dim=-1)
        out = self.gate * (attn @ self.values)
        self.last_attn = attn
        self.last_out = out
        return out


def register_hooks(model, memory, layer_indices):
    def hook_fn(module, inp, output):
        h = output[0]
        return (h + memory(h),) + output[1:]
    return [model.gpt_neox.layers[i].register_forward_hook(hook_fn)
            for i in layer_indices]


# ── Model + data ─────────────────────────────────────────────────────────────

def convert_checkpoint_if_needed():
    """One-shot LitGPT → HF conversion. No-op once HF_DIR/pytorch_model.bin exists."""
    if (HF_DIR / "pytorch_model.bin").exists():
        return
    print("Converting LitGPT checkpoint to HF format...")
    import shutil
    from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
    from litgpt.utils import copy_config_files
    HF_DIR.mkdir(parents=True, exist_ok=True)
    copy_config_files(source_dir=CHECKPOINT_DIR, out_dir=HF_DIR)
    convert_lit_checkpoint(checkpoint_dir=CHECKPOINT_DIR, output_dir=HF_DIR)
    state_dict = torch.load(HF_DIR / "model.pth")
    torch.save(state_dict, HF_DIR / "pytorch_model.bin")
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json"]:
        src = CHECKPOINT_DIR / f
        if src.exists():
            shutil.copy2(src, HF_DIR / f)
    print(f"  Saved HF checkpoint to {HF_DIR}")


def load_model_and_tokenizer(device):
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


def filter_data(examples, filt):
    if filt == "ffff":
        return [ex for ex in examples
                if all(d == "f" for d in ex.get("decision_funcs", []))]
    elif filt == "all":
        return list(examples)
    else:
        raise ValueError(f"unknown data filter: {filt}")


def prepare_training_data(examples, tokenizer, letter_tids, ce_mode, dp_target):
    """
    Returns (input_ids, attn_mask, labels, dp_mask).
      labels[i, pos] is the target token id at pos (or -100).
      dp_mask[i, pos] marks DP letter positions (1 = DP, 0 = not).

    ce_mode='dp_only': labels only set at DP letter positions.
    ce_mode='full_seq': labels set for every token after [TRACE].
    dp_target='gt': target at DP letter pos = GT letter.
    dp_target='f': target at DP letter pos = f(current_intermediate_vec).

    Intermediate vecs are walked through the GT trace (teacher forcing).
    """
    semi_id = tokenizer.encode(";", add_special_tokens=False)[0]
    processed = []

    for ex in examples:
        vec, _ = extract_input_output(ex["input"])
        if vec is None:
            continue
        ids = tokenizer.encode(
            f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]",
            add_special_tokens=False,
        )
        if TRACE_ID not in ids:
            continue
        trace_pos = ids.index(TRACE_ID)
        dp_positions = [trace_pos] + [
            i for i, t in enumerate(ids) if t == semi_id and i > trace_pos
        ]

        blocks = parse_trace(ex["output"])
        current = list(vec)
        dp_f_letters = {}
        dp_letter_positions = []
        for step, block in enumerate(blocks):
            if step >= len(dp_positions) or current is None or len(current) < 5:
                break
            letter_pos = dp_positions[step] + 1
            if letter_pos >= len(ids):
                break
            lf, _ = decision_letters(current)
            f_tid = letter_tids.get(lf)
            if f_tid is not None:
                dp_f_letters[letter_pos] = f_tid
            dp_letter_positions.append(letter_pos)
            if block.letter in LETTER_TO_OP:
                current, _ = apply_letter(block.letter, current)
            elif block.vec is not None:
                current = block.vec
            else:
                break

        label_dict = {}
        if ce_mode == "full_seq":
            for pos in range(trace_pos + 1, len(ids)):
                label_dict[pos] = ids[pos]
            if dp_target == "f":
                for letter_pos, f_tid in dp_f_letters.items():
                    label_dict[letter_pos] = f_tid
        else:  # dp_only
            for letter_pos, f_tid in dp_f_letters.items():
                if dp_target == "f":
                    label_dict[letter_pos] = f_tid
                else:
                    label_dict[letter_pos] = ids[letter_pos]

        processed.append((ids, label_dict, dp_letter_positions))

    N = len(processed)
    if N == 0:
        raise RuntimeError("No usable training examples after filtering/parsing.")
    max_len = max(len(ids) for ids, _, _ in processed)
    input_ids = torch.full((N, max_len), PAD_ID, dtype=torch.long)
    attn_mask = torch.zeros(N, max_len, dtype=torch.long)
    labels = torch.full((N, max_len), -100, dtype=torch.long)
    dp_mask = torch.zeros(N, max_len, dtype=torch.bool)

    for i, (ids, lab, dp_pos_list) in enumerate(processed):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn_mask[i, :len(ids)] = 1
        for pos, tid in lab.items():
            if pos < max_len:
                labels[i, pos] = tid
        for p in dp_pos_list:
            if p < max_len:
                dp_mask[i, p] = True

    n_supervised = (labels != -100).sum().item()
    n_dp = dp_mask.sum().item()
    print(f"  {N} examples, {n_supervised} supervised tokens, "
          f"{n_dp} DP letter positions ({n_dp / max(N, 1):.1f} per ex)")
    return input_ids, attn_mask, labels, dp_mask


# ── Training ─────────────────────────────────────────────────────────────────

def train_memory(model, memory, input_ids, attn_mask, labels, dp_mask,
                 epochs, batch_size, lr, sparsity_mode, sparsity_coeff,
                 device, vocab_size, wandb_run=None):
    params = [p for p in memory.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    N = input_ids.shape[0]
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(N)
        ce_sum, sp_sum, n_batches = 0.0, 0.0, 0
        dp_norm_sum, other_norm_sum = 0.0, 0.0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            b_ids = input_ids[idx].to(device)
            b_mask = attn_mask[idx].to(device)
            b_lab = labels[idx].to(device)
            b_dp = dp_mask[idx].to(device)

            logits = model(input_ids=b_ids, attention_mask=b_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_lab[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if sparsity_mode == "l2_nondp":
                non_dp = b_mask.bool() & ~b_dp
                if memory.last_out is not None and non_dp.any():
                    sp = memory.last_out[non_dp].norm(dim=-1).mean()
                else:
                    sp = torch.tensor(0.0, device=device)
            else:
                sp = torch.tensor(0.0, device=device)

            loss = ce + sparsity_coeff * sp
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ce_sum += ce.item()
            sp_sum += sp.item()
            n_batches += 1
            with torch.no_grad():
                if memory.last_out is not None:
                    if b_dp.any():
                        dp_norm_sum += memory.last_out[b_dp].norm(dim=-1).mean().item()
                    non_dp = b_mask.bool() & ~b_dp
                    if non_dp.any():
                        other_norm_sum += memory.last_out[non_dp].norm(dim=-1).mean().item()

        gate_val = memory.gate.item() if memory._gate_trainable else 1.0
        entry = {
            "epoch": epoch + 1,
            "ce": ce_sum / n_batches,
            "sparsity": sp_sum / n_batches,
            "out_norm_dp": dp_norm_sum / n_batches,
            "out_norm_other": other_norm_sum / n_batches,
            "gate": gate_val,
        }
        history.append(entry)
        print(f"  epoch {entry['epoch']}/{epochs}  CE={entry['ce']:.4f}  "
              f"sp={entry['sparsity']:.4f}  "
              f"||out||_DP={entry['out_norm_dp']:.3f}  "
              f"||out||_other={entry['out_norm_other']:.3f}  "
              f"gate={entry['gate']:.3f}")
        if wandb_run is not None:
            wandb_run.log({
                "train/ce": entry["ce"],
                "train/sparsity": entry["sparsity"],
                "train/out_norm_dp": entry["out_norm_dp"],
                "train/out_norm_other": entry["out_norm_other"],
                "train/gate": entry["gate"],
                "epoch": entry["epoch"],
            }, step=entry["epoch"])
    return history


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_trace(model, tokenizer, example, device):
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


def evaluate(model, tokenizer, examples, device, desc="Eval"):
    scores = []
    for ex in tqdm(examples, desc=desc):
        vec, ovec = extract_input_output(ex["input"])
        if vec is None:
            continue
        text = generate_trace(model, tokenizer, ex, device)
        scores.append(score_trace(vec, ovec, text))
    return aggregate_scores(scores)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data_filter", choices=["ffff", "all"], default="ffff")
    ap.add_argument("--ce_mode", choices=["dp_only", "full_seq"], default="dp_only")
    ap.add_argument("--dp_target", choices=["gt", "f"], default="f")
    ap.add_argument("--layers", default="10")
    ap.add_argument("--mem_entries", type=int, default=16)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--sparsity", choices=["none", "l2_nondp"], default="l2_nondp")
    ap.add_argument("--sparsity_coeff", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_eval", type=int, default=200,
                    help="Cap on val examples evaluated (both slices).")
    ap.add_argument("--ffff_val_file", default=None,
                    help="Optional path to a dedicated ffff-only val JSON "
                         "(e.g. val_ffff.json). When set, the ffff eval uses "
                         "this file instead of filtering the standard val.json.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true",
                    help="Log to Weights & Biases (requires `pip install wandb` + login).")
    ap.add_argument("--wandb_project", default="memory-experiment")
    ap.add_argument("--wandb_group", default=None,
                    help="Group label, e.g. 'sweep_layers' to cluster runs in the UI.")
    ap.add_argument("--save_memory", action="store_true",
                    help="Save trained memory state_dict + config to "
                         "outputs/experiments/memory_weights/{name}.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Experiment: {args.name} ===")
    print(f"  data={args.data_filter}  ce={args.ce_mode}  dp_target={args.dp_target}")
    print(f"  layers={layers}  mem_entries={args.mem_entries}  gate={args.gate}")
    print(f"  sparsity={args.sparsity} (coeff={args.sparsity_coeff})")
    print(f"  n_train={args.n_train}  n_eval={args.n_eval}  epochs={args.epochs}  "
          f"lr={args.lr}  seed={args.seed}")

    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("--wandb requested but wandb is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.name,
            group=args.wandb_group,
            config=vars(args),
            reinit=True,
        )

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.device)
    letter_tids = build_letter_tid_map(tokenizer)

    print("Loading data...")
    with open(DATA_DIR / "train.json") as f:
        train_all = json.load(f)
    with open(DATA_DIR / "val.json") as f:
        val_all = json.load(f)
    train_examples = filter_data(train_all, args.data_filter)[:args.n_train]
    # ffff eval: prefer dedicated file (2k disjoint ffff examples) if provided,
    # else filter the standard val.json (~663 ffff examples after filter).
    if args.ffff_val_file:
        with open(args.ffff_val_file) as f:
            val_fonly = json.load(f)[:args.n_eval]
    else:
        val_fonly = [ex for ex in val_all
                     if all(d == "f" for d in ex.get("decision_funcs", []))][:args.n_eval]
    val_full = val_all[:args.n_eval]
    print(f"  train: {len(train_examples)} (filter={args.data_filter})")
    print(f"  val ffff: {len(val_fonly)}"
          f"{' (from ' + args.ffff_val_file + ')' if args.ffff_val_file else ''}"
          f"   val full: {len(val_full)}")

    print("Preparing training data...")
    train_ids, train_mask, train_labels, train_dp_mask = prepare_training_data(
        train_examples, tokenizer, letter_tids, args.ce_mode, args.dp_target,
    )

    # Baseline depends only on (val data, n_eval), not on memory config —
    # cache it to disk and reuse across the sweep. Include a tag derived
    # from --ffff_val_file so switching val files invalidates the cache.
    import hashlib
    ffff_tag = ""
    if args.ffff_val_file:
        ffff_tag = "_" + hashlib.md5(
            str(Path(args.ffff_val_file).resolve()).encode()
        ).hexdigest()[:8]
    cache_path = RESULTS_DIR / f"_baseline_cache_n{args.n_eval}{ffff_tag}.json"
    if cache_path.exists():
        print(f"\n── Baseline (cached from {cache_path.name}) ──")
        cache = json.load(open(cache_path))
        baseline_fo = cache["ffff"]
        baseline_fu = cache["full"]
    else:
        print("\n── Baseline ──")
        baseline_fo = evaluate(model, tokenizer, val_fonly, args.device, "BL/ffff")
        baseline_fu = evaluate(model, tokenizer, val_full, args.device, "BL/full")
        with open(cache_path, "w") as f:
            json.dump({"ffff": baseline_fo, "full": baseline_fu, "n_eval": args.n_eval},
                      f, indent=2, default=str)
        print(f"  cached to {cache_path.name}")

    memory = MemoryAttention(
        model.config.hidden_size, args.mem_entries, use_gate=args.gate,
    ).to(args.device)
    n_params = sum(p.numel() for p in memory.parameters() if p.requires_grad)
    print(f"\n── Training ({args.mem_entries} entries, {n_params:,} params, "
          f"layers={layers}) ──")

    handles = register_hooks(model, memory, layers)
    t0 = time.time()
    history = train_memory(
        model, memory,
        train_ids, train_mask, train_labels, train_dp_mask,
        args.epochs, args.batch_size, args.lr,
        args.sparsity, args.sparsity_coeff,
        args.device, model.config.vocab_size,
        wandb_run=wandb_run,
    )
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s")

    print("\n── With memory ──")
    with_mem_fo = evaluate(model, tokenizer, val_fonly, args.device, "MEM/ffff")
    with_mem_fu = evaluate(model, tokenizer, val_full, args.device, "MEM/full")

    result = {
        "name": args.name,
        "config": vars(args),
        "n_train_actual": len(train_examples),
        "n_val_fonly_actual": len(val_fonly),
        "n_val_full_actual": len(val_full),
        "train_time_sec": train_time,
        "trainable_params": n_params,
        "history": history,
        "baseline_fonly": baseline_fo,
        "baseline_full": baseline_fu,
        "memory_fonly": with_mem_fo,
        "memory_full": with_mem_fu,
    }
    save_path = RESULTS_DIR / f"{args.name}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")

    if args.save_memory:
        weights_dir = RESULTS_DIR / "memory_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / f"{args.name}.pt"
        torch.save({
            "state_dict": memory.state_dict(),
            "hidden_dim": memory.keys.shape[1],
            "n_entries": memory.keys.shape[0],
            "use_gate": memory._gate_trainable,
            "layers": layers,
            "args": vars(args),
        }, weights_path)
        print(f"Saved memory weights to {weights_path}")

    # ── Log eval metrics to W&B FIRST (before anything that could crash).
    if wandb_run is not None:
        flat = {"train_time_sec": train_time, "trainable_params": n_params}
        for slice_, m in [("ffff_baseline", baseline_fo),
                          ("full_baseline", baseline_fu),
                          ("ffff_memory", with_mem_fo),
                          ("full_memory", with_mem_fu)]:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"eval/{slice_}/{k}"] = v
        for slice_, bl, me in [("ffff", baseline_fo, with_mem_fo),
                                ("full", baseline_fu, with_mem_fu)]:
            for k in ["operation_accuracy", "f_selection",
                       "full_f_alignment", "complete_solution",
                       "f_or_g_valid_selection"]:
                if k in bl and k in me:
                    flat[f"delta/{slice_}/{k}"] = me[k] - bl[k]
        wandb_run.log(flat)
        wandb_run.summary.update(flat)
        wandb_run.finish()

    print("\n── Summary ──")
    hdr = f"  {'metric':24s}  {'BL/ffff':>8s}  {'MEM/ffff':>8s}  {'Δ':>8s}    {'BL/full':>8s}  {'MEM/full':>8s}  {'Δ':>8s}"
    print(hdr)
    for k in ["operation_accuracy", "f_selection", "full_f_alignment", "complete_solution"]:
        b_fo = baseline_fo.get(k, 0); m_fo = with_mem_fo.get(k, 0)
        b_fu = baseline_fu.get(k, 0); m_fu = with_mem_fu.get(k, 0)
        print(f"  {k:24s}  {b_fo:>8.4f}  {m_fo:>8.4f}  {m_fo-b_fo:>+8.4f}    "
              f"{b_fu:>8.4f}  {m_fu:>8.4f}  {m_fu-b_fu:>+8.4f}")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
