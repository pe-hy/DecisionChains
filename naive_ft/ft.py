"""
ft.py — Naive finetuning baseline for the memory-experiment comparison.

Four variants:
  --method full --mode A:  full FT on ffff-only data with GT labels
  --method full --mode B:  full FT on all data with dp_target=f override
  --method lora --mode A:  LoRA on ffff-only data with GT labels
  --method lora --mode B:  LoRA on all data with dp_target=f override

Output JSON schema is intentionally identical to memory_experiment's exp.py
so ../memory_experiment/compare.py works cross-folder. (The keys
"memory_fonly" / "memory_full" here actually hold finetuned-model eval.)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Reuse data prep + evaluation from memory_experiment
SCRIPT_DIR = Path(__file__).resolve().parent
MEM_EXP_DIR = SCRIPT_DIR.parent / "memory_experiment"
sys.path.insert(0, str(MEM_EXP_DIR))
from exp import (
    prepare_training_data, evaluate, filter_data, build_letter_tid_map,
    load_model_and_tokenizer, convert_checkpoint_if_needed,
    TRACE_ID, BOS_ID, EOS_ID, PAD_ID,
    DATA_DIR as MEM_DATA_DIR,
)

RESULTS_DIR = SCRIPT_DIR / "outputs" / "experiments"
BASELINE_CACHE_SRC = MEM_EXP_DIR / "outputs" / "experiments"


# ── LoRA wrapping ────────────────────────────────────────────────────────────

def wrap_lora(model, rank, alpha, target="query_key_value,dense"):
    """Wrap base model with LoRA adapters on the listed target modules.

    GPT-NeoX uses a fused query_key_value projection and a dense output
    projection per attention layer. Targeting both gives LoRA full coverage
    of the attention block. MLPs are left untouched by default.
    """
    from peft import LoraConfig, get_peft_model
    targets = [t.strip() for t in target.split(",")]
    # Model weights were frozen by load_model_and_tokenizer; LoRA only needs
    # adapter params to be trainable, so this is already correct.
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    return model


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, input_ids, attn_mask, labels, epochs, batch_size, lr,
          weight_decay, warmup_ratio, device, vocab_size, wandb_run=None):
    """Standard AdamW + linear-warmup + cosine-decay on the trainable params."""
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    N = input_ids.shape[0]
    steps_per_epoch = max(1, (N + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    history = []
    model.train()
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(N)
        ce_sum, n_batches = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            b_ids = input_ids[idx].to(device)
            b_mask = attn_mask[idx].to(device)
            b_lab = labels[idx].to(device)

            logits = model(input_ids=b_ids, attention_mask=b_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_lab[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            optim.zero_grad()
            ce.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            step += 1

            ce_sum += ce.item()
            n_batches += 1

        avg_ce = ce_sum / max(1, n_batches)
        entry = {"epoch": epoch + 1, "ce": avg_ce, "lr": sched.get_last_lr()[0]}
        history.append(entry)
        print(f"  epoch {entry['epoch']}/{epochs}  CE={avg_ce:.4f}  lr={entry['lr']:.2e}")
        if wandb_run is not None:
            wandb_run.log({
                "train/ce": avg_ce,
                "train/lr": entry["lr"],
                "epoch": entry["epoch"],
            }, step=entry["epoch"])
    model.eval()
    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_target", default="query_key_value,dense")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="memory-experiment")
    ap.add_argument("--wandb_group", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Map mode → (data_filter, dp_target).
    data_filter = "ffff" if args.mode == "A" else "all"
    dp_target = "gt" if args.mode == "A" else "f"

    print(f"\n=== {args.name} ===")
    print(f"  method={args.method}  mode={args.mode}  (data={data_filter}, dp_target={dp_target})")
    print(f"  n_train={args.n_train}  epochs={args.epochs}  lr={args.lr}  bs={args.batch_size}")
    if args.method == "lora":
        print(f"  lora_rank={args.lora_rank}  lora_alpha={args.lora_alpha}  "
              f"target={args.lora_target}")

    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("--wandb but wandb not installed")
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.name, group=args.wandb_group,
            config=vars(args), reinit=True,
        )

    # ── Load model + tokenizer (unfreezes params; we'll refreeze selectively)
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.device)
    letter_tids = build_letter_tid_map(tokenizer)

    # load_model_and_tokenizer froze params. For full FT: unfreeze all.
    # For LoRA: wrap; the wrapper keeps everything frozen except adapters.
    if args.method == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_target)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,}  /  total: {total:,}  "
          f"({100*trainable/total:.2f}%)")

    # ── Data
    print("Loading data...")
    with open(MEM_DATA_DIR / "train.json") as f:
        train_all = json.load(f)
    with open(MEM_DATA_DIR / "val.json") as f:
        val_all = json.load(f)
    train_examples = filter_data(train_all, data_filter)[:args.n_train]
    val_fonly = [ex for ex in val_all
                 if all(d == "f" for d in ex.get("decision_funcs", []))][:args.n_eval]
    val_full = val_all[:args.n_eval]
    print(f"  train: {len(train_examples)}  (filter={data_filter})")
    print(f"  val ffff: {len(val_fonly)}   val full: {len(val_full)}")

    print("Preparing training data...")
    train_ids, train_mask, train_labels, _ = prepare_training_data(
        train_examples, tokenizer, letter_tids,
        ce_mode="full_seq", dp_target=dp_target,
    )

    # ── Baseline (reuse memory_experiment's cache if present)
    local_cache = RESULTS_DIR / f"_baseline_cache_n{args.n_eval}.json"
    shared_cache = BASELINE_CACHE_SRC / f"_baseline_cache_n{args.n_eval}.json"
    cache = None
    for c in (local_cache, shared_cache):
        if c.exists():
            cache = json.load(open(c)); break
    if cache is not None:
        print(f"\n── Baseline (cached) ──")
        baseline_fo, baseline_fu = cache["ffff"], cache["full"]
    else:
        print(f"\n── Baseline ──")
        baseline_fo = evaluate(model, tokenizer, val_fonly, args.device, "BL/ffff")
        baseline_fu = evaluate(model, tokenizer, val_full,  args.device, "BL/full")
        with open(local_cache, "w") as f:
            json.dump({"ffff": baseline_fo, "full": baseline_fu, "n_eval": args.n_eval},
                      f, indent=2, default=str)
        print(f"  cached to {local_cache.name}")

    # ── Train
    print(f"\n── Training ({args.method}/{args.mode}) ──")
    t0 = time.time()
    history = train(
        model, train_ids, train_mask, train_labels,
        args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.warmup_ratio,
        args.device, model.config.vocab_size if args.method == "full"
                     else model.base_model.config.vocab_size,
        wandb_run=wandb_run,
    )
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s")

    # ── Eval
    print(f"\n── With finetuned model ──")
    ft_fo = evaluate(model, tokenizer, val_fonly, args.device, "FT/ffff")
    ft_fu = evaluate(model, tokenizer, val_full,  args.device, "FT/full")

    # Save JSON (schema matches memory_experiment)
    result = {
        "name": args.name,
        "config": vars(args),
        "n_train_actual": len(train_examples),
        "n_val_fonly_actual": len(val_fonly),
        "n_val_full_actual": len(val_full),
        "train_time_sec": train_time,
        "trainable_params": trainable,
        "history": history,
        "baseline_fonly": baseline_fo,
        "baseline_full":  baseline_fu,
        "memory_fonly":   ft_fo,      # schema-compatible naming
        "memory_full":    ft_fu,      # schema-compatible naming
    }
    save_path = RESULTS_DIR / f"{args.name}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")

    # W&B eval metrics (logged before summary print to survive format errors)
    if wandb_run is not None:
        flat = {"train_time_sec": train_time, "trainable_params": trainable}
        for slice_, m in [("ffff_baseline", baseline_fo),
                          ("full_baseline", baseline_fu),
                          ("ffff_memory",   ft_fo),
                          ("full_memory",   ft_fu)]:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"eval/{slice_}/{k}"] = v
        for slice_, bl, me in [("ffff", baseline_fo, ft_fo),
                                ("full", baseline_fu, ft_fu)]:
            for k in ["operation_accuracy", "f_selection",
                       "full_f_alignment", "complete_solution",
                       "f_or_g_valid_selection"]:
                if k in bl and k in me:
                    flat[f"delta/{slice_}/{k}"] = me[k] - bl[k]
        wandb_run.log(flat)
        wandb_run.summary.update(flat)
        wandb_run.finish()

    # Summary
    print("\n── Summary ──")
    hdr = f"  {'metric':24s}  {'BL/ffff':>8s}  {'FT/ffff':>8s}  {'Δ':>8s}    " \
          f"{'BL/full':>8s}  {'FT/full':>8s}  {'Δ':>8s}"
    print(hdr)
    for k in ["operation_accuracy", "f_selection", "full_f_alignment", "complete_solution"]:
        b_fo = baseline_fo.get(k, 0); m_fo = ft_fo.get(k, 0)
        b_fu = baseline_fu.get(k, 0); m_fu = ft_fu.get(k, 0)
        print(f"  {k:24s}  {b_fo:>8.4f}  {m_fo:>8.4f}  {m_fo-b_fo:>+8.4f}    "
              f"{b_fu:>8.4f}  {m_fu:>8.4f}  {m_fu-b_fu:>+8.4f}")

    if args.save_model:
        model_dir = SCRIPT_DIR / "outputs" / "checkpoints" / args.name
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        print(f"\nModel saved to {model_dir}")


if __name__ == "__main__":
    main()
