"""Evaluate any HF checkpoint on val_full + val_ffff with the same metrics
used by exp.py. Output JSON in the same shape so make_tables.py picks it up.

Usage:
    python eval_method.py \
        --name grpo \
        --hf_dir /abs/path/to/hf_checkpoint \
        [--ffff_val_file ../outputs/data/decision_chains_extended/val_ffff.json] \
        [--n_eval 2000]
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from exp import (
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    evaluate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf_dir", required=True,
                    help="Path to HF checkpoint dir (must have config.json + weights).")
    ap.add_argument("--tokenizer", default=str(CHECKPOINT_DIR / "tokenizer.json"),
                    help="Tokenizer JSON. Default: shared decision-chains tokenizer.")
    ap.add_argument("--ffff_val_file",
                    default=str(DATA_DIR / "val_ffff.json"))
    ap.add_argument("--full_val_file",
                    default=str(DATA_DIR / "val.json"))
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"\n=== Eval: {args.name} ===")
    print(f"  hf_dir={args.hf_dir}")
    print(f"  n_eval={args.n_eval}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.float32, local_files_only=True,
    )
    model.to(args.device).eval()

    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tok.bos_token, tok.eos_token = "[BOS]", "[EOS]"
    tok.pad_token, tok.unk_token = "[PAD]", "[UNK]"

    print("Loading data...")
    with open(args.full_val_file) as f:
        val_full = json.load(f)[:args.n_eval]
    with open(args.ffff_val_file) as f:
        val_fonly = json.load(f)[:args.n_eval]
    print(f"  val ffff: {len(val_fonly)}   val full: {len(val_full)}")

    print("\n── ffff ──")
    res_fo = evaluate(model, tok, val_fonly, args.device, desc="ffff")
    print("\n── full ──")
    res_fu = evaluate(model, tok, val_full, args.device, desc="full")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "name": args.name,
        "hf_dir": args.hf_dir,
        "n_eval": args.n_eval,
        "memory_fonly": res_fo,
        "memory_full": res_fu,
    }
    save_path = RESULTS_DIR / f"{args.name}.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")

    print("\n── Summary ──")
    for k in ["operation_accuracy", "f_selection",
              "full_f_alignment", "complete_solution"]:
        fo = res_fo.get(k, 0); fu = res_fu.get(k, 0)
        print(f"  {k:24s}  ffff={fo:.4f}  full={fu:.4f}")


if __name__ == "__main__":
    main()
