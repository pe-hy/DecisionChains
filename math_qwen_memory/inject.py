#!/usr/bin/env python
"""Qwen3 thinking-block prefill with algorithm injection.

For each in-think span in the aligned traces, rebuild the prompt with:
  chat-template header (ends at <|im_start|>assistant\\n)
  + "<think>\\n"
  + original think text up to span.start_think
  + "\\n```\\n{algo.pseudo-code}\\n```\\n"

Model continues from there, closes </think>, emits final \\boxed{} answer.

Incremental save + resume: one JSONL record per (ex, algo, span).

Usage:
    python inject.py --aligned outputs/pilot/aime2026_test_n30.aligned.jsonl \\
                     --out outputs/pilot/aime2026_injections.jsonl \\
                     --max-examples 1 --max-new-tokens 2048
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

# Reuse from sibling module
from generate_traces import (  # noqa: E402
    MODEL_ID, SYSTEM_PROMPT,
    compute_token_entropy, decode_with_entropy,
    strip_thinking, extract_boxed, grade,
)


def build_prefill(tok, problem: str, think_text: str,
                  span_start: int, pseudo_code: str):
    """Return (prefill_text, injection_text, input_ids).

    Qwen3-8B template with enable_thinking=True DOES NOT emit <think>;
    we add it explicitly. Verified via template inspection + HF blog.
    """
    import torch  # noqa
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    header = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        enable_thinking=True, tokenize=False,
    )
    injection = f"\n```\n{pseudo_code.strip()}\n```\n"
    prefill = (
        header
        + "<think>\n"
        + think_text[:span_start]
        + injection
    )
    input_ids = tok(prefill, add_special_tokens=False,
                    return_tensors="pt").input_ids
    return prefill, injection, input_ids


def iter_tasks(aligned_path: Path, max_examples: int | None):
    """Yield (ex_idx, algo_idx, algo, span_idx, span, record) for in-think spans."""
    with open(aligned_path) as f:
        for ex_idx, line in enumerate(f):
            if max_examples is not None and ex_idx >= max_examples:
                break
            rec = json.loads(line)
            for algo_idx, algo in enumerate(rec["algorithms"]):
                for span_idx, span in enumerate(algo["spans"]):
                    if span.get("start_think") is None:
                        continue
                    yield ex_idx, algo_idx, algo, span_idx, span, rec


def already_done(out_path: Path) -> set:
    """Set of (ex_idx, algo_idx, span_idx) triples already in output file."""
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["ex_idx"], r["algo_idx"], r["span_idx"]))
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    aligned_path = Path(args.aligned)
    default_out = ROOT / "outputs" / "pilot" / "aime2026_injections.jsonl"
    out_path = Path(args.out) if args.out else default_out
    if args.n_shards > 1:
        out_path = out_path.with_name(
            out_path.stem + f"_shard{args.shard_idx}of{args.n_shards}"
            + out_path.suffix
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)

    print(f"Loading {MODEL_ID} (cache: {HF_CACHE})", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=str(HF_CACHE),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)

    eos_ids = {tok.eos_token_id}
    if tok.pad_token_id is not None:
        eos_ids.add(tok.pad_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end > 0:
        eos_ids.add(im_end)

    done = already_done(out_path)
    if done:
        print(f"Resuming — {len(done)} records already present", flush=True)
    out_f = open(out_path, "a")

    all_tasks = list(iter_tasks(aligned_path, args.max_examples))
    # round-robin shard by global task index
    tasks = [
        t for i, t in enumerate(all_tasks)
        if i % args.n_shards == args.shard_idx
    ]
    remaining = [t for t in tasks if (t[0], t[1], t[3]) not in done]
    print(f"shard {args.shard_idx}/{args.n_shards}: "
          f"{len(remaining)} tasks to run "
          f"(of {len(tasks)} in shard / {len(all_tasks)} total)", flush=True)

    for i, (ex_idx, algo_idx, algo, span_idx, span, rec) in enumerate(remaining):
        problem = rec["problem"]
        gold = rec["gold"]
        think_text = rec["think"]["text"]
        pseudo = algo.get("pseudo-code", "").strip()
        if not pseudo:
            print(f"[{i+1}/{len(remaining)}] skip: no pseudo-code "
                  f"(ex={ex_idx} algo={algo_idx} span={span_idx})", flush=True)
            continue

        prefill, injection, input_ids = build_prefill(
            tok, problem, think_text, span["start_think"], pseudo,
        )
        input_ids = input_ids.to(model.device)

        gen_list, entropies = decode_with_entropy(
            model, tok, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            eos_ids=eos_ids,
        )
        torch.cuda.empty_cache()

        continuation = tok.decode(gen_list, skip_special_tokens=False)
        full_gen = prefill + continuation
        # For parsing: remove all <think>..</think> blocks from the
        # text after the model closes thinking.
        answer_region = strip_thinking(continuation)
        pred = extract_boxed(answer_region)
        correct = grade(pred, gold)

        tokens = [tok.decode([t]) for t in gen_list]
        rec_out = {
            "ex_idx": ex_idx,
            "algo_idx": algo_idx,
            "algo_name": algo.get("name"),
            "span_idx": span_idx,
            "span_start_think": span["start_think"],
            "span_end_think": span["end_think"],
            "original_span_text": span.get("text", ""),
            "injection_text": injection,
            "prefill_text": prefill,
            "continuation": continuation,
            "full_generation": full_gen,
            "answer_region": answer_region,
            "pred": pred,
            "gold": gold,
            "correct": correct,
            "tokens": tokens,
            "token_ids": gen_list,
            "entropies": entropies,
        }
        out_f.write(json.dumps(rec_out) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())

        mean_h = sum(entropies) / max(1, len(entropies))
        print(
            f"[{i+1}/{len(remaining)}] ex={ex_idx} algo={algo_idx} "
            f"span={span_idx} len={len(tokens)} meanH={mean_h:.2f} "
            f"pred={pred!r} gold={gold!r} correct={correct}",
            flush=True,
        )

    out_f.close()
    print(f"Done. Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
