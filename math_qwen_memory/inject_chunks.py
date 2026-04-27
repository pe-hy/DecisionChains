#!/usr/bin/env python
"""Qwen3 thinking-block prefill with chunk-level rewrite injection.

For each (problem, chunk) pair where the chunk has been rewritten with a
canonical algorithm form, prefill the model with:
    chat-template header
  + concat(items[0..chunk_idx-1].original_text)   # exact original prefix
  + items[chunk_idx].rewritten_text                # algorithm-call form swap

Model continues from there, closes </think>, emits final \\boxed{} answer.
One prompt per (ex_idx, chunk_idx). Incremental save + resume.

Inputs:
  - rewrites dir: outputs/pilot/chunk_rewrites_unified/rewrites_{idx:03d}.json
    Each file has: {idx, n_chunks, model, stats, items[]}
    Each item has: {chunk, original_text, rewritten_text, unification_status,
                    has_algorithm, canonical_name, group_id, ...}
  - baseline JSONL: outputs/pilot/aime2026_test_n30.jsonl (for problem + gold)

Concatenating items[*].original_text reproduces baseline['generation'] exactly,
including the leading "<think>\\n" tag. We do NOT prepend "<think>\\n"
ourselves — chunk 0 already carries it.

Usage:
    python inject_chunks.py
    python inject_chunks.py --shard-idx 0 --n-shards 8
    python inject_chunks.py --max-examples 1   # smoke
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

from generate_traces import (  # noqa: E402
    MODEL_ID, SYSTEM_PROMPT,
    decode_with_entropy,
    strip_thinking, extract_boxed, grade,
)


KEEP_STATUSES = {"grouped", "unmatched"}


def load_baseline(baseline_path: Path) -> dict:
    """idx -> {problem, gold, generation}."""
    out = {}
    with open(baseline_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def iter_tasks(rewrites_dir: Path, baseline: dict, max_examples: int | None):
    """Yield (ex_idx, chunk_idx, item, items) for chunks we want to inject."""
    files = sorted(rewrites_dir.glob("rewrites_*.json"))
    seen_examples = 0
    for fp in files:
        d = json.load(open(fp))
        ex_idx = d["idx"]
        if max_examples is not None and seen_examples >= max_examples:
            break
        seen_examples += 1
        if ex_idx not in baseline:
            print(f"WARN: ex_idx={ex_idx} from {fp.name} not in baseline; skip",
                  flush=True)
            continue
        items = d["items"]
        for chunk_idx, it in enumerate(items):
            if it.get("unification_status") in KEEP_STATUSES:
                yield ex_idx, chunk_idx, it, items


def build_prefill(tok, problem: str, items: list, chunk_idx: int):
    """Return (prefill_text, input_ids).

    prefill = header + concat(items[0..chunk_idx-1].original_text)
              + items[chunk_idx].rewritten_text

    items[0].original_text already starts with "<think>\\n", so no manual tag.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    header = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        enable_thinking=True, tokenize=False,
    )
    prev = "".join(it["original_text"] for it in items[:chunk_idx])
    swap = items[chunk_idx]["rewritten_text"]
    prefill = header + prev + swap
    input_ids = tok(prefill, add_special_tokens=False,
                    return_tensors="pt").input_ids
    return prefill, input_ids


def already_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["ex_idx"], r["chunk_idx"]))
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rewrites-dir",
        default=str(ROOT / "outputs" / "pilot" / "chunk_rewrites_unified"),
    )
    ap.add_argument(
        "--baseline",
        default=str(ROOT / "outputs" / "pilot" / "aime2026_test_n30.jsonl"),
    )
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

    rewrites_dir = Path(args.rewrites_dir)
    baseline_path = Path(args.baseline)
    default_out = ROOT / "outputs" / "pilot" / "aime2026_chunk_injections.jsonl"
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
    torch.cuda.manual_seed_all(args.seed)

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

    baseline = load_baseline(baseline_path)
    print(f"Baseline loaded: {len(baseline)} problems", flush=True)

    all_tasks = list(iter_tasks(rewrites_dir, baseline, args.max_examples))
    tasks = [
        t for i, t in enumerate(all_tasks)
        if i % args.n_shards == args.shard_idx
    ]
    done = already_done(out_path)
    if done:
        print(f"Resuming — {len(done)} records already present", flush=True)
    remaining = [(e, c, it, items) for (e, c, it, items) in tasks
                 if (e, c) not in done]
    print(f"shard {args.shard_idx}/{args.n_shards}: "
          f"{len(remaining)} tasks to run "
          f"(of {len(tasks)} in shard / {len(all_tasks)} total)", flush=True)

    out_f = open(out_path, "a")
    for i, (ex_idx, chunk_idx, item, items) in enumerate(remaining):
        rec = baseline[ex_idx]
        problem = rec["problem"]
        gold = rec["gold"]

        prefill, input_ids = build_prefill(tok, problem, items, chunk_idx)
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
        answer_region = strip_thinking(continuation)
        pred = extract_boxed(answer_region)
        correct = grade(pred, gold)

        tokens = [tok.decode([t]) for t in gen_list]
        rec_out = {
            "ex_idx": ex_idx,
            "chunk_idx": chunk_idx,
            "label": item.get("label"),
            "unification_status": item.get("unification_status"),
            "has_algorithm": item.get("has_algorithm"),
            "cluster": item.get("cluster"),
            "group_id": item.get("group_id"),
            "canonical_name": item.get("canonical_name"),
            "original_call": item.get("original_call"),
            "original_text": item.get("original_text"),
            "rewritten_text": item.get("rewritten_text"),
            "rewrite_source": item.get("rewrite_source"),
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
            f"[{i+1}/{len(remaining)}] ex={ex_idx} chunk={chunk_idx} "
            f"status={item.get('unification_status')} "
            f"len={len(tokens)} meanH={mean_h:.2f} "
            f"pred={pred!r} gold={gold!r} correct={correct}",
            flush=True,
        )

    out_f.close()
    print(f"Done. Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
