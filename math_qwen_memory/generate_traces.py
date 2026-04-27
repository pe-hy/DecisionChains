#!/usr/bin/env python
"""Generate Qwen3 traces on MATH / algebra / AIME2026 with per-token entropies.

Single unified prompt: step-by-step reasoning, final answer in \\boxed{}.
Grader: math-verify (primary) + string-normalization fallback.

Usage:
    python generate_traces.py --dataset math --n 20 --max_new_tokens 8192
    python generate_traces.py --dataset aime2026 --n 30 --max_new_tokens 12288
    python generate_traces.py --rescore outputs/pilot/math_test_n20.jsonl
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

MODEL_ID = "Qwen/Qwen3-8B"
DATA_ROOT = ROOT / "data"

# Default system prompt — Qwen3 model-card recommended phrasing for math.
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Jarník is mostly proof-style problems. Ask the model to produce a short
# Conclusion: block at the end so post-hoc LLM-judging can read the conclusion
# instead of the full multi-thousand-token proof.
JARNIK_SYSTEM_PROMPT = (
    "Please reason step by step. If the problem asks for a numerical or "
    "closed-form answer, put it inside \\boxed{}. End your response with a "
    "section starting with 'Conclusion:' that states, in 1-3 sentences, the "
    "key claim being proved or the final value obtained."
)

DATASETS = {
    "math": {
        "path": "MATH/test_500.jsonl",
        "train_path": "MATH/train_1k.jsonl",
        "q_field": "problem",
        "a_field": "answer",
    },
    "algebra": {
        "path": "algebra__linear_1d/test.jsonl",
        "train_path": "algebra__linear_1d/train_1k.jsonl",
        "q_field": "question",
        "a_field": "answer",
    },
    "aime2026": {
        "path": "AIME_2026/test.jsonl",
        "train_path": None,
        "q_field": "problem",
        "a_field": "answer",
    },
    "jarnik": {
        "path": "jarnik/jarnik_2018to2026.jsonl",
        "train_path": None,
        "q_field": "problem",
        "a_field": "answer",
        "system_prompt": JARNIK_SYSTEM_PROMPT,
    },
}


# ---------- answer extraction ----------

def extract_boxed(text: str):
    idxs = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not idxs:
        return None
    start = idxs[-1]
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


# ---------- grading ----------

_math_verify_cache = {}

def _mv():
    """Lazy import so --rescore works even when generation env has issues."""
    if "fn" not in _math_verify_cache:
        from math_verify import parse, verify
        _math_verify_cache["fn"] = (parse, verify)
    return _math_verify_cache["fn"]


def normalize_str(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    for tok in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ",
                r"\displaystyle", "$"):
        s = s.replace(tok, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s


def grade(pred, gold) -> bool:
    """True iff pred == gold. math-verify first, string-normalize fallback."""
    if pred is None:
        return False
    gold_s, pred_s = str(gold).strip(), str(pred).strip()
    try:
        parse, verify = _mv()
        g = parse(f"${gold_s}$")
        p = parse(f"${pred_s}$")
        if verify(g, p):
            return True
    except Exception:
        pass
    return normalize_str(pred_s) == normalize_str(gold_s)


# ---------- data ----------

def load_examples(dataset: str, n: int, split: str):
    cfg = DATASETS[dataset]
    rel = cfg["path"] if split == "test" else cfg["train_path"]
    if rel is None:
        raise ValueError(f"no {split} split for {dataset}")
    path = DATA_ROOT / rel
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            out.append({
                "problem": ex[cfg["q_field"]],
                "gold": str(ex[cfg["a_field"]]),
            })
    return out


# ---------- generation ----------

def compute_token_entropy(logits):
    """Shannon entropy (nats) of predictive distribution at one step.

    Matches DecisionChains/scripts/inference_decision_chains_extended.py#L201.
    Numerically stable: uses log_softmax, not log(clamp_min(probs)).
    Expects RAW logits.
    """
    import torch
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum().item()


def decode_with_entropy(model, tok, input_ids, *, max_new_tokens, temperature,
                        top_p, top_k, eos_ids):
    """Manual decode loop. Stores only scalars — OOM-safe at 16K context.

    Computes entropy on RAW logits at each step; applies temp/top_p/top_k
    for sampling only. Returns (gen_ids[list[int]], entropies[list[float]]).
    """
    import torch
    from transformers import (
        LogitsProcessorList, TemperatureLogitsWarper,
        TopPLogitsWarper, TopKLogitsWarper,
    )
    warpers = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopPLogitsWarper(top_p),
        TopKLogitsWarper(top_k),
    ])
    gen_ids, entropies = [], []
    attn = torch.ones_like(input_ids)
    past = None
    next_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=next_ids, attention_mask=attn,
                        past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]          # [1, vocab] raw
            past = out.past_key_values
            entropies.append(compute_token_entropy(logits[0]))
            warped = warpers(torch.cat([input_ids, torch.tensor([gen_ids],
                                         device=input_ids.device)], dim=1)
                             if gen_ids else input_ids, logits.clone())
            probs = torch.softmax(warped, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # [1,1]
            tid = nxt.item()
            gen_ids.append(tid)
            if tid in eos_ids:
                break
            next_ids = nxt
            attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
    return gen_ids, entropies


def trace_seed(base_seed: int, idx: int, trace_idx: int) -> int:
    """Deterministic per-(problem, trace) seed.

    Replay key, not a guarantee of bit-identical output: same seed on different
    hardware (A100 vs MI250x) or attn impl can still diverge. 1000 head-room
    per problem covers reasonable n_traces growth.
    """
    return base_seed * 1_000_000 + idx * 1_000 + trace_idx


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_suffix = (
        f"_shard{args.shard_idx}of{args.n_shards}" if args.n_shards > 1 else ""
    )
    traces_suffix = f"_t{args.n_traces}" if args.n_traces > 1 else ""
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "outputs" / "pilot"
        / f"{args.dataset}_{args.split}_n{args.n}{traces_suffix}{shard_suffix}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} (cache: {HF_CACHE})")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    all_examples = load_examples(args.dataset, args.n, args.split)
    # shard: keep indices where i % n_shards == shard_idx
    examples = [
        (i, ex) for i, ex in enumerate(all_examples)
        if i % args.n_shards == args.shard_idx
    ]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(examples)} of "
          f"{len(all_examples)} examples, n_traces={args.n_traces}", flush=True)

    # resume: skip (idx, trace_idx) pairs already saved
    # backward-compat: old single-trace files lack trace_idx → treated as 0
    done_pairs = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_pairs.add((r["idx"], r.get("trace_idx", 0)))
                except Exception:
                    pass
        if done_pairs:
            print(f"resuming — {len(done_pairs)} (idx,trace) already saved",
                  flush=True)

    out_f = open(out_path, "a")
    results = []
    total = len(examples) * args.n_traces
    pos = 0
    sys_prompt = DATASETS[args.dataset].get("system_prompt", SYSTEM_PROMPT)
    print(f"system_prompt: {sys_prompt!r}", flush=True)
    for (idx, ex) in examples:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ex["problem"]},
        ]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to(model.device)

        eos_ids = {tok.eos_token_id}
        if tok.pad_token_id is not None:
            eos_ids.add(tok.pad_token_id)

        for trace_idx in range(args.n_traces):
            pos += 1
            if (idx, trace_idx) in done_pairs:
                continue
            seed = trace_seed(args.seed, idx, trace_idx)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            gen_list, entropies = decode_with_entropy(
                model, tok, inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_ids=eos_ids,
            )
            torch.cuda.empty_cache()

            gen_text = tok.decode(gen_list, skip_special_tokens=False)
            answer_region = strip_thinking(gen_text)
            pred = extract_boxed(answer_region)
            correct = grade(pred, ex["gold"])

            tokens = [tok.decode([t]) for t in gen_list]

            rec = {
                "idx": idx,
                "trace_idx": trace_idx,
                "seed": seed,
                "problem": ex["problem"],
                "gold": ex["gold"],
                "generation": gen_text,
                "answer_region": answer_region,
                "pred": pred,
                "correct": correct,
                "tokens": tokens,
                "token_ids": gen_list,
                "entropies": entropies,
            }
            results.append(rec)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            mean_h = sum(entropies) / max(1, len(entropies))
            print(
                f"[{pos}/{total}] idx={idx} t={trace_idx}/{args.n_traces} "
                f"seed={seed} len={len(tokens)} meanH={mean_h:.2f} "
                f"pred={pred!r} gold={ex['gold']!r} correct={correct}",
                flush=True,
            )

    out_f.close()
    acc = sum(r["correct"] for r in results) / max(1, len(results))
    print(f"Saved {len(results)} traces to {out_path} (accuracy={acc:.1%})")


# ---------- rescore ----------

def rescore(path: str):
    """Re-grade an existing traces JSONL with current grader. In place."""
    p = Path(path)
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    changed = 0
    for r in recs:
        new_correct = grade(r["pred"], r["gold"])
        if new_correct != r["correct"]:
            changed += 1
        r["correct"] = new_correct
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / max(1, len(recs))
    print(f"{p.name}: n={len(recs)} acc={acc:.1%} changed={changed}")


# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", default=None,
                    help="path to existing JSONL to re-grade in place")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-traces", type=int, default=1,
                    help="samples per problem (deterministic per-trace seed)")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore)
    else:
        if not args.dataset:
            ap.error("--dataset required (or use --rescore PATH)")
        generate(args)


if __name__ == "__main__":
    main()
