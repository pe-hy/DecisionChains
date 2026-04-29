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

DATA_ROOT = ROOT / "data"

# Default system prompt — Qwen3 model-card recommended phrasing for math.
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Per-model defaults from each model's HuggingFace card.
MODELS = {
    "qwen3-8b": {
        "id": "Qwen/Qwen3-8B",
        # Thinking-mode card recipe: T=0.6, top_p=0.95, top_k=20, min_p=0.
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        # 16384 default; bump to 38912 for benchmark-grade math/programming.
        "max_new_tokens": 16384,
        # 8B fits one MI250X GCD (~16GB).
        "device_map": "cuda:0",
    },
    "qwen3.6-27b": {
        "id": "Qwen/Qwen3.6-27B",
        # Thinking-mode default: T=1.0 (general). For coding precision drop to
        # 0.6. Math is general-thinking territory → keep 1.0.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        # 32768 = Qwen3.6's full thinking window from the card.  Empirically
        # 16384 was too short for Jarnik problems — many traces hit the cap
        # mid-think and produced no \boxed{} answer.  At 32k a single 64 GB
        # MI250X GCD is tight: bf16 weights ~54 GB + KV cache (GQA) ~10 GB
        # + framework overhead may OOM.  If `Loaded. VRAM:` reports >58 GB
        # at start, switch device_map to "auto" and request --gpus=2 in
        # sbatch/generate_traces_27b.sbatch.
        "max_new_tokens": 32768,
        "device_map": "cuda:0",
        # Qwen3.6 ships model_type='qwen3_5' which the singularity-bundled
        # transformers does not recognize. Loading the modeling files from the
        # HF repo bypasses the check.
        "trust_remote_code": True,
        "thinking_mode": True,
    },
    "gemma-3-27b": {
        # Largest Gemma 3 IT that fits a single MI250X GCD (~64 GB).  bf16
        # weights ~54 GB + KV cache + overhead is tight; if OOM, drop to
        # google/gemma-3-12b-it.  Gated repo — accept license at
        # https://huggingface.co/google/gemma-3-27b-it before downloading.
        "id": "google/gemma-3-27b-it",
        # Gemma 3 IT card: T=1.0, top_k=64, top_p=0.95.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.0,
        "max_new_tokens": 32768,
        "device_map": "cuda:0",
        # Gemma 3 has no <think>/</think> protocol; just CoT in the surface.
        "thinking_mode": False,
    },
    "gemma-4-26b": {
        # Gemma 4 26B-A4B-it: MoE with 4B active params, 26B total.  bf16
        # weights ~52 GB → fits a single MI250X GCD with KV cache room to
        # spare (vs 31B dense which is too tight at 62 GB).  Gated repo —
        # accept license at https://huggingface.co/google/gemma-4-26B-A4B-it
        # before downloading.
        "id": "google/gemma-4-26B-A4B-it",
        # Gemma card sampling defaults are inherited via generation_config
        # (T=1.0, top_p=0.95, top_k=64).
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.0,
        "max_new_tokens": 32768,
        "device_map": "cuda:0",
        "thinking_mode": False,
    },
}

# Backwards-compat: inject.py / inject_chunks.py import MODEL_ID. Default keeps
# them on Qwen3-8B without code changes.
MODEL_ID = MODELS["qwen3-8b"]["id"]

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
    },
    "jarnik_num": {
        "path": "jarnik/jarnik_numeric.jsonl",
        "train_path": None,
        "q_field": "problem",
        "a_field": "answer",
    },
    "jarnik_boxable": {
        "path": "jarnik/jarnik_boxable.jsonl",
        "train_path": None,
        "q_field": "problem",
        "a_field": "answer",
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


# ---------- Jarnik-specific grading ----------
#
# Jarnik problems have three answer flavours that the strict \boxed{} grader
# above gets wrong:
#   1. Booleans:  gold='yes' / 'no', pred often '\\text{Yes}', '**Yes**',
#                 '\\boxed{Yes}', or no \\boxed{} at all because the model
#                 ran out of think tokens before closing.
#   2. Numbers wrapped in LaTeX context:  gold='$K = 18$', pred='18'.
#   3. Multi-part 'a) yes; b) no' — too fragile to parse robustly without
#      examples-per-format, so we leave those to the strict grader.
#
# `grade_jarnik` extends `grade` with (1) and (2), plus a last-ditch scan of
# the full generation for explicit yes/no answer phrases when pred is None
# or unmatched (recovers truncated-thinking traces).

_JARNIK_BOOLISH = {
    "yes": "yes", "y": "yes", "true": "yes", "affirmative": "yes",
    "right": "yes", "correct": "yes", "possible": "yes", "exists": "yes",
    "no": "no", "n": "no", "false": "no", "negative": "no",
    "wrong": "no", "incorrect": "no", "impossible": "no",
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_text_macros(s: str) -> str:
    """Strip LaTeX text wrappers and surrounding $/$$ for boolean detection."""
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("**", "").replace("*", "")
    s = re.sub(r"^\s*\$+|\$+\s*$", "", s.strip())
    return s.strip().rstrip(".,!?:;")


def _bool_of(s) -> str | None:
    """Map a string answer to canonical 'yes' or 'no', else None."""
    if s is None:
        return None
    return _JARNIK_BOOLISH.get(_strip_text_macros(str(s)).lower())


def _single_num(s) -> float | None:
    """Return the unique number in `s` as a float, else None.

    Strips one level of `\\latex_macro{inner}`, dollars, braces, and equality
    signs first (so `'$K = 18$'` and `'$\\boxed{18}$'` both yield 18.0).
    """
    if s is None:
        return None
    t = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", str(s))
    t = re.sub(r"[\${}=]", " ", t)
    nums = _NUM_RE.findall(t)
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


# Phrase patterns the model uses near the end of a successful trace,
# even when no final \\boxed{} is emitted.  Order matters within a tier:
# more specific phrases first so we don't false-match on substrings.
_YES_PATTERNS = re.compile(
    r"(?:answer\s+is\s*\**\s*yes"
    r"|the\s+answer:?\s*\**\s*yes"
    r"|so\s*,?\s*\**\s*yes\b"
    r"|hence\s*,?\s*\**\s*yes\b"
    r"|therefore\s*,?\s*\**\s*yes\b"
    r"|\*\*yes\*\*"
    r"|\\text\{\s*yes\s*\}"
    r"|\\boxed\{\s*yes\s*\})",
    re.IGNORECASE,
)
_NO_PATTERNS = re.compile(
    r"(?:answer\s+is\s*\**\s*no\b"
    r"|the\s+answer:?\s*\**\s*no\b"
    r"|so\s*,?\s*\**\s*no\b"
    r"|hence\s*,?\s*\**\s*no\b"
    r"|therefore\s*,?\s*\**\s*no\b"
    r"|\*\*no\*\*"
    r"|\\text\{\s*no\s*\}"
    r"|\\boxed\{\s*no\s*\})",
    re.IGNORECASE,
)


def _scan_text_for_bool(text: str) -> str | None:
    """Return 'yes' or 'no' for the LAST explicit yes/no phrase in `text`.

    Scans the post-thinking region if the thinking block was closed; falls
    back to the whole text otherwise (covers traces that hit max_new_tokens
    before emitting `</think>`).  Returns None if neither pattern is found.
    """
    if not text:
        return None
    region = strip_thinking(text) or text
    # Bound the scan to the tail to avoid early "Yes, let's consider..." etc.
    tail = region[-1200:]
    last_yes = None
    for m in _YES_PATTERNS.finditer(tail):
        last_yes = m.start()
    last_no = None
    for m in _NO_PATTERNS.finditer(tail):
        last_no = m.start()
    if last_yes is None and last_no is None:
        return None
    if last_yes is None:
        return "no"
    if last_no is None:
        return "yes"
    return "yes" if last_yes > last_no else "no"


def grade_jarnik(pred, gold, full_text: str | None = None) -> bool:
    """Lenient grader for Jarnik (yes/no, single-number, truncated-thinking).

    Tries strict `grade()` first, then boolean equivalence on (pred, gold),
    then single-number equivalence, then a last-ditch scan of `full_text`
    when `gold` is yes/no — recovers traces that ran out of tokens
    mid-thinking.
    """
    if grade(pred, gold):
        return True

    gold_b = _bool_of(gold)
    pred_b = _bool_of(pred)
    if gold_b is not None and pred_b is not None and pred_b == gold_b:
        return True

    g_num = _single_num(gold)
    p_num = _single_num(pred)
    if g_num is not None and p_num is not None and abs(g_num - p_num) < 1e-9:
        return True

    if gold_b is not None and full_text:
        scanned = _scan_text_for_bool(full_text)
        if scanned == gold_b:
            return True

    return False


def grade_for(dataset: str, pred, gold, full_text=None) -> bool:
    """Dispatch grading to the dataset-appropriate function."""
    if dataset.startswith("jarnik"):
        return grade_jarnik(pred, gold, full_text)
    return grade(pred, gold)


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
                        top_p, top_k, eos_ids, min_p=0.0):
    """Manual decode loop. Stores only scalars — OOM-safe at 16K context.

    Computes entropy on RAW logits at each step; applies temp/top_p/top_k/min_p
    for sampling only. Returns (gen_ids[list[int]], entropies[list[float]]).
    """
    import torch
    from transformers import (
        LogitsProcessorList, TemperatureLogitsWarper,
        TopPLogitsWarper, TopKLogitsWarper, MinPLogitsWarper,
    )
    procs = [
        TemperatureLogitsWarper(temperature),
        TopPLogitsWarper(top_p),
        TopKLogitsWarper(top_k),
    ]
    if min_p > 0.0:
        procs.append(MinPLogitsWarper(min_p))
    warpers = LogitsProcessorList(procs)
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

    mcfg = MODELS[args.model]
    model_id = mcfg["id"]
    # CLI flags override per-model defaults; sentinel None means "use default".
    temperature = mcfg["temperature"] if args.temperature is None else args.temperature
    top_p = mcfg["top_p"] if args.top_p is None else args.top_p
    top_k = mcfg["top_k"] if args.top_k is None else args.top_k
    min_p = mcfg.get("min_p", 0.0) if args.min_p is None else args.min_p
    max_new_tokens = (
        mcfg["max_new_tokens"] if args.max_new_tokens is None else args.max_new_tokens
    )

    model_suffix = f"_{args.model}" if args.model != "qwen3-8b" else ""
    shard_suffix = (
        f"_shard{args.shard_idx}of{args.n_shards}" if args.n_shards > 1 else ""
    )
    traces_suffix = f"_t{args.n_traces}" if args.n_traces > 1 else ""
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "outputs" / "pilot"
        / f"{args.dataset}_{args.split}_n{args.n}{traces_suffix}{model_suffix}{shard_suffix}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device_map = mcfg.get("device_map", "cuda:0")
    trust_remote = mcfg.get("trust_remote_code", False)
    print(f"Loading {model_id} (cache: {HF_CACHE}, device_map={device_map}, "
          f"trust_remote_code={trust_remote})")
    print(f"sampling: T={temperature} top_p={top_p} top_k={top_k} "
          f"min_p={min_p} max_new={max_new_tokens}")
    tok = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(HF_CACHE), trust_remote_code=trust_remote,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(HF_CACHE),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        trust_remote_code=trust_remote,
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
    thinking_mode = mcfg.get("thinking_mode", True)
    print(f"system_prompt: {sys_prompt!r} thinking_mode={thinking_mode}", flush=True)
    for (idx, ex) in examples:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ex["problem"]},
        ]
        # transformers 5.x changed apply_chat_template: even with
        # return_tensors="pt" it returns a BatchEncoding when return_dict
        # defaults to True (Qwen3-style chat templates set this).  We want a
        # plain LongTensor of token ids for decode_with_entropy.
        _ct_kwargs = dict(
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # `enable_thinking=True` is a Qwen3 chat-template kwarg; Gemma's
        # template treats unknown kwargs as Jinja vars and rejects them.
        if thinking_mode:
            _ct_kwargs["enable_thinking"] = True
        _ct_out = tok.apply_chat_template(messages, **_ct_kwargs)
        if hasattr(_ct_out, "input_ids"):
            inputs = _ct_out["input_ids"].to(model.device)
        else:
            inputs = _ct_out.to(model.device)

        # Build EOS set from BOTH tokenizer and model.generation_config.
        # Gemma's tokenizer reports eos=<eos>(1), but its chat template ends turns
        # with <end_of_turn>(106) — only generation_config has the full list
        # [1, 106]. Without 106, decode runs to max_new_tokens spamming filler.
        eos_ids: set[int] = set()
        for src in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
            if src is None:
                continue
            if isinstance(src, (list, tuple)):
                eos_ids.update(int(x) for x in src)
            else:
                eos_ids.add(int(src))
        if tok.pad_token_id is not None:
            eos_ids.add(int(tok.pad_token_id))
        print(f"[eos] stop ids={sorted(eos_ids)} "
              f"({[tok.decode([i]) for i in sorted(eos_ids)]})", flush=True)

        for trace_idx in range(args.n_traces):
            pos += 1
            if (idx, trace_idx) in done_pairs:
                continue
            seed = trace_seed(args.seed, idx, trace_idx)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            gen_list, entropies = decode_with_entropy(
                model, tok, inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                eos_ids=eos_ids,
            )
            torch.cuda.empty_cache()

            gen_text = tok.decode(gen_list, skip_special_tokens=False)
            answer_region = strip_thinking(gen_text)
            pred = extract_boxed(answer_region)
            correct = grade_for(args.dataset, pred, ex["gold"], gen_text)

            tokens = [tok.decode([t]) for t in gen_list]

            rec = {
                "idx": idx,
                "trace_idx": trace_idx,
                "seed": seed,
                "model": args.model,
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
    """Re-grade an existing traces JSONL with the current grader. In place.

    Dataset is inferred from the filename (`<dataset>_<split>_n<N>...jsonl`)
    so that jarnik traces get the lenient grader.  If inference fails the
    strict grader is used.
    """
    p = Path(path)
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # Infer dataset from filename prefix: "jarnik_boxable_test_n27..." → "jarnik_boxable"
    name = p.stem
    dataset = next(
        (k for k in sorted(DATASETS, key=len, reverse=True) if name.startswith(k + "_")),
        "",
    )
    changed = 0
    for r in recs:
        new_correct = grade_for(dataset, r.get("pred"), r["gold"],
                                r.get("generation"))
        if new_correct != r.get("correct"):
            changed += 1
        r["correct"] = new_correct
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / max(1, len(recs))
    print(f"{p.name}: dataset={dataset or '?'} n={len(recs)} "
          f"acc={acc:.1%} changed={changed}")


# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=list(MODELS), default="qwen3-8b",
                    help="model registry key; sampling defaults from card")
    # Sampling args default to None so per-model card defaults apply.
    # Pass an explicit value to override the model default.
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--min_p", type=float, default=None)
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
