"""
BALANCED Experiment: Data Generation

For each full GT chain of length N, produces 2N examples:
  - N healthy variants:
    - (N-1) partial traces: execute blocks 0..s fully, then "peek" at the
      letter for step s+1. Loss will be masked on the peek letter.
    - 1 full chain (s = N-1): same as extended/stop variant, all blocks,
      no peek, no masking.
  - N unhealthy variants: one per step, wrong block + ; STOP.

Prompt (INPUT/OUTPUT) is identical across all 2N variants — it always
shows the full chain's input and final output vector. "One problem split
into 2N examples."

Only chains where every decision step is eligible (f != g at every step)
are kept, so each accepted GT contributes exactly 2N examples and the
dataset is perfectly balanced at every step.
"""

import sys
import os
import json
import random
import itertools
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ops.transformations import (
    reverse, add_1, double, negate, cumsum, scale_by_first,
    rotate_left, swap_pairs, position_multiply, diff,
    add_last_to_all, square, rotate_right, add_first_to_all, cumsum_reverse,
    prefix_product, sliding_sum, position_add, conditional_double,
    interleave_sum_diff,
    _format_vec,
)

LETTER_TO_FUNC = {
    "a": ("reverse", reverse),
    "b": ("add_1", add_1),
    "c": ("double", double),
    "d": ("negate", negate),
    "e": ("cumsum", cumsum),
    "f": ("scale_by_first", scale_by_first),
    "g": ("rotate_left", rotate_left),
    "h": ("swap_pairs", swap_pairs),
    "i": ("position_multiply", position_multiply),
    "j": ("diff", diff),
    "k": ("add_last_to_all", add_last_to_all),
    "l": ("square", square),
    "m": ("rotate_right", rotate_right),
    "n": ("add_first_to_all", add_first_to_all),
    "o": ("cumsum_reverse", cumsum_reverse),
    "p": ("prefix_product", prefix_product),
    "q": ("sliding_sum", sliding_sum),
    "r": ("position_add", position_add),
    "s": ("conditional_double", conditional_double),
    "t": ("interleave_sum_diff", interleave_sum_diff),
}

LETTERS = list(LETTER_TO_FUNC.keys())
K = 10


def is_even(x):
    return 1 if x % 2 == 0 else 0


def decision_func_f(L):
    return is_even(L[0]) * 10 + L[1]


def decision_func_g(L):
    return is_even(L[3]) * 10 + L[4]


DECISION_FUNCS = {"f": decision_func_f, "g": decision_func_g}
INT_TO_LETTER = {i: chr(ord("a") + i) for i in range(20)}


def resolve_chain(vec, df_choices):
    current = list(vec)
    letters = []
    for df_name in df_choices:
        idx = DECISION_FUNCS[df_name](current)
        letter = INT_TO_LETTER[idx]
        letters.append(letter)
        _, func = LETTER_TO_FUNC[letter]
        current = func(list(current), K, [])
    return tuple(letters)


def all_steps_eligible(vec, letters):
    """True iff f != g at every step — i.e. every step has a valid wrong branch."""
    current = list(vec)
    for letter in letters:
        if decision_func_f(current) == decision_func_g(current):
            return False
        _, func = LETTER_TO_FUNC[letter]
        current = func(list(current), K, [])
    return True


def build_balanced_examples(vec, letters, df_names, input_str):
    """
    Build 2N examples for a single eligible GT chain.
    Returns list of dicts.
    """
    N = len(letters)
    examples = []

    # Intermediate vectors: intermediate[s] = vector before step s
    intermediate = [list(vec)]
    for s in range(N):
        _, func = LETTER_TO_FUNC[letters[s]]
        intermediate.append(func(list(intermediate[-1]), K, []))

    # Correct block traces
    correct_traces = []
    for s in range(N):
        _, func = LETTER_TO_FUNC[letters[s]]
        trace_list = []
        func(list(intermediate[s]), K, trace_list)
        trace_str = trace_list[0].replace(
            LETTER_TO_FUNC[letters[s]][0], letters[s], 1
        )
        correct_traces.append(trace_str)

    # ── Healthy partials: execute blocks 0..s, then peek at letter_{s+1} ──
    for s in range(N - 1):
        blocks = correct_traces[: s + 1]
        peek_letter = letters[s + 1]
        output = " ; ".join(blocks) + " ; " + peek_letter
        examples.append({
            "input": input_str,
            "output": output,
            "letters": list(letters),
            "decision_funcs": list(df_names),
            "is_negative": False,
            "mask_last_n": 1,
            "variant": "healthy_partial",
            "peek_step": s + 1,
        })

    # ── Healthy full chain: all blocks, no peek, no mask ──
    full_output = " ; ".join(correct_traces)
    examples.append({
        "input": input_str,
        "output": full_output,
        "letters": list(letters),
        "decision_funcs": list(df_names),
        "is_negative": False,
        "mask_last_n": 0,
        "variant": "healthy_full",
    })

    # ── Unhealthy: wrong block at step s, then STOP ──
    for s in range(N):
        gt_df = df_names[s]
        wrong_df = "g" if gt_df == "f" else "f"
        wrong_idx = DECISION_FUNCS[wrong_df](intermediate[s])
        wrong_letter = INT_TO_LETTER[wrong_idx]
        assert wrong_letter != letters[s], \
            f"wrong_letter == gt_letter at step {s} — chain not eligible?"

        _, wrong_func = LETTER_TO_FUNC[wrong_letter]
        wt = []
        wrong_func(list(intermediate[s]), K, wt)
        wrong_trace = wt[0].replace(LETTER_TO_FUNC[wrong_letter][0], wrong_letter, 1)

        parts = correct_traces[:s] + [wrong_trace]
        output = " ; ".join(parts) + " ; STOP"
        examples.append({
            "input": input_str,
            "output": output,
            "letters": list(letters[:s]) + [wrong_letter],
            "decision_funcs": list(df_names[:s]) + [wrong_df],
            "is_negative": True,
            "mask_last_n": 0,
            "variant": "unhealthy",
            "mistake_at": s,
            "gt_letter_at_mistake": letters[s],
            "wrong_letter_at_mistake": wrong_letter,
            "gt_chain_length": N,
        })

    return examples


def split_tuples(all_tuples, test_fraction=0.25, seed=42, n_positions=None):
    rng = random.Random(seed)
    tuples_list = list(all_tuples)
    rng.shuffle(tuples_list)

    n_test = int(len(tuples_list) * test_fraction)
    n_train = len(tuples_list) - n_test

    if n_positions is None:
        n_positions = len(tuples_list[0]) if tuples_list else 0

    train_set = set()
    test_set = set()
    covered = [set() for _ in range(n_positions)]

    remaining = list(tuples_list)
    rng.shuffle(remaining)

    still_remaining = []
    for t in remaining:
        needs_coverage = any(t[pos] not in covered[pos] for pos in range(n_positions))
        if needs_coverage and len(train_set) < n_train:
            train_set.add(t)
            for pos in range(n_positions):
                covered[pos].add(t[pos])
        else:
            still_remaining.append(t)

    rng.shuffle(still_remaining)
    for t in still_remaining:
        if len(train_set) < n_train:
            train_set.add(t)
        else:
            test_set.add(t)

    for pos in range(n_positions):
        for letter in LETTERS:
            assert any(t[pos] == letter for t in train_set), \
                f"Letter {letter} missing at position {pos} in train tuples"

    assert len(train_set & test_set) == 0
    return sorted(train_set), sorted(test_set)


def collect_tokens(examples, n_sample=4000):
    tokens = set()
    sample = examples[:n_sample] if len(examples) > n_sample else examples
    for ex in sample:
        for field in ["input", "output"]:
            tokens.update(ex[field].split())
    return sorted(tokens)


def build_tokenizer_json(tokens, output_path):
    vocab = {}
    idx = 0
    for token in sorted(tokens):
        vocab[token] = idx
        idx += 1
    if "[TRACE]" not in vocab:
        vocab["[TRACE]"] = idx
        idx += 1
    special_start = idx
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": special_start + i, "content": name, "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for i, name in enumerate(["[BOS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]"])
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]",
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tokenizer, f, indent=2)
    print(f"Tokenizer saved to {output_path} (vocab_size={special_start + 5})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced healthy+unhealthy decision-chain data"
    )
    parser.add_argument("--train_vectors", type=int, default=95000)
    parser.add_argument("--test_vectors", type=int, default=5000)
    parser.add_argument("--target_train", type=int, default=2_000_000,
                        help="Target number of full GT chains (each expands to 2N examples)")
    parser.add_argument("--target_test", type=int, default=10_000,
                        help="Target number of full GT chains for test")
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--chain_lengths", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--vector_length", type=int, default=6)
    parser.add_argument("--reachable_sample", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    df_names = list(DECISION_FUNCS.keys())

    vec_len = args.vector_length
    all_vectors = list(itertools.product(range(10), repeat=vec_len))
    total_vectors = len(all_vectors)

    print("=" * 60)
    print("BALANCED Experiment: healthy+unhealthy pairs per chain")
    print(f"  vec_len={vec_len}, chain_lengths={args.chain_lengths}")
    print("=" * 60)

    # Reachable tuples
    if args.reachable_sample and args.reachable_sample < total_vectors:
        sample_vecs = random.sample(all_vectors, args.reachable_sample)
    else:
        sample_vecs = all_vectors
    print(f"\nComputing reachable tuples over {len(sample_vecs)} vectors...")
    reachable = {n: set() for n in args.chain_lengths}
    for vec in sample_vecs:
        for n in args.chain_lengths:
            for combo in itertools.product(df_names, repeat=n):
                reachable[n].add(resolve_chain(vec, combo))
    for n in args.chain_lengths:
        print(f"  Length {n}: {len(reachable[n])}/{20 ** n} reachable")

    # Split tuples
    print("\nSplitting tuples per length...")
    train_tuples = {}
    test_tuples = {}
    for n in args.chain_lengths:
        tr, te = split_tuples(reachable[n], args.test_fraction, args.seed + n, n)
        train_tuples[n] = set(tr)
        test_tuples[n] = set(te)
        print(f"  Length {n}: {len(tr)} train tuples, {len(te)} test tuples")

    # Split vectors
    random.shuffle(all_vectors)
    train_vectors = all_vectors[:args.train_vectors]
    test_vectors = all_vectors[args.train_vectors:args.train_vectors + args.test_vectors]
    print(f"\nVectors: {len(train_vectors)} train, {len(test_vectors)} test")

    def gen_chains(target, vectors, tuple_set, seed_offset):
        rng = random.Random(args.seed + seed_offset)
        examples = []
        skipped_ineligible = 0
        kept_gt_count = 0
        per_length_kept = defaultdict(int)
        target_per_length = target // len(args.chain_lengths)
        vec_idx = 0

        for n in args.chain_lengths:
            count = 0
            attempts = 0
            max_attempts = max(len(vectors) * 30, target_per_length * 15)
            while count < target_per_length and attempts < max_attempts:
                vec = vectors[vec_idx % len(vectors)]
                vec_idx += 1
                attempts += 1
                combo = tuple(rng.choice(df_names) for _ in range(n))
                letter_tuple = resolve_chain(vec, combo)
                if letter_tuple not in tuple_set[n]:
                    continue
                if not all_steps_eligible(vec, letter_tuple):
                    skipped_ineligible += 1
                    continue

                # Compute final vec for the prompt
                current = list(vec)
                for letter in letter_tuple:
                    _, func = LETTER_TO_FUNC[letter]
                    current = func(list(current), K, [])
                final_vec = current

                input_str = (f"INPUT : {_format_vec(list(vec))} "
                             f"OUTPUT : {_format_vec(final_vec)}")
                exs = build_balanced_examples(vec, letter_tuple, combo, input_str)
                examples.extend(exs)
                count += 1
                kept_gt_count += 1
                per_length_kept[n] += 1

            if count < target_per_length:
                print(f"  WARNING: length {n} only produced {count}/{target_per_length} "
                      f"GT chains after {attempts} attempts")

        return examples, skipped_ineligible, kept_gt_count, dict(per_length_kept)

    print(f"\nGenerating train (target GT: {args.target_train})...")
    train_data, train_skipped, train_kept_gt, train_per_len = gen_chains(
        args.target_train, train_vectors, train_tuples, 100
    )
    n_partial = sum(1 for ex in train_data if ex["variant"] == "healthy_partial")
    n_full = sum(1 for ex in train_data if ex["variant"] == "healthy_full")
    n_unhealthy = sum(1 for ex in train_data if ex["variant"] == "unhealthy")
    print(f"  Kept GT chains: {train_kept_gt}")
    print(f"  Per-length kept: {train_per_len}")
    print(f"  Skipped ineligible: {train_skipped}")
    print(f"  Total train examples: {len(train_data)}")
    print(f"    healthy_partial: {n_partial}")
    print(f"    healthy_full:    {n_full}")
    print(f"    unhealthy:       {n_unhealthy}")

    print(f"\nGenerating test (target GT: {args.target_test})...")
    test_data, test_skipped, test_kept_gt, test_per_len = gen_chains(
        args.target_test, test_vectors, test_tuples, 2000
    )
    te_partial = sum(1 for ex in test_data if ex["variant"] == "healthy_partial")
    te_full = sum(1 for ex in test_data if ex["variant"] == "healthy_full")
    te_unhealthy = sum(1 for ex in test_data if ex["variant"] == "unhealthy")
    print(f"  Kept GT chains: {test_kept_gt}")
    print(f"  Total test examples: {len(test_data)}")
    print(f"    healthy_partial: {te_partial}")
    print(f"    healthy_full:    {te_full}")
    print(f"    unhealthy:       {te_unhealthy}")

    # Verify disjointness
    for n in args.chain_lengths:
        assert len(train_tuples[n] & test_tuples[n]) == 0
    print("\nTrain/test tuple disjointness verified.")

    # Shuffle and write
    random.shuffle(train_data)
    random.shuffle(test_data)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    out_dir = args.output_dir or os.path.join(
        base_dir, "outputs", "data", "decision_chains_balanced"
    )
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.json")
    val_path = os.path.join(out_dir, "val.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"\n  Wrote {len(train_data)} train examples to {train_path}")
    print(f"  Wrote {len(test_data)} test examples to {val_path}")

    # Tokenizer
    all_samples = train_data[:3000] + test_data[:1000]
    tokens = collect_tokens(all_samples, n_sample=4000)
    tok_path = os.path.join(
        base_dir, "outputs", "tokenizer", "tokenizer_decision_chains_balanced.json"
    )
    build_tokenizer_json(tokens, tok_path)

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tok.unk_token = "[UNK]"
    stop_ids = tok.encode("STOP", add_special_tokens=False)
    assert len(stop_ids) == 1 and stop_ids[0] != tok.unk_token_id, \
        f"STOP token not in vocabulary! Got: {stop_ids}"
    print(f"  STOP token ID: {stop_ids[0]}")

    # Metadata
    meta = {
        "experiment": "balanced",
        "chain_lengths": args.chain_lengths,
        "vector_length": vec_len,
        "k": K,
        "train_full_gt_chains": train_kept_gt,
        "train_total_examples": len(train_data),
        "train_healthy_partial": n_partial,
        "train_healthy_full": n_full,
        "train_unhealthy": n_unhealthy,
        "train_skipped_ineligible": train_skipped,
        "test_full_gt_chains": test_kept_gt,
        "test_total_examples": len(test_data),
        "test_healthy_partial": te_partial,
        "test_healthy_full": te_full,
        "test_unhealthy": te_unhealthy,
        "test_skipped_ineligible": test_skipped,
        "letter_to_function": {k: v[0] for k, v in LETTER_TO_FUNC.items()},
        "decision_functions": {
            "f": "is_even(L[0]) * 10 + L[1]",
            "g": "is_even(L[3]) * 10 + L[4]",
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Samples
    print("\nSample healthy_partial:")
    for ex in [e for e in train_data if e["variant"] == "healthy_partial"][:2]:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    mask_last_n={ex['mask_last_n']}  peek_step={ex['peek_step']}")

    print("\nSample healthy_full:")
    for ex in [e for e in train_data if e["variant"] == "healthy_full"][:2]:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")

    print("\nSample unhealthy:")
    for ex in [e for e in train_data if e["variant"] == "unhealthy"][:2]:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    mistake_at={ex['mistake_at']}, "
              f"gt_letter={ex['gt_letter_at_mistake']}, "
              f"wrong={ex['wrong_letter_at_mistake']}")

    # Sequence length stats
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    lens_by_variant = defaultdict(list)
    for ex in train_data[:3000]:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tok.encode(text, add_special_tokens=False)
        lens_by_variant[ex["variant"]].append(len(ids))
    for v, lens in lens_by_variant.items():
        if lens:
            print(f"\n{v} token lengths: "
                  f"min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}")


if __name__ == "__main__":
    main()
