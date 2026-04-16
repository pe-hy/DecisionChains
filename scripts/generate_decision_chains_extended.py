"""
Extended Decision-Function-Driven Chain Experiment: Data Generation

Like generate_decision_chains.py, but chains 3-5 functions instead of 2.
Input includes both the input AND output (final) vector.

Decision functions:
  func_f(L) = is_even(L[0]) * 10 + L[1]   (output 0-19)
  func_g(L) = is_even(L[3]) * 10 + L[4]   (output 0-19)

For each example:
  1. Pick chain length n uniformly from {3, 4, 5}
  2. At each step: coin toss picks func_f or func_g → apply to current vector → letter
  3. Apply letter's operation → new current vector
  4. Repeat n times

Format:
  input:  "INPUT : [ ... ] OUTPUT : [ ... ]"
  output: "letter1 : trace1 : [vec1] ; letter2 : trace2 : [vec2] ; ..."

Train/test split: function tuples (per length) are split into disjoint sets.
"""

import sys
import os
import json
import random
import itertools
import argparse
from collections import defaultdict

# Add parent dir so we can import local transformations module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ops.transformations import (
    reverse, add_1, double, negate, cumsum, scale_by_first,
    rotate_left, swap_pairs, position_multiply, diff,
    add_last_to_all, square, rotate_right, add_first_to_all, cumsum_reverse,
    prefix_product, sliding_sum, position_add, conditional_double,
    interleave_sum_diff,
    _format_vec,
)

# ── Letter → function mapping (20 functions, a-t) ──
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
K = 10  # modular arithmetic base

# ── Decision functions (from superposition experiment) ──

def is_even(x: int) -> int:
    return 1 if x % 2 == 0 else 0


def decision_func_f(L) -> int:
    """f(L) = is_even(L[0]) * 10 + L[1]  →  0-19"""
    return is_even(L[0]) * 10 + L[1]


def decision_func_g(L) -> int:
    """g(L) = is_even(L[3]) * 10 + L[4]  →  0-19"""
    return is_even(L[3]) * 10 + L[4]


DECISION_FUNCS = {
    "f": decision_func_f,
    "g": decision_func_g,
}

# Integer (0-19) → letter (a-t)
INT_TO_LETTER = {i: chr(ord("a") + i) for i in range(20)}


# ── Chain helpers ──

def apply_chain_n(vec, letters):
    """Apply N functions in sequence.
    Returns list of (trace_str, result_vec) for each step."""
    current = list(vec)
    steps = []
    for letter in letters:
        name, func = LETTER_TO_FUNC[letter]
        trace = []
        result = func(list(current), K, trace)
        trace_str = trace[0].replace(name, letter, 1)
        steps.append((trace_str, list(result)))
        current = result
    return steps


def resolve_chain(vec, df_choices):
    """Given input vec and a sequence of decision function names ('f'/'g'),
    resolve to a tuple of letters by applying decision funcs at each step."""
    current = list(vec)
    letters = []
    for df_name in df_choices:
        idx = DECISION_FUNCS[df_name](current)
        letter = INT_TO_LETTER[idx]
        letters.append(letter)
        # Apply the function to get the next vector
        _, func = LETTER_TO_FUNC[letter]
        current = func(list(current), K, [])
    return tuple(letters)


def generate_example(vec, letters, df_names):
    """Generate a single chain example."""
    steps = apply_chain_n(vec, letters)
    final_vec = steps[-1][1]
    traces = " ; ".join(s[0] for s in steps)
    return {
        "input": f"INPUT : {_format_vec(list(vec))} OUTPUT : {_format_vec(final_vec)}",
        "output": traces,
        "letters": list(letters),
        "decision_funcs": list(df_names),
    }


# ── Tuple splitting ──

def split_tuples(all_tuples, test_fraction=0.25, seed=42, n_positions=None):
    """Split function tuples into train/test, ensuring every letter
    appears at every position in train tuples."""
    rng = random.Random(seed)
    tuples_list = list(all_tuples)
    rng.shuffle(tuples_list)

    n_test = int(len(tuples_list) * test_fraction)
    n_train = len(tuples_list) - n_test

    if n_positions is None:
        n_positions = len(tuples_list[0]) if tuples_list else 0

    train_set = set()
    test_set = set()

    # Track coverage: letter at each position
    covered = [set() for _ in range(n_positions)]

    remaining = list(tuples_list)
    rng.shuffle(remaining)

    # Phase 1: greedily add tuples that cover uncovered (letter, position) combos
    still_remaining = []
    for t in remaining:
        needs_coverage = any(t[pos] not in covered[pos] for pos in range(n_positions))
        if needs_coverage and len(train_set) < n_train:
            train_set.add(t)
            for pos in range(n_positions):
                covered[pos].add(t[pos])
        else:
            still_remaining.append(t)

    # Phase 2: fill train to target size
    rng.shuffle(still_remaining)
    for t in still_remaining:
        if len(train_set) < n_train:
            train_set.add(t)
        else:
            test_set.add(t)

    # Validate coverage
    for pos in range(n_positions):
        for letter in LETTERS:
            assert any(t[pos] == letter for t in train_set), \
                f"Letter {letter} missing at position {pos} in train tuples"

    assert len(train_set & test_set) == 0, "Train/test tuples overlap!"

    return sorted(train_set), sorted(test_set)


# ── Tokenizer builder ──

def collect_tokens(examples, n_sample=2000):
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


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate extended decision-chain data (3-5 functions)"
    )
    parser.add_argument("--train_vectors", type=int, default=95000)
    parser.add_argument("--test_vectors", type=int, default=5000)
    parser.add_argument("--target_train", type=int, default=1_000_000,
                        help="Target number of training examples")
    parser.add_argument("--target_test", type=int, default=10_000,
                        help="Target number of test examples")
    parser.add_argument("--test_fraction", type=float, default=0.25,
                        help="Fraction of function tuples held out for test")
    parser.add_argument("--chain_lengths", type=int, nargs="+", default=[3, 4, 5],
                        help="Allowed chain lengths")
    parser.add_argument("--vector_length", type=int, default=6,
                        help="Length of integer vectors")
    parser.add_argument("--reachable_sample", type=int, default=200_000,
                        help="Vectors to sample for reachable tuple discovery (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    df_names = list(DECISION_FUNCS.keys())  # ["f", "g"]

    # 1. Generate all vectors and split
    vec_len = args.vector_length
    all_vectors = list(itertools.product(range(10), repeat=vec_len))
    total_vectors = len(all_vectors)

    print("=" * 60)
    print(f"Extended Decision Chains: 3-5 function chains (vec_len={vec_len})")
    print("=" * 60)

    # 2. Compute reachable tuples per chain length
    if args.reachable_sample and args.reachable_sample < total_vectors:
        sample_vecs = random.sample(all_vectors, args.reachable_sample)
    else:
        sample_vecs = all_vectors
    print(f"\nComputing reachable tuples over {len(sample_vecs)} vectors "
          f"(total: {total_vectors})...")
    reachable_per_length = {n: set() for n in args.chain_lengths}

    for vec in sample_vecs:
        for n in args.chain_lengths:
            # Enumerate all 2^n coin-toss combos
            for combo in itertools.product(df_names, repeat=n):
                letter_tuple = resolve_chain(vec, combo)
                reachable_per_length[n].add(letter_tuple)

    for n in args.chain_lengths:
        total_possible = 20 ** n
        reached = len(reachable_per_length[n])
        print(f"  Length {n}: {reached}/{total_possible} reachable "
              f"({100*reached/total_possible:.1f}%)")

    # 3. Split tuples per length
    print("\nSplitting tuples per length...")
    train_tuples = {}  # length -> set of tuples
    test_tuples = {}
    for n in args.chain_lengths:
        tr, te = split_tuples(
            reachable_per_length[n],
            test_fraction=args.test_fraction,
            seed=args.seed + n,  # different seed per length
            n_positions=n,
        )
        train_tuples[n] = set(tr)
        test_tuples[n] = set(te)
        print(f"  Length {n}: {len(tr)} train, {len(te)} test tuples")

    # 4. Shuffle and split vectors
    random.shuffle(all_vectors)
    train_vectors = all_vectors[:args.train_vectors]
    test_vectors = all_vectors[args.train_vectors:args.train_vectors + args.test_vectors]
    print(f"\nVectors: {len(train_vectors)} train, {len(test_vectors)} test")

    # 5. Generate training data — uniform length distribution
    #    For each example: pick length uniformly, pick 1 random coin-toss combo,
    #    keep only if the resulting tuple is in train set. Repeat until target.
    print(f"\nGenerating train data (target: {args.target_train}, "
          f"uniform over lengths {args.chain_lengths})...")
    train_data = []
    train_tuple_counts = defaultdict(int)
    length_counts_train = defaultdict(int)

    target_per_length = args.target_train // len(args.chain_lengths)
    vec_idx = 0

    for n in args.chain_lengths:
        count = 0
        attempts = 0
        max_attempts = max(len(train_vectors) * 10, target_per_length * 3)
        while count < target_per_length and attempts < max_attempts:
            vec = train_vectors[vec_idx % len(train_vectors)]
            vec_idx += 1
            attempts += 1
            # Pick 1 random coin-toss combo
            combo = tuple(random.choice(df_names) for _ in range(n))
            letter_tuple = resolve_chain(vec, combo)
            if letter_tuple in train_tuples[n]:
                ex = generate_example(vec, letter_tuple, combo)
                train_data.append(ex)
                train_tuple_counts[(n, letter_tuple)] += 1
                length_counts_train[n] += 1
                count += 1
        if count < target_per_length:
            print(f"  WARNING: length {n} only generated {count}/{target_per_length} "
                  f"after {attempts} attempts")

    random.shuffle(train_data)
    print(f"  Generated {len(train_data)} train examples")

    for n in args.chain_lengths:
        used = sum(1 for (ln, _) in train_tuple_counts if ln == n)
        total = len(train_tuples[n])
        print(f"  Length {n}: {length_counts_train[n]} examples "
              f"({100*length_counts_train[n]/max(len(train_data),1):.1f}%), "
              f"{used}/{total} unique tuples used")

    # 6. Generate test data — uniform length distribution
    print(f"\nGenerating test data (target: {args.target_test}, "
          f"uniform over lengths {args.chain_lengths})...")
    test_data = []
    test_tuple_counts = defaultdict(int)
    length_counts_test = defaultdict(int)

    test_per_length = args.target_test // len(args.chain_lengths)
    vec_idx = 0

    for n in args.chain_lengths:
        count = 0
        attempts = 0
        max_attempts = max(len(test_vectors) * 10, test_per_length * 3)
        while count < test_per_length and attempts < max_attempts:
            vec = test_vectors[vec_idx % len(test_vectors)]
            vec_idx += 1
            attempts += 1
            combo = tuple(random.choice(df_names) for _ in range(n))
            letter_tuple = resolve_chain(vec, combo)
            if letter_tuple in test_tuples[n]:
                ex = generate_example(vec, letter_tuple, combo)
                test_data.append(ex)
                test_tuple_counts[(n, letter_tuple)] += 1
                length_counts_test[n] += 1
                count += 1

    random.shuffle(test_data)
    print(f"  Generated {len(test_data)} test examples")

    for n in args.chain_lengths:
        used = sum(1 for (ln, _) in test_tuple_counts if ln == n)
        total = len(test_tuples[n])
        print(f"  Length {n}: {length_counts_test[n]} examples, "
              f"{used}/{total} unique tuples used")

    # 7. Verify disjointness
    for n in args.chain_lengths:
        overlap = train_tuples[n] & test_tuples[n]
        assert len(overlap) == 0, f"Length {n}: {len(overlap)} overlapping tuples!"
    print("\nTrain/test tuple disjointness verified.")

    # 8. Output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    out_dir = args.output_dir or os.path.join(base_dir, "outputs", "data", "decision_chains_extended")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.json")
    val_path = os.path.join(out_dir, "val.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"\n  Wrote {len(train_data)} train examples to {train_path}")

    with open(val_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"  Wrote {len(test_data)} test examples to {val_path}")

    # 9. Save metadata
    meta = {
        "chain_lengths": args.chain_lengths,
        "train_tuples_per_length": {
            str(n): [list(t) for t in sorted(train_tuples[n])]
            for n in args.chain_lengths
        },
        "test_tuples_per_length": {
            str(n): [list(t) for t in sorted(test_tuples[n])]
            for n in args.chain_lengths
        },
        "reachable_per_length": {
            str(n): len(reachable_per_length[n]) for n in args.chain_lengths
        },
        "letter_to_function": {k: v[0] for k, v in LETTER_TO_FUNC.items()},
        "int_to_letter": INT_TO_LETTER,
        "decision_functions": {
            "f": "is_even(L[0]) * 10 + L[1]",
            "g": "is_even(L[3]) * 10 + L[4]",
        },
        "k": K,
        "vector_length": vec_len,
        "train_vectors": len(train_vectors),
        "test_vectors": len(test_vectors),
        "train_examples": len(train_data),
        "test_examples": len(test_data),
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote metadata to {meta_path}")

    # 10. Build tokenizer
    all_examples = train_data[:3000] + test_data[:1000]
    tokens = collect_tokens(all_examples, n_sample=4000)
    tok_path = os.path.join(base_dir, "outputs", "tokenizer", "tokenizer_decision_chains_extended.json")
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    build_tokenizer_json(tokens, tok_path)

    # 11. Print samples
    print("\nSample train examples:")
    for ex in train_data[:3]:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    letters={ex['letters']} decision={ex['decision_funcs']}")
        print()

    print("Sample test examples:")
    for ex in test_data[:3]:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    letters={ex['letters']} decision={ex['decision_funcs']}")
        print()

    # 12. Sequence length stats
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    lengths = []
    for ex in all_examples[:1000]:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tok.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
    print(f"\nSequence length stats (sample of {len(lengths)}):")
    print(f"  min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    # Per chain length
    for n in args.chain_lengths:
        n_lengths = []
        for ex in all_examples[:1000]:
            if len(ex["letters"]) == n:
                text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
                ids = tok.encode(text, add_special_tokens=False)
                n_lengths.append(len(ids))
        if n_lengths:
            print(f"  Length {n}: min={min(n_lengths)}, max={max(n_lengths)}, "
                  f"mean={sum(n_lengths)/len(n_lengths):.1f} ({len(n_lengths)} samples)")


if __name__ == "__main__":
    main()
