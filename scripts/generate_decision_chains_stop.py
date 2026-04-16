"""
STOP Token Experiment: Data Generation

Extends generate_decision_chains_extended.py with negative examples.
For each GT training example, generates additional "wrong branch" examples
where the opposite decision function is used at a specific step, producing
a wrong letter. The wrong step is fully computed, then terminated with STOP.

Decision functions:
  func_f(L) = is_even(L[0]) * 10 + L[1]   (output 0-19)
  func_g(L) = is_even(L[3]) * 10 + L[4]   (output 0-19)

Negative example format:
  correct_prefix ; wrong_step_trace ; STOP

Arguments:
  --mistake_at INT [INT ...]   Explicit 0-indexed steps for mistakes
  --num_mistakes INT           Random: pick N steps per example
  --neg_ratio FLOAT            Max negatives/positives ratio (default 1.0)

If neither --mistake_at nor --num_mistakes given, uses all eligible steps
(where f and g produce different letters), capped by --neg_ratio.
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

# ── Decision functions ──

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
    """Generate a single GT chain example."""
    steps = apply_chain_n(vec, letters)
    final_vec = steps[-1][1]
    traces = " ; ".join(s[0] for s in steps)
    return {
        "input": f"INPUT : {_format_vec(list(vec))} OUTPUT : {_format_vec(final_vec)}",
        "output": traces,
        "letters": list(letters),
        "decision_funcs": list(df_names),
        "is_negative": False,
    }


# ── Negative example generation ──

def extract_input_vector(input_str):
    """Parse input vector from 'INPUT : [ a , b , c , d , e , f ] OUTPUT : ...'"""
    try:
        # Find the first bracket pair
        start = input_str.index("[") + 1
        end = input_str.index("]")
        parts = input_str[start:end].split(",")
        return [int(p.strip()) for p in parts]
    except (ValueError, IndexError):
        return None


def build_negative_example(input_str, gt_letters, gt_df_names, intermediate_vecs, input_vec, mistake_at):
    """
    Build a single negative example with the wrong decision function at step mistake_at.

    Steps 0..mistake_at-1 are CORRECT (copied from GT trace computation).
    Step mistake_at uses the OPPOSITE decision function.
    Then ; STOP.
    """
    vec_before_mistake = intermediate_vecs[mistake_at]

    # Determine wrong decision function (opposite of what GT used)
    gt_df = gt_df_names[mistake_at]
    wrong_df = "g" if gt_df == "f" else "f"

    # Compute wrong letter from wrong decision function
    wrong_idx = DECISION_FUNCS[wrong_df](vec_before_mistake)
    wrong_letter = INT_TO_LETTER[wrong_idx]

    # Safety check: wrong letter must differ from GT letter
    if wrong_letter == gt_letters[mistake_at]:
        return None  # f and g agree at this step

    # Build the correct prefix (steps 0..mistake_at-1)
    if mistake_at > 0:
        correct_steps = apply_chain_n(input_vec, gt_letters[:mistake_at])
        correct_traces = [step[0] for step in correct_steps]
    else:
        correct_traces = []

    # Build the wrong step
    _, wrong_func = LETTER_TO_FUNC[wrong_letter]
    wrong_trace_list = []
    wrong_func(list(vec_before_mistake), K, wrong_trace_list)
    wrong_trace_str = wrong_trace_list[0].replace(
        LETTER_TO_FUNC[wrong_letter][0], wrong_letter, 1
    )

    # Combine into output string
    all_parts = correct_traces + [wrong_trace_str]
    output = " ; ".join(all_parts) + " ; STOP"

    return {
        "input": input_str,  # Same INPUT/OUTPUT prompt as GT
        "output": output,
        "letters": list(gt_letters[:mistake_at]) + [wrong_letter],
        "decision_funcs": list(gt_df_names[:mistake_at]) + [wrong_df],
        "is_negative": True,
        "mistake_at": mistake_at,
        "gt_letter_at_mistake": gt_letters[mistake_at],
        "wrong_letter_at_mistake": wrong_letter,
        "gt_chain_length": len(gt_letters),
    }


def generate_negative_examples(gt_example, mistake_steps_arg, num_mistakes_arg, rng):
    """
    Generate negative examples for a single GT example.

    1. Parse GT to get letters, decision_funcs, input vector
    2. Simulate chain to get intermediate vectors
    3. Find eligible steps (f != g)
    4. Select steps based on mode
    5. Build negative example for each selected step
    """
    letters = gt_example["letters"]
    df_names = gt_example["decision_funcs"]
    input_str = gt_example["input"]
    chain_len = len(letters)

    input_vec = extract_input_vector(input_str)
    if input_vec is None:
        return [], 0

    # Compute intermediate vectors and find eligible steps
    current = list(input_vec)
    intermediate_vecs = [list(current)]  # intermediate_vecs[s] = vector BEFORE step s
    eligible_steps = []

    for s in range(chain_len):
        idx_f = decision_func_f(current)
        idx_g = decision_func_g(current)

        if idx_f != idx_g:
            eligible_steps.append(s)

        # Advance using GT letter
        _, func = LETTER_TO_FUNC[letters[s]]
        current = func(list(current), K, [])
        intermediate_vecs.append(list(current))

    skipped = chain_len - len(eligible_steps)

    if not eligible_steps:
        return [], skipped

    # Select which steps get mistakes
    if mistake_steps_arg is not None:
        selected = [s for s in mistake_steps_arg if s in eligible_steps and s < chain_len]
    elif num_mistakes_arg is not None:
        n = min(num_mistakes_arg, len(eligible_steps))
        selected = sorted(rng.sample(eligible_steps, n))
    else:
        selected = eligible_steps

    # Build negative examples
    negatives = []
    for mistake_at in selected:
        neg = build_negative_example(
            input_str, letters, df_names, intermediate_vecs, input_vec, mistake_at
        )
        if neg is not None:
            negatives.append(neg)

    return negatives, skipped


def generate_all_negatives(gt_train_data, mistake_steps_arg, num_mistakes_arg, neg_ratio, seed):
    """Generate negative examples for all GT training data."""
    rng = random.Random(seed + 1000)
    all_negatives = []
    total_skipped_fg_equal = 0
    neg_per_step = defaultdict(int)

    for gt_ex in gt_train_data:
        negatives, skipped = generate_negative_examples(
            gt_ex, mistake_steps_arg, num_mistakes_arg, rng
        )
        total_skipped_fg_equal += skipped
        for neg in negatives:
            neg_per_step[neg["mistake_at"]] += 1
        all_negatives.extend(negatives)

    # Apply neg_ratio limit
    max_negatives = int(len(gt_train_data) * neg_ratio)
    if len(all_negatives) > max_negatives:
        rng.shuffle(all_negatives)
        all_negatives = all_negatives[:max_negatives]

    return all_negatives, total_skipped_fg_equal, dict(neg_per_step)


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
        description="Generate decision-chain data with STOP negative examples"
    )
    parser.add_argument("--train_vectors", type=int, default=95000)
    parser.add_argument("--test_vectors", type=int, default=5000)
    parser.add_argument("--target_train", type=int, default=1_000_000,
                        help="Target number of GT training examples")
    parser.add_argument("--target_test", type=int, default=10_000,
                        help="Target number of test examples (GT only)")
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

    # STOP-specific arguments
    parser.add_argument("--mistake_at", type=int, nargs="*", default=None,
                        help="Explicit 0-indexed step positions for mistakes. "
                             "E.g. --mistake_at 0 2 4. Takes precedence over --num_mistakes.")
    parser.add_argument("--num_mistakes", type=int, default=None,
                        help="Number of random steps to place mistakes at per example. "
                             "If None, uses all eligible steps.")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                        help="Max ratio of negative to positive examples (default 1.0)")

    args = parser.parse_args()

    random.seed(args.seed)
    df_names = list(DECISION_FUNCS.keys())  # ["f", "g"]

    # 1. Generate all vectors and split
    vec_len = args.vector_length
    all_vectors = list(itertools.product(range(10), repeat=vec_len))
    total_vectors = len(all_vectors)

    print("=" * 60)
    print(f"STOP Token Experiment: Decision chains with negative examples")
    print(f"  vec_len={vec_len}, chain_lengths={args.chain_lengths}")
    print(f"  mistake_at={args.mistake_at}, num_mistakes={args.num_mistakes}, "
          f"neg_ratio={args.neg_ratio}")
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
    train_tuples = {}
    test_tuples = {}
    for n in args.chain_lengths:
        tr, te = split_tuples(
            reachable_per_length[n],
            test_fraction=args.test_fraction,
            seed=args.seed + n,
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

    # 5. Generate GT training data
    print(f"\nGenerating GT train data (target: {args.target_train}, "
          f"uniform over lengths {args.chain_lengths})...")
    gt_train_data = []
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
            combo = tuple(random.choice(df_names) for _ in range(n))
            letter_tuple = resolve_chain(vec, combo)
            if letter_tuple in train_tuples[n]:
                ex = generate_example(vec, letter_tuple, combo)
                gt_train_data.append(ex)
                train_tuple_counts[(n, letter_tuple)] += 1
                length_counts_train[n] += 1
                count += 1
        if count < target_per_length:
            print(f"  WARNING: length {n} only generated {count}/{target_per_length} "
                  f"after {attempts} attempts")

    print(f"  Generated {len(gt_train_data)} GT train examples")

    for n in args.chain_lengths:
        used = sum(1 for (ln, _) in train_tuple_counts if ln == n)
        total = len(train_tuples[n])
        print(f"  Length {n}: {length_counts_train[n]} examples "
              f"({100*length_counts_train[n]/max(len(gt_train_data),1):.1f}%), "
              f"{used}/{total} unique tuples used")

    # 6. Generate GT test data
    print(f"\nGenerating GT test data (target: {args.target_test}, "
          f"uniform over lengths {args.chain_lengths})...")
    gt_test_data = []
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
                gt_test_data.append(ex)
                test_tuple_counts[(n, letter_tuple)] += 1
                length_counts_test[n] += 1
                count += 1

    random.shuffle(gt_test_data)
    print(f"  Generated {len(gt_test_data)} GT test examples")

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

    # 8. Generate negative training examples
    print(f"\nGenerating negative STOP examples from {len(gt_train_data)} GT examples...")
    neg_train_data, skipped_fg_equal, neg_per_step = generate_all_negatives(
        gt_train_data, args.mistake_at, args.num_mistakes, args.neg_ratio, args.seed
    )
    print(f"  Generated {len(neg_train_data)} negative examples")
    print(f"  Skipped steps (f==g): {skipped_fg_equal}")
    print(f"  Per-step distribution: {dict(sorted(neg_per_step.items()))}")
    print(f"  Neg/Pos ratio: {len(neg_train_data)/max(len(gt_train_data),1):.3f}")

    # 8b. Generate negative test examples
    print(f"\nGenerating negative STOP examples from {len(gt_test_data)} GT test examples...")
    neg_test_data, test_skipped_fg_equal, test_neg_per_step = generate_all_negatives(
        gt_test_data, args.mistake_at, args.num_mistakes, args.neg_ratio, args.seed + 2000
    )
    print(f"  Generated {len(neg_test_data)} negative test examples")
    print(f"  Skipped steps (f==g): {test_skipped_fg_equal}")
    print(f"  Per-step distribution: {dict(sorted(test_neg_per_step.items()))}")
    print(f"  Neg/Pos ratio: {len(neg_test_data)/max(len(gt_test_data),1):.3f}")

    # 9. Merge and shuffle training data
    train_data = gt_train_data + neg_train_data
    random.shuffle(train_data)
    print(f"\n  Total train examples: {len(train_data)} "
          f"({len(gt_train_data)} GT + {len(neg_train_data)} negative)")

    # 9b. Merge and shuffle test data
    test_data = gt_test_data + neg_test_data
    random.shuffle(test_data)
    print(f"  Total test examples: {len(test_data)} "
          f"({len(gt_test_data)} GT + {len(neg_test_data)} negative)")

    # 10. Output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    out_dir = args.output_dir or os.path.join(base_dir, "outputs", "data", "decision_chains_stop")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.json")
    val_path = os.path.join(out_dir, "val.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"\n  Wrote {len(train_data)} train examples to {train_path}")

    with open(val_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"  Wrote {len(test_data)} test examples ({len(gt_test_data)} GT + {len(neg_test_data)} neg) to {val_path}")

    # 11. Save metadata
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
        "gt_train_examples": len(gt_train_data),
        "neg_train_examples": len(neg_train_data),
        "total_train_examples": len(train_data),
        "gt_test_examples": len(gt_test_data),
        "neg_test_examples": len(neg_test_data),
        "total_test_examples": len(test_data),
        # STOP-specific metadata
        "stop_experiment": True,
        "mistake_mode": "explicit" if args.mistake_at is not None else
                        ("random" if args.num_mistakes is not None else "all_eligible"),
        "mistake_at_arg": args.mistake_at,
        "num_mistakes_arg": args.num_mistakes,
        "neg_ratio": args.neg_ratio,
        "neg_skipped_fg_equal": skipped_fg_equal,
        "neg_per_step_distribution": neg_per_step,
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote metadata to {meta_path}")

    # 12. Build tokenizer
    all_examples = train_data[:3000] + test_data[:1000]
    tokens = collect_tokens(all_examples, n_sample=4000)
    tok_path = os.path.join(base_dir, "outputs", "tokenizer", "tokenizer_decision_chains_stop.json")
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    build_tokenizer_json(tokens, tok_path)

    # Verify STOP is in tokenizer
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tok.unk_token = "[UNK]"
    stop_ids = tok.encode("STOP", add_special_tokens=False)
    assert len(stop_ids) == 1 and stop_ids[0] != tok.unk_token_id, \
        f"STOP token not in vocabulary! Got IDs: {stop_ids}"
    print(f"  STOP token verified in tokenizer (ID={stop_ids[0]})")

    # 13. Print samples
    print("\nSample GT train examples:")
    gt_samples = [ex for ex in train_data if not ex["is_negative"]][:3]
    for ex in gt_samples:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    letters={ex['letters']} decision={ex['decision_funcs']}")
        print()

    print("Sample NEGATIVE train examples:")
    neg_samples = [ex for ex in train_data if ex["is_negative"]][:5]
    for ex in neg_samples:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    mistake_at={ex['mistake_at']}, "
              f"gt_letter={ex['gt_letter_at_mistake']}, "
              f"wrong_letter={ex['wrong_letter_at_mistake']}")
        print()

    print("Sample GT test examples:")
    gt_test_samples = [ex for ex in test_data if not ex["is_negative"]][:3]
    for ex in gt_test_samples:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    letters={ex['letters']} decision={ex['decision_funcs']}")
        print()

    print("Sample NEGATIVE test examples:")
    neg_test_samples = [ex for ex in test_data if ex["is_negative"]][:5]
    for ex in neg_test_samples:
        print(f"  {ex['input']}")
        print(f"    -> {ex['output']}")
        print(f"    mistake_at={ex['mistake_at']}, "
              f"gt_letter={ex['gt_letter_at_mistake']}, "
              f"wrong_letter={ex['wrong_letter_at_mistake']}")
        print()

    # 14. Sequence length stats
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    lengths_gt = []
    lengths_neg = []
    for ex in train_data[:2000]:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tok.encode(text, add_special_tokens=False)
        if ex["is_negative"]:
            lengths_neg.append(len(ids))
        else:
            lengths_gt.append(len(ids))

    if lengths_gt:
        print(f"\nGT sequence length stats (train sample of {len(lengths_gt)}):")
        print(f"  min={min(lengths_gt)}, max={max(lengths_gt)}, "
              f"mean={sum(lengths_gt)/len(lengths_gt):.1f}")
    if lengths_neg:
        print(f"Negative sequence length stats (train sample of {len(lengths_neg)}):")
        print(f"  min={min(lengths_neg)}, max={max(lengths_neg)}, "
              f"mean={sum(lengths_neg)/len(lengths_neg):.1f}")

    # Test sequence length stats
    lengths_gt_test = []
    lengths_neg_test = []
    for ex in test_data[:2000]:
        text = f"[BOS] {ex['input']} [TRACE] {ex['output']} [EOS]"
        ids = tok.encode(text, add_special_tokens=False)
        if ex["is_negative"]:
            lengths_neg_test.append(len(ids))
        else:
            lengths_gt_test.append(len(ids))

    if lengths_gt_test:
        print(f"GT sequence length stats (test sample of {len(lengths_gt_test)}):")
        print(f"  min={min(lengths_gt_test)}, max={max(lengths_gt_test)}, "
              f"mean={sum(lengths_gt_test)/len(lengths_gt_test):.1f}")
    if lengths_neg_test:
        print(f"Negative sequence length stats (test sample of {len(lengths_neg_test)}):")
        print(f"  min={min(lengths_neg_test)}, max={max(lengths_neg_test)}, "
              f"mean={sum(lengths_neg_test)/len(lengths_neg_test):.1f}")


if __name__ == "__main__":
    main()
