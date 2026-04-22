"""
generate_val_ffff.py

Generate a dedicated ffff-only val file disjoint from existing train.json.

Approach (more robust than re-deriving the tuple split):
  1. Load the existing train.json, collect every (letter-tuple) that appears
     in any training example.
  2. Reproduce the vector train/test split from the main generator using the
     same seed (this IS deterministic since all_vectors is a fixed
     itertools.product order).
  3. For each test-side vector and chain length n ∈ {3,4,5}, compute
     resolve_chain(vec, ("f",)*n) — i.e. the all-f letter tuple.
  4. Keep the example if its letter tuple does NOT appear in train_tuples.
  5. Targets ~2000 examples balanced across chain lengths.

This guarantees disjointness with the actual train.json, regardless of
hash-ordering quirks in split_tuples.
"""

import argparse
import itertools
import json
import os
import random
from collections import defaultdict

from generate_decision_chains_extended import (
    DECISION_FUNCS, resolve_chain, generate_example,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--train_vectors", type=int, default=95000)
    ap.add_argument("--test_vectors", type=int, default=5000)
    ap.add_argument("--chain_lengths", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--vector_length", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "outputs", "data", "decision_chains_extended")
    train_path = os.path.join(data_dir, "train.json")

    print(f"Loading train set: {train_path}")
    with open(train_path) as f:
        train = json.load(f)
    print(f"  {len(train)} train examples")

    train_tuples = {n: set() for n in args.chain_lengths}
    for ex in train:
        letters = tuple(ex["letters"])
        n = len(letters)
        if n in train_tuples:
            train_tuples[n].add(letters)
    for n in args.chain_lengths:
        print(f"  train length {n}: {len(train_tuples[n])} unique letter-tuples")

    # Reproduce vector split (deterministic, independent of set hashing).
    vec_len = args.vector_length
    all_vectors = list(itertools.product(range(10), repeat=vec_len))
    random.shuffle(all_vectors)
    test_vectors = all_vectors[args.train_vectors:
                               args.train_vectors + args.test_vectors]
    print(f"\nTest-side vectors: {len(test_vectors)}")

    # Generate ffff-only candidates per length, keep only those whose letter
    # tuple is NOT in the train set.
    target_per_length = args.target // len(args.chain_lengths)
    print(f"\nGenerating ffff candidates (target {target_per_length} per length)...")
    examples = []
    stats = {}
    rng = random.Random(args.seed + 1000)

    for n in args.chain_lengths:
        ffff_combo = ("f",) * n
        shuffled_test = list(test_vectors)
        rng.shuffle(shuffled_test)

        count = 0
        skipped_overlap = 0
        for vec in shuffled_test:
            if count >= target_per_length:
                break
            letter_tuple = resolve_chain(vec, ffff_combo)
            if letter_tuple in train_tuples[n]:
                skipped_overlap += 1
                continue
            ex = generate_example(vec, letter_tuple, ffff_combo)
            examples.append(ex)
            count += 1
        stats[n] = (count, skipped_overlap)
        print(f"  Length {n}: {count} kept, {skipped_overlap} skipped "
              f"(letter-tuple in train)")

    rng.shuffle(examples)
    print(f"\nTotal: {len(examples)} ffff-only examples")

    # Verify disjointness
    gen_tuples_by_len = defaultdict(set)
    for ex in examples:
        gen_tuples_by_len[len(ex["letters"])].add(tuple(ex["letters"]))
    print("\nDisjointness check:")
    for n in args.chain_lengths:
        overlap = gen_tuples_by_len[n] & train_tuples[n]
        print(f"  Length {n}: {len(gen_tuples_by_len[n])} unique val tuples, "
              f"{len(overlap)} in train")
        assert len(overlap) == 0, f"length {n} overlaps!"

    out_path = args.output or os.path.join(data_dir, "val_ffff.json")
    with open(out_path, "w") as f:
        json.dump(examples, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
