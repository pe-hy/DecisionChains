"""
Pass@k evaluation for decision chain models.

Generates k independent completions per test input using temperature
sampling, checks correctness, and computes pass@k for various k values.
Results are broken down by chain length.

Uses the unbiased estimator from Chen et al. (Codex, 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples.

Usage:
    python scripts/eval_pass_at_k.py \\
        --model_path outputs/temp/hf_<model_name> \\
        --tokenizer_path outputs/tokenizer/tokenizer_decision_chains_extended.json \\
        --test_file outputs/data/decision_chains_extended/val.json \\
        --k_values 1,4,16,64,256 \\
        --n_samples 256 \\
        --temperature 0.8 \\
        --batch_size 128 \\
        --output_dir outputs/eval_results/pass_at_k/
"""

import sys
import os
import json
import argparse
import math
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import trange
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluator_chains_extended import extract_input_vector, extract_output_vector
from evaluation.correctness import check_completion_correct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ── Unbiased pass@k estimator ──

def pass_at_k(n, c, k):
    """Unbiased estimator of pass@k.

    Args:
        n: total number of samples per problem
        c: number of correct samples for this problem
        k: k value

    Returns:
        pass@k estimate for this problem (float in [0, 1])
    """
    if n - c < k:
        return 1.0
    # Use log-space for numerical stability with large combinatorics
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # log(C(n-c, k) / C(n, k)) = sum_{i=0}^{k-1} log(n-c-i) - log(n-i)
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


# ── Model / data loading ──

def load_model(model_path):
    log.info(f"Loading model from {model_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    hf_model.cuda()
    hf_model.eval()
    return hf_model


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"
    return tokenizer


def load_test_data(test_file):
    """Load test data and deduplicate by input string.

    Returns list of (input_str, chain_length, input_vec, output_vec).
    """
    with open(test_file) as f:
        raw_data = json.load(f)

    seen = {}
    unique_inputs = []
    for ex in raw_data:
        input_str = ex["input"]
        if input_str not in seen:
            input_vec = extract_input_vector(input_str)
            output_vec = extract_output_vector(input_str)
            chain_len = len(ex["letters"])
            seen[input_str] = len(unique_inputs)
            unique_inputs.append((input_str, chain_len, input_vec, output_vec))

    log.info(f"Loaded {len(raw_data)} examples, {len(unique_inputs)} unique inputs")

    len_counts = defaultdict(int)
    for _, cl, _, _ in unique_inputs:
        len_counts[cl] += 1
    for cl in sorted(len_counts):
        log.info(f"  Length {cl}: {len_counts[cl]} unique inputs")

    return unique_inputs


# ── Generation ──

def build_prompts(unique_inputs, tokenizer, split_str="[TRACE]"):
    """Build token-id prompts: [BOS] input_str [TRACE]."""
    prompts = []
    for input_str, _, _, _ in unique_inputs:
        text = f"{tokenizer.bos_token} {input_str} {split_str}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append(ids)
    return prompts


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, n_samples, temperature,
                     batch_size, max_length=512):
    """Generate n_samples completions per prompt using temperature sampling.

    Returns:
        all_predictions: list[list[str]] — per input, list of n_samples decoded strings
    """
    search_token_id = tokenizer.encode("[TRACE]", add_special_tokens=False)[0]
    n_inputs = len(prompts)

    # Flatten: repeat each prompt n_samples times, generate in batches
    all_predictions = [[] for _ in range(n_inputs)]

    # Process in chunks of prompts to manage memory
    # Each prompt gets n_samples generations
    for input_idx in trange(n_inputs, desc="Inputs"):
        prompt = prompts[input_idx]
        prompt_text = tokenizer.decode(prompt, skip_special_tokens=False)

        # Generate n_samples in batches
        remaining = n_samples
        while remaining > 0:
            cur_batch = min(remaining, batch_size)
            batch_texts = [prompt_text] * cur_batch

            tokenizer.padding_side = "left"
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True
            ).to("cuda")

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                max_length=max_length,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )

            for seq in outputs:
                seq = seq.tolist()
                try:
                    split_idx = seq.index(search_token_id)
                    end_idx = (seq.index(tokenizer.eos_token_id)
                               if tokenizer.eos_token_id in seq else len(seq))
                    pred = tokenizer.decode(
                        seq[split_idx + 1:end_idx], skip_special_tokens=True
                    ).strip()
                except ValueError:
                    pred = ""
                all_predictions[input_idx].append(pred)

            remaining -= cur_batch

    return all_predictions


# ── Evaluation ──

def evaluate_pass_at_k(unique_inputs, all_predictions, k_values):
    """Compute pass@k across all inputs, overall and per chain length.

    Returns:
        results: dict with overall and per-length pass@k
    """
    n_samples = len(all_predictions[0])
    n_inputs = len(unique_inputs)

    # Per-input: count correct samples
    per_input_correct = []
    per_input_length = []
    per_input_details = []

    for i, (input_str, chain_len, input_vec, output_vec) in enumerate(unique_inputs):
        preds = all_predictions[i]
        n_correct = 0
        for pred in preds:
            result = check_completion_correct(pred, input_vec, output_vec)
            if result["correct"]:
                n_correct += 1
        per_input_correct.append(n_correct)
        per_input_length.append(chain_len)
        per_input_details.append({
            "input": input_str,
            "chain_length": chain_len,
            "n_samples": n_samples,
            "n_correct": n_correct,
        })

    # Compute pass@k overall
    overall_pass_at_k = {}
    for k in k_values:
        if k > n_samples:
            continue
        scores = [pass_at_k(n_samples, c, k) for c in per_input_correct]
        overall_pass_at_k[str(k)] = float(np.mean(scores))

    # Compute pass@k per chain length
    lengths = sorted(set(per_input_length))
    per_length_pass_at_k = {}
    per_length_counts = {}
    for length in lengths:
        indices = [j for j, l in enumerate(per_input_length) if l == length]
        per_length_counts[str(length)] = len(indices)
        length_pass_at_k = {}
        for k in k_values:
            if k > n_samples:
                continue
            scores = [pass_at_k(n_samples, per_input_correct[j], k) for j in indices]
            length_pass_at_k[str(k)] = float(np.mean(scores))
        per_length_pass_at_k[str(length)] = length_pass_at_k

    # Saturation: k at which pass@k plateaus (< 1% abs gain from doubling)
    sorted_ks = sorted(k for k in k_values if k <= n_samples)
    saturation_k = sorted_ks[-1] if sorted_ks else None
    for i in range(len(sorted_ks) - 1):
        k1, k2 = sorted_ks[i], sorted_ks[i + 1]
        if k2 == 2 * k1:
            gain = overall_pass_at_k.get(str(k2), 0) - overall_pass_at_k.get(str(k1), 0)
            if gain < 0.01:
                saturation_k = k1
                break

    # Completion stats
    correct_fracs = [c / n_samples for c in per_input_correct]

    results = {
        "n_test_inputs": n_inputs,
        "n_samples_per_input": n_samples,
        "pass_at_k": overall_pass_at_k,
        "pass_at_k_by_length": per_length_pass_at_k,
        "n_inputs_by_length": per_length_counts,
        "saturation_k": saturation_k,
        "completion_stats": {
            "avg_correct_frac": float(np.mean(correct_fracs)),
            "median_correct_frac": float(np.median(correct_fracs)),
            "n_never_correct": int(sum(1 for c in per_input_correct if c == 0)),
            "n_always_correct": int(sum(1 for c in per_input_correct if c == n_samples)),
        },
        "per_input": per_input_details,
    }

    return results


def print_summary(results, temperature):
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print(f"  Pass@k Results  (T={temperature}, "
          f"n={results['n_samples_per_input']}, "
          f"inputs={results['n_test_inputs']})")
    print("=" * 60)

    # Overall
    print(f"\n{'k':>6}  {'pass@k':>8}")
    print("-" * 18)
    for k, v in sorted(results["pass_at_k"].items(), key=lambda x: int(x[0])):
        print(f"{k:>6}  {v:>8.4f}")

    # Per length
    print(f"\n  By chain length:")
    lengths = sorted(results["pass_at_k_by_length"].keys(), key=int)
    header = f"{'k':>6}"
    for l in lengths:
        n = results["n_inputs_by_length"].get(l, "?")
        header += f"  {'len=' + l + f' (n={n})':>18}"
    print(header)
    print("-" * (6 + 20 * len(lengths)))

    all_ks = set()
    for l in lengths:
        all_ks.update(int(k) for k in results["pass_at_k_by_length"][l])
    for k in sorted(all_ks):
        row = f"{k:>6}"
        for l in lengths:
            v = results["pass_at_k_by_length"][l].get(str(k))
            row += f"  {v:>18.4f}" if v is not None else f"  {'N/A':>18}"
        print(row)

    # Stats
    stats = results["completion_stats"]
    sat = results["saturation_k"]
    print(f"\n  Avg correct frac per input: {stats['avg_correct_frac']:.4f}")
    print(f"  Never correct: {stats['n_never_correct']} / {results['n_test_inputs']}")
    print(f"  Always correct: {stats['n_always_correct']} / {results['n_test_inputs']}")
    if sat is not None:
        print(f"  Saturation k: {sat}")
    print("=" * 60 + "\n")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Pass@k evaluation for decision chain models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HF model checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer JSON file")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test/val JSON file")
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16,32,64,128,256",
                        help="Comma-separated k values")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Number of samples to generate per input")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Generation batch size")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for generation")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results/pass_at_k/",
                        help="Output directory for results")
    parser.add_argument("--max_inputs", type=int, default=None,
                        help="Limit number of test inputs (for quick runs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_values = [int(k) for k in args.k_values.split(",")]

    # Load
    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.model_path)
    unique_inputs = load_test_data(args.test_file)
    unique_inputs = unique_inputs[:512]  # TEMP: limit for quick testing

    if args.max_inputs is not None and len(unique_inputs) > args.max_inputs:
        log.info(f"Limiting to {args.max_inputs} inputs (from {len(unique_inputs)})")
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(unique_inputs), args.max_inputs, replace=False)
        unique_inputs = [unique_inputs[i] for i in sorted(indices)]

    # Build prompts
    prompts = build_prompts(unique_inputs, tokenizer)
    log.info(f"Built {len(prompts)} prompts, generating {args.n_samples} samples each")

    # Generate
    all_predictions = generate_samples(
        model, tokenizer, prompts,
        n_samples=args.n_samples,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Evaluate
    results = evaluate_pass_at_k(unique_inputs, all_predictions, k_values)
    results["model_path"] = args.model_path
    results["temperature"] = args.temperature
    results["seed"] = args.seed

    # Print summary
    print_summary(results, args.temperature)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename encodes temperature and n_samples
    fname = f"pass_at_k_T{args.temperature}_n{args.n_samples}.json"
    output_path = output_dir / fname
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
