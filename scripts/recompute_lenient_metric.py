"""Post-process labeled_dataset.jsonl to compute a lenient
'output_match_any_length' metric per chain length.

Lenient = ignore predicted chain length mismatch. An example counts as
solved iff:
  1. Every chosen letter is the f-decision letter at that step's vec
  2. Every block's arithmetic is valid (re-applying letter reproduces text)
  3. After applying all chosen letters, final vec equals prompt OUTPUT vec

Writes per-method, per-length stats to outputs/eval_results/figures/lenient_metric.json.
"""

import json
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.evaluator_chains_extended import (
    parse_chain_output_n, apply_and_trace, extract_input_vector,
    LETTER_TO_FUNC, K,
)


def is_even(x): return 1 if x % 2 == 0 else 0
def decision_func_f(L): return is_even(L[0]) * 10 + L[1]
INT_TO_LETTER = {i: chr(ord("a") + i) for i in range(20)}


def extract_output_vector(input_str):
    m = re.search(r"OUTPUT\s*:\s*\[\s*([\d\s,\-]+)\s*\]", input_str)
    if not m:
        return None
    try:
        return [int(x.strip()) for x in m.group(1).split(",")]
    except ValueError:
        return None


def is_lenient_match(input_vec, output_vec, prediction_text):
    """Returns True iff every chosen letter is f's letter, every block is
    arithmetically valid, and final vec equals output_vec."""
    blocks = parse_chain_output_n(prediction_text)
    if not blocks:
        return False
    cur = list(input_vec)
    for b in blocks:
        if b is None or len(cur) < 5:
            return False
        letter = b["letter"]
        f_letter = INT_TO_LETTER[decision_func_f(cur)]
        if letter != f_letter:
            return False
        expected_result, expected_block_text = apply_and_trace(letter, cur)
        if expected_block_text is None or b["block"] != expected_block_text:
            return False
        cur = expected_result
    return cur is not None and list(cur) == list(output_vec)


METHODS = {
    "Pretrained":              "12l-8h-512d-decision-chains-ext_6_2M",
    "GRPO":                    "grpo_12l-8h-512d-decision-chains-ext_6_2M",
    "LoRA mode A":             "ft_for_fig_lora_A",
    "LoRA mode B":             "ft_for_fig_lora_B",
    "Naive FT mode A":         "ft_for_fig_full_A",
    "Naive FT mode B":         "ft_for_fig_full_B",
    "Memory align (L=2, N=32)":"memory_N32_12l-8h-512d-decision-chains-ext_6_2M",
}

results = {}
for method, sub in METHODS.items():
    path = REPO / "outputs/eval_results" / sub / "labeled_dataset.jsonl"
    by_len = defaultdict(lambda: {"hit": 0, "total": 0})
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            cl = int(ex["chain_length"])
            input_vec = extract_input_vector(ex["input"])
            out_vec = extract_output_vector(ex["input"])
            ok = (input_vec is not None
                  and out_vec is not None
                  and is_lenient_match(input_vec, out_vec, ex["prediction"]))
            by_len[cl]["total"] += 1
            if ok:
                by_len[cl]["hit"] += 1
    summary = {str(L): {
        "hit": by_len[L]["hit"],
        "total": by_len[L]["total"],
        "rate": (by_len[L]["hit"] / by_len[L]["total"]) if by_len[L]["total"] else None,
    } for L in sorted(by_len)}
    results[method] = summary
    pretty = "  ".join(f"L{L}: {summary[L]['hit']}/{summary[L]['total']} ({100*summary[L]['rate']:.1f}%)"
                      for L in summary)
    print(f"{method:35s} {pretty}")

out_path = REPO / "outputs/eval_results/figures/lenient_metric.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_path}")
