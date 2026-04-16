"""
Label model trajectories as correct or incorrect based on N-way match.

An example is correct if the model's predicted letter sequence AND computed
vectors match any of the 2^N valid ground-truth traces (all combinations
of decision functions f/g at each step).

Outputs labeled_dataset.jsonl with one JSON object per line.
"""

import json
import itertools
import logging

from evaluation.evaluator_chains_extended import (
    LETTER_TO_FUNC, K, INT_TO_LETTER,
    parse_chain_output_n, extract_input_vector,
)

log = logging.getLogger(__name__)

# ── Decision functions ──

def is_even(x: int) -> int:
    return 1 if x % 2 == 0 else 0


def decision_func_f(L) -> int:
    return is_even(L[0]) * 10 + L[1]


def decision_func_g(L) -> int:
    return is_even(L[3]) * 10 + L[4]


DECISION_FUNCS = {"f": decision_func_f, "g": decision_func_g}


def resolve_chain(vec, df_choices):
    """Resolve a sequence of decision function names to letters."""
    current = list(vec)
    letters = []
    for df_name in df_choices:
        idx = DECISION_FUNCS[df_name](current)
        letter = INT_TO_LETTER[idx]
        letters.append(letter)
        _, func = LETTER_TO_FUNC[letter]
        current = func(list(current), K, [])
    return tuple(letters)


def apply_chain_n(vec, letters):
    """Apply N functions in sequence. Returns list of (trace_str, result_vec)."""
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


def compute_all_gt_traces(vec, chain_length):
    """Compute all 2^N GT traces for an input vector."""
    df_names = list(DECISION_FUNCS.keys())
    results = []
    for combo in itertools.product(df_names, repeat=chain_length):
        letters = resolve_chain(vec, combo)
        steps = apply_chain_n(vec, letters)
        output = " ; ".join(s[0] for s in steps)
        results.append({
            "letters": letters,
            "df_combo": combo,
            "output": output,
        })
    return results


def check_n_way_match(pred, input_vec, chain_length):
    """Check if a prediction matches any of the 2^N valid GT traces.

    Returns (is_correct, matched_gt_or_None).
    """
    gt_traces = compute_all_gt_traces(input_vec, chain_length)

    pred_blocks = parse_chain_output_n(pred)
    pred_letters = (
        tuple(b["letter"] for b in pred_blocks if b is not None)
        if pred_blocks else ()
    )

    for gt in gt_traces:
        if pred_letters == tuple(gt["letters"]):
            gt_blocks = parse_chain_output_n(gt["output"])
            if (pred_blocks and gt_blocks
                    and len(pred_blocks) == len(gt_blocks)):
                all_vecs_match = True
                for pb, gb in zip(pred_blocks, gt_blocks):
                    if (pb is None or gb is None
                            or pb.get("vec") != gb.get("vec")):
                        all_vecs_match = False
                        break
                if all_vecs_match:
                    return True, gt
    return False, None


def label_trajectories(unique_inputs, predictions, output_path):
    """Label every example as correct/incorrect and write to JSONL.

    Args:
        unique_inputs: list of (input_str, chain_length, input_vec)
        predictions: list of prediction strings (same length)
        output_path: path to write labeled_dataset.jsonl
    """
    n_correct = 0
    n_total = len(unique_inputs)

    with open(output_path, "w") as f:
        for i, (input_str, chain_length, input_vec) in enumerate(unique_inputs):
            pred = predictions[i].strip()
            is_correct, matched_gt = check_n_way_match(
                pred, input_vec, chain_length
            )
            if is_correct:
                n_correct += 1

            pred_blocks = parse_chain_output_n(pred)
            gen_letters = (
                [b["letter"] for b in pred_blocks if b is not None]
                if pred_blocks else []
            )

            row = {
                "input": input_str,
                "input_vec": input_vec,
                "chain_length": chain_length,
                "prediction": pred,
                "gen_letters": gen_letters,
                "correct": is_correct,
                "matched_df_combo": (
                    list(matched_gt["df_combo"]) if matched_gt else None
                ),
            }
            f.write(json.dumps(row) + "\n")

    log.info(
        f"Labeled dataset saved to {output_path}: "
        f"{n_correct}/{n_total} correct ({100*n_correct/max(n_total,1):.1f}%)"
    )
    return n_correct, n_total
