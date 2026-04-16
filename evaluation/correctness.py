"""
Shared correctness checking for decision chain completions.

A completion is correct if every letter is a valid f/g decision,
every trace block is mathematically valid, and the final vector
matches the target OUTPUT vector.
"""

from evaluation.evaluator_chains_extended import (
    LETTER_TO_FUNC, LETTERS, K, INT_TO_LETTER,
    parse_chain_output_n, apply_and_trace,
    extract_input_vector, extract_output_vector,
    decision_func_f, decision_func_g,
)


def check_completion_correct(generated_text, input_vec, target_output_vec):
    """Check if a single generated completion is correct.

    A completion is correct if:
      1. It parses into trace blocks.
      2. Every letter is one of the two valid choices (from decision
         function f or g applied to the current intermediate vector).
      3. All blocks are mathematically valid (applying the letter's
         transformation to the previous vector produces the stated result).
      4. The final computed vector matches target_output_vec.

    Args:
        generated_text: The model's output string (after [TRACE]).
        input_vec: The INPUT vector (list of ints).
        target_output_vec: The expected OUTPUT vector (list of ints).

    Returns:
        dict with:
          - correct: bool
          - n_blocks: int (number of parsed blocks, 0 if unparseable)
          - n_valid_blocks: int
          - all_blocks_valid: bool
          - final_vec: list[int] | None
          - final_vec_matches: bool
          - all_chose_fg: bool (every letter is a valid f/g choice)
          - chose_fg: list[bool] (per block, whether letter is from f or g)
    """
    result = {
        "correct": False,
        "n_blocks": 0,
        "n_valid_blocks": 0,
        "all_blocks_valid": False,
        "final_vec": None,
        "final_vec_matches": False,
        "all_chose_fg": False,
        "all_chose_f": False,
        "chose_fg": [],
        "chose_f": [],
    }

    if input_vec is None or target_output_vec is None:
        return result

    pred_blocks = parse_chain_output_n(generated_text.strip())
    if pred_blocks is None or len(pred_blocks) == 0:
        return result

    n_blocks = len(pred_blocks)
    result["n_blocks"] = n_blocks

    current_vec = list(input_vec)
    n_valid = 0
    chose_fg = []
    chose_f = []

    for block in pred_blocks:
        if block is None or current_vec is None:
            chose_fg.append(False)
            chose_f.append(False)
            current_vec = None
            continue

        letter = block["letter"]

        # Check if letter is from f or g (and specifically from f)
        if len(current_vec) >= 5:
            idx_f = decision_func_f(current_vec)
            idx_g = decision_func_g(current_vec)
            if idx_f in INT_TO_LETTER and idx_g in INT_TO_LETTER:
                letter_f = INT_TO_LETTER[idx_f]
                letter_g = INT_TO_LETTER[idx_g]
                chose_fg.append(letter == letter_f or letter == letter_g)
                chose_f.append(letter == letter_f)
            else:
                chose_fg.append(False)
                chose_f.append(False)
        else:
            chose_fg.append(False)
            chose_f.append(False)

        # Validate block: recompute and compare
        expected_result, expected_block = apply_and_trace(letter, current_vec)
        if expected_block is not None and block["block"] == expected_block:
            n_valid += 1
            current_vec = expected_result
        elif block["vec"] is not None:
            # Block invalid but has a parseable vector; advance with it
            current_vec = block["vec"]
        else:
            current_vec = None

    result["n_valid_blocks"] = n_valid
    result["all_blocks_valid"] = (n_valid == n_blocks)
    result["final_vec"] = list(current_vec) if current_vec is not None else None
    result["chose_fg"] = chose_fg
    result["all_chose_fg"] = bool(chose_fg) and all(chose_fg)
    result["chose_f"] = chose_f
    result["all_chose_f"] = bool(chose_f) and all(chose_f)

    if current_vec is not None:
        result["final_vec_matches"] = (list(current_vec) == list(target_output_vec))

    # Correct = valid decision at every step + valid computation + final vec matches
    result["correct"] = (
        result["all_chose_fg"]
        and result["all_blocks_valid"]
        and result["final_vec_matches"]
    )

    return result
