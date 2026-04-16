"""
Focused tests for validate.py.

These tests cover the pure-Python pieces that are easy to get wrong and
painful to debug via end-to-end runs: the trace-block generator for each
letter, the parser, the per-example scorer, the letter-position locator, and
the intervened-prefix builder.

Run with:
    python -m pytest validate_pretrained/test_validate.py -v
"""

import json
from pathlib import Path

import pytest

from validate_pretrained.validate import (
    Block,
    LETTERS,
    LETTER_TO_OP,
    apply_letter,
    aggregate_scores,
    build_intervened_prefix,
    decision_f,
    decision_g,
    decision_letters,
    extract_input_output,
    find_letter_position,
    parse_generated_blocks,
    score_example,
)


# ---------------------------------------------------------------------------
# Transformations / decision functions
# ---------------------------------------------------------------------------

def test_all_letters_round_trip_with_val_json():
    """Re-executing every ground-truth step in val.json must reproduce the
    exact trace text and result vector. This is the strongest possible check
    that our inlined transformations match the ones used for data generation.
    """
    val_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "data"
        / "decision_chains_extended"
        / "val.json"
    )
    if not val_path.exists():
        pytest.skip(f"val.json not available at {val_path}")

    with open(val_path) as f:
        data = json.load(f)

    # Only check a subset to keep the test fast.
    sample = data[:500]
    for ex in sample:
        inp, _ = extract_input_output(ex["input"])
        assert inp is not None

        current = list(inp)
        gt_blocks = ex["output"].split(" ; ")
        assert len(gt_blocks) == len(ex["letters"])

        for letter, gt_block in zip(ex["letters"], gt_blocks):
            new_vec, block = apply_letter(letter, current)
            assert block == gt_block, (
                f"mismatch for letter {letter!r}\n"
                f"  expected: {gt_block}\n  got:      {block}"
            )
            current = new_vec


def test_decision_functions_cover_full_range():
    assert decision_f([0, 0, 0, 0, 0, 0]) == 10  # even L[0]=0 -> 10 + 0
    assert decision_f([1, 5, 0, 0, 0, 0]) == 5   # odd L[0]  -> 0 + 5
    assert decision_g([0, 0, 0, 2, 7, 0]) == 17  # even L[3]=2 -> 10 + 7
    assert decision_g([0, 0, 0, 1, 3, 0]) == 3   # odd L[3]  -> 0 + 3

    lf, lg = decision_letters([2, 4, 0, 3, 7, 0])
    # f: even 2 * 10 + 4 = 14 -> 'o'; g: odd 3 * 10 + 7 = 7 -> 'h'
    assert lf == "o"
    assert lg == "h"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_extract_input_output():
    s = "INPUT : [ 4 , 3 , 8 , 7 , 7 , 4 ] OUTPUT : [ 2 , 0 , 8 , 4 , 3 , 9 ]"
    inp, out = extract_input_output(s)
    assert inp == [4, 3, 8, 7, 7, 4]
    assert out == [2, 0, 8, 4, 3, 9]


def test_parse_blocks_mixed_formats():
    text = (
        "n : 4 + 4 = 8 , 3 + 4 = 7 , 8 + 4 = 2 , 7 + 4 = 1 , 7 + 4 = 1 , 4 + 4 = 8 "
        "R [ 8 , 7 , 2 , 1 , 1 , 8 ] ; "
        "m R [ 8 , 8 , 7 , 2 , 1 , 1 ] ; "
        "a R [ 1 , 1 , 2 , 7 , 8 , 8 ]"
    )
    blocks = parse_generated_blocks(text)
    assert len(blocks) == 3
    assert [b.letter for b in blocks] == ["n", "m", "a"]
    assert blocks[0].vec == [8, 7, 2, 1, 1, 8]
    assert blocks[1].vec == [8, 8, 7, 2, 1, 1]
    assert blocks[2].vec == [1, 1, 2, 7, 8, 8]


def test_parse_blocks_malformed_block_keeps_slot():
    """A garbage block must not crash parsing; it should produce an empty letter."""
    blocks = parse_generated_blocks("q : 1 + 1 = 2 R [ 2 , 2 ] ; GARBAGE")
    assert len(blocks) == 2
    assert blocks[0].letter == "q"
    assert blocks[1].letter == ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _trace_from_letters(start: list[int], letters: list[str]) -> str:
    cur = list(start)
    pieces = []
    for lt in letters:
        new, block = apply_letter(lt, cur)
        pieces.append(block)
        cur = new
    return " ; ".join(pieces), cur


def test_score_example_ground_truth_is_fully_correct():
    """A trace we generate ourselves from the same transformations must score
    as fully correct, provided the letters are valid f/g choices."""
    # Pick an input whose first letter matches f(v) so selection is valid.
    v = [2, 4, 0, 3, 7, 0]
    lf, _ = decision_letters(v)  # 'o'
    # Build a single-step trace with that letter.
    trace_text, final = _trace_from_letters(v, [lf])
    blocks = parse_generated_blocks(trace_text)
    score = score_example(v, final, blocks)

    assert score.n_steps == 1
    assert score.op_correct == 1
    assert score.sel_correct == 1
    assert score.chain_matches_output is True
    assert score.full_correct is True


def test_score_example_wrong_letter_breaks_selection_not_process():
    """If we force a letter that isn't f or g, the model's own trace for that
    letter is still a valid operation (process+result correct), but selection
    at that step is wrong and the final vector no longer matches OUTPUT."""
    v = [2, 4, 0, 3, 7, 0]
    lf, lg = decision_letters(v)
    # Pick a letter that is neither f nor g.
    bad = next(lt for lt in LETTERS if lt not in (lf, lg))

    # Fake trace: the bad letter, correctly executed.
    trace_text, final = _trace_from_letters(v, [bad])
    blocks = parse_generated_blocks(trace_text)

    # Score against the *original* correct-letter output.
    _, correct_final = _trace_from_letters(v, [lf])
    score = score_example(v, correct_final, blocks)

    assert score.op_correct == 1  # process and result are self-consistent
    assert score.sel_correct == 0  # letter was not selected by f or g
    assert score.chain_matches_output is False
    assert score.full_correct is False


def test_score_example_tampered_result_vector():
    """Flipping the result vector while keeping letter+process must drop the
    op_correct counter for that step."""
    v = [2, 4, 0, 3, 7, 0]
    lf, _ = decision_letters(v)
    trace_text, final = _trace_from_letters(v, [lf])
    # Tamper: replace the trailing result vector with zeros.
    tampered = trace_text.rsplit("[", 1)[0] + "[ 0 , 0 , 0 , 0 , 0 , 0 ]"
    blocks = parse_generated_blocks(tampered)
    score = score_example(v, final, blocks)

    assert score.op_correct == 0
    assert score.sel_correct == 1  # letter selection is still valid
    assert score.full_correct is False


def test_aggregate_scores_has_all_sections():
    a = score_example(*_score_fixture([2, 4, 0, 3, 7, 0], n_steps=2), gt_chain_length=2)
    b = score_example(*_score_fixture([0, 2, 0, 4, 6, 0], n_steps=3), gt_chain_length=3)
    agg = aggregate_scores([a, b])

    assert set(agg.keys()) == {"overall", "by_length", "by_step", "by_letter"}

    overall = agg["overall"]
    assert overall["num_examples"] == 2
    assert overall["total_predicted_ops"] == 5
    assert overall["operation_accuracy"] == 1.0
    assert overall["operation_selection"] == 1.0
    assert overall["complete_solution"] == 1.0
    assert overall["length_matches_gt"] == 1.0

    # Per-length breakdown keyed by ground-truth chain length.
    assert set(agg["by_length"].keys()) == {2, 3}
    assert agg["by_length"][2]["num_examples"] == 1
    assert agg["by_length"][3]["num_examples"] == 1
    assert agg["by_length"][2]["complete_solution"] == 1.0

    # Per-step breakdown must cover steps 0..2 (max chain length is 3).
    assert set(agg["by_step"].keys()) == {0, 1, 2}
    assert agg["by_step"][0]["n"] == 2  # both examples reach step 0
    assert agg["by_step"][2]["n"] == 1  # only the 3-step example reaches step 2
    assert agg["by_step"][0]["operation_accuracy"] == 1.0

    # By-letter op accuracy should sum to total ops.
    assert sum(b["n"] for b in agg["by_letter"].values()) == 5


def _score_fixture(start: list[int], n_steps: int):
    """Build a legit (input_vec, output_vec, blocks) triple that should score
    as fully correct. Chooses letters from f at each step."""
    cur = list(start)
    letters = []
    for _ in range(n_steps):
        lf, _ = decision_letters(cur)
        letters.append(lf)
        cur, _ = apply_letter(lf, cur)
    trace_text, final = _trace_from_letters(start, letters)
    blocks = parse_generated_blocks(trace_text)
    return start, final, blocks


# ---------------------------------------------------------------------------
# Letter position finder + intervention prefix
# ---------------------------------------------------------------------------

def test_find_letter_position_step0():
    assert find_letter_position([5, 1, 2, 3], semicolon_id=99, step=0) == 0


def test_find_letter_position_after_semicolons():
    # token 99 is the ';' token id for this test
    gen = [10, 11, 99, 20, 21, 99, 30, 31]
    assert find_letter_position(gen, 99, 0) == 0
    assert find_letter_position(gen, 99, 1) == 3
    assert find_letter_position(gen, 99, 2) == 6
    assert find_letter_position(gen, 99, 3) is None  # no third ';'


def test_find_letter_position_trailing_semicolon_returns_none():
    gen = [10, 99]  # semicolon is the last token, no letter follows
    assert find_letter_position(gen, 99, 1) is None


def test_build_intervened_prefix_splices_cleanly():
    prompt = [1, 2, 3]
    gen = [10, 11, 12, 13, 14]
    # letter_pos=2 means we keep gen[:2] = [10, 11] and replace gen[2] with 77.
    out = build_intervened_prefix(prompt, gen, letter_pos=2, new_letter_tid=77)
    assert out == [1, 2, 3, 10, 11, 77]
