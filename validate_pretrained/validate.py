"""
Standalone validator for a pretrained decision-chain GPT-NeoX model.

Loads an HF checkpoint, runs greedy free generation on val.json, and reports
three metrics per example (and in aggregate):

  * operation_accuracy   -- for each generated block, does the process text and
                            the result vector match a re-execution of the
                            chosen letter's transformation on the previous
                            vector?
  * operation_selection  -- is each chosen letter a valid output of decision
                            function f or g applied to the vector at that
                            step?
  * complete_solution    -- all operations correct AND all selections correct
                            AND the final result vector equals the OUTPUT
                            vector from the prompt.

Supports trace intervention: force a replacement letter at a chosen step,
then let the model finish the rest of the trace, and re-score with the same
metrics.

This file is intentionally self-contained (only torch + transformers as
external deps) so it can be debugged end-to-end without touching the rest of
the repo.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


# =============================================================================
# Domain: modular arithmetic, transformations, decision functions
# =============================================================================

K = 10  # modular base used in the training data


def _fmt_vec(v: list[int]) -> str:
    return "[ " + " , ".join(str(x) for x in v) + " ]"


# --- Transformations. Each one takes (vec, trace_sink) and returns new vec. ---
# The trace sink is a list with a single string appended in the same format
# the training data uses, prefixed with the function name. The name is later
# rewritten to the corresponding letter (a..t).

def _reverse(v, tr):
    r = v[::-1]
    tr.append(f"reverse R {_fmt_vec(r)}")
    return r


def _add_1(v, tr):
    r = [(x + 1) % K for x in v]
    ops = [f"{x} + 1 = {y}" for x, y in zip(v, r)]
    tr.append(f"add_1 : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _double(v, tr):
    r = [(2 * x) % K for x in v]
    ops = [f"2 * {x} = {y}" for x, y in zip(v, r)]
    tr.append(f"double : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _negate(v, tr):
    r = [(K - x) % K for x in v]
    ops = [f"{K} - {x} = {y}" for x, y in zip(v, r)]
    tr.append(f"negate : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _cumsum(v, tr):
    out, running, ops = [], 0, []
    for i, x in enumerate(v):
        if i == 0:
            out.append(x % K)
            ops.append(str(x % K))
        else:
            y = (running + x) % K
            ops.append(f"{running} + {x} = {y}")
            out.append(y)
        running = out[-1]
    tr.append(f"cumsum : {' , '.join(ops)} R {_fmt_vec(out)}")
    return out


def _scale_by_first(v, tr):
    r = [(x * v[0]) % K for x in v]
    ops = [f"{x} * {v[0]} = {y}" for x, y in zip(v, r)]
    tr.append(f"scale_by_first : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _rotate_left(v, tr):
    r = v[1:] + v[:1]
    tr.append(f"rotate_left R {_fmt_vec(r)}")
    return r


def _swap_pairs(v, tr):
    r = list(v)
    swaps = []
    for i in range(0, len(v) - 1, 2):
        r[i], r[i + 1] = r[i + 1], r[i]
        swaps.append(f"( {v[i]} , {v[i+1]} ) -> ( {r[i]} , {r[i+1]} )")
    if len(v) % 2 == 1:
        swaps.append(f"{v[-1]} -> stays")
    tr.append(f"swap_pairs : {' , '.join(swaps)} R {_fmt_vec(r)}")
    return r


def _position_multiply(v, tr):
    r = [(x * i) % K for i, x in enumerate(v)]
    ops = [f"{x} * {i} = {y}" for i, (x, y) in enumerate(zip(v, r))]
    tr.append(f"position_multiply : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _diff(v, tr):
    r = [v[0]]
    ops = [str(v[0])]
    for i in range(1, len(v)):
        d = (v[i] - v[i - 1]) % K
        ops.append(f"{v[i]} - {v[i-1]} = {d}")
        r.append(d)
    tr.append(f"diff : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _add_last_to_all(v, tr):
    r = [(x + v[-1]) % K for x in v]
    ops = [f"{x} + {v[-1]} = {y}" for x, y in zip(v, r)]
    tr.append(f"add_last_to_all : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _square(v, tr):
    r = [(x * x) % K for x in v]
    ops = [f"{x} ^ 2 = {y}" for x, y in zip(v, r)]
    tr.append(f"square : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _rotate_right(v, tr):
    r = v[-1:] + v[:-1]
    tr.append(f"rotate_right R {_fmt_vec(r)}")
    return r


def _add_first_to_all(v, tr):
    r = [(x + v[0]) % K for x in v]
    ops = [f"{x} + {v[0]} = {y}" for x, y in zip(v, r)]
    tr.append(f"add_first_to_all : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _cumsum_reverse(v, tr):
    rev = v[::-1]
    out, running, ops = [], 0, []
    for i, x in enumerate(rev):
        if i == 0:
            out.append(x % K)
            ops.append(str(x % K))
        else:
            y = (running + x) % K
            ops.append(f"{running} + {x} = {y}")
            out.append(y)
        running = out[-1]
    out = out[::-1]
    tr.append(f"cumsum_reverse : {' , '.join(ops)} R {_fmt_vec(out)}")
    return out


def _prefix_product(v, tr):
    out, running, ops = [], 1, []
    for i, x in enumerate(v):
        if i == 0:
            out.append(x % K)
            ops.append(str(x % K))
        else:
            y = (running * x) % K
            ops.append(f"{running} * {x} = {y}")
            out.append(y)
        running = out[-1]
    tr.append(f"prefix_product : {' , '.join(ops)} R {_fmt_vec(out)}")
    return out


def _sliding_sum(v, tr):
    r, ops = [], []
    n = len(v)
    for i in range(n):
        s = (v[i] + v[(i + 1) % n]) % K
        r.append(s)
        ops.append(f"{v[i]} + {v[(i+1) % n]} = {s}")
    tr.append(f"sliding_sum : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _position_add(v, tr):
    r = [(x + i) % K for i, x in enumerate(v)]
    ops = [f"{x} + {i} = {y}" for i, (x, y) in enumerate(zip(v, r))]
    tr.append(f"position_add : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _conditional_double(v, tr):
    threshold = K // 2
    r = [(2 * x) % K if x >= threshold else x for x in v]
    ops = [f"2 * {x} = {y}" if x >= threshold else str(x) for x, y in zip(v, r)]
    tr.append(f"conditional_double : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


def _interleave_sum_diff(v, tr):
    r, ops = [], []
    for i in range(0, len(v) - 1, 2):
        s = (v[i] + v[i + 1]) % K
        d = (v[i] - v[i + 1]) % K
        r.extend([s, d])
        ops.append(f"({v[i]} + {v[i+1]}, {v[i]} - {v[i+1]}) = ({s}, {d})")
    if len(v) % 2 == 1:
        r.append(v[-1])
        ops.append(f"{v[-1]} stays")
    tr.append(f"interleave_sum_diff : {' , '.join(ops)} R {_fmt_vec(r)}")
    return r


LETTER_TO_OP: dict[str, tuple[str, Callable]] = {
    "a": ("reverse", _reverse),
    "b": ("add_1", _add_1),
    "c": ("double", _double),
    "d": ("negate", _negate),
    "e": ("cumsum", _cumsum),
    "f": ("scale_by_first", _scale_by_first),
    "g": ("rotate_left", _rotate_left),
    "h": ("swap_pairs", _swap_pairs),
    "i": ("position_multiply", _position_multiply),
    "j": ("diff", _diff),
    "k": ("add_last_to_all", _add_last_to_all),
    "l": ("square", _square),
    "m": ("rotate_right", _rotate_right),
    "n": ("add_first_to_all", _add_first_to_all),
    "o": ("cumsum_reverse", _cumsum_reverse),
    "p": ("prefix_product", _prefix_product),
    "q": ("sliding_sum", _sliding_sum),
    "r": ("position_add", _position_add),
    "s": ("conditional_double", _conditional_double),
    "t": ("interleave_sum_diff", _interleave_sum_diff),
}

LETTERS = list(LETTER_TO_OP.keys())
INT_TO_LETTER = {i: LETTERS[i] for i in range(20)}


def decision_f(v: list[int]) -> int:
    """f(L) = is_even(L[0]) * 10 + L[1]  ->  0..19."""
    return (1 if v[0] % 2 == 0 else 0) * 10 + v[1]


def decision_g(v: list[int]) -> int:
    """g(L) = is_even(L[3]) * 10 + L[4]  ->  0..19."""
    return (1 if v[3] % 2 == 0 else 0) * 10 + v[4]


def decision_letters(v: list[int]) -> tuple[str, str]:
    return INT_TO_LETTER[decision_f(v)], INT_TO_LETTER[decision_g(v)]


def apply_letter(letter: str, vec: list[int]) -> tuple[list[int], str]:
    """Re-execute a letter's transformation and return (new_vec, block_text)."""
    name, func = LETTER_TO_OP[letter]
    sink: list[str] = []
    result = func(list(vec), sink)
    block = sink[0].replace(name, letter, 1)
    return result, block


# =============================================================================
# Parsing
# =============================================================================

_PROMPT_RE = re.compile(
    r"INPUT\s*:\s*\[\s*([\d\s,]+?)\s*\]\s*OUTPUT\s*:\s*\[\s*([\d\s,]+?)\s*\]"
)
_VEC_RE = re.compile(r"\[\s*([\d\s,]+?)\s*\]")


def extract_input_output(prompt_str: str) -> tuple[list[int] | None, list[int] | None]:
    m = _PROMPT_RE.search(prompt_str)
    if m is None:
        return None, None
    inp = [int(x) for x in m.group(1).split(",")]
    out = [int(x) for x in m.group(2).split(",")]
    return inp, out


@dataclass
class Block:
    letter: str
    block: str
    vec: list[int] | None


def parse_generated_blocks(gen_text: str) -> list[Block]:
    """Split a generated trace into ' ; '-separated blocks.

    Each block must start with a single letter followed by a space. The last
    '[ ... ]' in the block is parsed as the result vector.
    """
    blocks: list[Block] = []
    for part in gen_text.strip().split(" ; "):
        part = part.strip()
        if len(part) < 2 or part[1] != " ":
            blocks.append(Block(letter="", block=part, vec=None))
            continue
        letter = part[0]
        vec_matches = list(_VEC_RE.finditer(part))
        vec: list[int] | None = None
        if vec_matches:
            try:
                vec = [int(x) for x in vec_matches[-1].group(1).split(",")]
            except ValueError:
                vec = None
        blocks.append(Block(letter=letter, block=part, vec=vec))
    return blocks


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class ExampleScore:
    n_steps: int
    op_correct: int                 # number of blocks with correct process+result
    sel_correct: int                # number of letters matching f or g
    chain_matches_output: bool      # final vec == OUTPUT vec
    full_correct: bool              # all of the above, non-empty chain
    gt_chain_length: int | None = None     # expected chain length from data
    length_matches_gt: bool = False        # len(blocks) == gt_chain_length
    parseable: bool = True                 # input/output vectors were parseable
    per_step_op: list[bool] = field(default_factory=list)
    per_step_sel: list[bool] = field(default_factory=list)
    per_step_letter: list[str] = field(default_factory=list)


def score_example(
    input_vec: list[int],
    output_vec: list[int],
    blocks: list[Block],
    gt_chain_length: int | None = None,
) -> ExampleScore:
    """Compute per-example metrics from parsed blocks."""
    current: list[int] | None = list(input_vec)
    op_correct = 0
    sel_correct = 0
    per_step_op: list[bool] = []
    per_step_sel: list[bool] = []
    per_step_letter: list[str] = []

    for b in blocks:
        letter_known = b.letter in LETTER_TO_OP
        per_step_letter.append(b.letter)

        # Operation selection: does the chosen letter match f or g at this
        # step, given the vector the model should be working from?
        #
        # `current` may be a fallback from the model's parsed result vector,
        # which can contain out-of-range integers if the model hallucinates a
        # bad number. decision_f/g assume every element is in 0..9, so guard
        # against that explicitly and treat OOR vectors as un-scorable for
        # selection (sel_ok = False).
        if (
            current is not None
            and len(current) >= 5
            and letter_known
            and all(0 <= x < K for x in current[:5])
        ):
            lf, lg = decision_letters(current)
            sel_ok = b.letter in (lf, lg)
        else:
            sel_ok = False
        per_step_sel.append(sel_ok)
        if sel_ok:
            sel_correct += 1

        # Operation accuracy: re-execute the letter on `current` and compare
        # both the full block text (process) and the resulting vector.
        if current is not None and letter_known:
            expected_vec, expected_block = apply_letter(b.letter, current)
            op_ok = b.block == expected_block
        else:
            expected_vec, op_ok = None, False
        per_step_op.append(op_ok)
        if op_ok:
            op_correct += 1

        # Advance current_vec for the next step. Prefer the recomputed vector
        # when the block validated; fall back to the parsed vector if not, so
        # later steps still have something to measure against.
        if op_ok and expected_vec is not None:
            current = expected_vec
        elif b.vec is not None:
            current = b.vec
        else:
            current = None

    final_vec = blocks[-1].vec if blocks else None
    chain_matches_output = final_vec is not None and final_vec == list(output_vec)
    n_steps = len(blocks)
    length_matches_gt = (
        gt_chain_length is not None and n_steps == gt_chain_length
    )
    full_correct = (
        n_steps > 0
        and op_correct == n_steps
        and sel_correct == n_steps
        and chain_matches_output
    )

    return ExampleScore(
        n_steps=n_steps,
        op_correct=op_correct,
        sel_correct=sel_correct,
        chain_matches_output=chain_matches_output,
        full_correct=full_correct,
        gt_chain_length=gt_chain_length,
        length_matches_gt=length_matches_gt,
        parseable=True,
        per_step_op=per_step_op,
        per_step_sel=per_step_sel,
        per_step_letter=per_step_letter,
    )


def _safe_div(num: int, denom: int) -> float:
    return num / denom if denom > 0 else float("nan")


def aggregate_scores(scores: list[ExampleScore]) -> dict:
    """Roll per-example scores into a multi-level breakdown.

    Returns a dict with four sections:
      * overall          -- headline rates across all examples
      * by_length        -- per ground-truth chain length
      * by_step          -- per 0-indexed step position
      * by_letter        -- op accuracy for every letter the model produced
    """
    if not scores:
        return {}

    total_ops = sum(s.n_steps for s in scores)
    op_correct = sum(s.op_correct for s in scores)
    sel_correct = sum(s.sel_correct for s in scores)
    full_correct = sum(1 for s in scores if s.full_correct)
    chain_match = sum(1 for s in scores if s.chain_matches_output)
    length_match = sum(1 for s in scores if s.length_matches_gt)
    parseable = sum(1 for s in scores if s.parseable)

    overall = {
        "num_examples": len(scores),
        "total_predicted_ops": total_ops,
        "parseable_fraction": parseable / len(scores),
        "operation_accuracy": _safe_div(op_correct, total_ops),
        "operation_selection": _safe_div(sel_correct, total_ops),
        "chain_matches_output": chain_match / len(scores),
        "length_matches_gt": length_match / len(scores),
        "complete_solution": full_correct / len(scores),
    }

    # Per ground-truth chain length
    by_length: dict[int, dict] = {}
    for s in scores:
        if s.gt_chain_length is None:
            continue
        bucket = by_length.setdefault(
            s.gt_chain_length,
            {"n": 0, "ops": 0, "op_correct": 0, "sel_correct": 0,
             "complete": 0, "length_match": 0},
        )
        bucket["n"] += 1
        bucket["ops"] += s.n_steps
        bucket["op_correct"] += s.op_correct
        bucket["sel_correct"] += s.sel_correct
        if s.full_correct:
            bucket["complete"] += 1
        if s.length_matches_gt:
            bucket["length_match"] += 1
    by_length_out = {
        L: {
            "num_examples": b["n"],
            "operation_accuracy": _safe_div(b["op_correct"], b["ops"]),
            "operation_selection": _safe_div(b["sel_correct"], b["ops"]),
            "length_matches_gt": b["length_match"] / b["n"],
            "complete_solution": b["complete"] / b["n"],
        }
        for L, b in sorted(by_length.items())
    }

    # Per step index (0-indexed). A step is counted only for examples that
    # produced at least that many blocks.
    max_steps = max((s.n_steps for s in scores), default=0)
    by_step_out: dict[int, dict] = {}
    for step in range(max_steps):
        n = op_ok = sel_ok = 0
        for s in scores:
            if step < len(s.per_step_op):
                n += 1
                op_ok += int(s.per_step_op[step])
                sel_ok += int(s.per_step_sel[step])
        if n > 0:
            by_step_out[step] = {
                "n": n,
                "operation_accuracy": op_ok / n,
                "operation_selection": sel_ok / n,
            }

    # Per letter the model chose. Only op accuracy is meaningful here --
    # selection by letter would conflate with what f/g happened to produce.
    by_letter: dict[str, dict[str, int]] = {}
    for s in scores:
        for lt, op_ok in zip(s.per_step_letter, s.per_step_op):
            if not lt:
                continue
            b = by_letter.setdefault(lt, {"n": 0, "op_correct": 0})
            b["n"] += 1
            if op_ok:
                b["op_correct"] += 1
    by_letter_out = {
        lt: {
            "n": b["n"],
            "operation_accuracy": b["op_correct"] / b["n"],
        }
        for lt, b in sorted(by_letter.items())
    }

    return {
        "overall": overall,
        "by_length": by_length_out,
        "by_step": by_step_out,
        "by_letter": by_letter_out,
    }


def format_report(aggregate: dict, label: str) -> str:
    """Render the aggregate dict as a human-readable multi-section report."""
    if not aggregate:
        return f"=== {label} === (no examples)\n"

    lines: list[str] = []
    lines.append(f"=== Overall ({label}) ===")
    for k, v in aggregate["overall"].items():
        if isinstance(v, float):
            lines.append(f"  {k:24s} {v:.4f}")
        else:
            lines.append(f"  {k:24s} {v}")

    if aggregate["by_length"]:
        lines.append("")
        lines.append("=== By ground-truth chain length ===")
        lines.append(f"  {'length':>6}  {'n':>6}  {'op_acc':>8}  {'sel_acc':>8}  "
                     f"{'len_ok':>8}  {'complete':>9}")
        for L, b in aggregate["by_length"].items():
            lines.append(
                f"  {L:>6}  {b['num_examples']:>6}  "
                f"{b['operation_accuracy']:>8.4f}  "
                f"{b['operation_selection']:>8.4f}  "
                f"{b['length_matches_gt']:>8.4f}  "
                f"{b['complete_solution']:>9.4f}"
            )

    if aggregate["by_step"]:
        lines.append("")
        lines.append("=== By step index (over examples that reached that step) ===")
        lines.append(f"  {'step':>4}  {'n':>6}  {'op_acc':>8}  {'sel_acc':>8}")
        for step, b in aggregate["by_step"].items():
            lines.append(
                f"  {step:>4}  {b['n']:>6}  "
                f"{b['operation_accuracy']:>8.4f}  "
                f"{b['operation_selection']:>8.4f}"
            )

    if aggregate["by_letter"]:
        lines.append("")
        lines.append("=== By predicted letter (operation accuracy only) ===")
        lines.append(f"  {'letter':>6}  {'n':>6}  {'op_acc':>8}")
        for lt, b in aggregate["by_letter"].items():
            lines.append(
                f"  {lt:>6}  {b['n']:>6}  {b['operation_accuracy']:>8.4f}"
            )

    return "\n".join(lines) + "\n"


# =============================================================================
# Model loading and generation
# =============================================================================

def load_model_and_tokenizer(
    model_path: Path,
    tokenizer_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerFast]:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    return model, tokenizer


def build_prompt_text(input_str: str) -> str:
    return f"[BOS] {input_str} [TRACE]"


def find_letter_position(
    gen_ids: list[int],
    semicolon_id: int,
    step: int,
) -> int | None:
    """Return the index (within gen_ids) of the letter token at `step`.

    Step 0 is index 0. Step k (k>0) is the index immediately after the k-th
    semicolon. Returns None if the step does not exist in the generation.
    """
    if step == 0:
        return 0 if gen_ids else None
    seen = 0
    for i, tid in enumerate(gen_ids):
        if tid == semicolon_id:
            seen += 1
            if seen == step:
                pos = i + 1
                return pos if pos < len(gen_ids) else None
    return None


@torch.no_grad()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt_texts: list[str],
    max_length: int,
) -> list[list[int]]:
    """Greedy batched generation. Returns the generated tail (no prompt, no EOS)."""
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    padded_len = enc.input_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    results: list[list[int]] = []
    for seq in out.tolist():
        gen = seq[padded_len:]
        if eos_id in gen:
            gen = gen[: gen.index(eos_id)]
        results.append(gen)
    return results


@torch.no_grad()
def generate_continuation_from_ids(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prefix_id_lists: list[list[int]],
    max_length: int,
) -> list[list[int]]:
    """Greedy generation from raw ID prefixes. Returns only the tokens the
    model generated after each prefix, trimmed at EOS."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    max_prefix = max(len(p) for p in prefix_id_lists)

    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    for prefix in prefix_id_lists:
        pad_len = max_prefix - len(prefix)
        input_ids.append([pad_id] * pad_len + prefix)
        attention.append([0] * pad_len + [1] * len(prefix))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    mask_tensor = torch.tensor(attention, dtype=torch.long, device=model.device)

    out = model.generate(
        input_ids=input_tensor,
        attention_mask=mask_tensor,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )

    # With left padding, every row's generated tail starts at index max_prefix.
    results: list[list[int]] = []
    for row in out.tolist():
        tail = row[max_prefix:]
        if eos_id in tail:
            tail = tail[: tail.index(eos_id)]
        results.append(tail)
    return results


def build_intervened_prefix(
    prompt_ids: list[int],
    gen_ids: list[int],
    letter_pos: int,
    new_letter_tid: int,
) -> list[int]:
    """Return prompt_ids + gen_ids[:letter_pos] + [new_letter_tid]."""
    return list(prompt_ids) + list(gen_ids[:letter_pos]) + [new_letter_tid]


# =============================================================================
# Validation pipeline
# =============================================================================

@dataclass
class ValidationRun:
    scores: list[ExampleScore]
    generations: list[str]
    aggregate: dict


def run_validation(
    model,
    tokenizer: PreTrainedTokenizerFast,
    examples: list[dict],
    max_length: int,
    batch_size: int,
    intervene_step: int | None = None,
    intervene_letter: str | None = None,
) -> ValidationRun:
    """Run the model on the given examples, optionally with a forced letter.

    `examples` is a list of val.json entries (dicts with an "input" key).
    When `intervene_step`/`intervene_letter` are set, generation proceeds in
    two phases: a normal free-generation pass, followed by a second pass that
    forces `intervene_letter` at step `intervene_step` and lets the model
    continue. Metrics are computed on the intervened outputs.
    """
    if (intervene_step is None) != (intervene_letter is None):
        raise ValueError("intervene_step and intervene_letter must be set together")

    semicolon_id = tokenizer.encode(";", add_special_tokens=False)[0]

    prompt_texts = [build_prompt_text(ex["input"]) for ex in examples]
    prompt_id_lists = [
        tokenizer.encode(t, add_special_tokens=False) for t in prompt_texts
    ]

    scores: list[ExampleScore] = []
    decoded: list[str] = []

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_prompts = prompt_texts[start : start + batch_size]
        batch_prompt_ids = prompt_id_lists[start : start + batch_size]

        first_gens = generate_batch(model, tokenizer, batch_prompts, max_length)

        if intervene_step is None:
            final_gens = first_gens
        else:
            new_letter_tid = tokenizer.encode(
                intervene_letter, add_special_tokens=False
            )[0]
            prefix_lists: list[list[int]] = []
            forced_tails: list[list[int]] = []  # the gen-side slice of each prefix
            index_map: list[int] = []
            for j, (pids, gids) in enumerate(zip(batch_prompt_ids, first_gens)):
                pos = find_letter_position(gids, semicolon_id, intervene_step)
                if pos is None:
                    continue
                prefix = build_intervened_prefix(pids, gids, pos, new_letter_tid)
                prefix_lists.append(prefix)
                forced_tails.append(prefix[len(pids):])
                index_map.append(j)

            final_gens = [list(g) for g in first_gens]
            if prefix_lists:
                continuations = generate_continuation_from_ids(
                    model, tokenizer, prefix_lists, max_length
                )
                for j, forced, cont in zip(index_map, forced_tails, continuations):
                    final_gens[j] = forced + cont

        for ex, gen in zip(batch_examples, final_gens):
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            decoded.append(text)
            inp, out = extract_input_output(ex["input"])
            gt_len = len(ex["letters"]) if "letters" in ex else None
            if inp is None or out is None:
                unparsed = ExampleScore(0, 0, 0, False, False, gt_chain_length=gt_len)
                unparsed.parseable = False
                scores.append(unparsed)
                continue
            blocks = parse_generated_blocks(text)
            scores.append(score_example(inp, out, blocks, gt_chain_length=gt_len))

    return ValidationRun(
        scores=scores,
        generations=decoded,
        aggregate=aggregate_scores(scores),
    )


def print_sample_traces(
    examples: list[dict],
    run: "ValidationRun",
    n: int,
) -> None:
    """Print up to n sample traces: half from correct examples, half from
    incorrect ones (or whatever is available)."""
    correct_idx = [i for i, s in enumerate(run.scores) if s.full_correct]
    wrong_idx = [i for i, s in enumerate(run.scores) if not s.full_correct]
    n_correct = min(n // 2, len(correct_idx))
    n_wrong = min(n - n_correct, len(wrong_idx))
    chosen = correct_idx[:n_correct] + wrong_idx[:n_wrong]

    print("=== Sample traces ===")
    for idx in chosen:
        ex = examples[idx]
        s = run.scores[idx]
        tag = "OK" if s.full_correct else "FAIL"
        print(f"[{tag}] #{idx} gt_len={s.gt_chain_length} pred_len={s.n_steps}")
        print(f"  prompt: {ex['input']}")
        print(f"  gt:     {ex.get('output', '')}")
        print(f"  pred:   {run.generations[idx]}")
        print(
            f"  op={s.op_correct}/{s.n_steps} "
            f"sel={s.sel_correct}/{s.n_steps} "
            f"chain_out={s.chain_matches_output} "
            f"per_step_op={s.per_step_op} per_step_sel={s.per_step_sel}"
        )
        print()


# =============================================================================
# CLI
# =============================================================================

def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    model = repo_root / "outputs" / "temp" / "hf_12l-8h-512d-decision-chains-ext_6_2M"
    tokenizer = repo_root / "outputs" / "tokenizer" / "tokenizer_decision_chains_extended.json"
    val = repo_root / "outputs" / "data" / "decision_chains_extended" / "val.json"
    return model, tokenizer, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    default_model, default_tok, default_val = _default_paths(repo_root)

    parser.add_argument("--model_path", type=Path, default=default_model)
    parser.add_argument("--tokenizer_path", type=Path, default=default_tok)
    parser.add_argument("--val_file", type=Path, default=default_val)
    parser.add_argument("--num_examples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--intervene_step",
        type=int,
        default=None,
        help="0-indexed step at which to replace the model's chosen letter",
    )
    parser.add_argument(
        "--intervene_letter",
        type=str,
        default=None,
        help="Letter (a..t) to force at the intervention step",
    )
    parser.add_argument(
        "--dump_jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-example predictions and scores",
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=0,
        help="Print this many sample traces (half correct, half failing)",
    )
    args = parser.parse_args()

    if args.intervene_letter is not None and args.intervene_letter not in LETTER_TO_OP:
        parser.error(f"--intervene_letter must be one of {LETTERS}")

    with open(args.val_file) as f:
        raw = json.load(f)
    examples = raw[: args.num_examples]

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, device=args.device
    )

    print(f"Running validation on {len(examples)} examples (batch={args.batch_size})")
    run = run_validation(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        intervene_step=args.intervene_step,
        intervene_letter=args.intervene_letter,
    )

    label = "baseline"
    if args.intervene_step is not None:
        label = f"intervene@step{args.intervene_step}->{args.intervene_letter}"

    print()
    print(format_report(run.aggregate, label))

    if args.show_samples > 0:
        print_sample_traces(examples, run, args.show_samples)

    if args.dump_jsonl is not None:
        args.dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_jsonl, "w") as f:
            for ex, gen, score in zip(examples, run.generations, run.scores):
                rec = {
                    "input": ex["input"],
                    "gt_output": ex.get("output", ""),
                    "gt_letters": ex.get("letters", []),
                    "generation": gen,
                    "gt_chain_length": score.gt_chain_length,
                    "n_steps": score.n_steps,
                    "length_matches_gt": score.length_matches_gt,
                    "op_correct": score.op_correct,
                    "sel_correct": score.sel_correct,
                    "chain_matches_output": score.chain_matches_output,
                    "full_correct": score.full_correct,
                    "per_step_op": score.per_step_op,
                    "per_step_sel": score.per_step_sel,
                    "per_step_letter": score.per_step_letter,
                }
                # Post-intervention metrics (only for perturbed runs).
                # Measured strictly AFTER the intervened step -- the forced
                # letter itself is excluded, since its selection is wrong by
                # construction and its op accuracy only reflects the model's
                # ability to execute the forced transformation.
                if args.intervene_step is not None:
                    step = args.intervene_step
                    rec["intervene_step"] = step
                    rec["intervene_letter"] = args.intervene_letter
                    post_ops = score.per_step_op[step + 1:]
                    post_sels = score.per_step_sel[step + 1:]
                    rec["post_n_steps"] = len(post_ops)
                    rec["post_op_correct"] = sum(post_ops)
                    rec["post_sel_correct"] = sum(post_sels)
                    rec["post_op_rate"] = (
                        sum(post_ops) / len(post_ops) if post_ops else 0.0
                    )
                    rec["post_sel_rate"] = (
                        sum(post_sels) / len(post_sels) if post_sels else 0.0
                    )
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote per-example dump to {args.dump_jsonl}")


if __name__ == "__main__":
    main()
