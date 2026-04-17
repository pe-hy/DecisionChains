"""
Metrics for evaluating decision-chain traces.

Primary metrics for alignment analysis (f-only target):
  - operation_accuracy:       is the arithmetic in each block correct?
  - f_selection:              fraction of steps where letter == f's letter
                              (averaged over steps, so shorter chains count less)
  - full_f_alignment:         fraction of EXAMPLES where every step picked f
  - per_step_f_selection:     f-rate broken out by step index (diagnostic)

Sanity metrics (pre-alignment semantics):
  - f_or_g_valid_selection:   letter ∈ {f_letter, g_letter} — says the model is
                              still producing plausible decision-function output,
                              but doesn't care whether f or g was picked. This
                              used to be called `operation_selection`; the old
                              name was misleading under alignment.
  - chain_matches_output:     final vec = GT OUTPUT vec (only meaningful when
                              GT is the f-path, i.e. on ffff val)
  - complete_solution:        all ops correct AND all sels valid AND final=OUTPUT

Intermediate-vector advancement rule (used in both scoring paths):
  if the letter is known, advance via correct execution of that letter
  (apply_letter). This gives the alignment semantic: "what would f pick at the
  true next state, assuming the letter choice was the only action".

Usage:
    from metrics import score_trace, aggregate_scores, extract_input_output

    input_vec, output_vec = extract_input_output(example["input"])
    score = score_trace(input_vec, output_vec, generated_text)
    print(score)

    # Or aggregate over many examples:
    report = aggregate_scores(list_of_scores)
    print(format_report(report))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─── Constants ────────────────────────────────────────────────────────────────

K = 10  # modular base


# ─── Transformations (a-t) ────────────────────────────────────────────────────
# Each takes (vec, trace_sink) and returns new vec.
# trace_sink is a list; the function appends one string describing the operation.

def _fmt_vec(v):
    return "[ " + " , ".join(str(x) for x in v) + " ]"


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


LETTER_TO_OP = {
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


# ─── Decision functions ──────────────────────────────────────────────────────

def decision_f(v: list[int]) -> int:
    """f(L) = is_even(L[0]) * 10 + L[1]  ->  0..19."""
    return (1 if v[0] % 2 == 0 else 0) * 10 + v[1]


def decision_g(v: list[int]) -> int:
    """g(L) = is_even(L[3]) * 10 + L[4]  ->  0..19."""
    return (1 if v[3] % 2 == 0 else 0) * 10 + v[4]


def decision_letters(v: list[int]) -> tuple[str, str]:
    """Return (letter_f, letter_g) for the given vector."""
    return INT_TO_LETTER[decision_f(v)], INT_TO_LETTER[decision_g(v)]


def apply_letter(letter: str, vec: list[int]) -> tuple[list[int], str]:
    """Re-execute a letter's transformation. Returns (new_vec, block_text)."""
    name, func = LETTER_TO_OP[letter]
    sink: list[str] = []
    result = func(list(vec), sink)
    block = sink[0].replace(name, letter, 1)
    return result, block


# ─── Parsing ─────────────────────────────────────────────────────────────────

_PROMPT_RE = re.compile(
    r"INPUT\s*:\s*\[\s*([\d\s,]+?)\s*\]\s*OUTPUT\s*:\s*\[\s*([\d\s,]+?)\s*\]"
)
_VEC_RE = re.compile(r"\[\s*([\d\s,]+?)\s*\]")


def extract_input_output(prompt_str: str) -> tuple[list[int] | None, list[int] | None]:
    """Parse INPUT and OUTPUT vectors from a prompt string."""
    m = _PROMPT_RE.search(prompt_str)
    if m is None:
        return None, None
    inp = [int(x) for x in m.group(1).split(",")]
    out = [int(x) for x in m.group(2).split(",")]
    return inp, out


@dataclass
class Block:
    letter: str
    block: str          # full block text (e.g. "n : 4 + 4 = 8 , ... R [ 8 , ... ]")
    vec: list[int] | None  # parsed result vector from the block


def parse_trace(gen_text: str) -> list[Block]:
    """Split generated trace text into ` ; `-separated blocks.

    Each block starts with a single letter followed by a space.
    The last `[ ... ]` in the block is parsed as the result vector.
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


# ─── Scoring ─────────────────────────────────────────────────────────────────

@dataclass
class TraceScore:
    """Score for a single generated trace."""
    n_steps: int
    op_correct: int             # blocks with correct arithmetic
    sel_correct: int            # letters matching f or g
    f_hits: int                 # letters matching f specifically
    chain_matches_output: bool  # final vec == OUTPUT vec
    full_correct: bool          # all ops + all sels + output match
    per_step_op: list[bool] = field(default_factory=list)
    per_step_sel: list[bool] = field(default_factory=list)
    per_step_f: list[bool] = field(default_factory=list)
    per_step_letter: list[str] = field(default_factory=list)
    per_step_f_letter: list[str] = field(default_factory=list)

    @property
    def op_accuracy(self) -> float:
        return self.op_correct / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def sel_accuracy(self) -> float:
        """f-OR-g validity rate. Kept under the legacy name for callers."""
        return self.sel_correct / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def f_selection_rate(self) -> float:
        return self.f_hits / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def full_f_aligned(self) -> bool:
        return self.n_steps > 0 and self.f_hits == self.n_steps


def score_trace(
    input_vec: list[int],
    output_vec: list[int],
    generated_text: str,
) -> TraceScore:
    """Score a generated trace against input/output vectors.

    Args:
        input_vec: the INPUT vector from the prompt
        output_vec: the OUTPUT vector from the prompt
        generated_text: the model's generated trace text (after [TRACE])

    Returns:
        TraceScore with per-step and aggregate metrics.
    """
    blocks = parse_trace(generated_text)
    return _score_blocks(input_vec, output_vec, blocks)


def _score_blocks(
    input_vec: list[int],
    output_vec: list[int],
    blocks: list[Block],
) -> TraceScore:
    """Core scoring logic over parsed blocks."""
    current: list[int] | None = list(input_vec)
    op_correct = 0
    sel_correct = 0
    f_hits = 0
    per_step_op: list[bool] = []
    per_step_sel: list[bool] = []
    per_step_f: list[bool] = []
    per_step_letter: list[str] = []
    per_step_f_letter: list[str] = []

    for b in blocks:
        letter_known = b.letter in LETTER_TO_OP
        per_step_letter.append(b.letter)

        # --- Decision-function evaluation ---
        # Given the current intermediate vector, what would f and g pick?
        # sel_ok: letter is a valid f-or-g output (legacy "operation_selection").
        # is_f:   letter is specifically f's output (the alignment signal).
        if (
            current is not None
            and len(current) >= 5
            and letter_known
            and all(0 <= x < K for x in current[:5])
        ):
            lf, lg = decision_letters(current)
            sel_ok = b.letter in (lf, lg)
            is_f = (b.letter == lf)
            per_step_f_letter.append(lf)
        else:
            sel_ok = False
            is_f = False
            per_step_f_letter.append("")
        per_step_sel.append(sel_ok)
        per_step_f.append(is_f)
        if sel_ok:
            sel_correct += 1
        if is_f:
            f_hits += 1

        # --- Operation accuracy ---
        # Re-execute the letter on current vec, compare full block text.
        if current is not None and letter_known:
            expected_vec, expected_block = apply_letter(b.letter, current)
            op_ok = b.block == expected_block
        else:
            expected_vec, op_ok = None, False
        per_step_op.append(op_ok)
        if op_ok:
            op_correct += 1

        # Advance vector. Alignment semantic: given the letter the model picked,
        # advance as if it were executed correctly. This way the next-step f/g
        # comparison is against the true-next-state, not the model's own
        # possibly-buggy arithmetic. Falls back to parsed vec if the letter is
        # unknown.
        if letter_known and expected_vec is not None:
            current = expected_vec
        elif b.vec is not None:
            current = b.vec
        else:
            current = None

    final_vec = blocks[-1].vec if blocks else None
    chain_matches_output = final_vec is not None and final_vec == list(output_vec)
    n_steps = len(blocks)
    full_correct = (
        n_steps > 0
        and op_correct == n_steps
        and sel_correct == n_steps
        and chain_matches_output
    )

    return TraceScore(
        n_steps=n_steps,
        op_correct=op_correct,
        sel_correct=sel_correct,
        f_hits=f_hits,
        chain_matches_output=chain_matches_output,
        full_correct=full_correct,
        per_step_op=per_step_op,
        per_step_sel=per_step_sel,
        per_step_f=per_step_f,
        per_step_letter=per_step_letter,
        per_step_f_letter=per_step_f_letter,
    )


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_scores(scores: list[TraceScore]) -> dict:
    """Aggregate a list of TraceScores into summary statistics."""
    if not scores:
        return {}

    n = len(scores)
    total_ops = sum(s.n_steps for s in scores)
    op_correct = sum(s.op_correct for s in scores)
    sel_correct = sum(s.sel_correct for s in scores)
    f_hits = sum(s.f_hits for s in scores)
    full_correct = sum(1 for s in scores if s.full_correct)
    full_f = sum(1 for s in scores if s.full_f_aligned)
    chain_match = sum(1 for s in scores if s.chain_matches_output)

    # Per-step f_selection. Denominator at step i = number of examples whose
    # chain reached step i. Shorter chains contribute to fewer slots.
    max_steps = max((s.n_steps for s in scores), default=0)
    per_step_hits = [0] * max_steps
    per_step_denom = [0] * max_steps
    for s in scores:
        for i in range(s.n_steps):
            per_step_denom[i] += 1
            if i < len(s.per_step_f) and s.per_step_f[i]:
                per_step_hits[i] += 1
    per_step_f_rate = [
        h / d if d else 0.0 for h, d in zip(per_step_hits, per_step_denom)
    ]

    return {
        "num_examples": n,
        "operation_accuracy": op_correct / total_ops if total_ops > 0 else 0.0,
        "f_or_g_valid_selection": sel_correct / total_ops if total_ops > 0 else 0.0,
        "f_selection": f_hits / total_ops if total_ops > 0 else 0.0,
        "full_f_alignment": full_f / n,
        "chain_matches_output": chain_match / n,
        "complete_solution": full_correct / n,
        "per_step_f_selection": per_step_f_rate,
        "per_step_denominator": per_step_denom,
    }


def format_report(agg: dict, label: str = "results") -> str:
    """Format aggregated scores as a readable string."""
    if not agg:
        return f"=== {label} === (no examples)\n"
    lines = [f"=== {label} ({agg['num_examples']} examples) ==="]
    for k, v in agg.items():
        if k == "num_examples":
            continue
        if isinstance(v, float):
            lines.append(f"  {k:24s} {v:.4f}")
        else:
            lines.append(f"  {k:24s} {v}")
    return "\n".join(lines)
