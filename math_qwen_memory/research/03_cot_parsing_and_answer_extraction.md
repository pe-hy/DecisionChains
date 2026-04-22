# CoT Parsing & Answer Extraction — Literature & Tools Review

## Overview

Scoring a reasoning LLM's output against ground truth looks deceptively easy — "pull out the
number, compare it" — but the engineering surface is wide. Failures come from three layers:

1. **Extraction**: the answer may be buried in a `<think>` block, written in prose
   ("the answer is therefore 5"), formatted as `\boxed{\tfrac{1}{2}}`, repeated multiple
   times, or wrapped in redundant LaTeX (`\text{...}`, `\textbf{...}`).
2. **Normalization**: `1/2`, `0.5`, `\frac{1}{2}`, `\tfrac12`, `\dfrac 1 2`, and
   `\frac{2}{4}` are all the same answer. Unicode minus (`−`, U+2212) vs ASCII minus,
   smart quotes, `.0` vs integer, and stray trailing periods all cause spurious mismatches.
3. **Equivalence**: symbolic equivalence (`\sin^2 x + \cos^2 x` == `1`), numeric tolerance
   (floats), set equality (`{1,2,3}` == `{3,2,1}`), and relational flips (`a < 2` == `2 > a`).

The HuggingFace Open-LLM-Leaderboard found that replacing their Minerva-style grader with
`math-verify` **recovered on average 61 additional correct problems per model on MATH
(+4.66 points)**, sometimes up to +40 points — most losses were caused by the grader, not
the model. So this matters a lot for research reporting, and especially for training-signal
pipelines (SFT filtering, RL reward).

## Standard extraction patterns

| Pattern            | Dataset/Origin       | Regex sketch                                      | Notes |
| ------------------ | -------------------- | ------------------------------------------------- | ----- |
| `\boxed{...}`      | MATH, AIME, AMC      | `\\boxed\{((?:[^{}]\|\{[^{}]*\})*)\}`             | Take the **last** occurrence; brace-match for nesting. |
| `####`             | GSM8K                | `####\s*(-?\d[\d,]*\.?\d*)`                       | Plain number after the marker. |
| "answer is X"      | fallback prose       | `(?:answer is\|final answer:?)\s*\$?([^\s.]+)`    | Brittle; use only as last resort. |
| `<answer>...`      | DeepSeek-R1 training | `<answer>(.*?)</answer>`                          | DeepSeek's format-reward forces this. |
| `\fbox{...}`       | legacy MATH variant  | same as boxed                                     | Rare but present. |
| `<think>...</think>` | Qwen3 / R1 thinking | `<think>(.*?)</think>` (DOTALL)                  | Strip **before** answer extraction. |

**Brace matching for `\boxed`.** A regex with `[^{}]*` misses nested braces (e.g.
`\boxed{\frac{1}{2}}`). The canonical implementation scans character-by-character from
the `\boxed{` marker, tracking depth:

```python
def last_boxed_only_string(s: str) -> str | None:
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{": depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[idx:j+1]
    return None
```
(Copied in spirit from `lm-eval-harness/tasks/minerva_math/utils.py`.)

## Answer normalization

The Minerva normalizer (Lewkowycz et al. 2022, App. D, still used by `lm-eval-harness`
`minerva_math`) is a cascade of regexes and literal substitutions:

- Strip LaTeX delimiters: `$...$`, `\(...\)`, `\[...\]`.
- Unwrap `\text{X}`, `\textbf{X}`, `\overline{X}`, `\mathrm{X}`, `\boxed{X}`.
- Drop unit words: `square`, `mph`, `dollars`, `cm`, `feet`, `degrees`, ...
- Drop ellipses / decorations: `\ldots`, `\dots`, `^{\circ}`, leading articles `a`/`an`.
- Normalize `\frac` / `\sqrt` shorthand: `\frac12` → `\frac{1}{2}`, `\sqrt3` → `\sqrt{3}`.
- Strip commas from numbers (`1,000` → `1000`), `\,` spacers, trailing periods.
- Replace Unicode minus `−` (U+2212), en-dash `–`, em-dash `—` with ASCII `-`.
- Split on `=`, keep the **right-hand side** when the answer is written as
  `x = 5`.

Equivalence is then computed with `sympy.simplify(a - b) == 0`, with a 5 s timeout — the
timeout is essential because simplify can hang on adversarial inputs.

For **purely numeric** answers (our `algebra__linear_1d`), normalization reduces to:

```python
def normalize_numeric(s: str) -> str:
    s = s.strip().replace(",", "").replace(" ", "")
    s = s.replace("−", "-")              # unicode minus
    s = s.rstrip(".")                    # trailing period
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))            # "5.0" -> "5"
        return repr(f)
    except ValueError:
        return s
```

## Libraries — concrete recommendations

### math-verify (HuggingFace) — **recommended for MATH**

Three-step pipeline: regex extraction → SymPy/ANTLR4 LaTeX parsing → symbolic comparison.

```bash
pip install "math-verify[antlr4_13_2]"        # grading only
pip install "math-verify[inference]"          # + generation utilities
```

```python
from math_verify import parse, verify

gold   = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")
assert verify(gold, answer)                   # True

gold   = parse("$\\frac{1}{2}$")
answer = parse("0.5")
assert verify(gold, answer)                   # True — numeric ↔ symbolic

# Control extraction targets:
from math_verify import LatexExtractionConfig, ExprExtractionConfig
gold = parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
```

Strengths: sets, intervals, matrices, complex numbers, equations/inequalities with
flip-aware comparison, boxed extraction, configurable float tolerance. Handles the
ANTLR4 version-skew that bites `sympy.parsing.latex`. Includes a batteries-included
`evaluate_model_outputs.py` CSV grader.

Weaknesses: still opinionated about LaTeX delimiters — raw "the answer is 5" without
a `$...$` wrapper needs `ExprExtractionConfig`. SymPy simplify can hang; keep a
timeout wrapper.

### lm-eval-harness `minerva_math`

The reference implementation of Minerva/Lewkowycz normalization. Good to pair with
math-verify as a **fallback** when math-verify returns `False` — disagreements are
genuine grader edge cases.

```python
# Adapted from lm_eval/tasks/minerva_math/utils.py
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string, remove_boxed, normalize_final_answer, is_equiv
)
boxed = last_boxed_only_string(generation)
if boxed:
    pred = normalize_final_answer(remove_boxed(boxed))
    correct = is_equiv(pred, normalize_final_answer(gold))
```

### SymPy-based homebrew (for `algebra__linear_1d`)

Plain numeric targets don't need the full LaTeX stack:

```python
import re, sympy

NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")

def extract_algebra_answer(gen: str) -> str | None:
    # Prefer boxed if present
    m = re.search(r"\\boxed\{([^}]*)\}", gen)
    if m: return m.group(1).strip()
    # Otherwise last number in last line
    last_line = gen.strip().splitlines()[-1]
    nums = NUM_RE.findall(last_line)
    return nums[-1] if nums else None

def numeric_equiv(pred: str, gold: str, tol: float = 1e-9) -> bool:
    try:
        p = sympy.sympify(pred.replace("−", "-"))
        g = sympy.sympify(gold.replace("−", "-"))
        return bool(sympy.simplify(p - g) == 0) or abs(float(p) - float(g)) < tol
    except Exception:
        return pred.strip() == gold.strip()
```

### Others

- **MathArena** (`mathematics_dataset` / DeepMind dm-math): the algebra tasks ship with
  plain-string gold answers; exact string match *after* whitespace/minus normalization is
  sufficient and is what the original eval does.
- **Minerva** (2022): the original paper's appendix; identical to lm-eval's implementation.
- **HuMath / OpenMathInstruct**: both now use math-verify under the hood.

**Current best practice (2025-2026):** use `math-verify` as the primary grader for MATH;
add lm-eval-harness `is_equiv` as a fallback that turns a `True` into a `True` but never
flips a `True` to `False` (this is what the Open-LLM-Leaderboard effectively does).

## Step segmentation for CoT

- **PRM800K** (OpenAI, Lightman et al. 2023): solutions are segmented by the
  model/human labeller at each distinct reasoning step; in the released data, steps are
  stored as separate JSON fields. Practical heuristic from the paper: split on blank
  lines or sentence boundaries where an equation appears.
- **Math-Shepherd** (Wang et al. 2023): rule-based segmentation, **newline as the
  delimiter**. Simple `solution.split("\n")` with empty-string filtering. The paper
  explicitly recommends this and shows it works at scale (PRM training data 4× bigger
  than PRM800K, with better downstream results).
- **More recent** (e.g. OmegaPRM, ReST-MCTS): allow any token span to be a step; in
  practice still default to newline for open-source reproductions.

Pragmatic reusable rule for us:

```python
def split_steps(cot: str) -> list[str]:
    steps = [s.strip() for s in cot.split("\n")]
    return [s for s in steps if s]
```

For Qwen3 thinking output, split inside the `<think>` block; the post-`</think>` content
is the final answer, not a reasoning step.

## Qwen3 thinking-mode output format

**Delimiters:** `<think>...</think>`. Token IDs: `<think>=151667`, `</think>=151668` in the
Qwen3 tokenizer. The chat template adds them automatically when
`tokenizer.apply_chat_template(..., enable_thinking=True)`.

Quirks worth knowing:

- On "thinking-only" checkpoints (e.g. `Qwen3-4B-Thinking-2507`, `Qwen3-30B-A3B-Thinking-2507`)
  the template pre-emits an opening `<think>\n`, so **the model's output contains only
  `</think>`** followed by the final answer. Do not require a matching `<think>` tag.
- If `enable_thinking=False`, the model emits *no* think block at all.
- Soft toggles: appending `/think` or `/no_think` in the user message overrides per-turn
  when `enable_thinking=True`. When `enable_thinking=False`, the toggles are ignored.
- For stable answer extraction, always force a `\boxed{...}` via the prompt (as DeepSeek-R1
  does): `"Please reason step by step, and put your final answer within \\boxed{}."`

Canonical split, using token IDs to avoid string-matching false positives inside the
reasoning trace:

```python
THINK_END = 151668
try:
    idx = len(output_ids) - output_ids[::-1].index(THINK_END)
except ValueError:
    idx = 0
thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
answer   = tokenizer.decode(output_ids[idx:],  skip_special_tokens=True).strip("\n")
```

## Self-consistency (maj@k)

Sample `K` completions with temperature > 0 (0.6–0.8 is common for Qwen/DeepSeek R1-style
models), extract an answer from each, **normalize** (same pipeline as grading), then
majority-vote. Normalization is load-bearing: without it, `1/2` and `0.5` vote as
different candidates and self-consistency degrades.

```python
from collections import Counter
def majority_vote(generations, extract, normalize):
    votes = Counter()
    for g in generations:
        a = extract(g)
        if a is None: continue
        votes[normalize(a)] += 1
    return votes.most_common(1)[0][0] if votes else None
```

Wang et al. 2022 report +27.6 pp on GSM8K, +23.7 pp on MATH for CoT + self-consistency
vs greedy CoT. Diminishing returns set in around K=40; for MATH-hard, K=64 is the usual
ceiling.

Confidence-weighted variants (CISC, DeepConf) reweight votes by response probability or
verbal confidence — worth trying for efficiency but not a prerequisite.

## Actionable pipeline for our project

```python
import re
from math_verify import parse as mv_parse, verify as mv_verify

THINK_END_TOKEN = 151668  # Qwen3 </think>

def strip_thinking(text: str) -> str:
    # If </think> is present, answer is what comes after; otherwise whole text.
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i >= 0 else text

def extract_math_answer(generation: str) -> str | None:
    tail = strip_thinking(generation)
    # 1. boxed (preferred)
    m = _last_boxed(tail) or _last_boxed(generation)
    if m:
        return _remove_boxed(m)
    # 2. "answer is" fallback in the tail
    m = re.search(r"(?:final answer|answer is)[:\s]*\$?([^\s.$]+)", tail, re.I)
    if m: return m.group(1)
    # 3. last LaTeX-ish token on the last non-empty line
    for line in reversed([l for l in tail.splitlines() if l.strip()]):
        toks = re.findall(r"[^\s,]+", line)
        if toks: return toks[-1]
    return None

NUM_RE = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
def extract_algebra_answer(generation: str) -> str | None:
    tail = strip_thinking(generation)
    m = _last_boxed(tail)
    if m: return _remove_boxed(m).strip()
    nums = NUM_RE.findall(tail.replace("−", "-"))
    return nums[-1] if nums else None

def score(pred: str | None, gold: str, dataset: str) -> bool:
    if pred is None: return False
    pred = pred.replace("−", "-").strip().rstrip(".")
    gold = gold.replace("−", "-").strip()
    if dataset == "math":
        try:
            return bool(mv_verify(mv_parse(gold), mv_parse(pred)))
        except Exception:
            return _minerva_is_equiv(pred, gold)       # fallback
    if dataset == "algebra":
        return _numeric_equiv(pred, gold)
    raise ValueError(dataset)
```

Helpers `_last_boxed`, `_remove_boxed`, `_minerva_is_equiv`, `_numeric_equiv` are the ones
shown above. Wrap `mv_verify` and `sympy.simplify` in a 5 s signal-alarm / subprocess
timeout.

## Pitfalls

- **Unicode minus (U+2212) vs ASCII "-"**: Qwen3 likes emitting Unicode minus in LaTeX.
  Normalize at every stage.
- **Smart quotes** `’` vs `'` in string answers — hits dm-math word-answers rarely but
  silently.
- **`.0` vs integer**: `5.0` ≠ `5` in string compare; use sympy or `float.is_integer()`.
- **Trailing period / comma** at the end of a sentence ("the answer is 5.") — always strip.
- **Multiple boxed answers**: take the **last**, not the first. Models often hedge.
- **Nested braces**: `\boxed{\frac{1}{2}}` breaks naive `[^}]*` regex. Use brace-match.
- **`</think>` inside the answer** (e.g. the model quotes its reasoning): use `rfind` /
  reverse-index of the token ID, not first match.
- **`\left(`, `\right)` decorations** confuse naive LaTeX parsers; math-verify handles
  them, but homebrew grading will miss equivalences like `\left(\frac{1}{2}\right) == 1/2`.
- **Matrix / set element order**: `{1,2,3}` vs `{3,1,2}` must compare as sets. math-verify
  does this; a plain string compare does not.
- **SymPy timeout**: `sympy.simplify` can hang on adversarial diffs; always wrap.
- **Empty `<think>` block**: when `enable_thinking=False` the model emits no tags —
  `strip_thinking` must be a no-op in that case (our `rfind` handles this: returns -1,
  slice returns the full text).
- **Prompting for `\boxed{}`**: if the prompt does not explicitly request boxed answers,
  extraction accuracy on vanilla Qwen3 drops noticeably on MATH. Add the DeepSeek-R1
  instruction verbatim.

## Sources

- HuggingFace Math-Verify: https://github.com/huggingface/Math-Verify
- HF blog, *Fixing the Open LLM Leaderboard with Math-Verify*:
  https://huggingface.co/blog/math_verify_leaderboard
- lm-eval-harness `minerva_math`:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py
- Lewkowycz et al., *Solving Quantitative Reasoning Problems with Language Models*
  (Minerva), 2022.
- Hendrycks et al., *Measuring Mathematical Problem Solving with the MATH Dataset*, 2021.
- PRM800K: https://github.com/openai/prm800k
- Wang et al., *Math-Shepherd*, 2023: https://arxiv.org/abs/2312.08935
- Wang et al., *Self-Consistency Improves CoT Reasoning*, 2022.
- DeepSeek-R1 model card & paper: https://huggingface.co/deepseek-ai/DeepSeek-R1 ,
  https://arxiv.org/abs/2501.12948
- Qwen3-8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3-4B-Thinking-2507 model card (notes the pre-emitted `<think>`):
  https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- DeepMind `mathematics_dataset` (source for `algebra__linear_1d`):
  https://github.com/google-deepmind/mathematics_dataset
