#!/usr/bin/env python
"""Build Jarník Mathematical Competition dataset from PDFs.

Inputs: data/jarnik_raw/j{YY}{problems|solutions}{1|2}.pdf
Outputs:
    data/jarnik_raw/text/j{YY}{problems|solutions}{1|2}.txt   (pdftotext dump)
    data/jarnik/test.jsonl                                     (final dataset)

Schema (matches AIME extension):
    {
        "problem_idx": int,        # global running index
        "year_idx": int,           # j-edition (25..32)
        "category": int,           # 1 or 2
        "problem_num": int,        # problem number within year/category (1..4)
        "problem": str,            # problem statement
        "answer": str,             # "proof" placeholder, or extracted numeric/text
        "solution": str,           # full solution text
    }

Splitting heuristic: each PDF starts with header, then 'Problem N' markers.
We split on those, take text up to next 'Problem' or end of doc, strip header.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "jarnik_raw"
TXT_DIR = RAW / "text"
OUT_DIR = ROOT / "data" / "jarnik"
OUT_FILE = OUT_DIR / "test.jsonl"

YEARS = list(range(25, 33))   # j25..j32
CATEGORIES = [1, 2]

PROBLEM_MARK = re.compile(r"\bProblem\s+(\d+)\b")
POINTS_MARK = re.compile(r"\[\s*\d+\s*points\s*\]")
PAGE_BREAK = "\x0c"


def pdftotext(pdf: Path, txt: Path):
    txt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(txt)],
        check=True,
    )


def normalize(s: str) -> str:
    """Light cleanup: collapse runs of whitespace, drop page-break + footers."""
    # remove page breaks
    s = s.replace(PAGE_BREAK, "\n")
    # drop common footer dates like "01-May-2025 9:28"
    s = re.sub(r"\b\d{2}-[A-Z][a-z]+-\d{4}\s+\d{1,2}:\d{2}\b", "", s)
    # collapse blank-line runs
    s = re.sub(r"\n{3,}", "\n\n", s)
    # rstrip each line
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


def split_problems(text: str) -> dict[int, str]:
    """Return {problem_num: body_text}."""
    out = {}
    matches = list(PROBLEM_MARK.finditer(text))
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # strip leading whitespace
        body = body.lstrip(" \t\n")
        # drop trailing "[10 points]" markers and trailing whitespace
        body = POINTS_MARK.sub("", body).strip()
        # only first occurrence wins (problems doc has each Problem N once)
        if n not in out:
            out[n] = body
    return out


def main():
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    pidx = 0

    for y in YEARS:
        for c in CATEGORIES:
            pdf_p = RAW / f"j{y:02d}problems{c}.pdf"
            pdf_s = RAW / f"j{y:02d}solutions{c}.pdf"
            txt_p = TXT_DIR / f"j{y:02d}problems{c}.txt"
            txt_s = TXT_DIR / f"j{y:02d}solutions{c}.txt"
            if not pdf_p.exists() or not pdf_s.exists():
                print(f"skip y={y} c={c}: missing pdf")
                continue
            pdftotext(pdf_p, txt_p)
            pdftotext(pdf_s, txt_s)

            txt_p_data = normalize(txt_p.read_text())
            txt_s_data = normalize(txt_s.read_text())
            problems = split_problems(txt_p_data)
            solutions = split_problems(txt_s_data)

            for n in sorted(problems):
                if n not in solutions:
                    print(f"warn y={y} c={c} p={n}: problem present, "
                          f"solution missing")
                    continue
                records.append({
                    "problem_idx": pidx,
                    "year_idx": y,
                    "category": c,
                    "problem_num": n,
                    "problem": problems[n],
                    "answer": "proof",   # placeholder; user will refine
                    "solution": solutions[n],
                })
                pidx += 1
            print(f"j{y:02d} cat{c}: {len(problems)} problems, "
                  f"{len(solutions)} solutions, {len(records)-pidx+len(problems)} merged so far")

    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} records to {OUT_FILE}")
    # quick stats
    from collections import Counter
    print("by year:", Counter(r["year_idx"] for r in records))
    print("by cat:",  Counter(r["category"] for r in records))


if __name__ == "__main__":
    main()
