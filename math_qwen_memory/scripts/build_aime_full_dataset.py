"""Build AIME 2021-2025 single JSONL from public sources (last 5 years).

Sources (all already downloaded into data/AIME_2021_2025/_raw/):
  - di-zhang-fdu/AIME_1983_2024  (CSV; we filter to year>=2016 — covers
    2016-2023 fully + AIME 2024-II minus problem 9; missing 2023-I-15)
  - opencompass/AIME2025         (JSONLs; 2025-I + 2025-II, 30 problems)
  - AI-MO/aimo-validation-aime   (parquet; AIME 22-24 — fills 2024-I and the
    2023-I-15 / 2024-II-9 gaps via the URL field, since the parquet's `id`
    column is an opaque int)

Output schema (mirrors data/AIME_2026/test.jsonl):
  {"idx": int, "problem": str, "answer": str, "year": int, "part": str|None,
   "number": int|None, "source": str}

Records are sorted chronologically (year, part, number) and idx is assigned
in that order so downstream sharding by `i % n_shards` is deterministic.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

YEAR_FROM = 2021  # last 5 years, inclusive of YEAR_FROM..YEAR_TO
YEAR_TO   = 2025  # exclude 2026 (lives separately in data/AIME_2026/)

RAW = Path("data/AIME_2021_2025/_raw")
OUT_DIR = Path("data/AIME_2021_2025")
OUT = OUT_DIR / "test.jsonl"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def load_csv() -> list[dict]:
    out = []
    with open(RAW / "AIME_1983_2024.csv", newline="") as f:
        for r in csv.DictReader(f):
            year = int(r["Year"])
            if year < YEAR_FROM or year > YEAR_TO:
                continue
            part = r["Part"].strip() or None
            num = int(r["Problem Number"])
            out.append({
                "year": year,
                "part": part,
                "number": num,
                "problem": _norm(r["Question"]),
                "answer": _norm(r["Answer"]),
                "source": "di-zhang-fdu/AIME_1983_2024",
            })
    return out


def load_aime2025() -> list[dict]:
    out = []
    for part in ("I", "II"):
        path = RAW / f"aime2025-{part}.jsonl"
        with open(path) as f:
            for i, line in enumerate(f, start=1):
                r = json.loads(line)
                out.append({
                    "year": 2025,
                    "part": part,
                    "number": i,
                    "problem": _norm(r["question"]),
                    "answer": _norm(str(r["answer"])),
                    "source": "opencompass/AIME2025",
                })
    return out


def load_aimo_validation() -> list[dict]:
    """AIME 22-24, used as gap-filler.  The parquet's `id` is an opaque int,
    so we parse year/part/number from the wiki `url` field.
    """
    import pyarrow.parquet as pq
    t = pq.read_table(RAW / "aimo_validation_aime.parquet")
    rows = t.to_pylist()
    print(f"  aimo-validation columns: {list(rows[0].keys())}", flush=True)

    out = []
    skipped = 0
    for r in rows:
        url = r.get("url", "") or ""
        # ".../wiki/index.php/2024_AIME_I_Problems/Problem_3"
        m = re.search(r"/(\d{4})_AIME_(I+)_Problems/Problem_(\d+)", url)
        if not m:
            skipped += 1
            continue
        year = int(m.group(1))
        if year < YEAR_FROM or year > YEAR_TO:
            continue
        out.append({
            "year": year,
            "part": m.group(2),
            "number": int(m.group(3)),
            "problem": _norm(r["problem"]),
            "answer": _norm(str(r["answer"])),
            "source": "AI-MO/aimo-validation-aime",
        })
    if skipped:
        print(f"  WARN: skipped {skipped} aimo rows with unparseable url", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Merge + write
# --------------------------------------------------------------------------- #

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] CSV (year>={YEAR_FROM})…", flush=True)
    csv_rows = load_csv()
    print(f"      → {len(csv_rows)} rows", flush=True)

    print("[2/4] opencompass AIME 2025…", flush=True)
    y2025 = load_aime2025()
    print(f"      → {len(y2025)} rows", flush=True)

    print("[3/4] aimo-validation parquet (AIME 22-24)…", flush=True)
    aimo = load_aimo_validation()
    print(f"      → {len(aimo)} rows", flush=True)

    # Dedup by (year, part, number) with priority CSV > opencompass > aimo.
    by_key: dict[tuple, dict] = {}
    for r in csv_rows + y2025:
        by_key.setdefault((r["year"], r["part"], r["number"]), r)
    filled = 0
    for r in aimo:
        key = (r["year"], r["part"], r["number"])
        if key in by_key:
            continue
        by_key[key] = r
        filled += 1
    print(f"      → filled {filled} gaps from aimo validation", flush=True)

    def sort_key(r):
        part_rank = {None: 0, "": 0, "I": 1, "II": 2}.get(r["part"], 99)
        return (r["year"], part_rank, r["number"])
    rows = sorted(by_key.values(), key=sort_key)

    # Coverage report — 2016+ should all be 30/year.
    from collections import Counter
    per_year = Counter(r["year"] for r in rows)
    print("[4/4] coverage:", flush=True)
    expected = 30
    gaps = []
    for y in sorted(per_year):
        marker = "" if per_year[y] == expected else "  ← gap"
        print(f"      {y}: {per_year[y]} (expected {expected}){marker}", flush=True)
        if per_year[y] < expected:
            # Report which (part, number) tuples are missing for that year.
            seen = {(r["part"], r["number"]) for r in rows if r["year"] == y}
            missing = [(p, n) for p in ("I", "II") for n in range(1, 16)
                       if (p, n) not in seen]
            print(f"        missing: {missing}", flush=True)
            gaps.append((y, expected - per_year[y]))

    # Spot-check answers — they should all be 0..999 numeric for AIME.
    bad_ans = [r for r in rows if not re.fullmatch(r"\d{1,3}", r["answer"])]
    if bad_ans:
        print(f"[warn] {len(bad_ans)} non-numeric answers; sample: "
              f"{bad_ans[0]['answer']!r} (year {bad_ans[0]['year']})", flush=True)

    print(f"[write] {OUT}", flush=True)
    with OUT.open("w") as f:
        for i, r in enumerate(rows):
            rec = {
                "idx": i,
                "problem": r["problem"],
                "answer": r["answer"],
                "year": r["year"],
                "part": r["part"],
                "number": r["number"],
                "source": r["source"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} records to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
