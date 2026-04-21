#!/usr/bin/env python
"""
compare_vs_memory.py — Headline table: memory winner vs naive FT / LoRA.

Pulls runs from:
  - ./outputs/experiments/            (FT variants)
  - ../memory_experiment/outputs/experiments/  (memory winner)

Emits a single comparison table.
"""

import json
import statistics as stats
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FT_DIR = SCRIPT_DIR / "outputs" / "experiments"
MEM_DIR = SCRIPT_DIR.parent / "memory_experiment" / "outputs" / "experiments"


def load(path):
    if not path.exists():
        return None
    return json.load(open(path))


def delta(r, slice_, key):
    bl = "baseline_fonly" if slice_ == "ffff" else "baseline_full"
    me = "memory_fonly" if slice_ == "ffff" else "memory_full"
    return r[me][key] - r[bl][key]


def mean_seeds(paths, slice_, key):
    vs = [delta(load(p), slice_, key) for p in paths if load(p) is not None]
    if not vs:
        return None, None
    return stats.mean(vs), (stats.stdev(vs) if len(vs) > 1 else 0.0)


def fmt_pm(m, s):
    if m is None:
        return "-"
    return f"{m:+.3f} ±{s:.3f}"


def main():
    # Memory winner — 3 seeds at n=3000, e=10
    mem_paths = [MEM_DIR / f"P3_compound_n3000_e10_s{i}.json" for i in range(3)]
    mem_runs = [load(p) for p in mem_paths if load(p) is not None]

    ft_paths = {
        "FT_full_A": FT_DIR / "FT_full_A_n3000_e10.json",
        "FT_full_B": FT_DIR / "FT_full_B_n3000_e10.json",
        "FT_lora_A": FT_DIR / "FT_lora_A_n3000_e10.json",
        "FT_lora_B": FT_DIR / "FT_lora_B_n3000_e10.json",
    }

    # Collect rows
    rows = []
    if mem_runs:
        r0 = mem_runs[0]
        m_fs, s_fs = mean_seeds(mem_paths, "full", "f_selection")
        m_fa, s_fa = mean_seeds(mem_paths, "full", "full_f_alignment")
        m_oa, s_oa = mean_seeds(mem_paths, "full", "operation_accuracy")
        tt = stats.mean([r["train_time_sec"] for r in mem_runs])
        rows.append((
            "Memory (winner, 3 seeds)",
            r0["trainable_params"], tt,
            fmt_pm(m_fs, s_fs), fmt_pm(m_fa, s_fa), fmt_pm(m_oa, s_oa),
        ))

    for label, path in ft_paths.items():
        r = load(path)
        if r is None:
            rows.append((label, "-", "-", "(pending)", "(pending)", "(pending)"))
            continue
        rows.append((
            label,
            r["trainable_params"], r["train_time_sec"],
            f"{delta(r, 'full', 'f_selection'):+.3f}",
            f"{delta(r, 'full', 'full_f_alignment'):+.3f}",
            f"{delta(r, 'full', 'operation_accuracy'):+.3f}",
        ))

    # Print table
    col_w = (30, 14, 11, 18, 18, 18)
    hdr = ("method", "trainable", "time (s)", "full f_sel Δ", "full full_align Δ", "full op_acc Δ")
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(hdr, col_w)))
    print("-+-".join("-" * w for w in col_w))
    for row in rows:
        cells = []
        for v, w in zip(row, col_w):
            if isinstance(v, (int, float)):
                cells.append(f"{v:,.1f}".ljust(w) if isinstance(v, float) else f"{v:,}".ljust(w))
            else:
                cells.append(str(v).ljust(w))
        print(" | ".join(cells))
    print()


if __name__ == "__main__":
    main()
