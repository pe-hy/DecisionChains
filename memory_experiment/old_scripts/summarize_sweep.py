#!/usr/bin/env python
"""
summarize_sweep.py — Final results summary after all phases complete.

Generates outputs/final_results.md with:
  - Phase 1 top configurations per group (ingredient, layer, mem, sparsity, ...)
  - Phase 2 seed-confirmed top configs with mean ± std across seeds
  - Phase 3 compound best with mean ± std
  - Comparison vs E4 reference and the "minimum viable" D_n1000_e10
"""

import json
from pathlib import Path
import statistics as stats

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "outputs" / "experiments"
OUT = SCRIPT_DIR / "outputs" / "final_results.md"


def load_all():
    runs = {}
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        runs[p.stem] = json.load(open(p))
    return runs


def metric(r, key, slice_="memory_full"):
    return r[slice_].get(key, 0)


def delta(r, key, slice_=""):
    bl = "baseline_fonly" if slice_ == "ffff" else "baseline_full"
    me = "memory_fonly" if slice_ == "ffff" else "memory_full"
    return r[me].get(key, 0) - r[bl].get(key, 0)


def agg_seeds(runs_list, key, slice_=""):
    vals = [delta(r, key, slice_) for r in runs_list]
    if not vals:
        return None, None
    return stats.mean(vals), (stats.stdev(vals) if len(vals) > 1 else 0.0)


def fmt_pm(mean, sd):
    if mean is None:
        return "-"
    return f"{mean:+.3f} ±{sd:.3f}"


def main():
    runs = load_all()
    lines = ["# Final Sweep Results", ""]

    # ── Phase 1 best per group ────────────────────────────────────────────
    groups = [
        ("Ingredient (A1-A5)", "A"),
        ("Layer (B_)",       "B_"),
        ("Memory size (C_)", "C_"),
        ("Sparsity (D_)",    "D_"),
        ("Learning rate (E_)", "E_"),
        ("Batch size (F_)",  "F_"),
        ("Epochs @ n1000 (G_)", "G_"),
        ("Epochs @ n3000 (H_)", "H_"),
        ("Compound (I_)",    "I_"),
    ]
    lines.append("## Phase 1 — Top configuration per group")
    lines.append("")
    lines.append("Ranked by `full_val f_selection` delta. `op_acc` is the cost on full val.")
    lines.append("")
    lines.append("| Group | Winner | Δ full f_sel | Δ full full_align | Δ full op_acc |")
    lines.append("|---|---|---|---|---|")
    for label, prefix in groups:
        group = [r for n, r in runs.items()
                 if n.startswith(prefix) and not n.startswith("_")
                 and not n.startswith("P2_") and not n.startswith("P3_")]
        if not group:
            lines.append(f"| {label} | (no runs) | - | - | - |")
            continue
        w = max(group, key=lambda r: delta(r, "f_selection"))
        lines.append(
            f"| {label} | `{w['name']}` | "
            f"{delta(w, 'f_selection'):+.3f} | "
            f"{delta(w, 'full_f_alignment'):+.3f} | "
            f"{delta(w, 'operation_accuracy'):+.3f} |"
        )
    lines.append("")

    # ── Phase 2 seed-confirmed top 5 ──────────────────────────────────────
    phase2 = [r for n, r in runs.items() if n.startswith("P2_")]
    if phase2:
        lines.append("## Phase 2 — Top-5 configs, 3-seed confirmation")
        lines.append("")
        lines.append("At `n_eval=200`, each config × seeds {0,1,2}.")
        lines.append("")
        # Group by base config (strip _s0/_s1/_s2)
        by_base = {}
        for r in phase2:
            base = r["name"].rsplit("_s", 1)[0]
            by_base.setdefault(base, []).append(r)
        lines.append("| Config | Δ full f_sel | Δ full full_align | Δ full op_acc |")
        lines.append("|---|---|---|---|")
        order = sorted(by_base.items(),
                       key=lambda kv: -agg_seeds(kv[1], "f_selection")[0])
        for base, group in order:
            m, sd = agg_seeds(group, "f_selection")
            fm, fsd = agg_seeds(group, "full_f_alignment")
            om, osd = agg_seeds(group, "operation_accuracy")
            lines.append(f"| `{base}` ({len(group)} seeds) | {fmt_pm(m, sd)} | "
                         f"{fmt_pm(fm, fsd)} | {fmt_pm(om, osd)} |")
        lines.append("")

    # ── Phase 3 compound ──────────────────────────────────────────────────
    phase3 = [r for n, r in runs.items() if n.startswith("P3_")]
    if phase3:
        lines.append("## Phase 3 — Compound best ingredients")
        lines.append("")
        lines.append("At `n_eval=300`. `compound` config is (best layer, sparsity, mem, lr, bs) "
                     "from Phase 1 ingredient ablations, varying epoch budget.")
        lines.append("")
        by_base = {}
        for r in phase3:
            base = r["name"].rsplit("_s", 1)[0]
            by_base.setdefault(base, []).append(r)
        lines.append("| Config | Δ full f_sel | Δ full full_align | Δ full op_acc |")
        lines.append("|---|---|---|---|")
        order = sorted(by_base.items(),
                       key=lambda kv: -agg_seeds(kv[1], "f_selection")[0])
        for base, group in order:
            m, sd = agg_seeds(group, "f_selection")
            fm, fsd = agg_seeds(group, "full_f_alignment")
            om, osd = agg_seeds(group, "operation_accuracy")
            lines.append(f"| `{base}` ({len(group)} seeds) | {fmt_pm(m, sd)} | "
                         f"{fmt_pm(fm, fsd)} | {fmt_pm(om, osd)} |")
        lines.append("")

    # ── Anchors for context ───────────────────────────────────────────────
    lines.append("## Anchors")
    lines.append("")
    lines.append("| Reference | Δ full f_sel | Δ full full_align | Δ full op_acc |")
    lines.append("|---|---|---|---|")
    for anchor in ["E4_all_fullseq_f", "D_n1000_e10", "D_n3000_e10"]:
        if anchor not in runs:
            continue
        r = runs[anchor]
        lines.append(f"| `{anchor}` | {delta(r, 'f_selection'):+.3f} | "
                     f"{delta(r, 'full_f_alignment'):+.3f} | "
                     f"{delta(r, 'operation_accuracy'):+.3f} |")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
