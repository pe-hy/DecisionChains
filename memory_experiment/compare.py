"""
compare.py — Load all experiment JSONs and print ranked comparisons.

Usage:
  python compare.py                     # summarize all outputs/experiments/*.json
  python compare.py --only E1,E3,E4     # subset by name
"""

import argparse, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "outputs" / "experiments"

METRICS = ["operation_accuracy", "f_selection", "full_f_alignment", "complete_solution"]


def fmt_delta(bl, me):
    return f"{bl:.3f}→{me:.3f} ({me - bl:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated names to include")
    args = ap.parse_args()

    # Skip hidden/helper files (e.g. _baseline_cache_*.json).
    paths = sorted(p for p in RESULTS_DIR.glob("*.json") if not p.name.startswith("_"))
    runs = [json.load(open(p)) for p in paths]
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r["name"] in keep]
    if not runs:
        print(f"No experiment JSONs in {RESULTS_DIR}")
        return

    print(f"Loaded {len(runs)} experiment(s) from {RESULTS_DIR}\n")

    print("=" * 110)
    print("CONFIGURATIONS")
    print("=" * 110)
    print(f"{'name':18s}  {'data':5s}  {'ce':10s}  {'tgt':3s}  {'layers':8s}  "
          f"{'mem':4s}  {'gate':5s}  {'spars':8s}  {'epochs':6s}  {'n_train':7s}")
    for r in runs:
        c = r["config"]
        print(f"{r['name']:18s}  {c['data_filter']:5s}  {c['ce_mode']:10s}  "
              f"{c['dp_target']:3s}  {c['layers']:8s}  {c['mem_entries']:4d}  "
              f"{('yes' if c.get('gate') else 'no'):5s}  {c['sparsity']:8s}  "
              f"{c['epochs']:6d}  {c['n_train']:7d}")

    for slice_name, bl_key, mem_key in [("ffff val", "baseline_fonly", "memory_fonly"),
                                         ("full val", "baseline_full", "memory_full")]:
        print("\n" + "=" * 110)
        print(f"{slice_name.upper()}  (baseline → memory (delta))")
        print("=" * 110)
        header = f"{'name':18s}"
        for m in METRICS:
            header += f"  {m[:18]:>20s}"
        print(header)
        for r in runs:
            row = f"{r['name']:18s}"
            for m in METRICS:
                bl = r[bl_key].get(m, 0)
                me = r[mem_key].get(m, 0)
                row += f"  {fmt_delta(bl, me):>20s}"
            print(row)

    print("\n" + "=" * 110)
    print("WINNERS  (largest positive delta for each metric)")
    print("=" * 110)
    for slice_name, bl_key, mem_key in [("ffff val", "baseline_fonly", "memory_fonly"),
                                         ("full val", "baseline_full", "memory_full")]:
        for m in METRICS:
            # Skip near-zero metrics on full val
            if slice_name == "full val" and m in ("complete_solution",):
                continue
            best = max(runs, key=lambda r: r[mem_key].get(m, 0) - r[bl_key].get(m, 0))
            bl = best[bl_key].get(m, 0)
            me = best[mem_key].get(m, 0)
            print(f"  {slice_name:9s}  {m:24s}  winner={best['name']:18s}  "
                  f"{fmt_delta(bl, me)}")

    # Overall ranking by absolute f_selection on full val (true generalization test)
    print("\n" + "=" * 110)
    print("OVERALL RANKING  (absolute f_selection on full val — generalization)")
    print("=" * 110)
    ranked = sorted(runs, key=lambda r: -r["memory_full"].get("f_selection", 0))
    for i, r in enumerate(ranked, 1):
        fsel_full = r["memory_full"].get("f_selection", 0)
        ffa_full = r["memory_full"].get("full_f_alignment", 0)
        fsel_fo = r["memory_fonly"].get("f_selection", 0)
        ffa_fo = r["memory_fonly"].get("full_f_alignment", 0)
        opacc_full = r["memory_full"].get("operation_accuracy", 0)
        print(f"  {i}. {r['name']:18s}  "
              f"full: f_sel={fsel_full:.3f}  full_align={ffa_full:.3f}  op={opacc_full:.3f}   "
              f"ffff: f_sel={fsel_fo:.3f}  full_align={ffa_fo:.3f}")


if __name__ == "__main__":
    main()
