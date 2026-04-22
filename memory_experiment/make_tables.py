#!/usr/bin/env python
"""
make_tables.py — Single source of truth for all experiment results.

Reads every JSON in outputs/experiments/ (memory) and ../naive_ft/outputs/experiments/ (FT),
groups multi-seed runs, computes mean ± std, and prints clean tables.

Usage:
  python make_tables.py              # all runs
  python make_tables.py --v2         # only v2_ prefix runs (new 2k eval)
  python make_tables.py --prefix L2  # only runs starting with L2
"""

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MEM_DIR = SCRIPT_DIR / "outputs" / "experiments"
FT_DIR = SCRIPT_DIR.parent / "naive_ft" / "outputs" / "experiments"

METRICS = ["f_selection", "full_f_alignment", "operation_accuracy",
           "f_or_g_valid_selection", "chain_matches_output", "complete_solution"]
PRIMARY = ["f_selection", "full_f_alignment", "operation_accuracy"]


def load_all(dirs, prefix=None, v2_only=False):
    runs = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            name = p.stem
            if v2_only and not name.startswith("v2_"):
                continue
            if prefix and not name.startswith(prefix):
                continue
            try:
                r = json.load(open(p))
                if "config" not in r:
                    continue
                r["_source"] = "ft" if "naive_ft" in str(d) else "mem"
                runs[name] = r
            except Exception:
                pass
    return runs


def group_seeds(runs):
    """Group runs that differ only by _s{N} suffix."""
    groups = defaultdict(list)
    for name, r in runs.items():
        # Strip seed suffix to find base name
        base = name
        for suffix in ["_s0", "_s1", "_s2", "_s3"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        groups[base].append((name, r))
    return groups


def fmt_pct(val):
    return f"{val * 100:.1f}"


def fmt_delta(bl, mem):
    d = (mem - bl) * 100
    return f"{d:+.1f}"


def fmt_pm(vals_pct):
    """Format list of percentage values as mean ± std."""
    m = stats.mean(vals_pct)
    if len(vals_pct) > 1:
        s = stats.stdev(vals_pct)
        return f"{m:.1f}±{s:.1f}"
    return f"{m:.1f}"


def compute_row(group_runs):
    """Compute aggregated metrics for a group of runs (possibly multi-seed)."""
    row = {}
    n_seeds = len(group_runs)
    row["n_seeds"] = n_seeds

    # Config from first run
    first = group_runs[0][1]
    c = first["config"]
    row["source"] = first["_source"]
    row["method"] = c.get("method", "memory")
    row["mode"] = c.get("mode", "B")
    row["layers"] = str(c.get("layers", ""))
    row["mem_entries"] = c.get("mem_entries", c.get("lora_rank", ""))
    row["n_train"] = c.get("n_train", "")
    row["epochs"] = c.get("epochs", "")
    row["lr"] = c.get("lr", "")
    row["batch_size"] = c.get("batch_size", "")
    row["gate"] = c.get("gate", "")
    row["sparsity"] = c.get("sparsity", "")
    row["trainable_params"] = first.get("trainable_params", "")
    row["train_time"] = stats.mean([r.get("train_time_sec", 0) for _, r in group_runs])
    row["n_eval_ffff"] = first.get("n_val_fonly_actual", "")
    row["n_eval_full"] = first.get("n_val_full_actual", "")

    for sl_key, sl_prefix in [("memory_fonly", "ffff"),
                               ("memory_full", "full"),
                               ("baseline_fonly", "bl_ffff"),
                               ("baseline_full", "bl_full")]:
        for k in METRICS:
            vals = []
            for _, r in group_runs:
                v = r.get(sl_key, {}).get(k)
                if v is not None:
                    vals.append(v * 100)
            if vals:
                row[f"{sl_prefix}_{k}"] = fmt_pm(vals)
                row[f"{sl_prefix}_{k}_mean"] = stats.mean(vals)
            else:
                row[f"{sl_prefix}_{k}"] = "-"
                row[f"{sl_prefix}_{k}_mean"] = None

    # Deltas
    for sl in ["ffff", "full"]:
        for k in PRIMARY:
            mem_vals = []
            bl_vals = []
            for _, r in group_runs:
                bl_key = "baseline_fonly" if sl == "ffff" else "baseline_full"
                me_key = "memory_fonly" if sl == "ffff" else "memory_full"
                bv = r.get(bl_key, {}).get(k)
                mv = r.get(me_key, {}).get(k)
                if bv is not None and mv is not None:
                    mem_vals.append((mv - bv) * 100)
            if mem_vals:
                row[f"d_{sl}_{k}"] = fmt_pm(mem_vals)
                row[f"d_{sl}_{k}_mean"] = stats.mean(mem_vals)
            else:
                row[f"d_{sl}_{k}"] = "-"
                row[f"d_{sl}_{k}_mean"] = None

    return row


def print_table(groups, sort_key="d_full_f_selection_mean"):
    rows = []
    for base, group_runs in groups.items():
        row = compute_row(group_runs)
        row["name"] = base
        rows.append(row)

    # Sort by the chosen metric (descending), None at bottom
    rows.sort(key=lambda r: -(r.get(sort_key) or -9999))

    # Print header
    print(f"{'name':35s} {'src':>4s} {'seeds':>5s} {'params':>8s} {'time':>6s} "
          f"{'n_eval':>6s}  "
          f"{'Δfull f_sel':>12s} {'Δfull align':>12s} {'Δfull op':>12s}  "
          f"{'Δffff f_sel':>12s} {'Δffff op':>12s}  "
          f"{'full f_sel':>11s} {'full align':>11s} {'full op':>11s}")
    print("─" * 185)

    for r in rows:
        params_s = f"{r['trainable_params']//1000}K" if isinstance(r['trainable_params'], int) else str(r['trainable_params'])
        time_s = f"{r['train_time']:.0f}s" if r['train_time'] else ""
        n_eval_s = str(r.get('n_eval_full', ''))
        print(f"{r['name']:35s} {r['source']:>4s} {r['n_seeds']:>5d} {params_s:>8s} {time_s:>6s} "
              f"{n_eval_s:>6s}  "
              f"{r['d_full_f_selection']:>12s} {r['d_full_full_f_alignment']:>12s} {r['d_full_operation_accuracy']:>12s}  "
              f"{r['d_ffff_f_selection']:>12s} {r['d_ffff_operation_accuracy']:>12s}  "
              f"{r['full_f_selection']:>11s} {r['full_full_f_alignment']:>11s} {r['full_operation_accuracy']:>11s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="Only v2_ runs")
    ap.add_argument("--prefix", default=None, help="Filter by name prefix")
    ap.add_argument("--sort", default="d_full_f_selection_mean",
                    help="Sort key (default: d_full_f_selection_mean)")
    args = ap.parse_args()

    runs = load_all([MEM_DIR, FT_DIR], prefix=args.prefix, v2_only=args.v2)
    if not runs:
        print("No runs found.")
        return

    groups = group_seeds(runs)
    print(f"Loaded {len(runs)} runs → {len(groups)} groups\n")
    print_table(groups, sort_key=args.sort)


if __name__ == "__main__":
    main()
