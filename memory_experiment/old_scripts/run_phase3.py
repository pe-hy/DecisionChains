#!/usr/bin/env python
"""
Phase 3 — Compound best ingredients.

Takes the best layer, sparsity coefficient, memory size, learning rate, and
batch size from Phase 1 ablations, combines them, and runs 3 seeds at
n_eval=300 on two compute budgets (n=3000 × 10ep and n=3000 × 30ep).
Plus a "cheap E4 killer": n=3000 × 30ep at the best compound settings.
"""

import json
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "outputs" / "experiments"


def load(prefix):
    """Load all Phase 1 JSONs whose name starts with `prefix`."""
    out = []
    for p in sorted(RESULTS.glob(f"{prefix}*.json")):
        if p.name.startswith("_"):
            continue
        if p.stem in {"A1_ref_s1", "A1_ref_s2"}:
            continue
        out.append(json.load(open(p)))
    return out


def best_by(runs, key="f_selection", slice_="memory_full"):
    if not runs:
        return None
    return max(runs, key=lambda r: r[slice_].get(key, 0))


def run_cmd(name, args, wandb_group="p3_final"):
    if (RESULTS / f"{name}.json").exists():
        print(f">>> SKIP {name}")
        return 0
    cmd = ["python", "exp.py", "--name", name] + args + [
        "--wandb", "--wandb_project", "memory-experiment",
        "--wandb_group", wandb_group,
    ]
    print(f"==== START {name} ====")
    rc = subprocess.run(cmd).returncode
    print(f"==== {'DONE' if rc == 0 else f'FAILED rc={rc}'} {name} ====")
    return rc


def main():
    os.chdir(SCRIPT_DIR)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WANDB_DIR", "logs")

    # Determine each best ingredient independently (single-axis ablations).
    best_layer_run = best_by(load("B_"))
    best_sparsity_run = best_by(load("D_"))
    best_mem_run = best_by(load("C_"))
    best_lr_run = best_by(load("E_"))
    best_bs_run = best_by(load("F_"))
    best_compound_run = best_by(load("I_"))

    print("Best ingredient found by full-val f_selection:")
    for name, r in [("layer", best_layer_run), ("sparsity", best_sparsity_run),
                    ("mem_size", best_mem_run), ("lr", best_lr_run),
                    ("batch_size", best_bs_run), ("compound", best_compound_run)]:
        if r is None:
            print(f"  {name:10s}: (no Phase 1 runs)")
            continue
        fs = r["memory_full"].get("f_selection", 0)
        op = r["memory_full"].get("operation_accuracy", 0)
        print(f"  {name:10s}: {r['name']:25s}  full f_sel={fs:.3f}  op_acc={op:.3f}")

    if not all([best_layer_run, best_sparsity_run, best_mem_run,
                best_lr_run, best_bs_run]):
        print("\nERROR: Phase 1 did not produce enough runs to compound ingredients.")
        return

    layers = str(best_layer_run["config"]["layers"])
    mem_entries = str(best_mem_run["config"]["mem_entries"])
    lr = str(best_lr_run["config"]["lr"])
    batch_size = str(best_bs_run["config"]["batch_size"])
    sp_mode = best_sparsity_run["config"]["sparsity"]
    sp_coeff = str(best_sparsity_run["config"]["sparsity_coeff"])

    base_args = [
        "--data_filter", "all", "--ce_mode", "full_seq", "--dp_target", "f",
        "--gate",
        "--layers", layers,
        "--mem_entries", mem_entries,
        "--sparsity", sp_mode, "--sparsity_coeff", sp_coeff,
        "--lr", lr,
        "--batch_size", batch_size,
    ]
    print(f"\nCompound config: layers={layers}  mem={mem_entries}  "
          f"lr={lr}  bs={batch_size}  sparsity={sp_mode}(λ={sp_coeff})")

    print("\n=== Phase 3: compound × (n=3000 × 10ep, 20ep, 30ep) × 3 seeds × n_eval=300 ===\n")

    # Three compute budgets, each with 3 seeds.
    for n_train, epochs in [(3000, 10), (3000, 20), (3000, 30)]:
        for seed in [0, 1, 2]:
            name = f"P3_compound_n{n_train}_e{epochs}_s{seed}"
            args = base_args + [
                "--n_train", str(n_train),
                "--epochs", str(epochs),
                "--n_eval", "300",
                "--seed", str(seed),
            ]
            run_cmd(name, args, wandb_group="p3_final")

    # Bonus: the compound best_compound_run (hand-crafted small-mem + sparsity)
    # re-run at n=3000, e=20, 3 seeds, n_eval=300.
    if best_compound_run is not None:
        print("\n=== Phase 3 bonus: replay best I_* compound config at bigger compute ===\n")
        c = best_compound_run["config"]
        bonus_args = [
            "--data_filter", c["data_filter"], "--ce_mode", c["ce_mode"],
            "--dp_target", c["dp_target"],
            "--layers", str(c["layers"]),
            "--mem_entries", str(c["mem_entries"]),
            "--sparsity", c["sparsity"],
            "--sparsity_coeff", str(c.get("sparsity_coeff", 0.1)),
            "--lr", str(c["lr"]),
            "--batch_size", str(c.get("batch_size", 8)),
            "--n_train", "3000", "--epochs", "20", "--n_eval", "300",
        ]
        if c.get("gate"):
            bonus_args.append("--gate")
        for seed in [0, 1, 2]:
            name = f"P3_bonus_{best_compound_run['name']}_s{seed}"
            args = bonus_args + ["--seed", str(seed)]
            run_cmd(name, args, wandb_group="p3_final")

    print("\n==== PHASE 3 DONE ====")


if __name__ == "__main__":
    main()
