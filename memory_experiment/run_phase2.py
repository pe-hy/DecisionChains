#!/usr/bin/env python
"""
Phase 2 — Confirm top-5 Phase 1 configs with 3 seeds each at higher n_eval.

Reads outputs/experiments/{A,B,C,D,E,F,G,H,I}_*.json, ranks by full-val
f_selection, and runs 3 seeds × top 5 = 15 runs at n_eval=200.
"""

import json
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "outputs" / "experiments"

PHASE1_PREFIXES = ("A", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_")


def load_phase1_runs():
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        name = p.stem
        # Phase 1 runs: A* B_* C_* ... (prefixes above). Skip seed-duplicate A1_ref_s1/s2.
        if not (name.startswith(PHASE1_PREFIXES) or name == "A1_ref"):
            continue
        if name in {"A1_ref_s1", "A1_ref_s2"}:  # deduplicate seed copies
            continue
        runs.append(json.load(open(p)))
    return runs


def run_cmd(name, config, extra_flags=None, seed=0, n_eval=200,
            wandb_group="p2_top5_seeds"):
    if (RESULTS / f"{name}.json").exists():
        print(f">>> SKIP {name} (already done)")
        return 0
    cmd = [
        "python", "exp.py", "--name", name,
        "--data_filter", config["data_filter"],
        "--ce_mode", config["ce_mode"],
        "--dp_target", config["dp_target"],
        "--layers", str(config["layers"]),
        "--mem_entries", str(config["mem_entries"]),
        "--sparsity", config["sparsity"],
        "--sparsity_coeff", str(config.get("sparsity_coeff", 0.1)),
        "--n_train", str(config["n_train"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--batch_size", str(config.get("batch_size", 8)),
        "--n_eval", str(n_eval),
        "--seed", str(seed),
        "--wandb", "--wandb_project", "memory-experiment",
        "--wandb_group", wandb_group,
    ]
    if config.get("gate"):
        cmd.append("--gate")
    if extra_flags:
        cmd += extra_flags
    print(f"==== START {name} ====")
    rc = subprocess.run(cmd).returncode
    print(f"==== {'DONE' if rc == 0 else f'FAILED rc={rc}'} {name} ====")
    return rc


def main():
    os.chdir(SCRIPT_DIR)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WANDB_DIR", "logs")

    runs = load_phase1_runs()
    print(f"Loaded {len(runs)} Phase 1 runs.")
    if len(runs) < 5:
        print("ERROR: Need at least 5 Phase 1 runs to pick top-5.")
        return

    # Rank by full-val f_selection (primary generalization metric).
    ranked = sorted(runs, key=lambda r: -r["memory_full"].get("f_selection", 0))
    top5 = ranked[:5]

    print("\nTop 5 Phase 1 configs by full-val f_selection:")
    for i, r in enumerate(top5, 1):
        fs = r["memory_full"].get("f_selection", 0)
        op = r["memory_full"].get("operation_accuracy", 0)
        print(f"  {i}. {r['name']:28s}  full f_sel={fs:.3f}  op_acc={op:.3f}")

    print("\n=== Phase 2: 3 seeds × top 5 at n_eval=200 ===\n")
    for r in top5:
        base = r["name"]
        cfg = r["config"]
        for seed in [0, 1, 2]:
            name = f"P2_{base}_s{seed}"
            run_cmd(name, cfg, seed=seed, n_eval=200,
                    wandb_group="p2_top5_seeds")

    print("\n==== PHASE 2 DONE ====")


if __name__ == "__main__":
    main()
