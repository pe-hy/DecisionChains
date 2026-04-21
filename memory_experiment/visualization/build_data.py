#!/usr/bin/env python
"""
build_data.py — Dump all outputs/experiments/*.json into data.js for the viewer.

Writes a JS file assigning window.RUNS so the HTML can load via <script src>
without a server (avoids file:// CORS errors from fetch()).
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR.parent / "outputs" / "experiments"
OUT_JS = SCRIPT_DIR / "data.js"
OUT_JSON = SCRIPT_DIR / "runs.json"  # kept for compat / debugging


def summarize(r):
    c = r["config"]
    bf, bu = r["baseline_fonly"], r["baseline_full"]
    mf, mu = r["memory_fonly"], r["memory_full"]

    def keep(d):
        return {k: v for k, v in d.items() if isinstance(v, (int, float))}

    return {
        "name": r["name"],
        "config": {
            "data_filter": c.get("data_filter"),
            "ce_mode": c.get("ce_mode"),
            "dp_target": c.get("dp_target"),
            "layers": str(c.get("layers")),
            "mem_entries": c.get("mem_entries"),
            "gate": c.get("gate"),
            "sparsity": c.get("sparsity"),
            "sparsity_coeff": c.get("sparsity_coeff"),
            "n_train": c.get("n_train"),
            "epochs": c.get("epochs"),
            "batch_size": c.get("batch_size"),
            "lr": c.get("lr"),
            "seed": c.get("seed"),
            "wandb_group": c.get("wandb_group") or "",
        },
        "trainable_params": r.get("trainable_params"),
        "train_time_sec": r.get("train_time_sec"),
        "n_train_actual": r.get("n_train_actual"),
        "n_val_fonly_actual": r.get("n_val_fonly_actual"),
        "n_val_full_actual": r.get("n_val_full_actual"),
        "baseline_fonly": keep(bf),
        "memory_fonly": keep(mf),
        "baseline_full": keep(bu),
        "memory_full": keep(mu),
        "per_step_f_selection_baseline_full": bu.get("per_step_f_selection", []),
        "per_step_f_selection_memory_full":  mu.get("per_step_f_selection", []),
        "per_step_f_selection_baseline_fonly": bf.get("per_step_f_selection", []),
        "per_step_f_selection_memory_fonly":  mf.get("per_step_f_selection", []),
        "epoch_ce": [h.get("ce") for h in r.get("history", [])],
    }


def main():
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            data = json.load(open(p))
            if "config" not in data:
                continue
            runs.append(summarize(data))
        except Exception as e:
            print(f"skip {p.name}: {e}")

    payload = json.dumps(runs, indent=2)
    OUT_JS.write_text(f"window.RUNS = {payload};\n")
    OUT_JSON.write_text(payload)
    print(f"wrote {len(runs)} runs to {OUT_JS}")
    print(f"  also to {OUT_JSON} (for debugging)")


if __name__ == "__main__":
    main()
