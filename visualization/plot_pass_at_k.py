"""
Plot pass@k curves from eval_pass_at_k.py output JSON files.

Generates matplotlib figures with:
  1. Overall pass@k curve (log-x scale)
  2. Per-chain-length pass@k curves
  3. If multiple JSON files are given, overlay them for comparison

Usage:
    # Single model
    python visualization/plot_pass_at_k.py \\
        --input outputs/eval_results/pass_at_k/pass_at_k_T0.8_n256.json \\
        --output outputs/eval_results/pass_at_k/pass_at_k.png

    # Compare models / temperatures
    python visualization/plot_pass_at_k.py \\
        --input results_A/pass_at_k_T0.8_n256.json results_B/pass_at_k_T0.8_n256.json \\
        --labels "Model A" "Model B" \\
        --output comparison.png
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_single(results, output_path, title=None):
    """Plot pass@k for a single model: overall + per-length breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    temperature = results.get("temperature", "?")
    n_samples = results.get("n_samples_per_input", "?")
    n_inputs = results.get("n_test_inputs", "?")

    if title is None:
        model_name = Path(results.get("model_path", "unknown")).name
        title = model_name

    # --- Left: overall pass@k ---
    ax = axes[0]
    pak = results["pass_at_k"]
    ks = sorted(int(k) for k in pak)
    vals = [pak[str(k)] for k in ks]

    ax.plot(ks, vals, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], rotation=45)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"Overall pass@k")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Annotate values
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    # --- Right: per-chain-length ---
    ax = axes[1]
    pak_by_len = results["pass_at_k_by_length"]
    n_by_len = results.get("n_inputs_by_length", {})
    colors = ["#16a34a", "#2563eb", "#dc2626", "#9333ea", "#ea580c"]

    for idx, length in enumerate(sorted(pak_by_len.keys(), key=int)):
        data = pak_by_len[length]
        ks_l = sorted(int(k) for k in data)
        vals_l = [data[str(k)] for k in ks_l]
        n_l = n_by_len.get(length, "?")
        ax.plot(ks_l, vals_l, "o-", color=colors[idx % len(colors)],
                linewidth=2, markersize=5, label=f"len={length} (n={n_l})")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    all_ks = set()
    for data in pak_by_len.values():
        all_ks.update(int(k) for k in data)
    all_ks = sorted(all_ks)
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks], rotation=45)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k by chain length")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"{title}  |  T={temperature}, n_samples={n_samples}, n_inputs={n_inputs}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_comparison(results_list, labels, output_path, title="pass@k comparison"):
    """Plot overlaid pass@k curves from multiple result files."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
              "#0891b2", "#be185d", "#65a30d"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]

    # --- Left: overall comparison ---
    ax = axes[0]
    for idx, (results, label) in enumerate(zip(results_list, labels)):
        pak = results["pass_at_k"]
        ks = sorted(int(k) for k in pak)
        vals = [pak[str(k)] for k in ks]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.plot(ks, vals, f"{m}-", color=c, linewidth=2, markersize=6, label=label)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    all_ks = set()
    for r in results_list:
        all_ks.update(int(k) for k in r["pass_at_k"])
    all_ks = sorted(all_ks)
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks], rotation=45)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Overall pass@k")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --- Right: per-length for each model, using linestyle for model ---
    ax = axes[1]
    linestyles = ["-", "--", "-.", ":"]
    lengths_seen = set()
    for r in results_list:
        lengths_seen.update(r["pass_at_k_by_length"].keys())
    length_colors = {l: colors[i % len(colors)]
                     for i, l in enumerate(sorted(lengths_seen, key=int))}

    for model_idx, (results, label) in enumerate(zip(results_list, labels)):
        ls = linestyles[model_idx % len(linestyles)]
        m = markers[model_idx % len(markers)]
        pak_by_len = results["pass_at_k_by_length"]
        for length in sorted(pak_by_len.keys(), key=int):
            data = pak_by_len[length]
            ks_l = sorted(int(k) for k in data)
            vals_l = [data[str(k)] for k in ks_l]
            ax.plot(ks_l, vals_l, marker=m, linestyle=ls,
                    color=length_colors[length], linewidth=1.5, markersize=4,
                    label=f"{label} len={length}")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks], rotation=45)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k by chain length")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot pass@k curves")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="Path(s) to pass@k JSON result file(s)")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Labels for each input file (for comparison)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path (.png or .pdf)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    args = parser.parse_args()

    results_list = [load_results(p) for p in args.input]

    if len(results_list) == 1:
        plot_single(results_list[0], args.output, title=args.title)
    else:
        labels = args.labels or [Path(p).stem for p in args.input]
        if len(labels) < len(results_list):
            labels += [Path(p).stem for p in args.input[len(labels):]]
        plot_comparison(results_list, labels, args.output,
                        title=args.title or "pass@k comparison")


if __name__ == "__main__":
    main()
