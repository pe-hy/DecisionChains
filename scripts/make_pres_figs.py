"""Two presentation figures (PNG + PDF):

1) fig_metrics — Grouped bar chart of val_full metrics for Baseline / GRPO /
   Memory align. f-selection, full-alignment, op-accuracy.

2) fig_entropies — Per-decision-point Shannon entropy (mean across the
   128 ffff val examples, ±1 std) for each checkpoint. Lower = more
   confident decision.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
NPZ = {
    "Pretrained":   REPO / "outputs/eval_results/12l-8h-512d-decision-chains-ext_6_2M/decision_point_probs_extended.npz",
    "GRPO":         REPO / "outputs/eval_results/grpo_12l-8h-512d-decision-chains-ext_6_2M/decision_point_probs_extended.npz",
    "LoRA mode A":  REPO / "outputs/eval_results/ft_for_fig_lora_A/decision_point_probs_extended.npz",
    "LoRA mode B":  REPO / "outputs/eval_results/ft_for_fig_lora_B/decision_point_probs_extended.npz",
    "Naive FT mode A": REPO / "outputs/eval_results/ft_for_fig_full_A/decision_point_probs_extended.npz",
    "Naive FT mode B": REPO / "outputs/eval_results/ft_for_fig_full_B/decision_point_probs_extended.npz",
    "Memory align (L=2, N=32)": REPO / "outputs/eval_results/memory_N32_12l-8h-512d-decision-chains-ext_6_2M/decision_point_probs_extended.npz",
}
COLORS = {
    "Pretrained":              "#9aa3b1",
    "GRPO":                    "#e06c45",
    "LoRA mode A":             "#f1c453",
    "LoRA mode B":             "#d49a1f",
    "Naive FT mode A":         "#7e57c2",
    "Naive FT mode B":         "#4527a0",
    "Memory align (L=2, N=32)":"#2563eb",
}
OUT_DIR = REPO / "outputs/eval_results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Metric values from val_full @ n_eval=2000 (n_eval=500 for GRPO).
METRICS = {
    "f-selection":     {"Pretrained": 52.5, "GRPO": 60.2,
                        "LoRA mode A": 81.6, "LoRA mode B": 96.6,
                        "Naive FT mode A": 96.0, "Naive FT mode B": 92.9,
                        "Memory align (L=2, N=32)": 97.6},
    "Full alignment":  {"Pretrained":  8.8, "GRPO": 11.4,
                        "LoRA mode A": 37.2, "LoRA mode B": 83.5,
                        "Naive FT mode A": 84.2, "Naive FT mode B": 69.8,
                        "Memory align (L=2, N=32)": 88.0},
    "Op accuracy":     {"Pretrained": 94.8, "GRPO": 93.9,
                        "LoRA mode A": 94.3, "LoRA mode B": 89.3,
                        "Naive FT mode A": 92.6, "Naive FT mode B": 84.7,
                        "Memory align (L=2, N=32)": 94.1},
}

# ── Plot config
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_metrics():
    metrics = list(METRICS.keys())
    methods = list(NPZ.keys())
    vals = np.array([[METRICS[m][k] for k in methods] for m in metrics])  # (M, K)

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    n_metrics, n_methods = vals.shape
    bar_w = 0.10
    x = np.arange(n_metrics)
    for j, method in enumerate(methods):
        bars = ax.bar(
            x + (j - (n_methods - 1) / 2) * bar_w,
            vals[:, j],
            width=bar_w,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals[:, j]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color="#1b2330", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Method comparison — val_full (mixed pattern, n=2000;  GRPO@n=500)")
    ax.grid(axis="y", linestyle="--", color="#dde2eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)

    fig.tight_layout()
    out_png = OUT_DIR / "fig_metrics.png"
    out_pdf = OUT_DIR / "fig_metrics.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def per_step_entropy(npz_path: Path, max_steps: int = 5):
    """Return list of arrays — one per decision-point step, of entropies
    (one entry per example that has that step)."""
    d = np.load(npz_path, allow_pickle=True)
    per_step = d["per_step_probs"]
    by_step = [[] for _ in range(max_steps)]
    for ex in per_step:
        for s, p in enumerate(ex[:max_steps]):
            p = np.asarray(p, dtype=np.float64)
            if np.isnan(p).any():
                continue
            mask = p > 0
            h = -float(np.sum(p[mask] * np.log(p[mask])))
            by_step[s].append(h)
    return [np.asarray(xs) for xs in by_step]


def fig_entropies():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    max_steps = 5  # show 4 GT decision points + 1 overshoot

    # Compute per-method per-step mean+std
    summary = {}
    for method, path in NPZ.items():
        by_step = per_step_entropy(path, max_steps)
        means = np.array([xs.mean() if len(xs) else np.nan for xs in by_step])
        stds  = np.array([xs.std()  if len(xs) else np.nan for xs in by_step])
        ns    = np.array([len(xs)   for xs in by_step])
        summary[method] = (means, stds, ns)

    # Trim trailing all-nan steps
    last = max_steps
    while last > 0 and all(np.isnan(summary[m][0][last - 1]) for m in summary):
        last -= 1
    steps = np.arange(1, last + 1)

    width = 0.10
    methods = list(NPZ.keys())
    for j, m in enumerate(methods):
        means, _, ns = summary[m]
        offsets = (j - (len(methods) - 1) / 2) * width
        bars = ax.bar(steps + offsets, means[:last],
                      width=width, color=COLORS[m], label=m,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, means[:last]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#1b2330", rotation=90)

    ax.set_xticks(steps)
    # Sample-size annotations under each DP (n varies across methods at DP5)
    counts_text = []
    for s_i, _ in enumerate(steps):
        ns_at_step = [summary[m][2][s_i] for m in methods]
        if min(ns_at_step) == max(ns_at_step):
            counts_text.append(f"n={ns_at_step[0]}")
        else:
            counts_text.append(f"n={min(ns_at_step)}–{max(ns_at_step)}")
    ax.set_xticklabels([f"DP {s}\n{c}" for s, c in zip(steps, counts_text)])
    ax.set_xlabel("Decision point in chain")
    ax.set_ylabel("Mean Shannon entropy (nats)")
    ax.set_title(
        "Per-decision-point entropy — 128 $f$-only-solvable val examples\n"
        "lower = model more confident in its next-letter choice  ·  DP 5 = post-overshoot"
    )
    ymax = 0.0
    for m in methods:
        means, _, _ = summary[m]
        ymax = max(ymax, np.nanmax(means[:last]))
    ax.set_ylim(0, ymax * 1.5 + 0.02)
    ax.grid(axis="y", linestyle="--", color="#dde2eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    out_png = OUT_DIR / "fig_entropies.png"
    out_pdf = OUT_DIR / "fig_entropies.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    fig_metrics()
    fig_entropies()
