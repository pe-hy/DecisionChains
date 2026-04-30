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
                    fontsize=7.5, color="#1b2330")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Method comparison — val_full (mixed pattern, n=2000;  GRPO@n=500)")
    ax.grid(axis="y", linestyle="--", color="#dde2eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=4, fontsize=9)

    fig.tight_layout()
    out_png = OUT_DIR / "fig_metrics.png"
    out_pdf = OUT_DIR / "fig_metrics.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def per_step_entropy(npz_path: Path, max_steps: int = 5):
    """Per-example, per-step entropy. Shape (N_examples, max_steps),
    NaN when example has no valid probs at that step."""
    d = np.load(npz_path, allow_pickle=True)
    per_step = d["per_step_probs"]
    N = len(per_step)
    out = np.full((N, max_steps), np.nan, dtype=np.float64)
    for i, ex in enumerate(per_step):
        for s, p in enumerate(ex[:max_steps]):
            p = np.asarray(p, dtype=np.float64)
            if np.isnan(p).any():
                continue
            mask = p > 0
            out[i, s] = -float(np.sum(p[mask] * np.log(p[mask])))
    return out


def fig_entropies():
    max_steps = 5
    matrices = {m: per_step_entropy(p, max_steps) for m, p in NPZ.items()}
    valid_all = np.all(
        np.stack([~np.isnan(M) for M in matrices.values()], axis=0),
        axis=0,
    )
    summary = {}
    for method, M in matrices.items():
        means = np.full(max_steps, np.nan)
        ns    = np.zeros(max_steps, dtype=int)
        for s in range(max_steps):
            mask = valid_all[:, s]
            ns[s] = mask.sum()
            if ns[s] > 0:
                means[s] = M[mask, s].mean()
        summary[method] = (means, ns)

    last = max_steps
    while last > 0 and all(np.isnan(summary[m][0][last - 1]) for m in summary):
        last -= 1
    steps = np.arange(1, last + 1)

    methods = list(NPZ.keys())
    data = np.array([summary[m][0][:last] for m in methods])  # (M, last)
    ns_per_step = [summary[methods[0]][1][s] for s in range(last)]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.get_cmap("YlOrRd")
    vmax = float(np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(last))
    ax.set_xticklabels([f"DP {s}\nn={ns_per_step[s_i]}"
                        for s_i, s in enumerate(steps)],
                       fontsize=10)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    # Color row tick labels by method color
    for tick_label, m in zip(ax.get_yticklabels(), methods):
        tick_label.set_color(COLORS[m])
        tick_label.set_fontweight("bold")

    # Annotate each cell with value
    for i in range(len(methods)):
        for j in range(last):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > vmax * 0.55 else "#1b2330"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=10.5, color=txt_color, fontweight="bold")

    ax.set_title(
        "Per-decision-point entropy — 768 $f$-only-solvable val examples\n"
        "lower (lighter) = model more confident in its next-letter choice",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Mean Shannon entropy (nats)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out_png = OUT_DIR / "fig_entropies.png"
    out_pdf = OUT_DIR / "fig_entropies.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def fig_metrics_by_length():
    """Lenient solution rate per chain length, per method.
    Lenient = every step picked f-letter, every block arithmetically valid,
    final vec equals OUTPUT vec; chain length is allowed to differ from GT."""
    import json
    lenient_path = REPO / "outputs/eval_results/figures/lenient_metric.json"
    if not lenient_path.exists():
        raise FileNotFoundError(
            f"Run scripts/recompute_lenient_metric.py first ({lenient_path})"
        )
    data = json.load(open(lenient_path))
    lengths = [3, 4, 5]
    methods = list(NPZ.keys())
    vals = np.zeros((len(lengths), len(methods)))
    for j, m in enumerate(methods):
        for i, L in enumerate(lengths):
            vals[i, j] = 100.0 * data[m][str(L)]["rate"]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    bar_w = 0.10
    x = np.arange(len(lengths))
    for j, m in enumerate(methods):
        bars = ax.bar(
            x + (j - (len(methods) - 1) / 2) * bar_w,
            vals[:, j],
            width=bar_w,
            label=m,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals[:, j]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color="#1b2330")

    ax.set_xticks(x)
    ax.set_xticklabels([f"length {L}\n(n=256)" for L in lengths])
    ax.set_ylabel("Solution rate (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Solution rate by chain length — 768 $f$-only-solvable val (256 per length)\n"
        "all steps picked $f$ · all arithmetic valid · final vec matches OUTPUT (length-agnostic)"
    )
    ax.grid(axis="y", linestyle="--", color="#dde2eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    out_png = OUT_DIR / "fig_metrics_by_length.png"
    out_pdf = OUT_DIR / "fig_metrics_by_length.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def fig_metrics_aggregate():
    """Single bar per method: lenient solution rate aggregated across all
    768 examples (= weighted mean across length 3/4/5 since 256 each)."""
    import json
    lenient_path = REPO / "outputs/eval_results/figures/lenient_metric.json"
    data = json.load(open(lenient_path))
    methods = list(NPZ.keys())
    vals = []
    for m in methods:
        hit = sum(data[m][L]["hit"] for L in data[m])
        tot = sum(data[m][L]["total"] for L in data[m])
        vals.append(100.0 * hit / tot if tot else 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(methods))
    handles = []
    for j, m in enumerate(methods):
        bar = ax.bar(x[j], vals[j],
                     width=0.65,
                     color=COLORS[m],
                     edgecolor="white",
                     linewidth=0.8,
                     label=m)
        handles.append(bar)
        ax.text(x[j], vals[j] + 1.5,
                f"{vals[j]:.1f}", ha="center", va="bottom",
                fontsize=10, color="#1b2330")

    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(methods))
    ax.set_ylabel("Solution rate (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Aggregate solution rate — 768 $f$-only-solvable val (lengths 3, 4, 5)\n"
        "all steps picked $f$ · all arithmetic valid · final vec matches OUTPUT"
    )
    ax.grid(axis="y", linestyle="--", color="#dde2eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    out_png = OUT_DIR / "fig_metrics_aggregate.png"
    out_pdf = OUT_DIR / "fig_metrics_aggregate.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    fig_metrics()
    fig_entropies()
    fig_metrics_by_length()
    fig_metrics_aggregate()
