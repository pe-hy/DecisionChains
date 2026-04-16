"""
Visualization for validate_pretrained results (res.jsonl / res_perturb.jsonl).

Generates a multi-panel PDF/PNG summarising model performance on decision-chain
traces: overall accuracy, per-chain-length breakdown, per-step accuracy curves,
length distribution, letter-level operation accuracy, and baseline-vs-perturbed
comparison when both files are provided.

Usage:
    python validate_pretrained/visualize_results.py
    python validate_pretrained/visualize_results.py --baseline res.jsonl --perturb res_perturb.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
    }
)

# Friendly colour palette (colorblind-safe, 6 colours)
C = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "grey": "#949494",
    "teal": "#64B5CE",
}

METRIC_LABELS = {
    "full_correct": "Full Solution",
    "chain_matches_output": "Output Match",
    "operation_accuracy": "Op Accuracy",
    "operation_selection": "Selection Acc.",
    "length_matches_gt": "Length Match",
}

LETTERS = list("abcdefghijklmnopqrst")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def overall_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    total_ops = sum(r["n_steps"] for r in rows)
    return {
        "full_correct": sum(r["full_correct"] for r in rows) / n,
        "chain_matches_output": sum(r["chain_matches_output"] for r in rows) / n,
        "length_matches_gt": sum(r["length_matches_gt"] for r in rows) / n,
        "operation_accuracy": sum(r["op_correct"] for r in rows) / total_ops if total_ops else 0,
        "operation_selection": sum(r["sel_correct"] for r in rows) / total_ops if total_ops else 0,
    }


def by_length(rows: list[dict]) -> dict[int, dict[str, float]]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        cl = r.get("gt_chain_length")
        if cl is not None:
            buckets[cl].append(r)
    out = {}
    for cl in sorted(buckets):
        out[cl] = overall_metrics(buckets[cl])
        out[cl]["n"] = len(buckets[cl])
    return out


def by_step(rows: list[dict]) -> dict[int, dict[str, float]]:
    max_step = max((r["n_steps"] for r in rows), default=0)
    out: dict[int, dict[str, float]] = {}
    for s in range(max_step):
        op_vals, sel_vals = [], []
        for r in rows:
            if s < len(r["per_step_op"]):
                op_vals.append(float(r["per_step_op"][s]))
                sel_vals.append(float(r["per_step_sel"][s]))
        if op_vals:
            out[s] = {
                "operation_accuracy": np.mean(op_vals),
                "operation_selection": np.mean(sel_vals),
                "n": len(op_vals),
            }
    return out


def by_letter(rows: list[dict]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "op_correct": 0})
    for r in rows:
        for lt, op_ok in zip(r["per_step_letter"], r["per_step_op"]):
            if lt and lt in LETTERS:
                counts[lt]["n"] += 1
                if op_ok:
                    counts[lt]["op_correct"] += 1
    return {
        lt: {"n": v["n"], "operation_accuracy": v["op_correct"] / v["n"] if v["n"] else 0}
        for lt, v in sorted(counts.items())
    }


def length_distribution(rows: list[dict]) -> tuple[list[int], list[int]]:
    """Returns (gt_lengths, pred_lengths)."""
    gt = [r.get("gt_chain_length", 0) for r in rows]
    pred = [r["n_steps"] for r in rows]
    return gt, pred


def letter_confusion_at_step(rows: list[dict], step: int) -> np.ndarray:
    """20x20 confusion matrix: predicted letter vs GT letter at `step`."""
    mat = np.zeros((20, 20), dtype=int)
    for r in rows:
        gt_letters = r.get("gt_letters", [])
        pred_letters = r.get("per_step_letter", [])
        if step < len(gt_letters) and step < len(pred_letters):
            gi = LETTERS.index(gt_letters[step]) if gt_letters[step] in LETTERS else -1
            pi = LETTERS.index(pred_letters[step]) if pred_letters[step] in LETTERS else -1
            if gi >= 0 and pi >= 0:
                mat[pi, gi] += 1
    return mat


def step_offset_distribution(rows: list[dict]) -> dict[int, int]:
    """n_steps - gt_chain_length distribution."""
    counts: dict[int, int] = Counter()
    for r in rows:
        cl = r.get("gt_chain_length")
        if cl is not None:
            counts[r["n_steps"] - cl] += 1
    return counts


def has_post_intervene(rows: list[dict]) -> bool:
    """Check if any row carries post-intervention metrics."""
    return any("post_op_rate" in r for r in rows)


def post_intervene_metrics(rows: list[dict]) -> dict[str, float]:
    """Aggregate post-intervention rates over perturbed rows.

    Pools blocks across all applicable examples (same weighting as
    _base_post_metrics) and measures strictly after the intervened step
    using per_step_op[intervene_step+1:].
    """
    op_all: list[int] = []
    sel_all: list[int] = []
    examples_applicable = 0
    examples_fully_ok = 0
    for r in rows:
        if "intervene_step" not in r:
            continue
        s0 = r["intervene_step"]
        post_ops = r["per_step_op"][s0 + 1:]
        post_sels = r["per_step_sel"][s0 + 1:]
        if not post_ops:
            continue
        examples_applicable += 1
        op_all.extend(int(x) for x in post_ops)
        sel_all.extend(int(x) for x in post_sels)
        if all(post_ops) and all(post_sels):
            examples_fully_ok += 1
    if examples_applicable == 0:
        return {}
    n_blocks = len(op_all)
    return {
        "post_op_rate": sum(op_all) / n_blocks if n_blocks else 0.0,
        "post_sel_rate": sum(sel_all) / n_blocks if n_blocks else 0.0,
        "post_full_correct_rate": examples_fully_ok / examples_applicable,
        "n": examples_applicable,
        "n_blocks": n_blocks,
    }


def post_intervene_by_length(rows: list[dict]) -> dict[int, dict[str, float]]:
    """Post-intervention metrics broken down by gt_chain_length."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if "intervene_step" in r:
            cl = r.get("gt_chain_length")
            if cl is not None:
                buckets[cl].append(r)
    out = {}
    for cl in sorted(buckets):
        out[cl] = post_intervene_metrics(buckets[cl])
    return out


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_overall(metrics: dict[str, float], ax: plt.Axes, title: str = "Overall Metrics"):
    names = list(METRIC_LABELS.values())
    vals = [metrics[k] for k in METRIC_LABELS]
    colours = [C["blue"], C["green"], C["orange"], C["purple"], C["teal"]]
    bars = ax.barh(names, vals, color=colours, height=0.55, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1%}",
            va="center",
            fontsize=9,
            color="#333",
        )
    ax.set_title(title)


def plot_by_length(blen: dict[int, dict], ax: plt.Axes, title: str = "Metrics by Chain Length"):
    lengths = sorted(blen.keys())
    metrics_keys = list(METRIC_LABELS.keys())
    x = np.arange(len(lengths))
    width = 0.15
    colours = [C["blue"], C["green"], C["orange"], C["purple"], C["teal"]]
    for i, (mk, ml) in enumerate(zip(metrics_keys, METRIC_LABELS.values())):
        vals = [blen[L].get(mk, 0) for L in lengths]
        ax.bar(x + i * width, vals, width, label=ml, color=colours[i])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"L={L}\n(n={blen[L].get('n', '?')})" for L in lengths])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title(title)


def plot_per_step(bstep: dict[int, dict], ax: plt.Axes, title: str = "Accuracy by Step"):
    steps = sorted(bstep.keys())
    op_acc = [bstep[s]["operation_accuracy"] for s in steps]
    sel_acc = [bstep[s]["operation_selection"] for s in steps]
    ns = [bstep[s]["n"] for s in steps]
    ax.plot(steps, op_acc, "o-", color=C["blue"], label="Op Accuracy", markersize=5)
    ax.plot(steps, sel_acc, "s-", color=C["purple"], label="Selection Acc.", markersize=5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Step index")
    # Annotate sample counts
    for s, n in zip(steps, ns):
        ax.annotate(
            f"n={n}",
            (s, op_acc[s]),
            textcoords="offset points",
            xytext=(0, -14),
            fontsize=7,
            ha="center",
            color=C["grey"],
        )
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title)


def plot_length_distribution(
    gt: list[int], pred: list[int], ax: plt.Axes, title: str = "Chain Length Distribution"
):
    all_vals = sorted(set(gt + pred))
    bins = np.arange(min(all_vals) - 0.5, max(all_vals) + 1.5, 1)
    ax.hist(gt, bins=bins, alpha=0.55, label="Ground truth", color=C["blue"], edgecolor="white")
    ax.hist(pred, bins=bins, alpha=0.55, label="Predicted", color=C["orange"], edgecolor="white")
    ax.set_xlabel("Chain length (steps)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title)


def plot_step_offset(offsets: dict[int, int], ax: plt.Axes, title: str = "Step Offset (pred - gt)"):
    if not offsets:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ks = sorted(offsets.keys())
    vals = [offsets[k] for k in ks]
    colours = [C["green"] if k == 0 else C["red"] if k < 0 else C["orange"] for k in ks]
    ax.bar(ks, vals, color=colours, edgecolor="white", width=0.7)
    ax.set_xlabel("n_steps - gt_chain_length")
    ax.set_ylabel("Count")
    # Mark the correct column
    if 0 in offsets:
        ax.annotate(
            f"{offsets[0] / sum(offsets.values()):.0%} exact",
            (0, offsets[0]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=C["green"],
        )
    ax.set_title(title)


def plot_letter_accuracy(
    blet: dict[str, dict], ax: plt.Axes, title: str = "Op Accuracy by Letter"
):
    labels = [lt for lt in LETTERS if lt in blet]
    vals = [blet[lt]["operation_accuracy"] for lt in labels]
    ns = [blet[lt]["n"] for lt in labels]
    max_n = max(ns) if ns else 1
    bar_colors = [plt.cm.Blues(0.35 + 0.55 * n / max_n) for n in ns]
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white", width=0.7)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Letter")
    # Annotate counts
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            str(n),
            ha="center",
            fontsize=7,
            color="#555",
        )
    ax.set_title(title)


def plot_confusion_at_step(
    rows: list[dict], step: int, ax: plt.Axes, max_letter: int = 10
):
    mat = letter_confusion_at_step(rows, step)
    # Show only the top-N most frequent GT letters for readability
    gt_freq = mat.sum(axis=0)
    top_gt = np.argsort(gt_freq)[::-1][:max_letter]
    sub = mat[np.ix_(top_gt, top_gt)]
    labels = [LETTERS[i] for i in top_gt]
    im = ax.imshow(sub, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("GT letter")
    ax.set_ylabel("Predicted letter")
    ax.set_title(f"Letter Confusion @ Step {step}")
    # Annotate cells
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub[i, j]
            if v > 0:
                color = "white" if v > sub.max() * 0.6 else "#333"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_comparison(
    base_metrics: dict[str, float],
    perturb_metrics: dict[str, float],
    ax: plt.Axes,
    title: str = "Baseline vs Perturbed",
):
    names = list(METRIC_LABELS.values())
    keys = list(METRIC_LABELS.keys())
    x = np.arange(len(names))
    width = 0.35
    bv = [base_metrics.get(k, 0) for k in keys]
    pv = [perturb_metrics.get(k, 0) for k in keys]
    ax.bar(x - width / 2, bv, width, label="Baseline", color=C["blue"])
    ax.bar(x + width / 2, pv, width, label="Perturbed", color=C["red"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title)


def plot_comparison_by_length(
    bl_base: dict[int, dict],
    bl_perturb: dict[int, dict],
    ax: plt.Axes,
    metric_key: str = "full_correct",
    title_suffix: str = "Full Solution Rate",
):
    lengths = sorted(set(bl_base.keys()) | set(bl_perturb.keys()))
    x = np.arange(len(lengths))
    width = 0.35
    bv = [bl_base.get(L, {}).get(metric_key, 0) for L in lengths]
    pv = [bl_perturb.get(L, {}).get(metric_key, 0) for L in lengths]
    ax.bar(x - width / 2, bv, width, label="Baseline", color=C["blue"])
    ax.bar(x + width / 2, pv, width, label="Perturbed", color=C["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in lengths])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title(f"{title_suffix} by Chain Length")
    ax.legend(fontsize=9, frameon=False)


def plot_post_intervene_overall(
    post: dict[str, float],
    ax: plt.Axes,
    base_post: dict[str, float] | None = None,
    title: str = "Accuracy Strictly After Forced Letter",
):
    """Grouped horizontal bars: baseline vs post-intervention op and selection rates.

    Both bars are pooled over the same block population: per_step_*[intervene_step+1:]
    across all applicable examples. The forced-letter step itself is excluded.
    """
    names = ["Op Accuracy\n(arithmetic)", "Selection Acc.\n(decision fn)"]
    post_vals = [post.get("post_op_rate", 0), post.get("post_sel_rate", 0)]
    y = np.arange(len(names))
    height = 0.35

    if base_post is not None:
        base_vals = [base_post.get("post_op_rate", 0), base_post.get("post_sel_rate", 0)]
        ax.barh(y - height / 2, base_vals, height, label="Baseline", color=C["blue"], edgecolor="white")
        ax.barh(y + height / 2, post_vals, height, label="Post-intervene", color=C["red"], edgecolor="white")
        for i, (bv, pv) in enumerate(zip(base_vals, post_vals)):
            ax.text(bv + 0.012, y[i] - height / 2, f"{bv:.1%}", va="center", fontsize=9, color=C["blue"])
            ax.text(pv + 0.012, y[i] + height / 2, f"{pv:.1%}", va="center", fontsize=9, color=C["red"])
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.barh(y, post_vals, height * 2, color=C["red"], edgecolor="white")
        for i, v in enumerate(post_vals):
            ax.text(v + 0.012, y[i], f"{v:.1%}", va="center", fontsize=10, color="#333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    n = post.get("n", "?")
    ax.set_title(f"{title} (n={n})")


def plot_post_intervene_by_length(
    post_bl: dict[int, dict],
    ax: plt.Axes,
    base_bl: dict[int, dict] | None = None,
    title: str = "Accuracy After Forced Letter — by Chain Length",
):
    lengths = sorted(post_bl.keys())
    x = np.arange(len(lengths))

    if base_bl is not None:
        width = 0.2
        base_op = [base_bl.get(L, {}).get("post_op_rate", 0) for L in lengths]
        base_sel = [base_bl.get(L, {}).get("post_sel_rate", 0) for L in lengths]
        post_op = [post_bl[L].get("post_op_rate", 0) for L in lengths]
        post_sel = [post_bl[L].get("post_sel_rate", 0) for L in lengths]
        ax.bar(x - 1.5 * width, base_op, width, label="Base Op", color=C["blue"])
        ax.bar(x - 0.5 * width, post_op, width, label="Post Op", color=C["red"])
        ax.bar(x + 0.5 * width, base_sel, width, label="Base Sel", color=C["purple"], alpha=0.6)
        ax.bar(x + 1.5 * width, post_sel, width, label="Post Sel", color=C["orange"], alpha=0.6)
    else:
        width = 0.3
        op_vals = [post_bl[L].get("post_op_rate", 0) for L in lengths]
        sel_vals = [post_bl[L].get("post_sel_rate", 0) for L in lengths]
        ax.bar(x - width / 2, op_vals, width, label="Op Accuracy", color=C["blue"])
        ax.bar(x + width / 2, sel_vals, width, label="Selection Acc.", color=C["purple"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}\n(n={post_bl[L].get('n', '?')})" for L in lengths])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Baseline post-intervention helpers (compute baseline accuracy at steps >= intervene_step)
# ---------------------------------------------------------------------------
def _base_post_metrics(rows: list[dict], intervene_step: int) -> dict[str, float]:
    """Overall baseline op/sel accuracy for steps strictly AFTER intervene_step.

    Matches the post-intervention slicing (per_step_op[intervene_step+1:]) so
    the comparison uses the same block population and weighting.
    """
    op_all, sel_all = [], []
    start = intervene_step + 1
    for r in rows:
        for s in range(start, len(r["per_step_op"])):
            op_all.append(float(r["per_step_op"][s]))
            if s < len(r["per_step_sel"]):
                sel_all.append(float(r["per_step_sel"][s]))
    n = len(op_all)
    return {
        "post_op_rate": np.mean(op_all) if op_all else 0,
        "post_sel_rate": np.mean(sel_all) if sel_all else 0,
        "n": n,
    }


def _base_post_metrics_by_length(
    rows: list[dict], intervene_step: int
) -> dict[int, dict[str, float]]:
    """Per-chain-length baseline op/sel for steps >= intervene_step."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        cl = r.get("gt_chain_length")
        if cl is not None:
            buckets[cl].append(r)
    return {cl: _base_post_metrics(rs, intervene_step) for cl, rs in sorted(buckets.items())}


def _base_offsets_from_step(
    rows: list[dict], intervene_step: int
) -> dict[int, dict[str, list[float]]]:
    """Per-offset-from-intervene baseline op/sel accuracy.

    Offset 0 = first step AFTER the intervened step (i.e. intervene_step + 1).
    The intervened step itself is excluded to match post-intervention slicing.
    """
    offsets: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"op": [], "sel": []})
    start = intervene_step + 1
    for r in rows:
        for di in range(len(r["per_step_op"]) - start):
            idx = start + di
            if idx < len(r["per_step_op"]):
                offsets[di]["op"].append(float(r["per_step_op"][idx]))
            if idx < len(r["per_step_sel"]):
                offsets[di]["sel"].append(float(r["per_step_sel"][idx]))
    return offsets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_figure(
    rows: list[dict], label: str, rows_perturb: list[dict] | None = None
) -> plt.Figure:
    has_perturb = rows_perturb is not None and len(rows_perturb) > 0

    # Compute aggregates
    ov = overall_metrics(rows)
    bl = by_length(rows)
    bs = by_step(rows)
    blet = by_letter(rows)
    gt, pred = length_distribution(rows)
    offsets = step_offset_distribution(rows)

    if has_perturb:
        ov_p = overall_metrics(rows_perturb)
        bl_p = by_length(rows_perturb)
        bs_p = by_step(rows_perturb)
        gt_p, pred_p = length_distribution(rows_perturb)
        offsets_p = step_offset_distribution(rows_perturb)
        blet_p = by_letter(rows_perturb)
        has_post = has_post_intervene(rows_perturb)
        if has_post:
            post_ov = post_intervene_metrics(rows_perturb)
            post_bl = post_intervene_by_length(rows_perturb)
    else:
        has_post = False

    # Determine max GT chain length for confusion plot step selection
    max_gt = max((r.get("gt_chain_length", 0) for r in rows), default=3)
    conf_step = min(1, max_gt - 1)  # step 1 if possible, else 0

    # Layout
    if has_perturb:
        ncols, nrows = 3, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 20))
    else:
        ncols, nrows = 3, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(17, 14))

    fig.suptitle(
        f"Decision-Chain Validation — {label}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ax_flat = axes.flat

    # Row 0: overall + by_length + length_dist
    plot_overall(ov, ax_flat[0], title=f"Overall ({len(rows)} examples)")
    plot_by_length(bl, ax_flat[1])
    plot_length_distribution(gt, pred, ax_flat[2])

    # Row 1: per_step + step_offset + confusion
    plot_per_step(bs, ax_flat[3])
    plot_step_offset(offsets, ax_flat[4])
    plot_confusion_at_step(rows, conf_step, ax_flat[5])

    # Row 2: letter accuracy + confusion step 0 + confusion step 2 (or last)
    plot_letter_accuracy(blet, ax_flat[6])
    other_step = min(2, max_gt - 1)
    if other_step != conf_step:
        plot_confusion_at_step(rows, other_step, ax_flat[7])
    else:
        ax_flat[7].set_visible(False)

    # Fraction of correct-length chains reaching each step
    if has_perturb:
        ax_flat[8].set_visible(False)
    else:
        # Cumulative accuracy: fraction of examples that are fully correct up to step s
        max_s = max((r["n_steps"] for r in rows), default=0)
        cum_steps = list(range(max_s))
        cum_vals = []
        for s in cum_steps:
            ok = sum(
                1
                for r in rows
                if all(r["per_step_op"][i] and r["per_step_sel"][i] for i in range(s + 1) if i < len(r["per_step_op"]))
                and s < len(r["per_step_op"])
            )
            total = sum(1 for r in rows if s < len(r["per_step_op"]))
            cum_vals.append(ok / total if total else 0)
        ax_flat[8].plot(cum_steps, cum_vals, "o-", color=C["green"], markersize=4)
        ax_flat[8].set_ylim(0, 1.05)
        ax_flat[8].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_flat[8].set_xlabel("Step index")
        ax_flat[8].set_title("Cumulative Correct Rate\n(all steps up to s)")

    if has_perturb:
        # Row 3: comparison panels
        plot_comparison(ov, ov_p, ax_flat[9])
        plot_comparison_by_length(bl, bl_p, ax_flat[10], metric_key="full_correct")
        plot_comparison_by_length(
            bl, bl_p, ax_flat[11], metric_key="operation_accuracy", title_suffix="Op Accuracy"
        )

        # Extra row: per-step comparison. Mark the intervention boundary so
        # the pre/post regions are visually distinct.
        fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
        istep_for_line = (
            rows_perturb[0].get("intervene_step") if has_post else None
        )
        fig2.suptitle(
            f"Per-Step Comparison — {label}"
            + (
                f"  (intervention at step {istep_for_line};"
                f" post = steps > {istep_for_line})"
                if istep_for_line is not None
                else ""
            ),
            fontsize=14,
            fontweight="bold",
        )

        steps = sorted(set(bs.keys()) | set(bs_p.keys()))
        op_b = [bs.get(s, {}).get("operation_accuracy", 0) for s in steps]
        op_p = [bs_p.get(s, {}).get("operation_accuracy", 0) for s in steps]
        sel_b = [bs.get(s, {}).get("operation_selection", 0) for s in steps]
        sel_p = [bs_p.get(s, {}).get("operation_selection", 0) for s in steps]

        def _mark_intervention(ax):
            if istep_for_line is None:
                return
            ax.axvline(
                istep_for_line + 0.5,
                color=C["grey"],
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )
            # Shade the pre-intervention region lightly so the post-region
            # (the only one with causally affected data) reads at a glance.
            ymin, ymax = ax.get_ylim()
            ax.axvspan(
                -0.5,
                istep_for_line + 0.5,
                color=C["grey"],
                alpha=0.08,
                zorder=0,
            )
            ax.text(
                istep_for_line + 0.55,
                0.05,
                "post →",
                fontsize=8,
                color=C["grey"],
                ha="left",
                va="bottom",
            )

        axes2[0].plot(steps, op_b, "o-", color=C["blue"], label="Baseline op")
        axes2[0].plot(steps, op_p, "s--", color=C["red"], label="Perturbed op")
        axes2[0].set_ylim(0, 1.05)
        axes2[0].legend(fontsize=9, frameon=False)
        axes2[0].set_title("Op Accuracy by Step")
        axes2[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        axes2[0].set_xlabel("Step")
        _mark_intervention(axes2[0])

        axes2[1].plot(steps, sel_b, "o-", color=C["blue"], label="Baseline sel")
        axes2[1].plot(steps, sel_p, "s--", color=C["red"], label="Perturbed sel")
        axes2[1].set_ylim(0, 1.05)
        axes2[1].legend(fontsize=9, frameon=False)
        axes2[1].set_title("Selection Accuracy by Step")
        axes2[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        axes2[1].set_xlabel("Step")
        _mark_intervention(axes2[1])

        # Letter accuracy comparison
        all_letters = sorted(set(blet.keys()) | set(blet_p.keys()))
        if all_letters:
            la_b = [blet.get(lt, {}).get("operation_accuracy", 0) for lt in all_letters]
            la_p = [blet_p.get(lt, {}).get("operation_accuracy", 0) for lt in all_letters]
            x = np.arange(len(all_letters))
            w = 0.35
            axes2[2].bar(x - w / 2, la_b, w, color=C["blue"], label="Baseline")
            axes2[2].bar(x + w / 2, la_p, w, color=C["red"], label="Perturbed")
            axes2[2].set_xticks(x)
            axes2[2].set_xticklabels(all_letters)
            axes2[2].set_ylim(0, 1.05)
            axes2[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            axes2[2].legend(fontsize=9, frameon=False)
            axes2[2].set_title("Op Accuracy by Letter")
            axes2[2].set_xlabel("Letter")

        fig2.tight_layout(rect=[0, 0, 1, 0.94])

        # Optional third figure: post-intervention analysis
        fig3 = None
        if has_post:
            fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
            istep = rows_perturb[0].get("intervene_step", "?")
            iletter = rows_perturb[0].get("intervene_letter", "?")
            fig3.suptitle(
                f"Post-Intervention Only — strictly after forced "
                f"'{iletter}' at step {istep}  ({label})",
                fontsize=14,
                fontweight="bold",
            )

            # Compute baseline equivalent: op/sel accuracy at steps >= intervene_step
            base_post_ov = _base_post_metrics(rows, istep)
            base_post_bl = _base_post_metrics_by_length(rows, istep)

            plot_post_intervene_overall(post_ov, axes3[0], base_post=base_post_ov)
            plot_post_intervene_by_length(post_bl, axes3[1], base_bl=base_post_bl)

            # Per-step breakdown: baseline vs perturbed from intervene_step onward
            base_offsets = _base_offsets_from_step(rows, istep)
            perturb_offsets: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: {"op": [], "sel": []}
            )
            for r in rows_perturb:
                if "intervene_step" not in r:
                    continue
                s0 = r["intervene_step"] + 1  # strictly after intervened step
                for di in range(len(r["per_step_op"]) - s0):
                    idx = s0 + di
                    if idx < len(r["per_step_op"]):
                        perturb_offsets[di]["op"].append(float(r["per_step_op"][idx]))
                    if idx < len(r["per_step_sel"]):
                        perturb_offsets[di]["sel"].append(float(r["per_step_sel"][idx]))
            all_ds = sorted(set(base_offsets.keys()) | set(perturb_offsets.keys()))
            if all_ds:
                ax3 = axes3[2]
                for d in all_ds:
                    if d in base_offsets:
                        ax3.plot(d, np.mean(base_offsets[d]["op"]), "o", color=C["blue"], markersize=5)
                        ax3.plot(d, np.mean(base_offsets[d]["sel"]), "s", color=C["purple"], markersize=5)
                # Connect baseline
                b_ds = sorted(d for d in all_ds if d in base_offsets)
                if b_ds:
                    ax3.plot(
                        b_ds,
                        [np.mean(base_offsets[d]["op"]) for d in b_ds],
                        "-",
                        color=C["blue"],
                        label="Base Op",
                        markersize=5,
                    )
                    ax3.plot(
                        b_ds,
                        [np.mean(base_offsets[d]["sel"]) for d in b_ds],
                        "-",
                        color=C["purple"],
                        label="Base Sel",
                        markersize=5,
                    )
                # Connect perturbed
                p_ds = sorted(d for d in all_ds if d in perturb_offsets)
                if p_ds:
                    ax3.plot(
                        p_ds,
                        [np.mean(perturb_offsets[d]["op"]) for d in p_ds],
                        "o--",
                        color=C["red"],
                        label="Perturb Op",
                        markersize=5,
                    )
                    ax3.plot(
                        p_ds,
                        [np.mean(perturb_offsets[d]["sel"]) for d in p_ds],
                        "s--",
                        color=C["orange"],
                        label="Perturb Sel",
                        markersize=5,
                    )
                ax3.set_ylim(0, 1.05)
                ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                ax3.set_xlabel("Offset after forced step  (0 = first step after)")
                ax3.set_title("Accuracy by Offset After Forced Letter")
                ax3.legend(fontsize=8, frameon=False)
            fig3.tight_layout(rect=[0, 0, 1, 0.94])

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if has_perturb:
        if fig3 is not None:
            return fig, fig2, fig3
        return fig, fig2
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).parent / "output" / "res.jsonl",
        help="Path to baseline res.jsonl",
    )
    parser.add_argument(
        "--perturb",
        type=Path,
        default=None,
        help="Path to perturbed res_perturb.jsonl (optional)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Title label (auto-detected from filename if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: same dir as baseline, PDF)",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.baseline)
    print(f"Loaded {len(rows)} rows from {args.baseline}")

    rows_perturb = None
    if args.perturb is not None and args.perturb.exists():
        rows_perturb = load_jsonl(args.perturb)
        print(f"Loaded {len(rows_perturb)} rows from {args.perturb}")

    label = args.label or args.baseline.stem
    if rows_perturb is not None and args.label is None:
        label += "_with_perturb"
    result = build_figure(rows, label, rows_perturb)

    # Output goes next to perturb jsonl (if present), else next to baseline
    if args.output is not None:
        out_path = args.output
    elif rows_perturb is not None:
        out_path = args.perturb.parent / f"viz_{label}.pdf"
    else:
        out_path = args.baseline.parent / f"viz_{label}.pdf"

    # Collect all figures into a single multi-page PDF
    figures = result if isinstance(result, tuple) else (result,)
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {len(figures)} page(s): {out_path}")


# ---------------------------------------------------------------------------
# Aggregate: baseline vs all perturbation variants in one figure
# ---------------------------------------------------------------------------
def _post_metrics_for_variant(
    rows: list[dict], istep: int
) -> dict[str, float]:
    """Post-intervention op/sel rates pooled over per_step_op[istep+1:]."""
    op_all, sel_all = [], []
    for r in rows:
        post_ops = r["per_step_op"][istep + 1:]
        post_sels = r["per_step_sel"][istep + 1:]
        if not post_ops:
            continue
        op_all.extend(int(x) for x in post_ops)
        sel_all.extend(int(x) for x in post_sels)
    n = len(op_all)
    return {
        "post_op_rate": sum(op_all) / n if n else 0.0,
        "post_sel_rate": sum(sel_all) / n if n else 0.0,
        "n": n,
    }


def _post_metrics_by_length_for_variant(
    rows: list[dict], istep: int
) -> dict[int, dict[str, float]]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        cl = r.get("gt_chain_length")
        if cl is not None:
            buckets[cl].append(r)
    return {cl: _post_metrics_for_variant(rs, istep) for cl, rs in sorted(buckets.items())}


def build_aggregate(
    rows_baseline: list[dict],
    variant_dirs: dict[str, Path],
) -> plt.Figure:
    """Two-panel figure: (1) overall post-intervene accuracy, (2) by chain length.

    Each variant is one subdirectory under output/ containing res_perturb.jsonl.
    """
    # Discover all variants and load their data
    variants: list[tuple[str, int, str, list[dict]]] = []  # (label, step, letter, rows)
    for name, path in sorted(variant_dirs.items()):
        rows = load_jsonl(path)
        if not rows:
            continue
        istep = rows[0].get("intervene_step")
        iletter = rows[0].get("intervene_letter")
        if istep is None:
            continue
        label = name  # e.g. "step1_lettert"
        variants.append((label, istep, iletter, rows))

    if not variants:
        raise ValueError("No variant data found")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Baseline vs All Perturbation Variants — Accuracy After Forced Letter",
        fontsize=14,
        fontweight="bold",
    )

    # Colour palette for variants + baseline
    step_colours = {0: C["blue"], 1: C["orange"], 2: C["green"], 3: C["purple"]}
    baseline_colour = C["grey"]

    # Collect all chain lengths
    all_lengths: set[int] = set()
    for r in rows_baseline:
        cl = r.get("gt_chain_length")
        if cl is not None:
            all_lengths.add(cl)
    lengths = sorted(all_lengths)

    for ax, metric_key, metric_title in [
        (axes[0], "post_op_rate", "Op Accuracy After Forced Letter — by Chain Length"),
        (axes[1], "post_sel_rate", "Selection Accuracy After Forced Letter — by Chain Length"),
    ]:
        # Baseline: average across intervene steps
        base_by_len: dict[int, list[float]] = defaultdict(list)
        for _, istep, _, _ in variants:
            bm_bl = _post_metrics_by_length_for_variant(rows_baseline, istep)
            for cl, m in bm_bl.items():
                base_by_len[cl].append(m[metric_key])
                all_lengths.add(cl)
        lengths = sorted(all_lengths)
        base_avg = [np.mean(base_by_len.get(L, [0])) for L in lengths]
        ax.plot(
            lengths,
            base_avg,
            "o-",
            color=baseline_colour,
            linewidth=2.5,
            markersize=7,
            label="Baseline",
            zorder=10,
        )

        for vi, (label, istep, iletter, rows_p) in enumerate(variants):
            pm_bl = _post_metrics_by_length_for_variant(rows_p, istep)
            vals = [pm_bl.get(L, {}).get(metric_key, 0) for L in lengths]
            ax.plot(
                lengths,
                vals,
                "o--",
                color=step_colours.get(istep, C["red"]),
                alpha=0.6 + 0.4 * (vi % 2),
                markersize=5,
                label=label.replace("_", " "),
            )

        ax.set_xticks(lengths)
        ax.set_xticklabels([f"L={L}" for L in lengths])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_xlabel("Ground-truth chain length")
        ax.set_title(metric_title)
        ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main_aggregate():
    parser = argparse.ArgumentParser(
        description="Aggregate summary: baseline vs all perturbation variants."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Directory containing res.jsonl and step*/ subdirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <output_dir>/viz_aggregate.pdf)",
    )
    args = parser.parse_args()

    baseline_path = args.output_dir / "res.jsonl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    rows_baseline = load_jsonl(baseline_path)
    print(f"Loaded {len(rows_baseline)} baseline rows")

    # Discover variant subdirs
    variant_dirs: dict[str, Path] = {}
    for subdir in sorted(args.output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        perturb_jsonl = subdir / "res_perturb.jsonl"
        if perturb_jsonl.exists():
            variant_dirs[subdir.name] = perturb_jsonl

    if not variant_dirs:
        raise FileNotFoundError(f"No res_perturb.jsonl found under {args.output_dir}")

    print(f"Found {len(variant_dirs)} variants: {', '.join(variant_dirs.keys())}")

    fig = build_aggregate(rows_baseline, variant_dirs)

    out_path = args.output or args.output_dir / "viz_aggregate.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys
    if "--aggregate" in sys.argv:
        sys.argv.remove("--aggregate")
        main_aggregate()
        sys.exit(0)
    main()
    print(f"Saved {len(figures)} page(s): {out_path}")


if __name__ == "__main__":
    main()
