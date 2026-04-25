#!/usr/bin/env python
"""Compare baseline AIME traces vs algorithm-injected continuations.

Inputs:
    outputs/pilot/aime2026_test_n30.jsonl       (baseline, per example)
    outputs/pilot/aime2026_injections.jsonl     (one record per ex,algo,span)

Outputs:
    outputs/pilot/aime2026_injections_table.md  (human-readable)
    outputs/pilot/aime2026_injections_summary.json  (programmatic)
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "outputs" / "pilot"
BASELINE = PILOT / "aime2026_test_n30.jsonl"
INJECT = PILOT / "aime2026_injections.jsonl"
OUT_TABLE = PILOT / "aime2026_injections_table.md"
OUT_JSON = PILOT / "aime2026_injections_summary.json"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def verdict(b_correct, i_correct, b_pred):
    """6 buckets covering baseline×injection×truncation."""
    b_trunc = (b_pred is None) or (str(b_pred).strip() == "")
    if b_trunc and i_correct:
        return "rescued"
    if b_trunc and not i_correct:
        return "still_failed"
    if b_correct and i_correct:
        return "unchanged_correct"
    if b_correct and not i_correct:
        return "broken"
    if (not b_correct) and i_correct:
        return "fixed"
    return "unchanged_wrong"


VERDICT_ORDER = [
    "unchanged_correct", "fixed", "rescued",
    "broken", "unchanged_wrong", "still_failed",
]


def main():
    import sys
    inj_path = INJECT if not (len(sys.argv) > 1) else Path(sys.argv[1])

    baseline = load_jsonl(BASELINE)
    inj = load_jsonl(inj_path)

    by_idx = {r["idx"]: r for r in baseline}
    print(f"baseline: {len(baseline)}  injections: {len(inj)} ({inj_path.name})")

    rows = []
    for r in inj:
        b = by_idx.get(r["ex_idx"])
        if b is None:
            continue
        b_correct = b["correct"]
        b_pred = b.get("pred")
        rows.append({
            "ex_idx": r["ex_idx"],
            "algo_idx": r["algo_idx"],
            "algo_name": r["algo_name"],
            "span_idx": r["span_idx"],
            "baseline_correct": b_correct,
            "baseline_pred": b_pred,
            "injected_correct": r["correct"],
            "injected_pred": r["pred"],
            "gold": r["gold"],
            "verdict": verdict(b_correct, r["correct"], b_pred),
            "cont_len": len(r.get("tokens", [])),
        })

    overall = Counter(r["verdict"] for r in rows)
    n = len(rows)
    base_acc = sum(1 for r in rows if r["baseline_correct"]) / n
    inj_acc = sum(1 for r in rows if r["injected_correct"]) / n

    by_algo = defaultdict(Counter)
    for r in rows:
        by_algo[r["algo_name"]][r["verdict"]] += 1
    algo_totals = {
        name: {
            "n": sum(c.values()),
            "baseline_correct": sum(c[v] for v in
                ("unchanged_correct", "broken")),
            "injected_correct": sum(c[v] for v in
                ("unchanged_correct", "fixed", "rescued")),
            **{v: c[v] for v in VERDICT_ORDER},
        } for name, c in by_algo.items()
    }

    by_ex = defaultdict(Counter)
    for r in rows:
        by_ex[r["ex_idx"]][r["verdict"]] += 1
    ex_totals = {
        ex: {
            "n": sum(c.values()),
            "baseline_correct": sum(c[v] for v in
                ("unchanged_correct", "broken")),
            "injected_correct": sum(c[v] for v in
                ("unchanged_correct", "fixed", "rescued")),
            **{v: c[v] for v in VERDICT_ORDER},
        } for ex, c in by_ex.items()
    }

    # ---- per-PROBLEM table (SOLVED/UNSOLVED, baseline vs post-injection) ----
    n_problems = len(baseline)
    base_solved = sum(1 for b in baseline if b["correct"])

    inj_by_ex = {}
    for r in inj:
        inj_by_ex.setdefault(r["ex_idx"], []).append(r)

    def collapse(rule):
        solved = 0
        for b in baseline:
            ex = b["idx"]
            recs = inj_by_ex.get(ex, [])
            if not recs:
                # no injections for this problem → fall back to baseline
                if b["correct"]:
                    solved += 1
                continue
            if rule == "any":
                if any(r["correct"] for r in recs): solved += 1
            elif rule == "all":
                if all(r["correct"] for r in recs): solved += 1
            elif rule == "majority":
                preds = [str(r["pred"]) if r["pred"] is not None else "" for r in recs]
                gold = str(b["gold"]).strip()
                # most common pred
                cnt = Counter(preds)
                top, _ = cnt.most_common(1)[0]
                # use math-verify-equivalent grading: a pred is correct if any inj
                # record with that pred was graded correct
                top_correct = any(
                    (str(r["pred"]) if r["pred"] is not None else "") == top
                    and r["correct"] for r in recs
                )
                if top_correct: solved += 1
        return solved

    inj_any = collapse("any")
    inj_maj = collapse("majority")
    inj_all = collapse("all")

    print()
    print("=" * 60)
    print(f"Per-problem table (n={n_problems})")
    print("=" * 60)
    print(f"{'rule':<32}{'SOLVED':>8}{'UNSOLVED':>10}{'rate':>8}")
    print("-" * 60)
    print(f"{'baseline':<32}{base_solved:>8}{n_problems-base_solved:>10}{base_solved/n_problems:>7.1%}")
    print(f"{'post-injection (ANY)':<32}{inj_any:>8}{n_problems-inj_any:>10}{inj_any/n_problems:>7.1%}")
    print(f"{'post-injection (MAJORITY)':<32}{inj_maj:>8}{n_problems-inj_maj:>10}{inj_maj/n_problems:>7.1%}")
    print(f"{'post-injection (ALL)':<32}{inj_all:>8}{n_problems-inj_all:>10}{inj_all/n_problems:>7.1%}")
    print("=" * 60)
    print()

    summary = {
        "n_problems": n_problems,
        "baseline_solved": base_solved,
        "post_injection_any": inj_any,
        "post_injection_majority": inj_maj,
        "post_injection_all": inj_all,
        "n_prompts": n,
        "baseline_accuracy_per_prompt": base_acc,
        "injected_accuracy_per_prompt": inj_acc,
        "verdict_counts": dict(overall),
        "by_algorithm": algo_totals,
        "by_example": ex_totals,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = []
    lines.append("# AIME 2026 — Algorithm Injection vs Baseline")
    lines.append("")
    lines.append(f"- Records: **{n}** (one per `ex×algo×span` triple)")
    lines.append(f"- Baseline accuracy (per prompt): **{base_acc:.1%}**")
    lines.append(f"- Injected accuracy (per prompt): **{inj_acc:.1%}**")
    lines.append(f"- Net Δ: **{(inj_acc-base_acc)*100:+.1f} pp**")
    lines.append("")
    lines.append("## Verdict counts")
    lines.append("")
    lines.append("| verdict | count | % |")
    lines.append("|---------|------:|---:|")
    for v in VERDICT_ORDER:
        c = overall[v]
        lines.append(f"| {v} | {c} | {c/n:.1%} |")
    lines.append("")
    lines.append("Verdict legend:")
    lines.append("- **unchanged_correct**: baseline ✓ → injected ✓")
    lines.append("- **fixed**: baseline ✗ → injected ✓ (model finished both ways)")
    lines.append("- **rescued**: baseline truncated → injected ✓")
    lines.append("- **broken**: baseline ✓ → injected ✗")
    lines.append("- **unchanged_wrong**: baseline ✗ → injected ✗")
    lines.append("- **still_failed**: baseline truncated → injected ✗")
    lines.append("")

    lines.append("## By algorithm")
    lines.append("")
    lines.append("| algorithm | n | base✓ | inj✓ | Δ pp | unchg✓ | fixed | rescued | broken | unchg✗ | still✗ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    sorted_algos = sorted(
        algo_totals.items(),
        key=lambda kv: (kv[1]["injected_correct"] - kv[1]["baseline_correct"]) / max(1, kv[1]["n"]),
        reverse=True,
    )
    for name, t in sorted_algos:
        delta = (t["injected_correct"] - t["baseline_correct"]) / t["n"] * 100
        lines.append(
            f"| {name} | {t['n']} | {t['baseline_correct']} | {t['injected_correct']} | "
            f"{delta:+.1f} | {t['unchanged_correct']} | {t['fixed']} | {t['rescued']} | "
            f"{t['broken']} | {t['unchanged_wrong']} | {t['still_failed']} |"
        )
    lines.append("")

    lines.append("## By example")
    lines.append("")
    lines.append("| ex | gold | n | base✓ | inj✓ | Δ pp | unchg✓ | fixed | rescued | broken | unchg✗ | still✗ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ex in sorted(ex_totals):
        t = ex_totals[ex]
        delta = (t["injected_correct"] - t["baseline_correct"]) / t["n"] * 100
        gold = by_idx[ex]["gold"]
        lines.append(
            f"| {ex} | {gold} | {t['n']} | {t['baseline_correct']} | {t['injected_correct']} | "
            f"{delta:+.1f} | {t['unchanged_correct']} | {t['fixed']} | {t['rescued']} | "
            f"{t['broken']} | {t['unchanged_wrong']} | {t['still_failed']} |"
        )
    lines.append("")

    # detailed flips table — high signal cases
    lines.append("## Notable flips (rescued / fixed / broken)")
    lines.append("")
    flips = [r for r in rows if r["verdict"] in ("rescued", "fixed", "broken")]
    lines.append(f"Total flips: **{len(flips)}** of {n} prompts.")
    lines.append("")
    lines.append("| ex | algo | span | gold | base_pred | inj_pred | verdict |")
    lines.append("|---:|---|---:|---:|---|---|---|")
    for r in flips:
        lines.append(
            f"| {r['ex_idx']} | {r['algo_name']} | {r['span_idx']} | {r['gold']} | "
            f"{r['baseline_pred']!r} | {r['injected_pred']!r} | {r['verdict']} |"
        )

    OUT_TABLE.write_text("\n".join(lines))
    print(f"Wrote {OUT_TABLE} ({len(lines)} lines)")
    print(f"Wrote {OUT_JSON}")
    print()
    print(f"Overall: {n} prompts | baseline {base_acc:.1%} | injected {inj_acc:.1%} | "
          f"Δ {(inj_acc-base_acc)*100:+.1f} pp")
    print(f"Verdicts: {dict(overall)}")


if __name__ == "__main__":
    main()
