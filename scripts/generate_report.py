"""Generate technical report from experiment results.

Usage:
    uv run python scripts/generate_report.py --results-dir runs --output-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pdna.analysis.results import (
    load_all_results,
    compute_ablation_table,
    compute_degradation_table,
    compute_statistical_tests,
    compute_degradation_stats,
    compute_overhead_table,
    format_ablation_table,
    format_degradation_table,
)
from pdna.analysis.visualize import (
    plot_degradation_curves,
    plot_degradation_bars,
    plot_training_curves,
    plot_ablation_heatmap,
)


VARIANT_ORDER = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
VARIANT_LABELS = {
    "baseline": "A. Baseline CfC",
    "noise": "B. CfC + Noise",
    "pulse": "C. CfC + Pulse",
    "self_attend": "D. CfC + SelfAttend",
    "full_pdna": "E. Full PDNA",
    "full_idle": "F. Full + Idle",
}


def generate_report(results_dir: str = "runs", output_dir: str = "reports") -> None:
    """Generate full technical report with tables, figures, and statistical analysis."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load results
    results = load_all_results(results_dir)
    if not results:
        print("No results found!")
        return

    # Filter out errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    print(f"Loaded {len(valid_results)} valid experiment runs (of {len(results)} total)")

    # Compute all tables
    ablation = compute_ablation_table(valid_results)
    degradation = compute_degradation_table(valid_results)
    stat_tests = compute_statistical_tests(valid_results)
    deg_stats = compute_degradation_stats(valid_results)
    overhead = compute_overhead_table(valid_results)

    tasks = sorted({t for _, t in ablation.keys()})
    variants = [v for v in VARIANT_ORDER if any((v, t) in ablation for t in tasks)]

    # Print summary
    abl_text = format_ablation_table(ablation, tasks, variants)
    deg_text = format_degradation_table(degradation, tasks, variants)
    print("\n" + abl_text)
    print("\n" + deg_text)

    # Generate figures
    _generate_figures(degradation, valid_results, ablation, tasks, variants, figures_dir)

    # Evaluate success criteria
    success = evaluate_success_criteria(ablation, degradation, variants, tasks)

    # Build and save report
    report = _build_report(
        ablation, degradation, stat_tests, deg_stats, overhead,
        success, variants, tasks, valid_results, figures_dir,
    )
    report_path = out / "technical_report.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Save raw analysis data
    analysis_data = {
        "ablation": {f"{v}_{t}": d for (v, t), d in ablation.items()},
        "degradation": {f"{v}_{t}": d for (v, t), d in degradation.items()},
        "statistical_tests": stat_tests,
        "degradation_stats": deg_stats,
        "overhead": overhead,
        "success_criteria": success,
    }
    with open(out / "analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)
    print(f"Analysis data saved to {out / 'analysis_data.json'}")


def _generate_figures(degradation, results, ablation, tasks, variants, figures_dir):
    """Generate all figures, catching errors gracefully."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, fn, kwargs in [
        ("degradation_curves", plot_degradation_curves,
         dict(degradation_table=degradation, tasks=tasks)),
        ("degradation_bars", plot_degradation_bars,
         dict(degradation_table=degradation, tasks=tasks)),
        ("training_curves", plot_training_curves,
         dict(results=results)),
        ("ablation_heatmap", plot_ablation_heatmap,
         dict(ablation_table=ablation, tasks=tasks, variants=variants)),
    ]:
        try:
            fig = fn(**kwargs, save_path=str(figures_dir / f"{name}.png"))
            plt.close(fig)
            print(f"  Saved {name}.png")
        except Exception as e:
            print(f"  Could not generate {name}: {e}")


def evaluate_success_criteria(ablation, degradation, variants, tasks):
    """Evaluate results against defined success criteria."""
    criteria = {}

    baseline_accs = {t: ablation.get(("baseline", t), {}).get("mean", 0) for t in tasks}
    pdna_accs = {t: ablation.get(("full_pdna", t), {}).get("mean", 0) for t in tasks}
    noise_accs = {t: ablation.get(("noise", t), {}).get("mean", 0) for t in tasks}
    pulse_accs = {t: ablation.get(("pulse", t), {}).get("mean", 0) for t in tasks}

    improvements = {t: pdna_accs[t] - baseline_accs[t] for t in tasks}
    all_improved = all(v >= 0.02 for v in improvements.values()) if improvements else False
    tasks_improved = sum(1 for v in improvements.values() if v > 0)

    # Degradation comparison
    baseline_degs = {t: degradation.get(("baseline", t), {}).get("mean_degradation", 0) for t in tasks}
    pdna_degs = {t: degradation.get(("full_pdna", t), {}).get("mean_degradation", 0) for t in tasks}
    deg_ratios = {}
    for t in tasks:
        if baseline_degs.get(t, 0) > 0.001:
            deg_ratios[t] = pdna_degs.get(t, 0) / baseline_degs[t]

    pulse_beats_noise = sum(1 for t in tasks if pulse_accs.get(t, 0) > noise_accs.get(t, 0))

    criteria["improvements"] = improvements
    criteria["tasks_improved"] = tasks_improved
    criteria["total_tasks"] = len(tasks)
    criteria["all_improved_2pct"] = all_improved
    criteria["degradation_ratios"] = deg_ratios
    criteria["pulse_beats_noise"] = pulse_beats_noise
    criteria["pulse_beats_noise_total"] = len(tasks)

    # Determine level
    if all_improved and all(r <= 0.5 for r in deg_ratios.values() if deg_ratios):
        criteria["level"] = "Strong (Publishable)"
    elif tasks_improved >= max(1, len(tasks) * 0.6):
        criteria["level"] = "Moderate (Promising)"
    elif tasks_improved >= 1 or pulse_beats_noise > 0:
        criteria["level"] = "Minimal (Validated)"
    else:
        criteria["level"] = "Failure"

    return criteria


def _build_report(ablation, degradation, stat_tests, deg_stats, overhead,
                  success, variants, tasks, results, figures_dir):
    """Build the full technical report as markdown."""
    lines = []

    # Title and abstract
    lines.extend([
        "# PDNA: Pulse-Driven Neural Architecture",
        "",
        "## Experimental Results — Technical Report",
        "",
        "### Abstract",
        "",
        "This report presents the experimental evaluation of the Pulse-Driven Neural Architecture (PDNA),",
        "which augments Closed-form Continuous-time (CfC) recurrent networks with learnable oscillatory",
        "dynamics. We evaluate 6 architectural variants through a controlled ablation study on sequence",
        f"classification tasks ({', '.join(tasks)}), with gap-robustness evaluation at 5 difficulty levels.",
        "",
        "---",
        "",
    ])

    # 1. Ablation Table
    lines.extend([
        "## 1. Ablation Study Results",
        "",
        "Test accuracy across variants and tasks (mean +/- std across 3 seeds):",
        "",
    ])
    _add_markdown_table(lines, ablation, variants, tasks, metric="acc")

    # 2. Gap Degradation
    lines.extend([
        "",
        "## 2. Gap Robustness (Degradation = Gap0% acc - Gap30% acc)",
        "",
        "Lower degradation = more robust to input interruptions:",
        "",
    ])
    _add_markdown_table(lines, degradation, variants, tasks, metric="deg")

    # 3. THE KEY GRAPH description
    lines.extend([
        "",
        "## 3. Performance Under Increasing Input Gaps",
        "",
        "![Degradation Curves](figures/degradation_curves.png)",
        "",
        "![Degradation Bars](figures/degradation_bars.png)",
        "",
        "This is the core visualization of the PDNA hypothesis: models with structured internal",
        "dynamics (pulse) should degrade less gracefully when input is interrupted compared to",
        "baseline models that rely solely on input-driven state evolution.",
        "",
    ])

    # 4. Gap-level accuracy breakdown
    lines.extend(["## 4. Gap-Level Accuracy Breakdown", ""])
    for task in tasks:
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Variant | Gap 0% | Gap 5% | Gap 15% | Gap 30% | Multi-gap |")
        lines.append("|---------|--------|--------|---------|---------|-----------|")
        for v in variants:
            entry = degradation.get((v, task))
            if not entry:
                continue
            ga = entry.get("gap_accuracies", {})
            row = f"| {VARIANT_LABELS.get(v, v)} "
            for gl in ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"]:
                acc = ga.get(gl, {}).get("mean", 0)
                row += f"| {acc:.4f} "
            row += "|"
            lines.append(row)
        lines.append("")

    # 5. Statistical Analysis
    lines.extend(["## 5. Statistical Analysis", ""])
    for task in tasks:
        task_stats = stat_tests.get(task, {})
        if not task_stats:
            continue
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Comparison | Diff | t-stat | p-value | Cohen's d | Significant? | 95% CI |")
        lines.append("|------------|------|--------|---------|-----------|-------------|--------|")
        for label, s in task_stats.items():
            sig = "Yes" if s["significant_005"] else "No"
            lines.append(
                f"| {label} | {s['mean_diff']:+.4f} | {s['t_stat']:.3f} | "
                f"{s['p_value']:.4f} | {s['cohens_d']:.3f} | {sig} | "
                f"[{s['ci_95_low']:+.4f}, {s['ci_95_high']:+.4f}] |"
            )
        lines.append("")

    # 6. Degradation statistical comparison
    lines.extend(["## 6. Degradation Statistical Comparison", ""])
    for task in tasks:
        task_deg = deg_stats.get(task, {})
        if not task_deg:
            continue
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Comparison | V1 Deg | V2 Deg | Diff | p-value | Less degradation? |")
        lines.append("|------------|--------|--------|------|---------|-------------------|")
        for label, s in task_deg.items():
            lines.append(
                f"| {label} | {s['v1_mean_deg']:.4f} | {s['v2_mean_deg']:.4f} | "
                f"{s['diff']:+.4f} | {s['p_value']:.4f} | "
                f"{'V1' if s['v1_less_degradation'] else 'V2'} |"
            )
        lines.append("")

    # 7. Compute overhead
    lines.extend(["## 7. Compute Overhead", ""])
    lines.append("| Variant | Parameters | Avg Time (s) | Overhead Ratio |")
    lines.append("|---------|-----------|-------------|----------------|")
    for v in variants:
        entry = overhead.get(v)
        if not entry:
            continue
        ratio = entry.get("overhead_ratio", 1.0)
        lines.append(
            f"| {VARIANT_LABELS.get(v, v)} | {entry['params']:,} | "
            f"{entry['mean_wall_time']:.1f} +/- {entry['std_wall_time']:.1f} | "
            f"{ratio:.2f}x |"
        )
    lines.append("")

    # 8. Training curves
    lines.extend([
        "## 8. Training Convergence",
        "",
        "![Training Curves](figures/training_curves.png)",
        "",
        "![Ablation Heatmap](figures/ablation_heatmap.png)",
        "",
    ])

    # 9. Key findings
    lines.extend(["## 9. Key Findings", ""])
    for task, imp in success.get("improvements", {}).items():
        direction = "+" if imp > 0 else ""
        lines.append(f"- **{task}**: PDNA vs Baseline = {direction}{imp*100:.2f}%")
    lines.extend([
        "",
        f"- Tasks where PDNA outperforms baseline: **{success.get('tasks_improved', 0)}/{success.get('total_tasks', 0)}**",
        f"- Tasks where pulse beats noise (structured > random): **{success.get('pulse_beats_noise', 0)}/{success.get('pulse_beats_noise_total', 0)}**",
        "",
    ])

    for task, ratio in success.get("degradation_ratios", {}).items():
        better = "MORE" if ratio < 1.0 else "LESS"
        lines.append(f"- **{task}**: PDNA degradation = {ratio:.2f}x baseline ({better} robust)")
    lines.append("")

    # 10. Success criteria
    lines.extend([
        "## 10. Success Criteria Evaluation",
        "",
        f"### Overall Assessment: **{success.get('level', 'Unknown')}**",
        "",
        "| Level | Criteria | Met? |",
        "|-------|----------|------|",
        f"| Strong (Publishable) | >= 2% avg improvement on all tasks AND degradation <= 50% of baseline | {'Yes' if success.get('all_improved_2pct') else 'No'} |",
        f"| Moderate (Promising) | Outperforms on >= 60% of tasks OR clear gap advantage | {'Yes' if success.get('tasks_improved', 0) >= max(1, len(tasks) * 0.6) else 'No'} |",
        f"| Minimal (Validated) | Any improvement OR pulse > noise | {'Yes' if success.get('tasks_improved', 0) > 0 or success.get('pulse_beats_noise', 0) > 0 else 'No'} |",
        "",
        "---",
        "",
        "*Generated by PDNA analysis pipeline*",
    ])

    return "\n".join(lines)


def _add_markdown_table(lines, table, variants, tasks, metric="acc"):
    """Add a markdown-formatted table for ablation or degradation results."""
    if metric == "acc":
        header = "| Variant | " + " | ".join(tasks) + " | Avg |"
        sep = "|---------|" + "|".join("-" * (max(len(t), 18) + 2) for t in tasks) + "|-----|"
    else:
        header = "| Variant | " + " | ".join(tasks) + " | Avg |"
        sep = "|---------|" + "|".join("-" * (max(len(t), 18) + 2) for t in tasks) + "|-----|"

    lines.append(header)
    lines.append(sep)

    for v in variants:
        label = VARIANT_LABELS.get(v, v)
        row = f"| {label} "
        vals = []
        for t in tasks:
            entry = table.get((v, t))
            if entry:
                if metric == "acc":
                    m = entry.get("mean", 0)
                    s = entry.get("std", 0)
                    row += f"| {m:.4f} +/- {s:.4f} "
                    vals.append(m)
                else:
                    m = entry.get("mean_degradation", 0)
                    s = entry.get("std_degradation", 0)
                    row += f"| {m:.4f} +/- {s:.4f} "
                    vals.append(m)
            else:
                row += "| N/A "
        avg = np.mean(vals) if vals else 0
        row += f"| {avg:.4f} |"
        lines.append(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)
