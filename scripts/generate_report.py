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
    format_ablation_table,
    format_degradation_table,
)
from pdna.analysis.visualize import (
    plot_degradation_curves,
    plot_training_curves,
    plot_ablation_heatmap,
)


def generate_report(results_dir: str = "runs", output_dir: str = "reports") -> None:
    """Generate full technical report with tables and figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load results
    results = load_all_results(results_dir)
    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} experiment runs")

    # Compute tables
    ablation = compute_ablation_table(results)
    degradation = compute_degradation_table(results)

    # Determine tasks and variants from results
    tasks = sorted({t for _, t in ablation.keys()})
    variants = sorted({v for v, _ in ablation.keys()})

    # Format text tables
    abl_text = format_ablation_table(ablation, tasks, variants)
    deg_text = format_degradation_table(degradation, tasks, variants)

    print("\n" + abl_text)
    print("\n" + deg_text)

    # Generate figures
    try:
        fig = plot_degradation_curves(degradation, tasks, save_path=str(figures_dir / "degradation_curves.png"))
        print("Saved degradation curves")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate degradation curves: {e}")

    try:
        fig = plot_training_curves(results, save_path=str(figures_dir / "training_curves.png"))
        print("Saved training curves")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate training curves: {e}")

    try:
        fig = plot_ablation_heatmap(ablation, tasks, variants, save_path=str(figures_dir / "ablation_heatmap.png"))
        print("Saved ablation heatmap")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate ablation heatmap: {e}")

    # Compute success criteria
    success = evaluate_success_criteria(ablation, degradation, variants, tasks)

    # Generate markdown report
    report = _build_report_markdown(abl_text, deg_text, success, variants, tasks, results, ablation, degradation)
    report_path = out / "technical_report.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


def evaluate_success_criteria(ablation, degradation, variants, tasks):
    """Evaluate against defined success criteria."""
    criteria = {}

    # Get baseline and full_pdna accuracies
    baseline_accs = {t: ablation.get(("baseline", t), {}).get("mean", 0) for t in tasks}
    pdna_accs = {t: ablation.get(("full_pdna", t), {}).get("mean", 0) for t in tasks}

    # Check: Full PDNA >= 2% improvement over baseline on all tasks
    improvements = {t: pdna_accs[t] - baseline_accs[t] for t in tasks}
    all_improved = all(v >= 0.02 for v in improvements.values()) if improvements else False
    tasks_improved = sum(1 for v in improvements.values() if v > 0)

    # Check degradation
    baseline_degs = {t: degradation.get(("baseline", t), {}).get("mean_degradation", 0) for t in tasks}
    pdna_degs = {t: degradation.get(("full_pdna", t), {}).get("mean_degradation", 0) for t in tasks}
    deg_ratios = {}
    for t in tasks:
        if baseline_degs.get(t, 0) > 0:
            deg_ratios[t] = pdna_degs.get(t, 0) / baseline_degs[t]

    # Check noise vs pulse
    noise_accs = {t: ablation.get(("noise", t), {}).get("mean", 0) for t in tasks}
    pulse_accs = {t: ablation.get(("pulse", t), {}).get("mean", 0) for t in tasks}
    pulse_beats_noise = sum(1 for t in tasks if pulse_accs.get(t, 0) > noise_accs.get(t, 0))

    criteria["improvements"] = improvements
    criteria["tasks_improved"] = tasks_improved
    criteria["all_improved_2pct"] = all_improved
    criteria["degradation_ratios"] = deg_ratios
    criteria["pulse_beats_noise"] = pulse_beats_noise

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


def _build_report_markdown(abl_text, deg_text, success, variants, tasks, results, ablation, degradation):
    """Build the full technical report as markdown."""
    lines = [
        "# PDNA — Technical Report",
        "",
        "## Pulse-Driven Neural Architecture: Experimental Results",
        "",
        "---",
        "",
        "## 1. Ablation Table",
        "",
        "Test accuracy across variants and tasks (mean +/- std across seeds):",
        "",
        "```",
        abl_text,
        "```",
        "",
        "## 2. Gapped-LRA Degradation",
        "",
        "Performance degradation (gap_0 - gap_30 accuracy):",
        "",
        "```",
        deg_text,
        "```",
        "",
        "## 3. Key Findings",
        "",
    ]

    # Add improvement details
    for task, imp in success.get("improvements", {}).items():
        direction = "+" if imp > 0 else ""
        lines.append(f"- **{task}**: PDNA vs Baseline = {direction}{imp*100:.2f}%")

    lines.extend([
        "",
        f"- Tasks where PDNA beats baseline: {success.get('tasks_improved', 0)}/{len(tasks)}",
        f"- Tasks where pulse beats noise: {success.get('pulse_beats_noise', 0)}/{len(tasks)}",
        "",
        "## 4. Degradation Analysis",
        "",
    ])

    for task, ratio in success.get("degradation_ratios", {}).items():
        lines.append(f"- **{task}**: PDNA degradation = {ratio:.2f}x baseline degradation")

    lines.extend([
        "",
        "## 5. Compute Overhead",
        "",
    ])

    # Compute wall times
    for v in variants:
        times = []
        for key, data in results.items():
            if key.startswith(v + "_") and "wall_time" in data:
                times.append(data["wall_time"])
        if times:
            lines.append(f"- **{v}**: {np.mean(times):.1f}s avg per run")

    # Param counts
    lines.extend(["", "### Parameter Counts", ""])
    for v in variants:
        for key, data in results.items():
            if key.startswith(v + "_") and "params" in data:
                lines.append(f"- **{v}**: {data['params']:,} parameters")
                break

    lines.extend([
        "",
        "## 6. Success Criteria Evaluation",
        "",
        f"**Overall Assessment: {success.get('level', 'Unknown')}**",
        "",
        "| Level | Criteria | Met? |",
        "|-------|----------|------|",
        f"| Strong | >= 2% avg improvement on all tasks | {'Yes' if success.get('all_improved_2pct') else 'No'} |",
        f"| Moderate | Outperforms on >= 60% of tasks | {'Yes' if success.get('tasks_improved', 0) >= max(1, len(tasks) * 0.6) else 'No'} |",
        f"| Minimal | Any improvement OR pulse > noise | {'Yes' if success.get('tasks_improved', 0) > 0 or success.get('pulse_beats_noise', 0) > 0 else 'No'} |",
        "",
        "## 7. Figures",
        "",
        "- `figures/degradation_curves.png` — THE KEY GRAPH: Performance under increasing gaps",
        "- `figures/training_curves.png` — Training convergence across variants",
        "- `figures/ablation_heatmap.png` — Ablation results heatmap",
        "",
        "---",
        "",
        "*Generated by PDNA analysis pipeline*",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)
