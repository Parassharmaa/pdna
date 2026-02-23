#!/usr/bin/env python
"""Populate paper tables and figures from v5 experiment results.

Reads runs_v5/all_results.json and generates:
1. LaTeX tables for inclusion in paper/main.tex
2. Publication-quality figures
3. Statistical analysis summary
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna"]
VARIANT_LABELS = {
    "baseline": "A. Baseline \\cfc{}",
    "noise": "B. \\cfc{} + Noise",
    "pulse": "C. \\cfc{} + Pulse",
    "self_attend": "D. \\cfc{} + SelfAttend",
    "full_pdna": "E. Full \\pdna{}",
}
TASKS = ["smnist"]
TASK_LABELS = {"smnist": "sMNIST"}
SEEDS = [42, 123, 456, 789, 1337]

GAP_LEVELS = ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"]


def load_results(results_path="runs_v5/all_results.json"):
    with open(results_path) as f:
        return json.load(f)


def get_values(results, variant, task, key, seeds=SEEDS):
    """Get values for a variant-task pair across seeds."""
    vals = []
    for s in seeds:
        run_key = f"{variant}_{task}_seed{s}"
        data = results.get(run_key, {})
        if "error" not in data and key in data:
            vals.append(data[key])
    return vals


def get_gap_values(results, variant, task, gap_level, seeds=SEEDS):
    """Get gap accuracy values."""
    vals = []
    for s in seeds:
        run_key = f"{variant}_{task}_seed{s}"
        data = results.get(run_key, {})
        if "error" not in data and "gapped" in data:
            acc = data["gapped"].get(gap_level, {}).get("accuracy")
            if acc is not None:
                vals.append(acc)
    return vals


def fmt(mean, std, percent=True):
    if percent:
        return f"{mean*100:.2f} $\\pm$ {std*100:.2f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def generate_accuracy_table(results):
    """Generate LaTeX accuracy table."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Test accuracy (\\%, mean $\\pm$ std across 5 seeds). Bold indicates best per task.}")
    lines.append("\\label{tab:accuracy}")
    lines.append("\\begin{tabular}{l" + "c" * len(TASKS) + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variant} & " + " & ".join(f"\\textbf{{{TASK_LABELS[t]}}}" for t in TASKS) + " \\\\")
    lines.append("\\midrule")

    # Find best per task
    best = {}
    for t in TASKS:
        best_mean = -1
        for v in VARIANTS:
            vals = get_values(results, v, t, "test_acc")
            if vals and np.mean(vals) > best_mean:
                best_mean = np.mean(vals)
                best[t] = v

    for v in VARIANTS:
        row = [VARIANT_LABELS[v]]
        for t in TASKS:
            vals = get_values(results, v, t, "test_acc")
            if vals:
                m, s = np.mean(vals), np.std(vals)
                cell = fmt(m, s)
                if best.get(t) == v:
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_degradation_table(results):
    """Generate LaTeX degradation table."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Degradation (Gap 0\\% $-$ Gap 30\\% accuracy, in \\%). Lower = more robust. Bold indicates least degradation per task.}")
    lines.append("\\label{tab:degradation}")
    lines.append("\\begin{tabular}{l" + "c" * len(TASKS) + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variant} & " + " & ".join(f"\\textbf{{{TASK_LABELS[t]}}}" for t in TASKS) + " \\\\")
    lines.append("\\midrule")

    best = {}
    for t in TASKS:
        best_deg = float("inf")
        for v in VARIANTS:
            vals = get_values(results, v, t, "degradation")
            if vals and np.mean(vals) < best_deg:
                best_deg = np.mean(vals)
                best[t] = v

    for v in VARIANTS:
        row = [VARIANT_LABELS[v]]
        for t in TASKS:
            vals = get_values(results, v, t, "degradation")
            if vals:
                m, s = np.mean(vals), np.std(vals)
                cell = fmt(m, s)
                if best.get(t) == v:
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_gap_table(results, task):
    """Generate per-gap-level table for a specific task."""
    lines = []
    lines.append(f"\\begin{{table}}[h]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{Accuracy (\\%) at each gap level for {TASK_LABELS[task]}.}}")
    lines.append(f"\\label{{tab:gap_{task}}}")
    lines.append("\\begin{tabular}{l" + "c" * len(GAP_LEVELS) + "}")
    lines.append("\\toprule")
    gap_headers = ["0\\%", "5\\%", "15\\%", "30\\%", "Multi"]
    lines.append("\\textbf{Variant} & " + " & ".join(f"\\textbf{{{h}}}" for h in gap_headers) + " \\\\")
    lines.append("\\midrule")

    for v in VARIANTS:
        row = [VARIANT_LABELS[v]]
        for gl in GAP_LEVELS:
            vals = get_gap_values(results, v, task, gl)
            if vals:
                row.append(f"{np.mean(vals)*100:.2f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_stats_table(results):
    """Generate statistical comparison table."""
    comparisons = [
        ("full_pdna", "baseline", "PDNA vs Baseline"),
        ("pulse", "noise", "Pulse vs Noise"),
        ("pulse", "baseline", "Pulse vs Baseline"),
        ("self_attend", "baseline", "SelfAttend vs Baseline"),
        ("full_pdna", "pulse", "Full PDNA vs Pulse-only"),
    ]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical comparisons (paired $t$-test, 5 seeds). $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}")
    lines.append("\\label{tab:stats}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Task} & \\textbf{Comparison} & \\textbf{$\\Delta$ Acc (\\%)} & \\textbf{$\\Delta$ Deg (\\%)} & \\textbf{$p$ (Acc)} & \\textbf{$p$ (Deg)} \\\\")
    lines.append("\\midrule")

    for t in TASKS:
        first = True
        for v1, v2, label in comparisons:
            a1 = get_values(results, v1, t, "test_acc")
            a2 = get_values(results, v2, t, "test_acc")
            d1 = get_values(results, v1, t, "degradation")
            d2 = get_values(results, v2, t, "degradation")

            if len(a1) < 2 or len(a2) < 2 or len(a1) != len(a2):
                continue

            _, p_acc = stats.ttest_rel(a1, a2)
            _, p_deg = stats.ttest_rel(d1, d2)
            diff_acc = (np.mean(a1) - np.mean(a2)) * 100
            diff_deg = (np.mean(d1) - np.mean(d2)) * 100

            sig_acc = "^{***}" if p_acc < 0.01 else ("^{**}" if p_acc < 0.05 else ("^{*}" if p_acc < 0.1 else ""))
            sig_deg = "^{***}" if p_deg < 0.01 else ("^{**}" if p_deg < 0.05 else ("^{*}" if p_deg < 0.1 else ""))

            task_col = TASK_LABELS[t] if first else ""
            first = False
            lines.append(
                f"{task_col} & {label} & "
                f"${diff_acc:+.2f}{sig_acc}$ & "
                f"${diff_deg:+.2f}{sig_deg}$ & "
                f"{p_acc:.4f} & {p_deg:.4f} \\\\"
            )

        if t != TASKS[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_overhead_table(results):
    """Generate compute overhead table."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Compute overhead comparison (sMNIST task).}")
    lines.append("\\label{tab:overhead}")
    lines.append("\\begin{tabular}{lccr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variant} & \\textbf{Parameters} & \\textbf{Overhead} & \\textbf{Avg Time (s)} \\\\")
    lines.append("\\midrule")

    baseline_time = None
    for v in VARIANTS:
        times = get_values(results, v, "smnist", "wall_time")
        params_list = get_values(results, v, "smnist", "params")
        params = params_list[0] if params_list else 0
        mean_time = np.mean(times) if times else 0
        std_time = np.std(times) if times else 0

        if v == "baseline":
            baseline_time = mean_time

        ratio = f"{mean_time/baseline_time:.2f}$\\times$" if baseline_time and baseline_time > 0 else "--"
        if v == "baseline":
            ratio = "1.00$\\times$"

        lines.append(
            f"{VARIANT_LABELS[v]} & {params:,} & {ratio} & "
            f"{mean_time:.1f} $\\pm$ {std_time:.1f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_pulse_analysis(results):
    """Summarize learned pulse parameters."""
    lines = ["\\subsection{Learned Pulse Parameters}", ""]

    for t in TASKS:
        lines.append(f"\\paragraph{{{TASK_LABELS[t]}.}}")
        for v in ["pulse", "full_pdna"]:
            alphas = []
            omegas_all = []
            for s in SEEDS:
                key = f"{v}_{t}_seed{s}"
                data = results.get(key, {})
                pp = data.get("pulse_params", {})
                if "alpha" in pp:
                    alphas.append(pp["alpha"])
                if "omega" in pp:
                    omegas_all.append(pp["omega"])

            if alphas:
                lines.append(
                    f"{VARIANT_LABELS[v]}: $\\alpha = {np.mean(alphas):.4f} \\pm {np.std(alphas):.4f}$"
                )
            if omegas_all:
                all_omegas = np.array(omegas_all)
                lines.append(
                    f"($\\omega$ range: [{all_omegas.min():.2f}, {all_omegas.max():.2f}], "
                    f"mean={all_omegas.mean():.2f})"
                )
        lines.append("")

    return "\n".join(lines)


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "runs_v5/all_results.json"
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    valid = sum(1 for v in results.values() if "error" not in v)
    total = len(results)
    print(f"Loaded {valid}/{total} valid runs\n")

    # Print completed runs summary
    print("=== COMPLETED RUNS ===")
    for t in TASKS:
        for v in VARIANTS:
            seeds_done = []
            for s in SEEDS:
                key = f"{v}_{t}_seed{s}"
                if key in results and "error" not in results[key]:
                    seeds_done.append(s)
            if seeds_done:
                print(f"  {v}_{t}: {len(seeds_done)}/5 seeds")

    print("\n=== ACCURACY TABLE ===")
    print(generate_accuracy_table(results))

    print("\n=== DEGRADATION TABLE ===")
    print(generate_degradation_table(results))

    print("\n=== STATISTICAL COMPARISONS ===")
    print(generate_stats_table(results))

    print("\n=== OVERHEAD TABLE ===")
    print(generate_overhead_table(results))

    for t in TASKS:
        vals = get_values(results, "baseline", t, "test_acc")
        if vals:
            print(f"\n=== GAP TABLE: {t} ===")
            print(generate_gap_table(results, t))

    print("\n=== PULSE PARAMETER ANALYSIS ===")
    print(generate_pulse_analysis(results))

    # Save all tables to a file
    out_path = Path("paper/tables_generated.tex")
    with open(out_path, "w") as f:
        f.write("% Auto-generated tables from experiment results\n")
        f.write("% Generated by scripts/populate_paper.py\n\n")
        f.write("% === ACCURACY TABLE ===\n")
        f.write(generate_accuracy_table(results))
        f.write("\n\n% === DEGRADATION TABLE ===\n")
        f.write(generate_degradation_table(results))
        f.write("\n\n% === STATISTICAL COMPARISONS ===\n")
        f.write(generate_stats_table(results))
        f.write("\n\n% === OVERHEAD TABLE ===\n")
        f.write(generate_overhead_table(results))
        for t in TASKS:
            vals = get_values(results, "baseline", t, "test_acc")
            if vals:
                f.write(f"\n\n% === GAP TABLE: {t} ===\n")
                f.write(generate_gap_table(results, t))
        f.write(f"\n\n% === PULSE ANALYSIS ===\n")
        f.write(generate_pulse_analysis(results))

    print(f"\nTables written to {out_path}")


if __name__ == "__main__":
    main()
