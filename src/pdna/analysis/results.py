"""Results loading, aggregation, and statistical analysis for PDNA experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_all_results(runs_dir: str = "runs") -> dict:
    """Load all experiment results from runs directory."""
    runs_path = Path(runs_dir)
    all_results = {}

    # Try aggregated file first
    summary = runs_path / "all_results.json"
    if summary.exists():
        with open(summary) as f:
            return json.load(f)

    # Load individual results
    for run_dir in sorted(runs_path.iterdir()):
        if run_dir.is_dir():
            result_file = run_dir / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    all_results[run_dir.name] = json.load(f)

    return all_results


def _parse_run_key(key: str) -> tuple[str, str, int]:
    """Parse 'variant_task_seedN' into (variant, task, seed)."""
    parts = key.rsplit("_seed", 1)
    seed = int(parts[1])
    rest = parts[0]
    # Find variant/task split — variants may contain underscores
    from pdna.models.variants import Variant
    for v in Variant:
        if rest.startswith(v.value + "_"):
            task = rest[len(v.value) + 1:]
            return v.value, task, seed
    # Fallback: split at first underscore
    variant, task = rest.split("_", 1)
    return variant, task, seed


def compute_ablation_table(results: dict) -> dict:
    """Compute mean +/- std accuracy table across seeds.

    Returns:
        Dict mapping (variant, task) -> {"mean": float, "std": float, "values": list}
    """
    # Group by (variant, task)
    grouped: dict[tuple[str, str], list[float]] = {}
    for key, data in results.items():
        if "error" in data:
            continue
        variant, task, seed = _parse_run_key(key)
        test_acc = data.get("test_acc")
        if test_acc is not None:
            grouped.setdefault((variant, task), []).append(test_acc)

    table = {}
    for (variant, task), values in grouped.items():
        table[(variant, task)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return table


def compute_degradation_table(results: dict) -> dict:
    """Compute degradation scores (gap_0 - gap_30 accuracy) by variant and task.

    Returns:
        Dict mapping (variant, task) -> {"mean_degradation": float, "gap_results": dict}
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for key, data in results.items():
        if "error" in data:
            continue
        variant, task, seed = _parse_run_key(key)
        gapped = data.get("gapped")
        if gapped:
            grouped.setdefault((variant, task), []).append(gapped)

    table = {}
    for (variant, task), gap_results_list in grouped.items():
        degradations = []
        avg_gap = {}
        for gap_results in gap_results_list:
            g0 = gap_results.get("gap_0", {}).get("accuracy", 0)
            g30 = gap_results.get("gap_30", {}).get("accuracy", 0)
            degradations.append(g0 - g30)

            for level, data in gap_results.items():
                avg_gap.setdefault(level, []).append(data["accuracy"])

        table[(variant, task)] = {
            "mean_degradation": float(np.mean(degradations)),
            "std_degradation": float(np.std(degradations)),
            "gap_accuracies": {level: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                              for level, vals in avg_gap.items()},
        }
    return table


def format_ablation_table(table: dict, tasks: list[str] | None = None, variants: list[str] | None = None) -> str:
    """Format ablation table as readable text."""
    if tasks is None:
        tasks = sorted({t for _, t in table.keys()})
    if variants is None:
        variants = sorted({v for v, _ in table.keys()})

    # Header
    header = f"{'Variant':<15}" + "".join(f"{t:<20}" for t in tasks) + "Avg"
    lines = [header, "=" * len(header)]

    for v in variants:
        row = f"{v:<15}"
        accs = []
        for t in tasks:
            entry = table.get((v, t))
            if entry:
                row += f"{entry['mean']:.4f} +/- {entry['std']:.4f}  "
                accs.append(entry["mean"])
            else:
                row += f"{'N/A':<20}"
        avg = np.mean(accs) if accs else 0
        row += f"{avg:.4f}"
        lines.append(row)

    return "\n".join(lines)


def format_degradation_table(table: dict, tasks: list[str] | None = None, variants: list[str] | None = None) -> str:
    """Format degradation table as readable text."""
    if tasks is None:
        tasks = sorted({t for _, t in table.keys()})
    if variants is None:
        variants = sorted({v for v, _ in table.keys()})

    header = f"{'Variant':<15}" + "".join(f"{t:<20}" for t in tasks) + "Avg Degradation"
    lines = [header, "=" * len(header)]

    for v in variants:
        row = f"{v:<15}"
        degs = []
        for t in tasks:
            entry = table.get((v, t))
            if entry:
                row += f"{entry['mean_degradation']:.4f} +/- {entry['std_degradation']:.4f}  "
                degs.append(entry["mean_degradation"])
            else:
                row += f"{'N/A':<20}"
        avg = np.mean(degs) if degs else 0
        row += f"{avg:.4f}"
        lines.append(row)

    return "\n".join(lines)


def compute_statistical_tests(results: dict) -> dict:
    """Compute pairwise statistical tests between variants.

    For each task, computes:
    - Paired t-test (PDNA vs baseline, pulse vs noise)
    - Cohen's d effect size
    - 95% confidence intervals for differences

    Returns dict with test results per task.
    """
    comparisons = [
        ("full_pdna", "baseline", "PDNA vs Baseline"),
        ("pulse", "noise", "Pulse vs Noise"),
        ("full_pdna", "pulse", "Full PDNA vs Pulse-only"),
        ("full_idle", "full_pdna", "Idle vs PDNA"),
        ("self_attend", "baseline", "SelfAttend vs Baseline"),
    ]

    # Group results by (variant, task, seed)
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for key, data in results.items():
        if "error" in data or "test_acc" not in data:
            continue
        variant, task, seed = _parse_run_key(key)
        grouped.setdefault((variant, task), []).append((seed, data["test_acc"]))

    tasks = sorted({t for _, t in grouped.keys()})
    stat_results = {}

    for task in tasks:
        task_stats = {}
        for v1, v2, label in comparisons:
            vals1 = sorted(grouped.get((v1, task), []))
            vals2 = sorted(grouped.get((v2, task), []))

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            a = np.array([v for _, v in vals1])
            b = np.array([v for _, v in vals2])

            # Paired t-test (if same seeds, pair them)
            if len(a) == len(b):
                t_stat, p_value = stats.ttest_rel(a, b)
            else:
                t_stat, p_value = stats.ttest_ind(a, b)

            # Cohen's d
            diff = a - b if len(a) == len(b) else None
            if diff is not None and np.std(diff) > 0:
                cohens_d = float(np.mean(diff) / np.std(diff))
            else:
                pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
                cohens_d = float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0

            # 95% CI for mean difference
            mean_diff = float(np.mean(a) - np.mean(b))
            if diff is not None:
                se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
            else:
                se = float(np.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b)))
            ci_95 = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)

            task_stats[label] = {
                "v1_mean": float(np.mean(a)),
                "v2_mean": float(np.mean(b)),
                "mean_diff": mean_diff,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": cohens_d,
                "ci_95_low": ci_95[0],
                "ci_95_high": ci_95[1],
                "significant_005": p_value < 0.05,
                "n_samples": min(len(a), len(b)),
            }

        stat_results[task] = task_stats

    return stat_results


def compute_degradation_stats(results: dict) -> dict:
    """Compute statistical comparison of degradation between variants."""
    comparisons = [
        ("full_pdna", "baseline", "PDNA vs Baseline degradation"),
        ("pulse", "noise", "Pulse vs Noise degradation"),
        ("full_idle", "full_pdna", "Idle vs PDNA degradation"),
    ]

    grouped: dict[tuple[str, str], list[float]] = {}
    for key, data in results.items():
        if "error" in data or "degradation" not in data:
            continue
        variant, task, seed = _parse_run_key(key)
        grouped.setdefault((variant, task), []).append(data["degradation"])

    tasks = sorted({t for _, t in grouped.keys()})
    deg_stats = {}

    for task in tasks:
        task_stats = {}
        for v1, v2, label in comparisons:
            a = np.array(grouped.get((v1, task), []))
            b = np.array(grouped.get((v2, task), []))

            if len(a) < 2 or len(b) < 2:
                continue

            if len(a) == len(b):
                t_stat, p_value = stats.ttest_rel(a, b)
            else:
                t_stat, p_value = stats.ttest_ind(a, b)

            task_stats[label] = {
                "v1_mean_deg": float(np.mean(a)),
                "v2_mean_deg": float(np.mean(b)),
                "diff": float(np.mean(a) - np.mean(b)),
                "p_value": float(p_value),
                "v1_less_degradation": float(np.mean(a)) < float(np.mean(b)),
            }

        deg_stats[task] = task_stats

    return deg_stats


def compute_overhead_table(results: dict, task_filter: str | None = None) -> dict:
    """Compute parameter count and wall time overhead by variant.

    Args:
        results: Experiment results dict.
        task_filter: If set, only include runs for this task (avoids mixing
            tasks with different run times).
    """
    overhead: dict[str, dict] = {}

    for key, data in results.items():
        if "error" in data:
            continue
        variant, task, seed = _parse_run_key(key)
        if task_filter and task != task_filter:
            continue
        overhead.setdefault(variant, {"params": [], "wall_times": [], "tasks": set()})
        if "params" in data:
            overhead[variant]["params"].append(data["params"])
        if "wall_time" in data:
            overhead[variant]["wall_times"].append(data["wall_time"])
        overhead[variant]["tasks"].add(task)

    table = {}
    baseline_time = None
    for v, d in overhead.items():
        params = d["params"][0] if d["params"] else 0
        mean_time = float(np.mean(d["wall_times"])) if d["wall_times"] else 0
        if v == "baseline":
            baseline_time = mean_time
        table[v] = {
            "params": params,
            "mean_wall_time": mean_time,
            "std_wall_time": float(np.std(d["wall_times"])) if d["wall_times"] else 0,
            "n_runs": len(d["wall_times"]),
        }

    # Add overhead ratio relative to baseline
    if baseline_time and baseline_time > 0:
        for v in table:
            table[v]["overhead_ratio"] = table[v]["mean_wall_time"] / baseline_time

    return table
