"""Results loading and aggregation for PDNA experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
