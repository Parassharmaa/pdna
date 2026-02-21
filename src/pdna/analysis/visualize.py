"""Visualization functions for PDNA experiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
VARIANT_LABELS = {
    "baseline": "A. Baseline",
    "noise": "B. Noise",
    "pulse": "C. Pulse",
    "self_attend": "D. SelfAttend",
    "full_pdna": "E. PDNA",
    "full_idle": "F. Idle",
}
VARIANT_COLORS = {
    "baseline": "#1f77b4",
    "noise": "#ff7f0e",
    "pulse": "#2ca02c",
    "self_attend": "#d62728",
    "full_pdna": "#9467bd",
    "full_idle": "#8c564b",
}


def plot_degradation_curves(
    degradation_table: dict,
    tasks: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """THE KEY GRAPH: Performance degradation under increasing input gaps.

    X-axis: Gap level (0%, 5%, 15%, 30%, multi)
    Y-axis: Test accuracy
    Lines: One per variant
    """
    gap_levels = ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"]
    gap_labels = ["0%", "5%", "15%", "30%", "Multi"]

    if tasks is None:
        tasks = sorted({t for _, t in degradation_table.keys()})

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), squeeze=False)

    variants = [v for v in VARIANT_ORDER if any((v, t) in degradation_table for t in tasks)]

    for idx, task in enumerate(tasks):
        ax = axes[0, idx]
        for variant in variants:
            entry = degradation_table.get((variant, task))
            if entry is None:
                continue
            accs = []
            for gl in gap_levels:
                ga = entry.get("gap_accuracies", {}).get(gl, {})
                accs.append(ga.get("mean", 0))
            label = VARIANT_LABELS.get(variant, variant)
            color = VARIANT_COLORS.get(variant, "#333333")
            ax.plot(gap_labels, accs, "o-", label=label, color=color, linewidth=2, markersize=6)

        ax.set_title(f"{task}", fontsize=14)
        ax.set_xlabel("Gap Level")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Performance Degradation Under Increasing Input Gaps", fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    results: dict,
    metric: str = "val_acc",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training convergence curves grouped by task."""
    from pdna.analysis.results import _parse_run_key

    # Group by (variant, task)
    grouped: dict[tuple[str, str], list] = {}
    for key, data in results.items():
        if "error" in data or "history" not in data:
            continue
        variant, task, seed = _parse_run_key(key)
        history = data["history"]
        if metric in history:
            grouped.setdefault((variant, task), []).append(history[metric])

    tasks = sorted({t for _, t in grouped.keys()})
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), squeeze=False)

    variants = [v for v in VARIANT_ORDER if any((v, t) in grouped for t in tasks)]

    for idx, task in enumerate(tasks):
        ax = axes[0, idx]
        for variant in variants:
            curves = grouped.get((variant, task), [])
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            aligned = np.array([c[:min_len] for c in curves])
            mean = aligned.mean(axis=0)
            std = aligned.std(axis=0)
            epochs = np.arange(1, min_len + 1)
            label = VARIANT_LABELS.get(variant, variant)
            color = VARIANT_COLORS.get(variant, "#333333")
            ax.plot(epochs, mean, label=label, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"{task}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Convergence", fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_heatmap(
    ablation_table: dict,
    tasks: list[str] | None = None,
    variants: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot ablation results as a heatmap."""
    if tasks is None:
        tasks = sorted({t for _, t in ablation_table.keys()})
    if variants is None:
        variants = sorted({v for v, _ in ablation_table.keys()})

    data = np.zeros((len(variants), len(tasks)))
    for i, v in enumerate(variants):
        for j, t in enumerate(tasks):
            entry = ablation_table.get((v, t))
            data[i, j] = entry["mean"] if entry else 0

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2), max(4, len(variants) * 0.8)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)

    # Add text annotations
    for i in range(len(variants)):
        for j in range(len(tasks)):
            text = f"{data[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, label="Test Accuracy")
    ax.set_title("Ablation Results (Test Accuracy)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_degradation_bars(
    degradation_table: dict,
    tasks: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing degradation (gap_0 - gap_30) across variants per task."""
    if tasks is None:
        tasks = sorted({t for _, t in degradation_table.keys()})

    variants = [v for v in VARIANT_ORDER if any((v, t) in degradation_table for t in tasks)]
    n_variants = len(variants)
    x = np.arange(len(tasks))
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 3), 5))

    for i, v in enumerate(variants):
        degs = []
        errs = []
        for t in tasks:
            entry = degradation_table.get((v, t))
            if entry:
                degs.append(entry["mean_degradation"])
                errs.append(entry.get("std_degradation", 0))
            else:
                degs.append(0)
                errs.append(0)

        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, "#333333")
        offset = (i - n_variants / 2 + 0.5) * width
        ax.bar(x + offset, degs, width, yerr=errs, label=label, color=color, alpha=0.85, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylabel("Degradation (Gap0% - Gap30% accuracy)")
    ax.set_title("Gap Robustness: Lower = More Robust", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frequency_spectrum(
    model,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot learned frequency spectrum (omega values) from pulse models."""
    fig, ax = plt.subplots(figsize=(8, 4))

    omegas = []
    for name, param in model.named_parameters():
        if "omega" in name:
            omegas.extend(param.detach().cpu().numpy().flatten())

    if omegas:
        ax.hist(omegas, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_xlabel("Frequency (omega)")
        ax.set_ylabel("Count")
        ax.set_title("Learned Frequency Spectrum")
        ax.axvline(np.mean(omegas), color="red", linestyle="--", label=f"Mean: {np.mean(omegas):.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No omega parameters found", ha="center", va="center", transform=ax.transAxes)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
