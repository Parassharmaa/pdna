"""Visualization functions for PDNA experiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    variants = sorted({v for v, _ in degradation_table.keys()})
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    variant_colors = dict(zip(variants, colors))

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
            ax.plot(gap_labels, accs, "o-", label=variant, color=variant_colors[variant], linewidth=2)

        ax.set_title(f"{task}", fontsize=14)
        ax.set_xlabel("Gap Level")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=8)
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
    """Plot training convergence curves for all runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by variant
    variant_data: dict[str, list] = {}
    for key, data in results.items():
        if "error" in data or "history" not in data:
            continue
        variant = key.rsplit("_seed", 1)[0].rsplit("_", 1)[0]
        history = data["history"]
        if metric in history:
            variant_data.setdefault(variant, []).append(history[metric])

    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_data)))
    for (variant, curves), color in zip(sorted(variant_data.items()), colors):
        # Plot mean with std band
        min_len = min(len(c) for c in curves)
        aligned = np.array([c[:min_len] for c in curves])
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)
        epochs = np.arange(1, min_len + 1)
        ax.plot(epochs, mean, label=variant, color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
