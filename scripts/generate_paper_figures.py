#!/usr/bin/env python
"""Generate publication-quality figures from v5 experiment results.

Produces:
1. degradation_curves.pdf — Accuracy vs gap level (the core result)
2. ablation_heatmap.pdf — Heatmap of accuracy across variants and tasks
3. frequency_spectrum.pdf — Learned omega distribution
4. state_norms.pdf — State norms during gaps vs non-gaps
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna"]
VARIANT_LABELS = {
    "baseline": "A. Baseline CfC",
    "noise": "B. CfC + Noise",
    "pulse": "C. CfC + Pulse",
    "self_attend": "D. CfC + SelfAttend",
    "full_pdna": "E. Full PDNA",
}
VARIANT_COLORS = {
    "baseline": "#1f77b4",
    "noise": "#ff7f0e",
    "pulse": "#2ca02c",
    "self_attend": "#9467bd",
    "full_pdna": "#d62728",
}
VARIANT_MARKERS = {
    "baseline": "o",
    "noise": "s",
    "pulse": "^",
    "self_attend": "D",
    "full_pdna": "*",
}

TASKS = ["smnist", "psmnist", "scifar10"]
TASK_LABELS = {"smnist": "sMNIST", "psmnist": "psMNIST", "scifar10": "sCIFAR-10"}
SEEDS = [42, 123, 456, 789, 1337]
GAP_LEVELS = ["gap_0", "gap_5", "gap_15", "gap_30", "multi_gap"]
GAP_LABELS = ["0%", "5%", "15%", "30%", "Multi"]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_gap_values(results, variant, task, gap_level):
    vals = []
    for s in SEEDS:
        key = f"{variant}_{task}_seed{s}"
        data = results.get(key, {})
        if "error" not in data and "gapped" in data:
            acc = data["gapped"].get(gap_level, {}).get("accuracy")
            if acc is not None:
                vals.append(acc)
    return vals


def get_values(results, variant, task, key):
    vals = []
    for s in SEEDS:
        run_key = f"{variant}_{task}_seed{s}"
        data = results.get(run_key, {})
        if "error" not in data and key in data:
            vals.append(data[key])
    return vals


def plot_degradation_curves(results, out_dir):
    """The core visualization: accuracy under increasing gap severity."""
    active_tasks = [t for t in TASKS if get_gap_values(results, "baseline", t, "gap_0")]

    if not active_tasks:
        print("No gap data available for degradation curves")
        return

    fig, axes = plt.subplots(1, len(active_tasks), figsize=(5.5 * len(active_tasks), 4.5))
    if len(active_tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, active_tasks):
        for v in VARIANTS:
            means = []
            stds = []
            for gl in GAP_LEVELS:
                vals = get_gap_values(results, v, task, gl)
                if vals:
                    means.append(np.mean(vals) * 100)
                    stds.append(np.std(vals) * 100)
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(range(len(GAP_LEVELS)), means,
                    marker=VARIANT_MARKERS[v], color=VARIANT_COLORS[v],
                    label=VARIANT_LABELS[v], linewidth=1.8, markersize=7)
            ax.fill_between(range(len(GAP_LEVELS)),
                           means - stds, means + stds,
                           alpha=0.15, color=VARIANT_COLORS[v])

        ax.set_xticks(range(len(GAP_LEVELS)))
        ax.set_xticklabels(GAP_LABELS)
        ax.set_xlabel("Gap Severity")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(TASK_LABELS[task])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "degradation_curves.pdf")
    fig.savefig(out_dir / "degradation_curves.png")
    plt.close(fig)
    print(f"  Saved degradation_curves.pdf/png")


def plot_degradation_bars(results, out_dir):
    """Bar chart comparing degradation across variants."""
    active_tasks = [t for t in TASKS if get_values(results, "baseline", t, "degradation")]

    if not active_tasks:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(active_tasks))
    width = 0.15
    offsets = np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2

    for i, v in enumerate(VARIANTS):
        means = []
        stds = []
        for t in active_tasks:
            vals = get_values(results, v, t, "degradation")
            if vals:
                means.append(np.mean(vals) * 100)
                stds.append(np.std(vals) * 100)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + offsets[i] * width, means, width, yerr=stds,
               label=VARIANT_LABELS[v], color=VARIANT_COLORS[v],
               alpha=0.85, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in active_tasks])
    ax.set_ylabel("Degradation (%, Gap 0% − Gap 30%)")
    ax.set_title("Gap Robustness: Lower = More Robust")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "degradation_bars.pdf")
    fig.savefig(out_dir / "degradation_bars.png")
    plt.close(fig)
    print(f"  Saved degradation_bars.pdf/png")


def plot_ablation_heatmap(results, out_dir):
    """Heatmap showing accuracy across variants and tasks."""
    active_tasks = [t for t in TASKS if get_values(results, "baseline", t, "test_acc")]

    if not active_tasks:
        return

    data = np.zeros((len(VARIANTS), len(active_tasks)))
    for i, v in enumerate(VARIANTS):
        for j, t in enumerate(active_tasks):
            vals = get_values(results, v, t, "test_acc")
            data[i, j] = np.mean(vals) * 100 if vals else 0

    fig, ax = plt.subplots(figsize=(3.5 + 1.2 * len(active_tasks), 4))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(active_tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in active_tasks])
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS])

    for i in range(len(VARIANTS)):
        for j in range(len(active_tasks)):
            text_color = "white" if data[i, j] > (data.max() + data.min()) / 2 else "black"
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                   color=text_color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Test Accuracy (%)")
    ax.set_title("Ablation Study: Test Accuracy")
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_heatmap.pdf")
    fig.savefig(out_dir / "ablation_heatmap.png")
    plt.close(fig)
    print(f"  Saved ablation_heatmap.pdf/png")


def plot_frequency_spectrum(results, out_dir):
    """Plot learned omega (frequency) distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, v in zip(axes, ["pulse", "full_pdna"]):
        all_omegas = []
        for t in TASKS:
            for s in SEEDS:
                key = f"{v}_{t}_seed{s}"
                data = results.get(key, {})
                pp = data.get("pulse_params", {})
                if "omega" in pp:
                    all_omegas.extend(pp["omega"])

        if all_omegas:
            ax.hist(all_omegas, bins=40, alpha=0.7, color=VARIANT_COLORS[v], edgecolor="black", linewidth=0.5)
            ax.axvline(np.mean(all_omegas), color="red", linestyle="--", label=f"Mean={np.mean(all_omegas):.2f}")
            ax.set_xlabel("Learned $\\omega$ (frequency)")
            ax.set_ylabel("Count")
            ax.set_title(f"{VARIANT_LABELS[v]}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(f"{VARIANT_LABELS[v]}")

    plt.suptitle("Learned Oscillation Frequencies", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "frequency_spectrum.pdf")
    fig.savefig(out_dir / "frequency_spectrum.png")
    plt.close(fig)
    print(f"  Saved frequency_spectrum.pdf/png")


def plot_state_norms(results, out_dir):
    """Plot state norms during gap vs non-gap positions."""
    active_tasks = [t for t in TASKS if get_values(results, "baseline", t, "test_acc")]

    if not active_tasks:
        return

    fig, axes = plt.subplots(1, len(active_tasks), figsize=(5 * len(active_tasks), 4))
    if len(active_tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, active_tasks):
        gap_norms = []
        nongap_norms = []
        labels = []

        for v in VARIANTS:
            gnorms = []
            ngnorms = []
            for s in SEEDS:
                key = f"{v}_{task}_seed{s}"
                data = results.get(key, {})
                sn = data.get("state_norms", {})
                if sn.get("gap_norm_mean", 0) > 0:
                    gnorms.append(sn["gap_norm_mean"])
                    ngnorms.append(sn["nongap_norm_mean"])

            if gnorms:
                gap_norms.append(np.mean(gnorms))
                nongap_norms.append(np.mean(ngnorms))
                labels.append(VARIANT_LABELS[v].split(". ")[1] if ". " in VARIANT_LABELS[v] else VARIANT_LABELS[v])

        if gap_norms:
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, nongap_norms, width, label="Non-gap", color="#2ca02c", alpha=0.7)
            ax.bar(x + width/2, gap_norms, width, label="Gap", color="#d62728", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Mean Hidden State Norm")
            ax.set_title(f"State Norms: {TASK_LABELS[task]}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "state_norms.pdf")
    fig.savefig(out_dir / "state_norms.png")
    plt.close(fig)
    print(f"  Saved state_norms.pdf/png")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "runs_v5/all_results.json"
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    valid = sum(1 for v in results.values() if "error" not in v)
    print(f"Loaded {valid}/{len(results)} valid runs")

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures:")
    plot_degradation_curves(results, out_dir)
    plot_degradation_bars(results, out_dir)
    plot_ablation_heatmap(results, out_dir)
    plot_frequency_spectrum(results, out_dir)
    plot_state_norms(results, out_dir)

    print("\nDone! Figures in paper/figures/")


if __name__ == "__main__":
    main()
