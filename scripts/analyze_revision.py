#!/usr/bin/env python
"""Generate figures and statistics for paper revision from v6 experiment results.

Produces 3 new figures:
1. alpha_A_product.png — α·||A|| over training epochs
2. phase_magnitude.png — |W_φ·h|/ω vs timestep t
3. nyquist_analysis.png — ω histogram with Nyquist line + clamping impact
"""

import json
import math
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

VARIANTS = ["pulse", "full_pdna"]
VARIANT_LABELS = {
    "pulse": "C. CfC + Pulse",
    "full_pdna": "E. Full PDNA",
}
VARIANT_COLORS = {
    "pulse": "#2ca02c",
    "full_pdna": "#d62728",
}
SEEDS = [42, 123, 456, 789, 1337]


def load_results(path="runs_v6/all_results.json"):
    with open(path) as f:
        return json.load(f)


def plot_alpha_A_product(results, out_dir):
    """Figure 1: α·||A|| over training epochs, with α and ||A|| individually."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for v in VARIANTS:
        # Collect per-epoch histories across seeds
        histories = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            h = data.get("pulse_param_history", [])
            if h:
                histories.append(h)

        if not histories:
            continue

        # Align by epoch — find max epoch across seeds
        max_epochs = max(len(h) for h in histories)

        for ax_idx, (metric, label, ylabel) in enumerate([
            ("alpha_times_A_norm", r"$\alpha \cdot \|A\|_2$", "Effective pulse magnitude"),
            ("alpha", r"$\alpha$", "Pulse gate"),
            ("A_norm", r"$\|A\|_2$", "Amplitude norm"),
        ]):
            ax = axes[ax_idx]
            # Get values per seed, per epoch
            all_values = []
            for h in histories:
                vals = [snap.get(metric, 0) for snap in h]
                all_values.append(vals)

            # Pad shorter histories with last value
            for vals in all_values:
                while len(vals) < max_epochs:
                    vals.append(vals[-1])

            arr = np.array(all_values)  # (n_seeds, n_epochs)
            epochs = np.arange(max_epochs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)

            color = VARIANT_COLORS[v]
            ax.plot(epochs, mean, color=color, label=VARIANT_LABELS[v], linewidth=1.5)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "alpha_A_product.png")
    fig.savefig(out_dir / "alpha_A_product.pdf")
    plt.close(fig)
    print(f"  Saved alpha_A_product.{{png,pdf}}")

    # Print statistics for paper
    for v in VARIANTS:
        histories = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            h = data.get("pulse_param_history", [])
            if h:
                histories.append(h)
        if histories:
            initial_aA = [h[0].get("alpha_times_A_norm", 0) for h in histories]
            final_aA = [h[-1].get("alpha_times_A_norm", 0) for h in histories]
            initial_alpha = [h[0].get("alpha", 0) for h in histories]
            final_alpha = [h[-1].get("alpha", 0) for h in histories]
            initial_A = [h[0].get("A_norm", 0) for h in histories]
            final_A = [h[-1].get("A_norm", 0) for h in histories]
            print(f"\n  {VARIANT_LABELS[v]}:")
            print(f"    α·||A||: {np.mean(initial_aA):.4f} → {np.mean(final_aA):.4f} ± {np.std(final_aA):.4f} ({np.mean(final_aA)/np.mean(initial_aA):.0f}x)")
            print(f"    α:       {np.mean(initial_alpha):.4f} → {np.mean(final_alpha):.4f} ± {np.std(final_alpha):.4f}")
            print(f"    ||A||:   {np.mean(initial_A):.4f} → {np.mean(final_A):.4f} ± {np.std(final_A):.4f}")


def plot_phase_magnitude(results, out_dir):
    """Figure 2: |W_φ·h|/ω vs timestep t."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, v in enumerate(VARIANTS):
        ax = axes[ax_idx]

        # Collect phase analysis across seeds
        all_ratio_means = []
        all_phase_mags = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            pma = data.get("phase_magnitude_analysis", {})
            per_t = pma.get("per_timestep", [])
            if per_t:
                all_ratio_means.append([r['ratio_mean'] for r in per_t])
                all_phase_mags.append([r['phase_mag_mean'] for r in per_t])

        if not all_ratio_means:
            continue

        T = len(all_ratio_means[0])
        timesteps = np.arange(T)

        ratio_arr = np.array(all_ratio_means)  # (n_seeds, T)
        ratio_mean = ratio_arr.mean(axis=0)
        ratio_std = ratio_arr.std(axis=0)

        color = VARIANT_COLORS[v]

        # Plot |W_phi * h + b| / |omega| vs t
        ax.plot(timesteps, ratio_mean, color=color, linewidth=1.5, label=r"$|\varphi(h)| / |\omega|$")
        ax.fill_between(timesteps, ratio_mean - ratio_std, ratio_mean + ratio_std, alpha=0.2, color=color)

        # Plot t for reference
        ax.plot(timesteps, timesteps, '--', color='gray', linewidth=1, alpha=0.7, label="$t$ (reference)")

        ax.set_xlabel("Timestep $t$")
        ax.set_ylabel(r"$|\varphi(h)| / |\omega|$")
        ax.set_title(VARIANT_LABELS[v])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary annotation
        summary = results.get(f"{v}_smnist_seed42", {}).get("phase_magnitude_analysis", {}).get("summary", {})
        if summary:
            mid_ratio = summary.get("ratio_at_midpoint", 0)
            mid_t = summary.get("t_at_midpoint", 14)
            ax.annotate(
                f"At $t={mid_t}$: ratio={mid_ratio:.2f}\n(vs $t={mid_t}$)",
                xy=(mid_t, ratio_mean[mid_t] if mid_t < len(ratio_mean) else 0),
                xytext=(mid_t + 3, ratio_mean.max() * 0.7),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    plt.tight_layout()
    fig.savefig(out_dir / "phase_magnitude.png")
    fig.savefig(out_dir / "phase_magnitude.pdf")
    plt.close(fig)
    print(f"  Saved phase_magnitude.{{png,pdf}}")

    # Print statistics
    for v in VARIANTS:
        summaries = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            pma = data.get("phase_magnitude_analysis", {})
            summary = pma.get("summary", {})
            if summary:
                summaries.append(summary)
        if summaries:
            mid_ratios = [s["ratio_at_midpoint"] for s in summaries]
            mean_ratios = [s["mean_ratio_across_time"] for s in summaries]
            print(f"\n  {VARIANT_LABELS[v]} phase analysis:")
            print(f"    Mean |φ(h)|/|ω| across time: {np.mean(mean_ratios):.3f} ± {np.std(mean_ratios):.3f}")
            print(f"    At midpoint (t=14): {np.mean(mid_ratios):.3f} ± {np.std(mid_ratios):.3f}")
            frac_dom = [s.get("frac_timesteps_phase_dominates", 0) for s in summaries]
            print(f"    Fraction of timesteps where phase > t: {np.mean(frac_dom):.1%}")


def plot_nyquist_analysis(results, out_dir):
    """Figure 3: ω histogram with Nyquist line + clamping impact."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nyquist = math.pi

    # Left: Frequency histogram
    ax = axes[0]
    for v in VARIANTS:
        all_omega = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            pp = data.get("pulse_params", {})
            if "omega" in pp:
                all_omega.extend(pp["omega"])

        if not all_omega:
            continue

        omega = np.array(all_omega)
        color = VARIANT_COLORS[v]
        ax.hist(omega, bins=50, alpha=0.5, color=color, label=VARIANT_LABELS[v], density=True)

        # Stats
        above = (np.abs(omega) > nyquist).mean()
        median = np.median(omega)
        q25, q75 = np.percentile(omega, [25, 75])
        print(f"\n  {VARIANT_LABELS[v]} frequency stats:")
        print(f"    Mean: {omega.mean():.3f}, Median: {median:.3f}, IQR: [{q25:.3f}, {q75:.3f}]")
        print(f"    Above Nyquist (|ω| > π): {above:.1%} ({int(above * len(omega))}/{len(omega)})")

    ax.axvline(x=nyquist, color='red', linestyle='--', linewidth=1.5, label=f"Nyquist ($\\omega = \\pi$)")
    ax.axvline(x=-nyquist, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel(r"Learned frequency $\omega$")
    ax.set_ylabel("Density")
    ax.set_title("Learned Frequency Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Clamping impact bar chart
    ax = axes[1]
    bar_data = {}
    for v in VARIANTS:
        orig_mg = []
        clamp_mg = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            nq = data.get("nyquist_analysis", {})
            if nq:
                orig_mg.append(nq.get("original_multigap", 0))
                clamp_mg.append(nq.get("clamped_multigap", 0))
        if orig_mg:
            bar_data[v] = {
                "original": (np.mean(orig_mg), np.std(orig_mg)),
                "clamped": (np.mean(clamp_mg), np.std(clamp_mg)),
            }

    if bar_data:
        x = np.arange(len(bar_data))
        width = 0.35
        for i, (v, data) in enumerate(bar_data.items()):
            color = VARIANT_COLORS[v]
            ax.bar(i - width/2, data["original"][0] * 100, width,
                   yerr=data["original"][1] * 100, label="Original" if i == 0 else "",
                   color=color, alpha=0.8, capsize=3)
            ax.bar(i + width/2, data["clamped"][0] * 100, width,
                   yerr=data["clamped"][1] * 100, label="Nyquist-clamped" if i == 0 else "",
                   color=color, alpha=0.4, hatch="//", capsize=3)

            # Delta annotation
            delta = (data["clamped"][0] - data["original"][0]) * 100
            ax.annotate(
                f"$\\Delta$={delta:+.2f}pp",
                xy=(i, max(data["original"][0], data["clamped"][0]) * 100 + data["original"][1] * 100 + 0.5),
                ha='center', fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in bar_data.keys()], fontsize=9)
        ax.set_ylabel("Multi-Gap Accuracy (%)")
        ax.set_title("Nyquist Clamping Impact")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(out_dir / "nyquist_analysis.png")
    fig.savefig(out_dir / "nyquist_analysis.pdf")
    plt.close(fig)
    print(f"  Saved nyquist_analysis.{{png,pdf}}")

    # Print clamping statistics
    for v in VARIANTS:
        deltas = []
        for s in SEEDS:
            key = f"{v}_smnist_seed{s}"
            data = results.get(key, {})
            nq = data.get("nyquist_analysis", {})
            if nq:
                deltas.append(nq.get("multigap_delta", 0))
        if deltas:
            print(f"\n  {VARIANT_LABELS[v]} Nyquist clamping:")
            print(f"    Multi-gap delta: {np.mean(deltas)*100:+.2f} ± {np.std(deltas)*100:.2f} pp")


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs_v6/all_results.json")
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating revision figures...")
    plot_alpha_A_product(results, out_dir)
    plot_phase_magnitude(results, out_dir)
    plot_nyquist_analysis(results, out_dir)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
