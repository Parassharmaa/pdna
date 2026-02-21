"""Tests for the analysis and report generation pipeline."""

import json
import numpy as np
import pytest
from pathlib import Path

from pdna.analysis.results import (
    _parse_run_key,
    compute_ablation_table,
    compute_degradation_table,
    compute_statistical_tests,
    compute_degradation_stats,
    compute_overhead_table,
)


def _make_mock_results():
    """Create mock experiment results matching the proper experiment output format."""
    variants = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
    tasks = ["smnist", "adding"]
    seeds = [42, 123, 456]
    rng = np.random.RandomState(0)

    results = {}
    for v in variants:
        for t in tasks:
            for s in seeds:
                key = f"{v}_{t}_seed{s}"
                base_acc = 0.95 if t == "smnist" else 0.88

                # Add small variant-dependent offsets
                if v == "full_pdna":
                    offset = 0.02
                elif v == "pulse":
                    offset = 0.01
                elif v == "noise":
                    offset = 0.005
                else:
                    offset = 0.0

                test_acc = base_acc + offset + rng.normal(0, 0.005)

                # Gap results with variant-dependent degradation
                gap_0_acc = test_acc
                if v in ("full_pdna", "full_idle", "pulse"):
                    deg = 0.02 + rng.normal(0, 0.005)
                else:
                    deg = 0.05 + rng.normal(0, 0.005)

                results[key] = {
                    "test_acc": test_acc,
                    "test_loss": 0.1,
                    "best_val_acc": test_acc + 0.005,
                    "final_epoch": 30 + rng.randint(0, 20),
                    "params": 87434 if v == "baseline" else 88000 + rng.randint(0, 500),
                    "wall_time": 300 + rng.randint(0, 100),
                    "degradation": deg,
                    "gapped": {
                        "gap_0": {"accuracy": gap_0_acc},
                        "gap_5": {"accuracy": gap_0_acc - deg * 0.2},
                        "gap_15": {"accuracy": gap_0_acc - deg * 0.5},
                        "gap_30": {"accuracy": gap_0_acc - deg},
                        "multi_gap": {"accuracy": gap_0_acc - deg * 1.2},
                    },
                    "history": {
                        "train_loss": [0.5 - i * 0.01 for i in range(30)],
                        "train_acc": [0.5 + i * 0.015 for i in range(30)],
                        "val_loss": [0.6 - i * 0.01 for i in range(30)],
                        "val_acc": [0.5 + i * 0.015 for i in range(30)],
                        "lr": [5e-4] * 30,
                        "epoch_time": [8.5] * 30,
                    },
                }

    return results


class TestParseRunKey:
    def test_baseline_smnist(self):
        v, t, s = _parse_run_key("baseline_smnist_seed42")
        assert v == "baseline"
        assert t == "smnist"
        assert s == 42

    def test_full_pdna_adding(self):
        v, t, s = _parse_run_key("full_pdna_adding_seed123")
        assert v == "full_pdna"
        assert t == "adding"
        assert s == 123

    def test_self_attend(self):
        v, t, s = _parse_run_key("self_attend_pmnist_seed456")
        assert v == "self_attend"
        assert t == "pmnist"
        assert s == 456

    def test_full_idle(self):
        v, t, s = _parse_run_key("full_idle_smnist_seed42")
        assert v == "full_idle"
        assert t == "smnist"
        assert s == 42


class TestComputeAblationTable:
    def test_has_all_variant_task_pairs(self):
        results = _make_mock_results()
        table = compute_ablation_table(results)
        assert ("baseline", "smnist") in table
        assert ("full_pdna", "adding") in table

    def test_mean_and_std(self):
        results = _make_mock_results()
        table = compute_ablation_table(results)
        entry = table[("baseline", "smnist")]
        assert "mean" in entry
        assert "std" in entry
        assert len(entry["values"]) == 3  # 3 seeds

    def test_accuracy_range(self):
        results = _make_mock_results()
        table = compute_ablation_table(results)
        for (v, t), entry in table.items():
            assert 0.0 <= entry["mean"] <= 1.0


class TestComputeDegradationTable:
    def test_has_gap_accuracies(self):
        results = _make_mock_results()
        table = compute_degradation_table(results)
        entry = table[("baseline", "smnist")]
        assert "gap_accuracies" in entry
        assert "gap_0" in entry["gap_accuracies"]
        assert "gap_30" in entry["gap_accuracies"]

    def test_degradation_positive(self):
        results = _make_mock_results()
        table = compute_degradation_table(results)
        for (v, t), entry in table.items():
            # Degradation = gap_0 - gap_30, should be positive (gaps hurt)
            assert entry["mean_degradation"] > 0


class TestStatisticalTests:
    def test_returns_comparisons(self):
        results = _make_mock_results()
        stats = compute_statistical_tests(results)
        assert "smnist" in stats
        assert "PDNA vs Baseline" in stats["smnist"]

    def test_comparison_fields(self):
        results = _make_mock_results()
        stats = compute_statistical_tests(results)
        comp = stats["smnist"]["PDNA vs Baseline"]
        assert "p_value" in comp
        assert "cohens_d" in comp
        assert "ci_95_low" in comp
        assert "ci_95_high" in comp
        assert "significant_005" in comp
        assert 0.0 <= comp["p_value"] <= 1.0

    def test_pulse_vs_noise(self):
        results = _make_mock_results()
        stats = compute_statistical_tests(results)
        assert "Pulse vs Noise" in stats["smnist"]


class TestOverheadTable:
    def test_has_all_variants(self):
        results = _make_mock_results()
        table = compute_overhead_table(results)
        assert "baseline" in table
        assert "full_pdna" in table

    def test_overhead_ratio(self):
        results = _make_mock_results()
        table = compute_overhead_table(results)
        assert table["baseline"]["overhead_ratio"] == pytest.approx(1.0, abs=0.3)

    def test_param_count(self):
        results = _make_mock_results()
        table = compute_overhead_table(results)
        assert table["baseline"]["params"] > 0
