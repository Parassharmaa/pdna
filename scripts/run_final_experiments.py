#!/usr/bin/env python
"""PDNA Final Experiments — tasks designed to reveal pulse mechanism benefits.

Three tasks with CLEAR signal for the model to learn:
1. Frequency Classification (freq_class): Classify multi-cycle oscillations
   - Low freq (1-3 cycles) vs high freq (8-15 cycles)
   - Pulse mechanism should naturally encode periodic structure
2. Gap Memory (gap_memory): Remember a pattern shown before a mandatory gap
   - Pattern → gap → classify based on remembered pattern
   - Tests state preservation during input interruptions (core PDNA hypothesis)
3. Temporal Order (temporal_order): Detect which of two events came first
   - Tests temporal awareness, which pulse timing should support

6 variants × 3 tasks × 3 seeds = 54 runs
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from pdna.data.gapped import GapLevel, create_gap_mask
from pdna.models.variants import Variant, build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer

VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
SEEDS = [42, 123, 456]


# ==================== Task 1: Frequency Classification ====================

class FreqClassDataset(Dataset):
    """Classify signals by frequency: low (1-3 cycles) vs high (8-15 cycles).

    The signal clearly oscillates multiple times so the model can detect frequency.
    """

    def __init__(self, n_samples, seq_len=128, n_features=4, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            t = np.linspace(0, 1, seq_len)
            label = rng.randint(0, 2)

            if label == 0:
                # Low frequency: 1-3 full cycles
                n_cycles = rng.uniform(1.0, 3.0)
            else:
                # High frequency: 8-15 full cycles
                n_cycles = rng.uniform(8.0, 15.0)

            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(0.5, 1.5)
            signal = amplitude * np.sin(2 * np.pi * n_cycles * t + phase)

            features = np.zeros((seq_len, n_features), dtype=np.float32)
            features[:, 0] = signal + rng.randn(seq_len) * 0.2
            for i in range(1, n_features):
                features[:, i] = signal * rng.uniform(0.3, 1.0) + rng.randn(seq_len) * 0.3

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== Task 2: Gap Memory ====================

class GapMemoryDataset(Dataset):
    """Remember a pattern shown in the first quarter, classify after a mandatory gap.

    Structure: [pattern region | mandatory gap (zeros) | distractor noise]
    Pattern types: 'rising' (label=0) or 'falling' (label=1)

    This DIRECTLY tests the PDNA hypothesis: models that can maintain state
    during the gap should classify better. The pulse provides internal activity
    during the gap that helps preserve the memory of the pattern.
    """

    def __init__(self, n_samples, seq_len=128, n_features=4, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        pattern_len = seq_len // 4
        gap_len = seq_len // 4
        distractor_len = seq_len - pattern_len - gap_len

        for _ in range(n_samples):
            features = np.zeros((seq_len, n_features), dtype=np.float32)
            label = rng.randint(0, 2)

            # Pattern region: rising or falling ramp with noise
            if label == 0:
                pattern = np.linspace(0, 2, pattern_len) + rng.randn(pattern_len) * 0.2
            else:
                pattern = np.linspace(2, 0, pattern_len) + rng.randn(pattern_len) * 0.2

            features[:pattern_len, 0] = pattern
            for i in range(1, n_features):
                features[:pattern_len, i] = pattern * rng.uniform(0.5, 1.0) + rng.randn(pattern_len) * 0.3

            # Gap region: zeros (already initialized to zeros)
            # Distractor: random noise
            start = pattern_len + gap_len
            features[start:, :] = rng.randn(distractor_len, n_features).astype(np.float32) * 0.5

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== Task 3: Temporal Order ====================

class TemporalOrderDataset(Dataset):
    """Detect which of two marker events came first.

    Two distinct markers are embedded in noise at random positions.
    Task: classify which appeared first (A-then-B = 0, B-then-A = 1).
    """

    def __init__(self, n_samples, seq_len=128, n_features=4, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            features = rng.randn(seq_len, n_features).astype(np.float32) * 0.2

            # Two markers at random non-overlapping positions
            pos1 = rng.randint(seq_len // 6, seq_len // 3)
            pos2 = rng.randint(2 * seq_len // 3, 5 * seq_len // 6)

            label = rng.randint(0, 2)
            if label == 0:
                # A first (positive spike in ch0), then B (positive spike in ch1)
                features[pos1, 0] = 3.0
                features[pos1, 1] = -1.0
                features[pos2, 0] = -1.0
                features[pos2, 1] = 3.0
            else:
                # B first, then A
                features[pos1, 0] = -1.0
                features[pos1, 1] = 3.0
                features[pos2, 0] = 3.0
                features[pos2, 1] = -1.0

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== Gapped Wrapper for float data ====================

class FloatGappedWrapper(Dataset):
    """Wraps float-tensor datasets with gap masks for gapped evaluation."""

    def __init__(self, base_dataset, gap_level):
        from pdna.data.gapped import GapLevel
        self.base = base_dataset
        self.gap_level = GapLevel(gap_level) if isinstance(gap_level, str) else gap_level

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, label = self.base[idx]
        seq_len = x.shape[0]
        gap_mask = create_gap_mask(seq_len, self.gap_level, batch_size=1, seed=idx).squeeze(0)
        x = x.clone()
        x[gap_mask] = 0
        return x, label, gap_mask


# ==================== Task config ====================

TASKS = {
    "freq_class": {
        "dataset_cls": FreqClassDataset,
        "input_size": 4, "output_size": 2,
        "seq_len": 128, "n_features": 4,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
    },
    "gap_memory": {
        "dataset_cls": GapMemoryDataset,
        "input_size": 4, "output_size": 2,
        "seq_len": 128, "n_features": 4,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
    },
    "temporal_order": {
        "dataset_cls": TemporalOrderDataset,
        "input_size": 4, "output_size": 2,
        "seq_len": 128, "n_features": 4,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
    },
}


def get_data(task_name, cfg, seed):
    tc = TASKS[task_name]
    cls = tc["dataset_cls"]
    bs = cfg.training.batch_size

    kwargs = {"n_samples": tc["train_size"], "seq_len": tc["seq_len"], "seed": seed}
    if "n_features" in tc:
        kwargs["n_features"] = tc["n_features"]

    train_ds = cls(**kwargs)
    kwargs["n_samples"] = tc["val_size"]
    kwargs["seed"] = seed + 1000
    val_ds = cls(**kwargs)
    kwargs["n_samples"] = tc["test_size"]
    kwargs["seed"] = seed + 2000
    test_ds = cls(**kwargs)

    kw = dict(batch_size=bs, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        test_ds,
    )


def gapped_eval(model, test_ds, device, bs=64):
    gap_results = {}
    for gl in GapLevel:
        gds = FloatGappedWrapper(test_ds, gl)
        loader = DataLoader(gds, batch_size=bs, num_workers=0)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, labels, mask in loader:
                x, labels, mask = x.to(device), labels.to(device), mask.to(device)
                logits, _ = model(x, gap_mask=mask)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        gap_results[gl.value] = {"accuracy": correct / total}
    return gap_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    cfg = ExperimentConfig()
    cfg.model.hidden_size = 64
    cfg.training.max_epochs = 30
    cfg.training.batch_size = 128
    cfg.training.warmup_epochs = 2
    cfg.training.early_stopping_patience = 8
    cfg.training.lr = 1e-3
    cfg.logging.backend = "none"  # Disable TensorBoard for portability

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    all_results = {}

    total = len(VARIANTS) * len(TASKS) * len(SEEDS)
    run_num = 0

    for task_name in TASKS:
        tc = TASKS[task_name]
        for variant_name in VARIANTS:
            for seed in SEEDS:
                run_num += 1
                key = f"{variant_name}_{task_name}_seed{seed}"
                print(f"\n[{run_num}/{total}] {key}", flush=True)

                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = build_variant(
                        variant_name,
                        input_size=tc["input_size"],
                        hidden_size=cfg.model.hidden_size,
                        output_size=tc["output_size"],
                        num_layers=cfg.model.num_layers,
                        dropout=cfg.model.dropout,
                    )

                    train_ld, val_ld, test_ld, test_ds = get_data(task_name, cfg, seed)

                    trainer = Trainer(
                        model=model, config=cfg,
                        train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                        device=device, run_name=key, output_dir=str(out_dir),
                        vocab_size=256, embed_dim=tc["input_size"],
                        use_embedding=False,
                    )

                    t0 = time.time()
                    results = trainer.train()
                    elapsed = time.time() - t0

                    # Gapped eval
                    gap_results = gapped_eval(trainer.model, test_ds, device)
                    results["gapped"] = gap_results
                    results["degradation"] = gap_results["gap_0"]["accuracy"] - gap_results["gap_30"]["accuracy"]
                    results["params"] = sum(p.numel() for p in model.parameters())
                    results["wall_time"] = elapsed

                    all_results[key] = results
                    print(
                        f"  Test={results['test_acc']:.4f} "
                        f"Gap0={gap_results['gap_0']['accuracy']:.4f} "
                        f"Gap30={gap_results['gap_30']['accuracy']:.4f} "
                        f"Deg={results['degradation']:.4f} "
                        f"{elapsed:.0f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    all_results[key] = {"error": str(e)}

                # Save incrementally
                with open(out_dir / "all_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("ABLATION TABLE (mean +/- std across 3 seeds)", flush=True)
    print("=" * 70, flush=True)
    for task_name in TASKS:
        print(f"\n--- {task_name} ---", flush=True)
        for v in VARIANTS:
            accs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
            degs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]
            g0s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_0", {}).get("accuracy", 0) for s in SEEDS]
            g30s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_30", {}).get("accuracy", 0) for s in SEEDS]
            print(
                f"  {v:15s} | Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} "
                f"| Gap0: {np.mean(g0s):.4f} Gap30: {np.mean(g30s):.4f} "
                f"| Deg: {np.mean(degs):.4f}+/-{np.std(degs):.4f}",
                flush=True,
            )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
