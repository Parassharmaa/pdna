#!/usr/bin/env python
"""PDNA Final Experiments — tasks designed to reveal pulse mechanism benefits.

Three tasks that test different aspects of temporal processing:
1. Periodic Signal Classification (seq_periodic): Can the model detect periodic patterns?
   - Binary classification of oscillating signals with noise
   - Pulse mechanism should naturally encode periodic structure
2. Adding Problem (adding): Classic RNN long-range dependency benchmark
   - Sum two marked values in a long sequence
   - Tests ability to maintain information over time
3. Temporal Order (temporal_order): Remember order of events across gaps
   - 3-class: which of 3 marker patterns appeared first?
   - Gaps disrupt temporal context — pulse should help maintain timing

6 variants × 3 tasks × 3 seeds = 54 runs
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from pdna.data.gapped import GapLevel, GappedWrapper
from pdna.models.variants import Variant, build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer

VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
SEEDS = [42, 123, 456]


# ==================== Task 1: Periodic Signal Classification ====================

class PeriodicSignalDataset(Dataset):
    """Classify signals by their dominant frequency band.

    Each sample is a multi-channel signal with a primary oscillation.
    Label = which frequency band the signal belongs to (binary: low vs high freq).

    The pulse mechanism should have a natural advantage here since it
    generates oscillatory dynamics matching the input structure.
    """

    def __init__(self, n_samples, seq_len=200, n_features=16, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            t = np.linspace(0, 8 * np.pi, seq_len)
            label = rng.randint(0, 2)

            if label == 0:
                # Low frequency: 0.3-1.0 Hz
                freq = rng.uniform(0.3, 1.0)
            else:
                # High frequency: 2.0-5.0 Hz
                freq = rng.uniform(2.0, 5.0)

            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(0.5, 1.5)

            # Base signal
            signal = amplitude * np.sin(2 * np.pi * freq * t / seq_len + phase)

            # Multi-channel: signal + correlated noise channels
            features = np.zeros((seq_len, n_features), dtype=np.float32)
            features[:, 0] = signal + rng.randn(seq_len) * 0.3
            for i in range(1, n_features):
                delay = rng.randint(0, 5)
                features[:, i] = np.roll(signal, delay) * rng.uniform(0.3, 1.0) + rng.randn(seq_len) * 0.5

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== Task 2: Adding Problem ====================

class AddingProblemDataset(Dataset):
    """Classic adding problem for RNNs.

    Input: (seq_len, 2) where channel 0 = random values [0,1],
    channel 1 = binary indicator (1 at exactly two positions).
    Target: sum of the two indicated values.
    Converted to binary classification: sum > 1.0 or not.
    """

    def __init__(self, n_samples, seq_len=200, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            values = rng.uniform(0, 1, seq_len).astype(np.float32)
            indicator = np.zeros(seq_len, dtype=np.float32)

            # Place two markers at random positions (not too close)
            pos1 = rng.randint(0, seq_len // 2)
            pos2 = rng.randint(seq_len // 2, seq_len)
            indicator[pos1] = 1.0
            indicator[pos2] = 1.0

            features = np.stack([values, indicator], axis=1)  # (seq_len, 2)
            target_sum = values[pos1] + values[pos2]
            label = 1 if target_sum > 1.0 else 0

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== Task 3: Temporal Order ====================

class TemporalOrderDataset(Dataset):
    """Remember the order of marker events across a sequence.

    Two distinct markers (A and B) are embedded in noise.
    Task: classify which appeared first (A-then-B vs B-then-A vs same-position).
    This tests temporal memory, which gaps should disrupt.
    """

    def __init__(self, n_samples, seq_len=200, n_features=8, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            features = rng.randn(seq_len, n_features).astype(np.float32) * 0.3

            # Place marker A and B at random positions
            pos1 = rng.randint(seq_len // 4, seq_len // 2)
            pos2 = rng.randint(seq_len // 2, 3 * seq_len // 4)

            # Randomly decide order
            label = rng.randint(0, 2)
            if label == 0:
                # A first, then B
                features[pos1, :n_features // 2] += 2.0  # Marker A
                features[pos2, n_features // 2:] += 2.0   # Marker B
            else:
                # B first, then A
                features[pos1, n_features // 2:] += 2.0   # Marker B
                features[pos2, :n_features // 2] += 2.0    # Marker A

            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== GappedWrapper for float tensors ====================

class FloatGappedWrapper(Dataset):
    """Wraps float-tensor datasets with gap masks."""

    def __init__(self, base_dataset, gap_level):
        from pdna.data.gapped import GapLevel, create_gap_mask
        self.base = base_dataset
        self.gap_level = GapLevel(gap_level) if isinstance(gap_level, str) else gap_level
        self.create_gap_mask = create_gap_mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, label = self.base[idx]
        seq_len = x.shape[0]
        gap_mask = self.create_gap_mask(seq_len, self.gap_level, batch_size=1, seed=idx).squeeze(0)
        x = x.clone()
        x[gap_mask] = 0
        return x, label, gap_mask


# ==================== Task config ====================

TASKS = {
    "seq_periodic": {
        "dataset_cls": PeriodicSignalDataset,
        "input_size": 16, "output_size": 2,
        "seq_len": 200, "n_features": 16,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
        "use_embedding": False,
    },
    "adding": {
        "dataset_cls": AddingProblemDataset,
        "input_size": 2, "output_size": 2,
        "seq_len": 200,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
        "use_embedding": False,
    },
    "temporal_order": {
        "dataset_cls": TemporalOrderDataset,
        "input_size": 8, "output_size": 2,
        "seq_len": 200, "n_features": 8,
        "train_size": 8000, "val_size": 1000, "test_size": 1000,
        "use_embedding": False,
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
    cfg.model.hidden_size = 128
    cfg.training.max_epochs = 40
    cfg.training.batch_size = 128
    cfg.training.warmup_epochs = 3
    cfg.training.early_stopping_patience = 10
    cfg.training.lr = 5e-4

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
