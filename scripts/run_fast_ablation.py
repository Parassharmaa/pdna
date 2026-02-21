#!/usr/bin/env python
"""PDNA Fast Ablation — streamlined experiment for quick results.

Baseline already validated (sMNIST 97.98%, pMNIST 96%).
Skips validation, goes straight to full ablation.

6 variants × 2 tasks × 2 seeds = 24 runs (~2-3 hours on RTX A4000)
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from pdna.data.gapped import GapLevel, create_gap_mask
from pdna.models.variants import build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer

VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
SEEDS = [42, 123]


# ==================== Datasets ====================

class RowMNIST(Dataset):
    """Row-by-row MNIST: 28 rows × 28 pixels = 28 timesteps."""

    def __init__(self, train=True, data_dir="/root/data", permute=False, perm_seed=42):
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        if permute:
            rng = np.random.RandomState(perm_seed)
            self.perm = torch.from_numpy(rng.permutation(784))
        else:
            self.perm = None

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        pixels = img.view(-1)  # (784,)
        if self.perm is not None:
            pixels = pixels[self.perm]
        return pixels.view(28, 28), label  # (28 rows, 28 features)


class AddingProblem(Dataset):
    """Sum two marked values — binary classification (sum > 1.0?)."""

    def __init__(self, n_samples=10000, seq_len=50, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []
        for _ in range(n_samples):
            values = rng.uniform(0, 1, seq_len).astype(np.float32)
            markers = np.zeros(seq_len, dtype=np.float32)
            pos1 = rng.randint(0, seq_len // 3)
            pos2 = rng.randint(seq_len // 2, seq_len - seq_len // 6)
            markers[pos1] = 1.0
            markers[pos2] = 1.0
            features = np.stack([values, markers], axis=1)
            label = 1 if values[pos1] + values[pos2] > 1.0 else 0
            self.data.append(torch.tensor(features))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FloatGappedWrapper(Dataset):
    def __init__(self, base_dataset, gap_level):
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


TASKS = {
    "smnist": {"input_size": 28, "output_size": 10, "seq_len": 28},
    "adding": {"input_size": 2, "output_size": 2, "seq_len": 50},
}


def get_data(task_name, batch_size, seed):
    if task_name == "smnist":
        train_ds = RowMNIST(train=True)
        test_ds = RowMNIST(train=False)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [54000, 6000],
            generator=torch.Generator().manual_seed(seed),
        )
    elif task_name == "adding":
        train_ds = AddingProblem(n_samples=20000, seq_len=50, seed=seed)
        val_ds = AddingProblem(n_samples=2000, seq_len=50, seed=seed + 1000)
        test_ds = AddingProblem(n_samples=2000, seq_len=50, seed=seed + 2000)
    else:
        raise ValueError(task_name)

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        test_ds,
    )


def gapped_eval(model, test_ds, device, batch_size=256):
    gap_results = {}
    for gl in GapLevel:
        gds = FloatGappedWrapper(test_ds, gl)
        loader = DataLoader(gds, batch_size=batch_size, num_workers=0)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, labels, mask in loader:
                x, labels, mask = x.to(device), labels.to(device), mask.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                logits, _ = model(x, gap_mask=mask)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        gap_results[gl.value] = {"accuracy": correct / total if total > 0 else 0}
    return gap_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    cfg = ExperimentConfig()
    cfg.model.hidden_size = 128
    cfg.training.max_epochs = 40          # Reduced from 80
    cfg.training.batch_size = 512
    cfg.training.warmup_epochs = 3
    cfg.training.early_stopping_patience = 8  # Reduced from 15
    cfg.training.lr = 5e-4
    cfg.training.grad_clip_norm = 1.0
    cfg.logging.backend = "none"

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    all_results = {}

    total = len(VARIANTS) * len(TASKS) * len(SEEDS)
    run_num = 0

    for task_name, tc in TASKS.items():
        for variant_name in VARIANTS:
            for seed in SEEDS:
                run_num += 1
                key = f"{variant_name}_{task_name}_seed{seed}"
                print(f"\n[{run_num}/{total}] {key}", flush=True)

                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    model = build_variant(
                        variant_name,
                        input_size=tc["input_size"],
                        hidden_size=cfg.model.hidden_size,
                        output_size=tc["output_size"],
                        num_layers=cfg.model.num_layers,
                        dropout=cfg.model.dropout,
                    )

                    train_ld, val_ld, test_ld, test_ds = get_data(task_name, cfg.training.batch_size, seed)

                    trainer = Trainer(
                        model=model, config=cfg,
                        train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                        device=device, run_name=key, output_dir=str(out_dir),
                        use_embedding=False, vocab_size=256, embed_dim=tc["input_size"],
                    )

                    t0 = time.time()
                    results = trainer.train()
                    elapsed = time.time() - t0

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
                    import traceback
                    traceback.print_exc()
                    all_results[key] = {"error": str(e)}

                # Save incrementally
                with open(out_dir / "all_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    # ==================== Summary ====================
    print("\n" + "=" * 70, flush=True)
    print("ABLATION RESULTS (mean +/- std across seeds)", flush=True)
    print("=" * 70, flush=True)

    for task_name in TASKS:
        print(f"\n--- {task_name} ---", flush=True)
        print(f"  {'Variant':<15} | {'Test Acc':>18} | {'Gap0':>8} {'Gap30':>8} | {'Degradation':>18}", flush=True)
        print(f"  {'-'*15}-+-{'-'*18}-+-{'-'*8}-{'-'*8}-+-{'-'*18}", flush=True)
        for v in VARIANTS:
            accs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
            degs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]
            g0s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_0", {}).get("accuracy", 0) for s in SEEDS]
            g30s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_30", {}).get("accuracy", 0) for s in SEEDS]
            print(
                f"  {v:15s} | {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
                f"| {np.mean(g0s):.4f} {np.mean(g30s):.4f} "
                f"| {np.mean(degs):.4f} +/- {np.std(degs):.4f}",
                flush=True,
            )

    # Key comparisons
    print("\n" + "=" * 70, flush=True)
    print("KEY COMPARISONS", flush=True)
    print("=" * 70, flush=True)
    for task_name in TASKS:
        baseline_accs = [all_results.get(f"baseline_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
        pdna_accs = [all_results.get(f"full_pdna_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
        noise_accs = [all_results.get(f"noise_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
        pulse_accs = [all_results.get(f"pulse_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
        baseline_deg = [all_results.get(f"baseline_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]
        pdna_deg = [all_results.get(f"full_pdna_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]

        print(f"\n{task_name}:", flush=True)
        print(f"  PDNA vs Baseline: {np.mean(pdna_accs):.4f} vs {np.mean(baseline_accs):.4f} (delta: {np.mean(pdna_accs)-np.mean(baseline_accs):+.4f})", flush=True)
        print(f"  Pulse vs Noise:   {np.mean(pulse_accs):.4f} vs {np.mean(noise_accs):.4f} (structured > random? {'YES' if np.mean(pulse_accs) > np.mean(noise_accs) else 'NO'})", flush=True)
        print(f"  Gap degradation:  PDNA={np.mean(pdna_deg):.4f} vs Baseline={np.mean(baseline_deg):.4f}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
