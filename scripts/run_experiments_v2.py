#!/usr/bin/env python
"""Run PDNA ablation experiments (v2) — improved data and hyperparameters.

Key improvements over v1:
- Larger training sets (5000+ samples) for better generalization
- hidden_size=128 (CfC is efficient enough)
- 30 epochs with early stopping
- 3 seeds for statistical robustness
- Better synthetic data with clearer signal
"""

import json
import sys
import time

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from pdna.data.gapped import GapLevel, GappedWrapper
from pdna.data.listops import ListOpsDataset
from pdna.data.pathfinder import PathfinderDataset, _generate_synthetic_pathfinder
from pdna.models.variants import Variant, build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer

VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna", "full_idle"]
SEEDS = [42, 123, 456]

TASKS = {
    "listops": {
        "input_size": 17, "output_size": 10, "vocab_size": 17,
        "max_len": 128, "train_size": 5000, "val_size": 500, "test_size": 500,
    },
    "pathfinder": {
        "input_size": 256, "output_size": 2, "vocab_size": 256,
        "max_len": 1024, "train_size": 5000, "val_size": 500, "test_size": 500,
    },
    "sequence_class": {
        "input_size": 16, "output_size": 2, "vocab_size": None,
        "max_len": 200, "train_size": 8000, "val_size": 1000, "test_size": 1000,
    },
}


class SequenceClassificationDataset(torch.utils.data.Dataset):
    """Synthetic sequence classification with clear signal.

    Task: classify whether the mean of a noisy signal exceeds a threshold.
    The signal has periodic structure that the pulse mechanism should exploit.
    Gaps in the signal make the task harder — models that maintain state
    during gaps should perform better.
    """

    def __init__(self, n_samples, seq_len=200, n_features=16, seed=42):
        rng = np.random.RandomState(seed)
        self.data = []
        self.labels = []

        for _ in range(n_samples):
            # Generate base signal with periodic component
            t = np.linspace(0, 4 * np.pi, seq_len)
            freq = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(0.3, 1.0)

            # Signal: periodic + noise
            signal = amplitude * np.sin(freq * t + phase)
            noise = rng.randn(seq_len) * 0.3

            # Create multi-feature input
            features = np.zeros((seq_len, n_features), dtype=np.float32)
            features[:, 0] = signal + noise
            # Add correlated features
            for i in range(1, n_features):
                features[:, i] = signal * rng.uniform(-0.5, 0.5) + rng.randn(seq_len) * 0.5

            # Label: 1 if mean amplitude > 0, 0 otherwise
            label = 1 if amplitude * np.sin(phase + 2 * np.pi) > 0.1 else 0

            self.data.append(torch.tensor(features, dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def make_text_synth_data(n, max_len, seed):
    """Generate synthetic text with learnable byte-level patterns."""
    rng = np.random.RandomState(seed)
    texts, labels = [], []

    for _ in range(n):
        length = rng.randint(50, max_len)
        label = rng.randint(0, 2)
        if label == 1:
            # Positive: more uppercase, more punctuation
            chars = []
            for _ in range(length):
                r = rng.random()
                if r < 0.4:
                    chars.append(chr(rng.randint(65, 91)))   # A-Z
                elif r < 0.6:
                    chars.append(chr(rng.randint(33, 48)))   # punctuation
                else:
                    chars.append(chr(rng.randint(97, 123)))  # a-z
            text = "".join(chars)
        else:
            # Negative: mostly lowercase, fewer punctuation
            chars = []
            for _ in range(length):
                r = rng.random()
                if r < 0.1:
                    chars.append(chr(rng.randint(65, 91)))   # A-Z
                elif r < 0.15:
                    chars.append(chr(rng.randint(33, 48)))   # punctuation
                else:
                    chars.append(chr(rng.randint(97, 123)))  # a-z
            text = "".join(chars)
        texts.append(text)
        labels.append(label)

    from pdna.data.text import IMDBByteDataset
    return IMDBByteDataset(texts, labels, max_len)


def get_data(task, cfg, seed):
    tc = TASKS[task]
    bs = cfg.training.batch_size
    if task == "listops":
        train_ds = ListOpsDataset.from_generated(tc["train_size"], max_len=tc["max_len"], seed=seed)
        val_ds = ListOpsDataset.from_generated(tc["val_size"], max_len=tc["max_len"], seed=seed + 1000)
        test_ds = ListOpsDataset.from_generated(tc["test_size"], max_len=tc["max_len"], seed=seed + 2000)
    elif task == "pathfinder":
        ti, tl = _generate_synthetic_pathfinder(tc["train_size"], seed=seed)
        vi, vl = _generate_synthetic_pathfinder(tc["val_size"], seed=seed + 1000)
        tei, tel = _generate_synthetic_pathfinder(tc["test_size"], seed=seed + 2000)
        train_ds = PathfinderDataset(ti, tl)
        val_ds = PathfinderDataset(vi, vl)
        test_ds = PathfinderDataset(tei, tel)
    elif task == "sequence_class":
        train_ds = SequenceClassificationDataset(tc["train_size"], tc["max_len"], tc["input_size"], seed)
        val_ds = SequenceClassificationDataset(tc["val_size"], tc["max_len"], tc["input_size"], seed + 1000)
        test_ds = SequenceClassificationDataset(tc["test_size"], tc["max_len"], tc["input_size"], seed + 2000)
    else:
        raise ValueError(task)
    kw = dict(batch_size=bs, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        test_ds,
    )


def gapped_eval(model, embedding, test_ds, device, bs=32):
    gap_results = {}
    for gl in GapLevel:
        gds = GappedWrapper(test_ds, gl)
        loader = DataLoader(gds, batch_size=bs, num_workers=0)
        model.eval()
        if embedding is not None:
            embedding.eval()
        correct = total = 0
        with torch.no_grad():
            for tokens, labels, mask in loader:
                tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                if embedding is not None:
                    x = embedding(tokens)
                else:
                    x = tokens.float() if tokens.dtype == torch.float32 else tokens.float().unsqueeze(-1)
                # Ensure 3D input
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
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
    cfg.training.max_epochs = 30
    cfg.training.batch_size = 64
    cfg.training.warmup_epochs = 3
    cfg.training.early_stopping_patience = 8
    cfg.training.lr = 1e-3

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

                # Determine if we need embedding
                use_embedding = tc.get("vocab_size") is not None
                trainer = Trainer(
                    model=model, config=cfg,
                    train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                    device=device, run_name=key, output_dir=str(out_dir),
                    vocab_size=tc.get("vocab_size", 256),
                    embed_dim=tc["input_size"],
                    use_embedding=use_embedding,
                )

                t0 = time.time()
                results = trainer.train()
                elapsed = time.time() - t0

                # Gapped eval
                gap_results = gapped_eval(
                    trainer.model,
                    trainer.embedding if use_embedding else None,
                    test_ds, device,
                )
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
    print("ABLATION TABLE (mean +/- std across seeds)", flush=True)
    print("=" * 70, flush=True)
    for task_name in TASKS:
        print(f"\n--- {task_name} ---", flush=True)
        for v in VARIANTS:
            accs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
            degs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]
            print(
                f"  {v:15s} | Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} "
                f"| Deg: {np.mean(degs):.4f}+/-{np.std(degs):.4f}",
                flush=True,
            )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
