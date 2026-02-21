#!/usr/bin/env python
"""PDNA Extended Ablation — publishable-quality experiments.

Addresses peer review gaps:
  - 5 seeds per config (up from 2) for statistical significance
  - 3 tasks: sMNIST, psMNIST (permuted), sCIFAR-10
  - Saves model checkpoints for post-hoc analysis (frequency spectrum, state norms)
  - Extracts learned pulse parameters (alpha, omega, amplitude) per variant

5 variants × 3 tasks × 5 seeds = 75 runs
(Dropping Variant F / full_idle since it's identical to E due to CfC parallel processing)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from pdna.data.gapped import GapLevel, create_gap_mask
from pdna.models.variants import build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer

# Drop full_idle (same as full_pdna due to CfC parallel processing)
VARIANTS = ["baseline", "noise", "pulse", "self_attend", "full_pdna"]
SEEDS = [42, 123, 456, 789, 1337]


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


class PixelMNIST(Dataset):
    """Pixel-by-pixel MNIST: 784 timesteps × 1 feature.

    Harder than row-by-row because the model must remember across
    784 steps instead of 28. Used for psMNIST (permuted pixel MNIST).
    """

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
        return pixels.unsqueeze(-1), label  # (784, 1)


class SequentialCIFAR10(Dataset):
    """Sequential CIFAR-10: 1024 timesteps × 3 features.

    Flatten 32×32 image to 1024 pixels, each with 3 color channels.
    Much harder than MNIST — tests whether pulse helps on real-world images.
    """

    def __init__(self, train=True, data_dir="/root/data"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.cifar = datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]
        # img: (3, 32, 32) -> (1024, 3) — each pixel is a 3D input
        pixels = img.permute(1, 2, 0).reshape(-1, 3)  # (1024, 3)
        return pixels, label


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
    "psmnist": {"input_size": 1, "output_size": 10, "seq_len": 784},
    "scifar10": {"input_size": 3, "output_size": 10, "seq_len": 1024},
}


def get_data(task_name, batch_size, seed):
    if task_name == "smnist":
        train_ds = RowMNIST(train=True)
        test_ds = RowMNIST(train=False)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [54000, 6000],
            generator=torch.Generator().manual_seed(seed),
        )
    elif task_name == "psmnist":
        train_ds = PixelMNIST(train=True, permute=True, perm_seed=0)
        test_ds = PixelMNIST(train=False, permute=True, perm_seed=0)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [54000, 6000],
            generator=torch.Generator().manual_seed(seed),
        )
    elif task_name == "scifar10":
        train_ds = SequentialCIFAR10(train=True)
        test_ds = SequentialCIFAR10(train=False)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [45000, 5000],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        raise ValueError(task_name)

    num_workers = 4
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
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


def extract_pulse_params(model):
    """Extract learned pulse parameters for analysis."""
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'omega'):
            params['omega'] = module.omega.detach().cpu().numpy().tolist()
        if hasattr(module, 'amplitude'):
            params['amplitude'] = module.amplitude.detach().cpu().numpy().tolist()
        if hasattr(module, 'alpha') and not isinstance(module, nn.Module.__class__):
            try:
                params['alpha'] = module.alpha.item()
            except (AttributeError, RuntimeError):
                pass
        if hasattr(module, 'beta') and not isinstance(module, nn.Module.__class__):
            try:
                params['beta'] = module.beta.item()
            except (AttributeError, RuntimeError):
                pass
        if hasattr(module, 'noise_scale'):
            params['noise_scale'] = module.noise_scale.item()
    return params


def compute_state_norms(model, test_ds, device, n_samples=100):
    """Compute hidden state norms during gap vs non-gap positions."""
    model.eval()
    gap_norms = []
    nongap_norms = []

    gds = FloatGappedWrapper(test_ds, GapLevel.LARGE)
    loader = DataLoader(gds, batch_size=min(n_samples, 64), num_workers=0)

    with torch.no_grad():
        for x, labels, mask in loader:
            x, mask = x.to(device), mask.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(-1)

            # Get hidden states from forward pass
            # Hook into backbone to get intermediate states
            h_states = []
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h_states.append(output[0].detach())

            handles = []
            for m in model.modules():
                if hasattr(m, 'cfc'):
                    handles.append(m.cfc.register_forward_hook(hook_fn))

            model(x, gap_mask=mask)

            for h in handles:
                h.remove()

            if h_states:
                h = h_states[0]  # (batch, seq_len, hidden)
                norms = h.norm(dim=-1)  # (batch, seq_len)

                gap_positions = mask.bool()
                nongap_positions = ~mask.bool()

                if gap_positions.any():
                    gap_norms.append(norms[gap_positions].mean().item())
                if nongap_positions.any():
                    nongap_norms.append(norms[nongap_positions].mean().item())
            break  # One batch is enough

    return {
        "gap_norm_mean": float(np.mean(gap_norms)) if gap_norms else 0,
        "nongap_norm_mean": float(np.mean(nongap_norms)) if nongap_norms else 0,
        "norm_ratio": float(np.mean(gap_norms) / np.mean(nongap_norms)) if gap_norms and nongap_norms else 0,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)

    out_dir = Path("runs_v5")
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    all_results = {}

    # Resume from existing results if available
    results_file = out_dir / "all_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resumed from {len(all_results)} existing runs", flush=True)

    total = len(VARIANTS) * len(TASKS) * len(SEEDS)
    run_num = 0

    for task_name, tc in TASKS.items():
        # Task-specific hyperparameters
        if task_name == "scifar10":
            # sCIFAR needs more capacity and longer training for 1024-step sequences
            hidden_size = 128
            batch_size = 128   # smaller batch for longer sequences
            max_epochs = 60
            lr = 3e-4
            patience = 12
        elif task_name == "psmnist":
            # psMNIST: 784-step sequences
            hidden_size = 128
            batch_size = 256
            max_epochs = 50
            lr = 5e-4
            patience = 10
        else:
            # sMNIST: quick 28-step sequences
            hidden_size = 128
            batch_size = 512
            max_epochs = 40
            lr = 5e-4
            patience = 8

        for variant_name in VARIANTS:
            for seed in SEEDS:
                run_num += 1
                key = f"{variant_name}_{task_name}_seed{seed}"

                # Skip already completed runs
                if key in all_results and "error" not in all_results[key]:
                    print(f"\n[{run_num}/{total}] {key} — SKIPPED (already done)", flush=True)
                    continue

                print(f"\n[{run_num}/{total}] {key}", flush=True)

                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    cfg = ExperimentConfig()
                    cfg.model.hidden_size = hidden_size
                    cfg.training.max_epochs = max_epochs
                    cfg.training.batch_size = batch_size
                    cfg.training.warmup_epochs = 3
                    cfg.training.early_stopping_patience = patience
                    cfg.training.lr = lr
                    cfg.training.grad_clip_norm = 1.0
                    cfg.logging.backend = "none"

                    model = build_variant(
                        variant_name,
                        input_size=tc["input_size"],
                        hidden_size=hidden_size,
                        output_size=tc["output_size"],
                        num_layers=cfg.model.num_layers,
                        dropout=cfg.model.dropout,
                    )

                    train_ld, val_ld, test_ld, test_ds = get_data(task_name, batch_size, seed)

                    trainer = Trainer(
                        model=model, config=cfg,
                        train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                        device=device, run_name=key, output_dir=str(out_dir),
                        use_embedding=False, vocab_size=256, embed_dim=tc["input_size"],
                    )

                    t0 = time.time()
                    results = trainer.train()
                    elapsed = time.time() - t0

                    # Gap robustness evaluation
                    gap_results = gapped_eval(trainer.model, test_ds, device)
                    results["gapped"] = gap_results
                    results["degradation"] = gap_results["gap_0"]["accuracy"] - gap_results["gap_30"]["accuracy"]
                    results["params"] = sum(p.numel() for p in model.parameters())
                    results["wall_time"] = elapsed

                    # Extract pulse parameters for analysis
                    results["pulse_params"] = extract_pulse_params(model)

                    # Compute state norms during gaps
                    results["state_norms"] = compute_state_norms(model, test_ds, device)

                    # Save model checkpoint
                    ckpt_path = ckpt_dir / f"{key}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "variant": variant_name,
                        "task": task_name,
                        "seed": seed,
                        "test_acc": results["test_acc"],
                    }, ckpt_path)

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
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)

    # ==================== Summary ====================
    print("\n" + "=" * 80, flush=True)
    print("EXTENDED ABLATION RESULTS (mean +/- std across 5 seeds)", flush=True)
    print("=" * 80, flush=True)

    for task_name in TASKS:
        print(f"\n--- {task_name} ---", flush=True)
        print(f"  {'Variant':<15} | {'Test Acc':>18} | {'Gap0':>8} {'Gap30':>8} | {'Degradation':>18}", flush=True)
        print(f"  {'-'*15}-+-{'-'*18}-+-{'-'*8}-{'-'*8}-+-{'-'*18}", flush=True)
        for v in VARIANTS:
            accs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
            degs = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("degradation", 0) for s in SEEDS]
            g0s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_0", {}).get("accuracy", 0) for s in SEEDS]
            g30s = [all_results.get(f"{v}_{task_name}_seed{s}", {}).get("gapped", {}).get("gap_30", {}).get("accuracy", 0) for s in SEEDS]
            valid = [a for a in accs if a > 0]
            if valid:
                print(
                    f"  {v:15s} | {np.mean(valid):.4f} +/- {np.std(valid):.4f} "
                    f"| {np.mean(g0s):.4f} {np.mean(g30s):.4f} "
                    f"| {np.mean(degs):.4f} +/- {np.std(degs):.4f}",
                    flush=True,
                )
            else:
                print(f"  {v:15s} | NO VALID RESULTS", flush=True)

    # Key comparisons with statistical tests
    from scipy import stats as sp_stats

    print("\n" + "=" * 80, flush=True)
    print("STATISTICAL COMPARISONS (paired t-test, 5 seeds)", flush=True)
    print("=" * 80, flush=True)

    comparisons = [
        ("full_pdna", "baseline", "PDNA vs Baseline"),
        ("pulse", "noise", "Pulse vs Noise (structured > random?)"),
        ("pulse", "baseline", "Pulse vs Baseline"),
        ("self_attend", "baseline", "SelfAttend vs Baseline"),
        ("full_pdna", "pulse", "Full PDNA vs Pulse-only"),
    ]

    for task_name in TASKS:
        print(f"\n--- {task_name} ---", flush=True)
        for v1, v2, label in comparisons:
            a1 = [all_results.get(f"{v1}_{task_name}_seed{s}", {}).get("test_acc") for s in SEEDS]
            a2 = [all_results.get(f"{v2}_{task_name}_seed{s}", {}).get("test_acc") for s in SEEDS]
            d1 = [all_results.get(f"{v1}_{task_name}_seed{s}", {}).get("degradation") for s in SEEDS]
            d2 = [all_results.get(f"{v2}_{task_name}_seed{s}", {}).get("degradation") for s in SEEDS]

            a1 = [x for x in a1 if x is not None]
            a2 = [x for x in a2 if x is not None]
            d1 = [x for x in d1 if x is not None]
            d2 = [x for x in d2 if x is not None]

            if len(a1) >= 2 and len(a2) >= 2 and len(a1) == len(a2):
                t_acc, p_acc = sp_stats.ttest_rel(a1, a2)
                t_deg, p_deg = sp_stats.ttest_rel(d1, d2)
                diff_acc = np.mean(a1) - np.mean(a2)
                diff_deg = np.mean(d1) - np.mean(d2)
                sig_acc = "***" if p_acc < 0.01 else ("**" if p_acc < 0.05 else ("*" if p_acc < 0.1 else ""))
                sig_deg = "***" if p_deg < 0.01 else ("**" if p_deg < 0.05 else ("*" if p_deg < 0.1 else ""))
                print(
                    f"  {label}: acc Δ={diff_acc:+.4f} (p={p_acc:.4f}{sig_acc}), "
                    f"deg Δ={diff_deg:+.4f} (p={p_deg:.4f}{sig_deg})",
                    flush=True,
                )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
