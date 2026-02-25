#!/usr/bin/env python
"""PDNA Revision Experiments — per-epoch pulse parameter tracking + post-hoc analyses.

Addresses reviewer feedback:
  - Tracks alpha * ||A|| over training (reviewer: "alpha can be absorbed into A")
  - Analyzes |W_phi * h| / omega vs t (reviewer: "phase degeneracy")
  - Tests Nyquist clamping (reviewer: "frequencies above Nyquist")

Only runs pulse + full_pdna variants × 5 seeds × sMNIST = 10 runs.
"""

import json
import math
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

torch.backends.cudnn.benchmark = True

VARIANTS = ["pulse", "full_pdna"]
SEEDS = [42, 123, 456, 789, 1337]

# ==================== Datasets (from run_extended_ablation.py) ====================

class RowMNIST(Dataset):
    def __init__(self, train=True, data_dir="/root/data"):
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist = datasets.MNIST(data_dir, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        pixels = img.view(-1)
        return pixels.view(28, 28), label


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


# ==================== Parameter Extraction ====================

def extract_pulse_params_extended(model):
    """Extract pulse parameters including W_phi for phase analysis."""
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'omega'):
            omega = module.omega.detach().cpu()
            params['omega'] = omega.numpy().tolist()
        if hasattr(module, 'amplitude'):
            A = module.amplitude.detach().cpu()
            params['amplitude'] = A.numpy().tolist()
            params['A_norm'] = A.norm().item()
        if hasattr(module, 'alpha') and isinstance(getattr(module, 'alpha', None), nn.Parameter):
            params['alpha'] = module.alpha.item()
        if hasattr(module, 'phase_net') and isinstance(module.phase_net, nn.Linear):
            W_phi = module.phase_net.weight.detach().cpu()
            b_phi = module.phase_net.bias.detach().cpu()
            params['W_phi_frobenius'] = W_phi.norm().item()
            params['W_phi'] = W_phi.numpy().tolist()
            params['b_phi'] = b_phi.numpy().tolist()
        if hasattr(module, 'beta') and isinstance(getattr(module, 'beta', None), nn.Parameter):
            params['beta'] = module.beta.item()
    if 'alpha' in params and 'A_norm' in params:
        params['alpha_times_A_norm'] = params['alpha'] * params['A_norm']
    return params


def extract_epoch_snapshot(model):
    """Lightweight per-epoch snapshot (no large matrices)."""
    snap = {}
    for name, module in model.named_modules():
        if hasattr(module, 'omega'):
            omega = module.omega.detach().cpu()
            snap['omega_mean'] = omega.mean().item()
            snap['omega_std'] = omega.std().item()
            snap['omega_median'] = omega.median().item()
            snap['omega_q25'] = omega.quantile(0.25).item()
            snap['omega_q75'] = omega.quantile(0.75).item()
        if hasattr(module, 'amplitude'):
            A = module.amplitude.detach().cpu()
            snap['A_norm'] = A.norm().item()
            snap['A_mean'] = A.mean().item()
            snap['A_std'] = A.std().item()
        if hasattr(module, 'alpha') and isinstance(getattr(module, 'alpha', None), nn.Parameter):
            snap['alpha'] = module.alpha.item()
        if hasattr(module, 'phase_net') and isinstance(module.phase_net, nn.Linear):
            snap['W_phi_frobenius'] = module.phase_net.weight.detach().norm().item()
        if hasattr(module, 'beta') and isinstance(getattr(module, 'beta', None), nn.Parameter):
            snap['beta'] = module.beta.item()
    if 'alpha' in snap and 'A_norm' in snap:
        snap['alpha_times_A_norm'] = snap['alpha'] * snap['A_norm']
    return snap


# ==================== Post-Training Analyses ====================

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
                logits, _ = model(x, gap_mask=mask)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        gap_results[gl.value] = {"accuracy": correct / total if total > 0 else 0}
    return gap_results


def analyze_phase_magnitude(model, test_ds, device, n_batches=5):
    """Compute |W_phi * h| / omega vs t — answers the reviewer's phase degeneracy question."""
    model.eval()

    # Extract W_phi and omega
    W_phi = omega = None
    for name, module in model.named_modules():
        if hasattr(module, 'phase_net') and isinstance(module.phase_net, nn.Linear):
            W_phi = module.phase_net.weight.detach().to(device)  # (d, d)
            b_phi = module.phase_net.bias.detach().to(device)  # (d,)
        if hasattr(module, 'omega'):
            omega = module.omega.detach().to(device)  # (d,)

    if W_phi is None or omega is None:
        return {}

    loader = DataLoader(test_ds, batch_size=64, num_workers=0, shuffle=False)

    # Collect stats per timestep
    T = 28
    phase_mag_per_t = [[] for _ in range(T)]  # |W_phi * h + b_phi| per timestep
    ratio_per_t = [[] for _ in range(T)]       # |W_phi * h + b_phi| / |omega| per timestep

    batch_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            x = batch_x.to(device).float()

            # Get h_cfc (pre-pulse hidden states) via CfC backbone
            h_cfc_list = []
            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    h_cfc_list.append(output[0].detach())
                else:
                    h_cfc_list.append(output.detach())

            # Hook the CfC inside the first backbone
            backbone = model.backbones[0]
            handle = backbone.cfc.register_forward_hook(hook_fn)
            model(x)  # forward pass to trigger hook
            handle.remove()

            if not h_cfc_list:
                continue

            h_cfc = h_cfc_list[0]  # (B, T, d)

            for t_idx in range(min(T, h_cfc.shape[1])):
                h_t = h_cfc[:, t_idx, :]  # (B, d)
                phi_h = h_t @ W_phi.T + b_phi  # (B, d) — W_phi * h + b_phi
                phi_mag = phi_h.abs()  # (B, d)
                omega_abs = omega.abs().unsqueeze(0)  # (1, d)
                ratio = phi_mag / (omega_abs + 1e-8)  # (B, d)

                phase_mag_per_t[t_idx].append(phi_mag.mean(dim=0).cpu().numpy())  # (d,)
                ratio_per_t[t_idx].append(ratio.mean(dim=0).cpu().numpy())  # (d,)

            batch_count += 1
            if batch_count >= n_batches:
                break

    # Aggregate
    results = {'per_timestep': []}
    for t_idx in range(T):
        if not phase_mag_per_t[t_idx]:
            continue
        phase_mags = np.stack(phase_mag_per_t[t_idx]).mean(axis=0)  # (d,)
        ratios = np.stack(ratio_per_t[t_idx]).mean(axis=0)  # (d,)
        results['per_timestep'].append({
            't': t_idx,
            'phase_mag_mean': float(phase_mags.mean()),
            'phase_mag_std': float(phase_mags.std()),
            'phase_mag_median': float(np.median(phase_mags)),
            'ratio_mean': float(ratios.mean()),
            'ratio_std': float(ratios.std()),
            'ratio_median': float(np.median(ratios)),
            'ratio_vs_t': float(ratios.mean()) / (t_idx + 1e-8),
            'frac_phase_dominates': float((ratios > t_idx).mean()) if t_idx > 0 else 1.0,
        })

    # Summary
    all_ratio_means = [r['ratio_mean'] for r in results['per_timestep']]
    all_ts = [r['t'] for r in results['per_timestep']]
    frac_dominates = sum(1 for r, t in zip(all_ratio_means, all_ts) if t > 0 and r > t) / max(1, sum(1 for t in all_ts if t > 0))
    results['summary'] = {
        'mean_ratio_across_time': float(np.mean(all_ratio_means)),
        'frac_timesteps_phase_dominates': frac_dominates,
        'ratio_at_midpoint': results['per_timestep'][T // 2]['ratio_mean'] if len(results['per_timestep']) > T // 2 else 0,
        't_at_midpoint': T // 2,
    }

    return results


def nyquist_analysis(model, test_ds, device):
    """Analyze fraction of omega above Nyquist and test clamping impact."""
    model.eval()
    nyquist = math.pi  # dt=1, so Nyquist angular frequency = pi

    omega = None
    omega_param = None
    for name, module in model.named_modules():
        if hasattr(module, 'omega'):
            omega = module.omega.detach().cpu().numpy()
            omega_param = module.omega  # keep reference for clamping

    if omega is None:
        return {}

    above_nyquist = np.abs(omega) > nyquist

    results = {
        'nyquist_freq': float(nyquist),
        'fraction_above': float(above_nyquist.mean()),
        'n_above': int(above_nyquist.sum()),
        'n_total': len(omega),
        'omega_above_nyquist_mean': float(np.abs(omega[above_nyquist]).mean()) if above_nyquist.any() else 0,
        'omega_below_nyquist_mean': float(np.abs(omega[~above_nyquist]).mean()) if (~above_nyquist).any() else 0,
    }

    # Evaluate with original omega
    original_gap = gapped_eval(model, test_ds, device)
    results['original_multigap'] = original_gap['multi_gap']['accuracy']
    results['original_gap5'] = original_gap['gap_5']['accuracy']

    # Clamp omega and re-evaluate
    original_omega_data = omega_param.data.clone()
    with torch.no_grad():
        omega_param.data.clamp_(-nyquist, nyquist)

    clamped_gap = gapped_eval(model, test_ds, device)
    results['clamped_multigap'] = clamped_gap['multi_gap']['accuracy']
    results['clamped_gap5'] = clamped_gap['gap_5']['accuracy']
    results['multigap_delta'] = clamped_gap['multi_gap']['accuracy'] - original_gap['multi_gap']['accuracy']
    results['gap5_delta'] = clamped_gap['gap_5']['accuracy'] - original_gap['gap_5']['accuracy']

    # Restore original omega
    with torch.no_grad():
        omega_param.data.copy_(original_omega_data)

    return results


# ==================== Custom Training Loop ====================

def train_with_tracking(model, config, train_loader, val_loader, test_loader, device, run_name, out_dir):
    """Custom training loop that extracts pulse params after each epoch."""
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, run_name=run_name, output_dir=str(out_dir),
        use_embedding=False, vocab_size=256, embed_dim=28,
        use_amp=True,
    )

    pulse_param_history = []
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Training {run_name} on {device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Snapshot at epoch 0 (before any training)
    snap = extract_epoch_snapshot(model)
    snap['epoch'] = 0
    pulse_param_history.append(snap)

    for epoch in range(config.training.max_epochs):
        t0 = time.time()
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.evaluate(trainer.val_loader)
        elapsed = time.time() - t0

        # Log to trainer history
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        trainer.history["lr"].append(trainer.optimizer.param_groups[0]["lr"])
        trainer.history["epoch_time"].append(elapsed)

        # Per-epoch pulse parameter snapshot
        snap = extract_epoch_snapshot(model)
        snap['epoch'] = epoch + 1
        snap['train_acc'] = train_acc
        snap['val_acc'] = val_acc
        pulse_param_history.append(snap)

        print(
            f"  Epoch {epoch+1:3d}/{config.training.max_epochs} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"alpha={snap.get('alpha', 0):.4f} A_norm={snap.get('A_norm', 0):.3f} "
            f"a*A={snap.get('alpha_times_A_norm', 0):.4f} | {elapsed:.1f}s",
            flush=True,
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch + 1
            trainer._save_checkpoint("best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best model and evaluate
    trainer._load_checkpoint("best.pt")
    test_loss, test_acc = trainer.evaluate(trainer.test_loader)
    print(f"  Test: {test_loss:.4f}/{test_acc:.4f} (best epoch {best_epoch})", flush=True)

    results = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_epoch": len(trainer.history["train_loss"]),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": trainer.history,
        "pulse_param_history": pulse_param_history,
    }

    return results, model


# ==================== Main ====================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    out_dir = Path("runs_v6")
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    all_results = {}
    results_file = out_dir / "all_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resumed from {len(all_results)} existing runs", flush=True)

    total = len(VARIANTS) * len(SEEDS)
    run_num = 0

    for variant_name in VARIANTS:
        for seed in SEEDS:
            run_num += 1
            key = f"{variant_name}_smnist_seed{seed}"

            if key in all_results and "error" not in all_results[key]:
                print(f"\n[{run_num}/{total}] {key} — SKIPPED (already done)", flush=True)
                continue

            print(f"\n[{run_num}/{total}] {key}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                cfg = ExperimentConfig()
                cfg.model.hidden_size = 128
                cfg.training.max_epochs = 40
                cfg.training.batch_size = 512
                cfg.training.warmup_epochs = 3
                cfg.training.early_stopping_patience = 8
                cfg.training.lr = 5e-4
                cfg.training.grad_clip_norm = 1.0
                cfg.logging.backend = "none"

                model = build_variant(
                    variant_name, input_size=28, hidden_size=128, output_size=10,
                    num_layers=cfg.model.num_layers, dropout=cfg.model.dropout,
                )

                # Data
                train_ds = RowMNIST(train=True)
                test_ds = RowMNIST(train=False)
                train_ds, val_ds = torch.utils.data.random_split(
                    train_ds, [54000, 6000],
                    generator=torch.Generator().manual_seed(seed),
                )
                kw = dict(batch_size=512, num_workers=8, pin_memory=True, persistent_workers=True)
                train_ld = DataLoader(train_ds, shuffle=True, **kw)
                val_ld = DataLoader(val_ds, shuffle=False, **kw)
                test_ld = DataLoader(test_ds, shuffle=False, **kw)

                # Train with per-epoch tracking
                t0 = time.time()
                results, model = train_with_tracking(
                    model, cfg, train_ld, val_ld, test_ld, device, key, out_dir,
                )
                elapsed = time.time() - t0
                results["wall_time"] = elapsed
                results["params"] = sum(p.numel() for p in model.parameters())

                # Gap evaluation
                print("  Running gapped evaluation...", flush=True)
                gap_results = gapped_eval(model, test_ds, device)
                results["gapped"] = gap_results
                results["degradation"] = gap_results["gap_0"]["accuracy"] - gap_results["gap_30"]["accuracy"]

                # Final pulse params (full, with W_phi matrix)
                results["pulse_params"] = extract_pulse_params_extended(model)

                # Phase magnitude analysis
                print("  Running phase magnitude analysis...", flush=True)
                results["phase_magnitude_analysis"] = analyze_phase_magnitude(model, test_ds, device)

                # Nyquist analysis
                print("  Running Nyquist analysis...", flush=True)
                results["nyquist_analysis"] = nyquist_analysis(model, test_ds, device)

                # Save checkpoint
                ckpt_path = ckpt_dir / f"{key}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "variant": variant_name,
                    "seed": seed,
                    "test_acc": results["test_acc"],
                }, ckpt_path)

                all_results[key] = results

                print(
                    f"  Test={results['test_acc']:.4f} "
                    f"MultiGap={gap_results['multi_gap']['accuracy']:.4f} "
                    f"Nyquist_above={results['nyquist_analysis'].get('fraction_above', 0):.1%} "
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
    print("\n" + "=" * 80)
    print("REVISION EXPERIMENT SUMMARY")
    print("=" * 80)

    for v in VARIANTS:
        accs = [all_results.get(f"{v}_smnist_seed{s}", {}).get("test_acc", 0) for s in SEEDS]
        valid = [a for a in accs if a > 0]
        if not valid:
            continue

        # Alpha * ||A|| at convergence
        final_aA = []
        for s in SEEDS:
            r = all_results.get(f"{v}_smnist_seed{s}", {})
            history = r.get("pulse_param_history", [])
            if history:
                final_aA.append(history[-1].get("alpha_times_A_norm", 0))

        # Nyquist
        nyquist_fracs = []
        for s in SEEDS:
            r = all_results.get(f"{v}_smnist_seed{s}", {})
            nq = r.get("nyquist_analysis", {})
            if nq:
                nyquist_fracs.append(nq.get("fraction_above", 0))

        print(f"\n{v}:")
        print(f"  Test acc: {np.mean(valid):.4f} ± {np.std(valid):.4f}")
        if final_aA:
            print(f"  Final α·||A||: {np.mean(final_aA):.4f} ± {np.std(final_aA):.4f}")
        if nyquist_fracs:
            print(f"  Fraction ω > π: {np.mean(nyquist_fracs):.1%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
