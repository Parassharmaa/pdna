"""Run all ablation experiments for PDNA.

This script runs all 6 variants on all LRA tasks with multiple seeds.
Results are saved to JSON files for analysis.

Usage:
    # Full experiments (GPU recommended):
    uv run python scripts/run_experiments.py --tier 1

    # Quick test (CPU, small models):
    uv run python scripts/run_experiments.py --tier 1 --quick

    # Single variant/task:
    uv run python scripts/run_experiments.py --variants baseline full_pdna --tasks listops
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pdna.data.gapped import GapLevel, GappedWrapper
from pdna.data.listops import ListOpsDataset
from pdna.data.pathfinder import PathfinderDataset, _generate_synthetic_pathfinder
from pdna.models.variants import Variant, build_variant
from pdna.training.config import ExperimentConfig
from pdna.training.trainer import Trainer


TIER1_TASKS = ["listops", "pathfinder", "text_synthetic"]
TIER2_TASKS = ["image", "retrieval"]

TASK_CONFIGS = {
    "listops": {"input_size": 17, "output_size": 10, "vocab_size": 17, "max_len": 512},
    "pathfinder": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 1024},
    "text_synthetic": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 512},
    "image": {"input_size": 256, "output_size": 10, "vocab_size": 256, "max_len": 1024},
    "retrieval": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 512},
}


def _make_synthetic_text_data(n_samples: int, max_len: int, seed: int):
    """Generate synthetic text classification data for testing."""
    import numpy as np
    rng = np.random.RandomState(seed)
    texts, labels = [], []
    for _ in range(n_samples):
        length = rng.randint(50, max_len)
        text = bytes(rng.randint(32, 127, size=length)).decode("ascii")
        texts.append(text)
        labels.append(rng.randint(0, 2))

    from pdna.data.text import IMDBByteDataset
    return IMDBByteDataset(texts, labels, max_len)


def get_dataloaders_for_experiment(task: str, config: ExperimentConfig, quick: bool = False):
    """Build dataloaders for a given task."""
    batch_size = config.training.batch_size
    task_cfg = TASK_CONFIGS[task]
    max_len = task_cfg["max_len"]

    if quick:
        n_train, n_val, n_test = 200, 50, 50
    else:
        n_train, n_val, n_test = 2000, 500, 500

    if task == "listops":
        train_ds = ListOpsDataset.from_generated(n_train, max_len=max_len, seed=42)
        val_ds = ListOpsDataset.from_generated(n_val, max_len=max_len, seed=123)
        test_ds = ListOpsDataset.from_generated(n_test, max_len=max_len, seed=456)
    elif task == "pathfinder":
        train_imgs, train_labels = _generate_synthetic_pathfinder(n_train, seed=42)
        val_imgs, val_labels = _generate_synthetic_pathfinder(n_val, seed=123)
        test_imgs, test_labels = _generate_synthetic_pathfinder(n_test, seed=456)
        train_ds = PathfinderDataset(train_imgs, train_labels)
        val_ds = PathfinderDataset(val_imgs, val_labels)
        test_ds = PathfinderDataset(test_imgs, test_labels)
    elif task == "text_synthetic":
        train_ds = _make_synthetic_text_data(n_train, max_len, seed=42)
        val_ds = _make_synthetic_text_data(n_val, max_len, seed=123)
        test_ds = _make_synthetic_text_data(n_test, max_len, seed=456)
    elif task == "retrieval":
        from pdna.data.retrieval import _generate_synthetic_retrieval
        train_ds = _generate_synthetic_retrieval(n_train, max_len=max_len, seed=42)
        val_ds = _generate_synthetic_retrieval(n_val, max_len=max_len, seed=123)
        test_ds = _generate_synthetic_retrieval(n_test, max_len=max_len, seed=456)
    else:
        raise ValueError(f"Unknown task: {task}")

    loader_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, train_ds, test_ds


def run_gapped_evaluation(model, embedding, test_ds, task_cfg, device, config):
    """Evaluate model on all gap levels."""
    from pdna.training.trainer import Trainer
    import torch.nn as nn

    gap_results = {}
    for gap_level in GapLevel:
        gapped_ds = GappedWrapper(test_ds, gap_level)
        gapped_loader = DataLoader(gapped_ds, batch_size=config.training.batch_size, num_workers=0)

        model.eval()
        if embedding is not None:
            embedding.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in gapped_loader:
                tokens, labels, gap_mask = batch
                tokens = tokens.to(device)
                labels = labels.to(device)
                gap_mask = gap_mask.to(device)

                if embedding is not None:
                    x = embedding(tokens)
                else:
                    x = tokens.float().unsqueeze(-1)

                logits, _ = model(x, gap_mask=gap_mask)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        gap_results[gap_level.value] = {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    return gap_results


def run_single_experiment(variant_name, task, seed, config, device, output_dir, quick=False):
    """Run a single experiment (one variant, one task, one seed)."""
    task_cfg = TASK_CONFIGS[task]

    torch.manual_seed(seed)
    model = build_variant(
        variant=variant_name,
        input_size=task_cfg["input_size"],
        hidden_size=config.model.hidden_size,
        output_size=task_cfg["output_size"],
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )

    train_loader, val_loader, test_loader, train_ds, test_ds = get_dataloaders_for_experiment(task, config, quick)
    run_name = f"{variant_name}_{task}_seed{seed}"

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        run_name=run_name,
        output_dir=output_dir,
        vocab_size=task_cfg["vocab_size"],
        embed_dim=task_cfg["input_size"],
        use_embedding=True,
    )

    results = trainer.train()

    # Run gapped evaluation
    gap_results = run_gapped_evaluation(
        model=trainer.model,
        embedding=trainer.embedding,
        test_ds=test_ds,
        task_cfg=task_cfg,
        device=device,
        config=config,
    )
    results["gapped"] = gap_results

    # Compute degradation: gap_0 acc - gap_30 acc
    if "gap_0" in gap_results and "gap_30" in gap_results:
        results["degradation"] = gap_results["gap_0"]["accuracy"] - gap_results["gap_30"]["accuracy"]

    # Save
    result_path = Path(output_dir) / run_name / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run PDNA ablation experiments")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2])
    parser.add_argument("--quick", action="store_true", help="Quick test with small models")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    from pdna.training.config import load_config
    config = load_config(args.config)

    if args.quick:
        config.model.hidden_size = 32
        config.training.max_epochs = 5
        config.training.batch_size = 16
        config.training.warmup_epochs = 1

    # Determine tasks
    if args.tasks:
        tasks = args.tasks
    elif args.tier == 1:
        tasks = TIER1_TASKS
    else:
        tasks = TIER1_TASKS + TIER2_TASKS

    # Determine variants
    if args.variants:
        variants = args.variants
    else:
        variants = [v.value for v in Variant]

    # Seeds
    seeds = args.seeds or config.training.seeds

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Running experiments:")
    print(f"  Variants: {variants}")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Total runs: {len(variants) * len(tasks) * len(seeds)}")
    print()

    all_results = {}
    start_time = time.time()

    for task in tasks:
        for variant_name in variants:
            for seed in seeds:
                run_key = f"{variant_name}_{task}_seed{seed}"
                print(f"{'='*60}")
                print(f"Running: {run_key}")
                print(f"{'='*60}")

                try:
                    results = run_single_experiment(
                        variant_name=variant_name,
                        task=task,
                        seed=seed,
                        config=config,
                        device=device,
                        output_dir=args.output_dir,
                        quick=args.quick,
                    )
                    all_results[run_key] = results
                    print(f"  -> Test acc: {results.get('test_acc', 'N/A')}")
                    if "degradation" in results:
                        print(f"  -> Degradation: {results['degradation']:.4f}")
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    all_results[run_key] = {"error": str(e)}

                print()

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.1f}s")

    # Save aggregated results
    summary_path = Path(args.output_dir) / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
