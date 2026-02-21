"""Main training script for PDNA ablation experiments.

Usage:
    uv run python scripts/train.py --variant baseline --task listops --seed 42
    uv run python scripts/train.py --variant full_pdna --task listops --seed 42 --epochs 5
"""

from __future__ import annotations

import argparse
import sys

import torch

from pdna.models.variants import Variant, build_variant
from pdna.training.config import ExperimentConfig, load_config
from pdna.training.trainer import Trainer


TASK_CONFIGS = {
    "listops": {"input_size": 17, "output_size": 10, "vocab_size": 17, "max_len": 2048},
    "pathfinder": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 1024},
    "text": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 4096},
    "image": {"input_size": 256, "output_size": 10, "vocab_size": 256, "max_len": 1024},
    "retrieval": {"input_size": 256, "output_size": 2, "vocab_size": 256, "max_len": 8000},
}


def get_dataloaders(task: str, config: ExperimentConfig, data_dir: str | None = None, use_gapped: bool = False, gap_level: str = "gap_0"):
    """Get dataloaders for a given task."""
    batch_size = config.training.batch_size

    if task == "listops":
        from pdna.data.listops import get_listops_dataloaders
        train_loader, val_loader, test_loader = get_listops_dataloaders(
            data_dir=data_dir, batch_size=batch_size, max_len=TASK_CONFIGS[task]["max_len"],
            generate=True, train_size=2000, val_size=500, test_size=500,
        )
    elif task == "image":
        from pdna.data.image import get_cifar10_dataloaders
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            data_dir=data_dir or "data", batch_size=batch_size,
        )
    elif task == "text":
        from pdna.data.text import get_imdb_dataloaders
        train_loader, val_loader, test_loader = get_imdb_dataloaders(
            data_dir=data_dir or "data", batch_size=batch_size, max_len=TASK_CONFIGS[task]["max_len"],
        )
    elif task == "pathfinder":
        from pdna.data.pathfinder import get_pathfinder_dataloaders
        train_loader, val_loader, test_loader = get_pathfinder_dataloaders(
            data_dir=data_dir, batch_size=batch_size,
            generate=True, train_size=2000, val_size=500, test_size=500,
        )
    elif task == "retrieval":
        from pdna.data.retrieval import get_retrieval_dataloaders
        train_loader, val_loader, test_loader = get_retrieval_dataloaders(
            data_dir=data_dir, batch_size=batch_size,
            generate=True, train_size=2000, val_size=500, test_size=500,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train PDNA variant on LRA task")
    parser.add_argument("--variant", type=str, required=True, choices=[v.value for v in Variant])
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)
    if args.epochs is not None:
        config.training.max_epochs = args.epochs

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Task config
    task_cfg = TASK_CONFIGS[args.task]

    # Build model
    model = build_variant(
        variant=args.variant,
        input_size=task_cfg["input_size"],
        hidden_size=config.model.hidden_size,
        output_size=task_cfg["output_size"],
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        alpha_init=config.model.alpha_init,
        beta_init=config.model.beta_init,
        idle_ticks_per_gap=config.model.idle_ticks_per_gap,
        ode_unfolds=config.model.ode_unfolds,
    )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(args.task, config, args.data_dir)

    # Run name
    run_name = f"{args.variant}_{args.task}_seed{args.seed}"

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        run_name=run_name,
        output_dir=args.output_dir,
        vocab_size=task_cfg["vocab_size"],
        embed_dim=task_cfg["input_size"],
        use_embedding=True,
    )

    results = trainer.train()
    print(f"\nFinal results: {results.get('test_acc', 'N/A')}")
    return results


if __name__ == "__main__":
    main()
