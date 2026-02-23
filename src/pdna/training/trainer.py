"""Training loop with AdamW, cosine annealing, early stopping, checkpointing."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pdna.training.config import ExperimentConfig


class Trainer:
    """Trains and evaluates a PDNA variant model."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        device: str | torch.device = "cpu",
        run_name: str = "run",
        output_dir: str = "runs",
        vocab_size: int = 256,
        embed_dim: int = 128,
        use_embedding: bool = True,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.run_name = run_name
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_embedding = use_embedding

        # AMP (Automatic Mixed Precision)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Embedding layer for token inputs
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        else:
            self.embedding = None

        # Optimizer
        params = list(model.parameters())
        if self.embedding is not None:
            params += list(self.embedding.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Cosine annealing with warmup
        total_steps = config.training.max_epochs * len(train_loader)
        warmup_steps = config.training.warmup_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_schedule(step, warmup_steps, total_steps),
        )

        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

        # TensorBoard writer
        self.writer = None
        if config.logging.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError:
                pass

    def _prepare_batch(self, batch):
        """Unpack batch and move to device. Handles both gapped and non-gapped datasets."""
        if len(batch) == 3:
            tokens, labels, gap_mask = batch
            gap_mask = gap_mask.to(self.device)
        else:
            tokens, labels = batch
            gap_mask = None

        tokens = tokens.to(self.device)
        labels = labels.to(self.device)

        # Embed tokens or pass through as float
        if self.embedding is not None:
            x = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        else:
            x = tokens.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # (batch, seq_len, 1) for scalar sequences

        return x, labels, gap_mask

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        if self.embedding is not None:
            self.embedding.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            x, labels, gap_mask = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(x, gap_mask=gap_mask)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            self.scaler.unscale_(self.optimizer)
            params = list(self.model.parameters())
            if self.embedding is not None:
                params += list(self.embedding.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.config.training.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate on a data loader. Returns (avg_loss, accuracy)."""
        self.model.eval()
        if self.embedding is not None:
            self.embedding.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            x, labels, gap_mask = self._prepare_batch(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(x, gap_mask=gap_mask)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self) -> dict:
        """Run full training loop with early stopping.

        Returns:
            Dictionary with final metrics and training history.
        """
        print(f"Training {self.run_name} on {self.device}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Epochs: {self.config.training.max_epochs}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  AMP: {self.use_amp}")

        for epoch in range(self.config.training.max_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)
            epoch_time = time.time() - start_time

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["epoch_time"].append(epoch_time)

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            print(
                f"  Epoch {epoch+1:3d}/{self.config.training.max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and evaluate on test
        self._load_checkpoint("best.pt")
        test_results = {}
        if self.test_loader is not None:
            test_loss, test_acc = self.evaluate(self.test_loader)
            test_results = {"test_loss": test_loss, "test_acc": test_acc}
            print(f"  Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # Save results
        results = {
            "best_val_acc": self.best_val_acc,
            "final_epoch": len(self.history["train_loss"]),
            "history": self.history,
            **test_results,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        if self.writer:
            self.writer.close()

        return results

    def _save_checkpoint(self, filename: str) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if self.embedding is not None:
            state["embedding"] = self.embedding.state_dict()
        torch.save(state, self.output_dir / filename)

    def _load_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"])
            if self.embedding is not None and "embedding" in state:
                self.embedding.load_state_dict(state["embedding"])
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])


def _warmup_cosine_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
