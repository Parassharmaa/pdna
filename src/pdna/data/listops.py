"""ListOps dataset for LRA benchmark.

10-class classification on nested mathematical expressions.
Sequences of ~2K tokens, vocabulary of ~17 tokens.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# ListOps vocabulary
LISTOPS_TOKENS = ["<pad>", "<unk>", "[MAX", "[MIN", "[MED", "[SM", "]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(LISTOPS_TOKENS)}
VOCAB_SIZE = len(LISTOPS_TOKENS)
PAD_ID = 0


def tokenize_listops(expression: str, max_len: int = 2048) -> list[int]:
    """Tokenize a ListOps expression into integer token IDs."""
    tokens = expression.replace("(", "").replace(")", "").split()
    ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["<unk>"]) for t in tokens]
    # Truncate and pad
    ids = ids[:max_len]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def _generate_expression(depth: int = 0, max_depth: int = 10, max_args: int = 5) -> tuple[str, int]:
    """Recursively generate a ListOps expression and its value."""
    ops = {
        "[MAX": max,
        "[MIN": min,
        "[MED": lambda xs: sorted(xs)[len(xs) // 2],
        "[SM": lambda xs: sum(xs) % 10,
    }
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        val = random.randint(0, 9)
        return str(val), val

    op_name = random.choice(list(ops.keys()))
    n_args = random.randint(2, max_args)
    args_strs = []
    args_vals = []
    for _ in range(n_args):
        s, v = _generate_expression(depth + 1, max_depth, max_args)
        args_strs.append(s)
        args_vals.append(v)

    result = ops[op_name](args_vals)
    expr = f"{op_name} {' '.join(args_strs)} ]"
    return expr, result


def generate_listops_dataset(n_samples: int, max_depth: int = 6, max_args: int = 5, seed: int = 42) -> list[tuple[str, int]]:
    """Generate a ListOps dataset."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        expr, val = _generate_expression(0, max_depth, max_args)
        data.append((expr, val))
    return data


class ListOpsDataset(Dataset):
    """ListOps dataset — either from TSV files or generated."""

    def __init__(self, data: list[tuple[str, int]], max_len: int = 2048) -> None:
        self.data = data
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        expr, label = self.data[idx]
        tokens = tokenize_listops(expr, self.max_len)
        return torch.tensor(tokens, dtype=torch.long), label

    @classmethod
    def from_tsv(cls, path: str | Path, max_len: int = 2048) -> ListOpsDataset:
        """Load from TSV file (LRA release format)."""
        data = []
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                expr = row["Source"]
                label = int(row["Target"])
                data.append((expr, label))
        return cls(data, max_len)

    @classmethod
    def from_generated(cls, n_samples: int, max_len: int = 2048, seed: int = 42) -> ListOpsDataset:
        """Generate synthetic ListOps data."""
        data = generate_listops_dataset(n_samples, seed=seed)
        return cls(data, max_len)


def get_listops_dataloaders(
    data_dir: str | Path | None = None,
    batch_size: int = 32,
    max_len: int = 2048,
    num_workers: int = 0,
    generate: bool = True,
    train_size: int = 96000,
    val_size: int = 2000,
    test_size: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders for ListOps.

    If data_dir is provided and contains TSV files, loads from them.
    Otherwise generates synthetic data.
    """
    if data_dir and os.path.exists(data_dir):
        train_ds = ListOpsDataset.from_tsv(Path(data_dir) / "basic_train.tsv", max_len)
        val_ds = ListOpsDataset.from_tsv(Path(data_dir) / "basic_val.tsv", max_len)
        test_ds = ListOpsDataset.from_tsv(Path(data_dir) / "basic_test.tsv", max_len)
    elif generate:
        train_ds = ListOpsDataset.from_generated(train_size, max_len, seed=42)
        val_ds = ListOpsDataset.from_generated(val_size, max_len, seed=123)
        test_ds = ListOpsDataset.from_generated(test_size, max_len, seed=456)
    else:
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
