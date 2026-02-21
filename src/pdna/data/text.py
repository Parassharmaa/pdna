"""IMDB byte-level text classification for LRA benchmark.

Binary sentiment classification on IMDB reviews at byte level.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

VOCAB_SIZE = 256  # Byte-level: 0-255
PAD_ID = 0


class IMDBByteDataset(Dataset):
    """IMDB reviews encoded at byte level for LRA text classification."""

    def __init__(self, texts: list[str], labels: list[int], max_len: int = 4096) -> None:
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        # Byte-level encoding
        byte_ids = [b for b in text.encode("utf-8", errors="replace")]
        # Truncate and pad
        byte_ids = byte_ids[: self.max_len]
        byte_ids = byte_ids + [PAD_ID] * (self.max_len - len(byte_ids))
        return torch.tensor(byte_ids, dtype=torch.long), label


def _load_imdb_from_dir(data_dir: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load IMDB from extracted directory structure."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for split, texts, labels in [("train", train_texts, train_labels), ("test", test_texts, test_labels)]:
        for sentiment, label in [("pos", 1), ("neg", 0)]:
            folder = data_dir / "aclImdb" / split / sentiment
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix == ".txt":
                    texts.append(f.read_text(encoding="utf-8", errors="replace"))
                    labels.append(label)

    return train_texts, train_labels, test_texts, test_labels


def _download_imdb(data_dir: Path) -> None:
    """Download and extract IMDB dataset."""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_dir / "aclImdb_v1.tar.gz"

    if not (data_dir / "aclImdb").exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading IMDB to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
        os.remove(tar_path)


def get_imdb_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    max_len: int = 4096,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders for IMDB byte-level classification."""
    data_path = Path(data_dir)
    _download_imdb(data_path)

    train_texts, train_labels, test_texts, test_labels = _load_imdb_from_dir(data_path)

    # Split train into train/val (90/10)
    n_val = len(train_texts) // 10
    val_texts, val_labels = train_texts[:n_val], train_labels[:n_val]
    train_texts, train_labels = train_texts[n_val:], train_labels[n_val:]

    train_ds = IMDBByteDataset(train_texts, train_labels, max_len)
    val_ds = IMDBByteDataset(val_texts, val_labels, max_len)
    test_ds = IMDBByteDataset(test_texts, test_labels, max_len)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
