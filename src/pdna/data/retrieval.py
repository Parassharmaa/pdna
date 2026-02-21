"""AAN Document Retrieval dataset for LRA benchmark.

Binary classification: does document 1 cite document 2?
Two documents encoded at byte level and concatenated.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VOCAB_SIZE = 256
PAD_ID = 0


class AANRetrievalDataset(Dataset):
    """AAN document retrieval — byte-level dual document classification."""

    def __init__(
        self,
        doc1_list: list[str],
        doc2_list: list[str],
        labels: list[int],
        max_len_per_doc: int = 4000,
    ) -> None:
        self.doc1_list = doc1_list
        self.doc2_list = doc2_list
        self.labels = labels
        self.max_len_per_doc = max_len_per_doc

    def __len__(self) -> int:
        return len(self.labels)

    def _encode(self, text: str) -> list[int]:
        byte_ids = list(text.encode("utf-8", errors="replace"))
        byte_ids = byte_ids[: self.max_len_per_doc]
        byte_ids = byte_ids + [PAD_ID] * (self.max_len_per_doc - len(byte_ids))
        return byte_ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        doc1_ids = self._encode(self.doc1_list[idx])
        doc2_ids = self._encode(self.doc2_list[idx])
        # Concatenate both documents
        combined = doc1_ids + doc2_ids
        return torch.tensor(combined, dtype=torch.long), self.labels[idx]

    @classmethod
    def from_tsv(cls, path: str | Path, max_len_per_doc: int = 4000) -> AANRetrievalDataset:
        """Load from LRA release TSV format."""
        doc1_list, doc2_list, labels = [], [], []
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 5:
                    labels.append(int(float(row[0])))
                    doc1_list.append(row[3])
                    doc2_list.append(row[4])
        return cls(doc1_list, doc2_list, labels, max_len_per_doc)


def _generate_synthetic_retrieval(n_samples: int, max_len: int = 512, seed: int = 42) -> AANRetrievalDataset:
    """Generate synthetic retrieval data for development/testing."""
    rng = np.random.RandomState(seed)
    doc1_list, doc2_list, labels = [], [], []
    for _ in range(n_samples):
        # Random byte strings
        d1 = bytes(rng.randint(32, 127, size=rng.randint(100, max_len))).decode("ascii")
        d2 = bytes(rng.randint(32, 127, size=rng.randint(100, max_len))).decode("ascii")
        doc1_list.append(d1)
        doc2_list.append(d2)
        labels.append(rng.randint(0, 2))
    return AANRetrievalDataset(doc1_list, doc2_list, labels, max_len)


def get_retrieval_dataloaders(
    data_dir: str | Path | None = None,
    batch_size: int = 32,
    max_len_per_doc: int = 4000,
    num_workers: int = 0,
    generate: bool = True,
    train_size: int = 10000,
    val_size: int = 2000,
    test_size: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders for AAN retrieval."""
    if data_dir and os.path.exists(Path(data_dir) / "aan"):
        aan_dir = Path(data_dir) / "aan"
        # Find TSV files
        train_files = sorted(aan_dir.glob("*.train.tsv"))
        val_files = sorted(aan_dir.glob("*.eval.tsv"))
        test_files = sorted(aan_dir.glob("*.test.tsv"))

        if train_files:
            train_ds = AANRetrievalDataset.from_tsv(train_files[0], max_len_per_doc)
            val_ds = AANRetrievalDataset.from_tsv(val_files[0], max_len_per_doc)
            test_ds = AANRetrievalDataset.from_tsv(test_files[0], max_len_per_doc)
        else:
            raise FileNotFoundError(f"No TSV files found in {aan_dir}")
    elif generate:
        train_ds = _generate_synthetic_retrieval(train_size, max_len=512, seed=42)
        val_ds = _generate_synthetic_retrieval(val_size, max_len=512, seed=123)
        test_ds = _generate_synthetic_retrieval(test_size, max_len=512, seed=456)
    else:
        raise FileNotFoundError(f"AAN data not found in {data_dir}")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
