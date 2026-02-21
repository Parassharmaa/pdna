"""Pathfinder dataset for LRA benchmark.

Binary classification: are two endpoints connected by a path?
32x32 grayscale images flattened to 1024-length sequences.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VOCAB_SIZE = 256  # 8-bit pixel intensities


class PathfinderDataset(Dataset):
    """Pathfinder dataset — synthetic images with connected/disconnected paths."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            images: Array of shape (N, 1024) with pixel values [0, 255].
            labels: Array of shape (N,) with labels {0, 1}.
        """
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pixels = torch.tensor(self.images[idx], dtype=torch.long)
        label = int(self.labels[idx])
        return pixels, label

    @classmethod
    def from_directory(cls, data_dir: str | Path, difficulty: str = "curv_contour_length_14") -> PathfinderDataset:
        """Load from LRA release directory structure.

        Expected layout: data_dir/pathfinder32/{difficulty}/metadata/
        """
        base = Path(data_dir) / "pathfinder32" / difficulty
        metadata_dir = base / "metadata"

        images = []
        labels = []

        for npy_file in sorted(metadata_dir.glob("*.npy")):
            meta = np.load(npy_file, allow_pickle=True)
            for entry in meta:
                img_dir = entry[0] if isinstance(entry[0], str) else str(entry[0])
                img_file = entry[1] if isinstance(entry[1], str) else str(entry[1])
                label = int(entry[3])
                img_path = base / img_dir / img_file
                if img_path.exists():
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert("L")).flatten()
                    images.append(img)
                    labels.append(label)

        return cls(np.array(images), np.array(labels))


def _generate_synthetic_pathfinder(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple synthetic pathfinder-like data for development/testing.

    Creates 32x32 images with random lines — not the real task, but useful
    for pipeline testing when real data isn't available.
    """
    rng = np.random.RandomState(seed)
    images = rng.randint(0, 50, (n_samples, 1024), dtype=np.uint8)
    labels = rng.randint(0, 2, (n_samples,))

    # Add some signal: for positive examples, add a bright line
    for i in range(n_samples):
        if labels[i] == 1:
            # Add a simple path (bright pixels along a row)
            row = rng.randint(0, 32)
            images[i, row * 32: (row + 1) * 32] = 200

    return images, labels


def get_pathfinder_dataloaders(
    data_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    generate: bool = True,
    train_size: int = 160000,
    val_size: int = 20000,
    test_size: int = 20000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders for Pathfinder."""
    if data_dir and os.path.exists(Path(data_dir) / "pathfinder32"):
        full_ds = PathfinderDataset.from_directory(data_dir)
        # Split 80/10/10
        n = len(full_ds)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n - n_train - n_val],
            generator=torch.Generator().manual_seed(42),
        )
    elif generate:
        train_imgs, train_labels = _generate_synthetic_pathfinder(train_size, seed=42)
        val_imgs, val_labels = _generate_synthetic_pathfinder(val_size, seed=123)
        test_imgs, test_labels = _generate_synthetic_pathfinder(test_size, seed=456)
        train_ds = PathfinderDataset(train_imgs, train_labels)
        val_ds = PathfinderDataset(val_imgs, val_labels)
        test_ds = PathfinderDataset(test_imgs, test_labels)
    else:
        raise FileNotFoundError(f"Pathfinder data not found in {data_dir}")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
