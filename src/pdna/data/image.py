"""CIFAR-10 sequential image classification for LRA benchmark.

Flattens 32x32 grayscale images into sequences of 1024 pixel tokens.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CIFAR10SequenceDataset(Dataset):
    """CIFAR-10 as a sequence classification task (flattened grayscale pixels)."""

    def __init__(self, train: bool = True, data_dir: str = "data") -> None:
        try:
            from torchvision.datasets import CIFAR10
        except ImportError:
            raise ImportError("torchvision required: pip install torchvision")

        self.dataset = CIFAR10(root=data_dir, train=train, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        # Convert PIL image to grayscale numpy array
        img_array = np.array(img)
        # RGB to grayscale: 0.2989*R + 0.5870*G + 0.1140*B
        gray = (0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2])
        # Flatten to 1D sequence of 1024 pixels, quantize to [0, 255]
        pixels = gray.flatten().astype(np.uint8)
        return torch.tensor(pixels, dtype=torch.long), label


def get_cifar10_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders for CIFAR-10 sequential task.

    Splits training set 45K/5K for train/val, uses standard test set.
    """
    full_train = CIFAR10SequenceDataset(train=True, data_dir=data_dir)
    test_ds = CIFAR10SequenceDataset(train=False, data_dir=data_dir)

    # Split train into train/val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
