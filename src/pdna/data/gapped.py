"""Gapped-LRA benchmark variant.

Creates input gaps of varying sizes to test robustness to interrupted sequences.
Gap levels: 0%, 5%, 15%, 30%, and multi-gap.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch.utils.data import Dataset


class GapLevel(str, Enum):
    """Gap difficulty levels for Gapped-LRA."""
    NONE = "gap_0"        # 0% gap (standard)
    SMALL = "gap_5"       # 5% gap
    MEDIUM = "gap_15"     # 15% gap
    LARGE = "gap_30"      # 30% gap
    MULTI = "multi_gap"   # Multiple scattered gaps


def create_gap_mask(
    seq_len: int,
    gap_level: GapLevel | str,
    batch_size: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Create a boolean gap mask for a sequence.

    Args:
        seq_len: Sequence length.
        gap_level: Gap difficulty level.
        batch_size: Number of sequences in batch.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask of shape (batch_size, seq_len) where True = gap position.
    """
    if isinstance(gap_level, str):
        gap_level = GapLevel(gap_level)

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    if gap_level == GapLevel.NONE:
        return mask

    # Determine gap fraction
    gap_fractions = {
        GapLevel.SMALL: 0.05,
        GapLevel.MEDIUM: 0.15,
        GapLevel.LARGE: 0.30,
    }

    if gap_level in gap_fractions:
        gap_frac = gap_fractions[gap_level]
        gap_len = max(1, int(seq_len * gap_frac))
        # Place a contiguous gap in the middle of the sequence
        start = (seq_len - gap_len) // 2
        mask[:, start: start + gap_len] = True

    elif gap_level == GapLevel.MULTI:
        # Scatter 3-5 gaps of varying sizes, totaling ~20% of sequence
        n_gaps = 4
        total_gap = int(seq_len * 0.20)
        gap_size = total_gap // n_gaps

        # Distribute gaps evenly across the sequence
        segment_len = seq_len // (n_gaps + 1)
        for i in range(n_gaps):
            start = segment_len * (i + 1) - gap_size // 2
            start = max(0, min(start, seq_len - gap_size))
            mask[:, start: start + gap_size] = True

    return mask


class GappedWrapper(Dataset):
    """Wraps an existing dataset to add gap masks at a given difficulty level."""

    def __init__(
        self,
        base_dataset: Dataset,
        gap_level: GapLevel | str,
        zero_gapped_inputs: bool = True,
    ) -> None:
        """
        Args:
            base_dataset: The underlying LRA dataset.
            gap_level: Gap difficulty level.
            zero_gapped_inputs: If True, zero out input at gap positions
                               (for non-idle variants). If False, keep original.
        """
        self.base_dataset = base_dataset
        self.gap_level = GapLevel(gap_level) if isinstance(gap_level, str) else gap_level
        self.zero_gapped_inputs = zero_gapped_inputs

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Returns (input_tokens, label, gap_mask)."""
        tokens, label = self.base_dataset[idx]
        seq_len = tokens.shape[0]
        gap_mask = create_gap_mask(seq_len, self.gap_level, batch_size=1, seed=idx).squeeze(0)

        if self.zero_gapped_inputs:
            # Zero out input at gap positions
            tokens = tokens.clone()
            tokens[gap_mask] = 0

        return tokens, label, gap_mask
