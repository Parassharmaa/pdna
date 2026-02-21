"""Variant A: Baseline LTC model — control group for ablation study."""

from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP, FullyConnected


class BaselineLTC(nn.Module):
    """Baseline LTC classifier using AutoNCP wiring.

    This is Variant A in the ablation study — a standard LTC network
    with no pulse, no self-attention, and no idle ticks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 10,
        num_layers: int = 1,
        dropout: float = 0.1,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Build stacked LTC layers
        self.ltc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            # Use FullyConnected wiring for all layers (AutoNCP requires output < units-2)
            wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
            ltc = LTC(
                input_size=layer_input,
                units=wiring,
                return_sequences=True,
                batch_first=True,
                ode_unfolds=ode_unfolds,
            )
            self.ltc_layers.append(ltc)
            if i < num_layers - 1:
                self.dropout_layers.append(nn.Dropout(dropout))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hx: Optional list of hidden states per layer.

        Returns:
            Tuple of (logits, hidden_states) where logits has shape
            (batch, output_size) and hidden_states is a list of
            per-layer hidden tensors.
        """
        if hx is None:
            hx = [None] * self.num_layers

        new_hx = []
        out = x
        for i, ltc in enumerate(self.ltc_layers):
            out, h = ltc(out, hx[i])
            new_hx.append(h)
            if i < self.num_layers - 1:
                out = self.dropout_layers[i](out)

        # Use the last timestep output for classification
        last_out = out[:, -1, :]
        logits = self.classifier(last_out)
        return logits, new_hx
