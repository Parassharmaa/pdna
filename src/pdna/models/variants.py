"""Factory for all 6 ablation variants with unified interface."""

from __future__ import annotations

from enum import Enum

import torch.nn as nn

from pdna.models.baseline_ltc import BaselineLTC
from pdna.models.noise_ltc import NoiseLTC
from pdna.models.pulse_ltc import PulseLTC


class Variant(str, Enum):
    """The 6 ablation variants."""

    A = "baseline"           # Baseline LTC
    B = "noise"              # LTC + Noise
    C = "pulse"              # LTC + Pulse only
    D = "self_attend"        # LTC + SelfAttend only
    E = "full_pdna"          # Full PDNA (pulse + self-attend)
    F = "full_idle"          # Full + Idle ticks


class VariantModel(nn.Module):
    """Unified wrapper around any variant's RNN backbone + classification head.

    All variants share the same hidden_size, num_layers, dropout, and output_size.
    Only the backbone architecture differs.
    """

    def __init__(
        self,
        variant: Variant | str,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 10,
        num_layers: int = 1,
        dropout: float = 0.1,
        alpha_init: float = 0.01,
        beta_init: float = 0.01,
        idle_ticks_per_gap: int = 10,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        if isinstance(variant, str):
            variant = Variant(variant)
        self.variant = variant
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build stacked backbone layers
        self.backbones = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            backbone = _build_backbone(
                variant=variant,
                input_size=layer_input,
                hidden_size=hidden_size,
                alpha_init=alpha_init,
                beta_init=beta_init,
                idle_ticks_per_gap=idle_ticks_per_gap,
                ode_unfolds=ode_unfolds,
            )
            self.backbones.append(backbone)
            if i < num_layers - 1:
                self.dropout_layers.append(nn.Dropout(dropout))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, hx=None, gap_mask=None):
        """Forward pass through stacked backbone layers + classifier.

        Args:
            x: Input (batch, seq_len, input_size).
            hx: Optional list of hidden states per layer.
            gap_mask: Optional boolean mask for gap positions.

        Returns:
            Tuple of (logits, hidden_states_list).
        """
        if hx is None:
            hx = [None] * self.num_layers

        new_hx = []
        out = x
        for i, backbone in enumerate(self.backbones):
            if isinstance(backbone, PulseLTC):
                out, h = backbone(out, hx[i], gap_mask=gap_mask)
            elif isinstance(backbone, NoiseLTC):
                out, h = backbone(out, hx[i], gap_mask=gap_mask)
            else:
                # BaselineLTC-style LTC
                out, h = backbone(out, hx[i])
            new_hx.append(h)
            if i < self.num_layers - 1:
                out = self.dropout_layers[i](out)

        # Last timestep for classification
        last_out = out[:, -1, :]
        logits = self.classifier(last_out)
        return logits, new_hx


def _build_backbone(
    variant: Variant,
    input_size: int,
    hidden_size: int,
    alpha_init: float,
    beta_init: float,
    idle_ticks_per_gap: int,
    ode_unfolds: int,
) -> nn.Module:
    """Build a single backbone layer for a given variant."""
    from ncps.torch import LTC
    from ncps.wirings import FullyConnected

    if variant == Variant.A:
        # Baseline: standard LTC, no pulse, no self-attend
        wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
        return LTC(
            input_size=input_size,
            units=wiring,
            return_sequences=True,
            batch_first=True,
            ode_unfolds=ode_unfolds,
        )

    elif variant == Variant.B:
        # LTC + Noise
        return NoiseLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            noise_scale_init=alpha_init,
            return_sequences=True,
            ode_unfolds=ode_unfolds,
        )

    elif variant == Variant.C:
        # LTC + Pulse only
        return PulseLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=False,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            ode_unfolds=ode_unfolds,
        )

    elif variant == Variant.D:
        # LTC + SelfAttend only
        return PulseLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=False,
            enable_self_attend=True,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            ode_unfolds=ode_unfolds,
        )

    elif variant == Variant.E:
        # Full PDNA (pulse + self-attend)
        return PulseLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=True,
            enable_idle=False,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            ode_unfolds=ode_unfolds,
        )

    elif variant == Variant.F:
        # Full + Idle ticks
        return PulseLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=True,
            enable_idle=True,
            idle_ticks_per_gap=idle_ticks_per_gap,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            ode_unfolds=ode_unfolds,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


def build_variant(
    variant: Variant | str,
    input_size: int,
    hidden_size: int = 128,
    output_size: int = 10,
    num_layers: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> VariantModel:
    """Convenience function to build a variant model."""
    return VariantModel(
        variant=variant,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
    )
