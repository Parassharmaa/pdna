"""Factory for all 6 ablation variants with unified interface.

Uses CfC (Closed-form Continuous-time) as the backbone instead of LTC
for ~20x faster training while preserving continuous-time dynamics.
"""

from __future__ import annotations

from enum import Enum

import torch.nn as nn
from ncps.torch import CfC

from pdna.models.pulse_cfc import NoiseCfC, PulseCfC


class Variant(str, Enum):
    """The 6 ablation variants."""

    A = "baseline"           # Baseline CfC
    B = "noise"              # CfC + Noise
    C = "pulse"              # CfC + Pulse only
    D = "self_attend"        # CfC + SelfAttend only
    E = "full_pdna"          # Full PDNA (pulse + self-attend)
    F = "full_idle"          # Full + Idle ticks


class VariantModel(nn.Module):
    """Unified wrapper around any variant's backbone + classification head."""

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
        backbone_units: int = 128,
        backbone_layers: int = 1,
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
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
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
        """Forward pass through stacked backbone layers + classifier."""
        if hx is None:
            hx = [None] * self.num_layers

        new_hx = []
        out = x
        for i, backbone in enumerate(self.backbones):
            if isinstance(backbone, (PulseCfC, NoiseCfC)):
                out, h = backbone(out, hx[i], gap_mask=gap_mask)
            else:
                # Baseline CfC — zero out gaps manually
                if gap_mask is not None:
                    out_masked = out * (~gap_mask).unsqueeze(-1).float()
                else:
                    out_masked = out
                out, h = backbone(out_masked, hx[i])
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
    backbone_units: int,
    backbone_layers: int,
) -> nn.Module:
    """Build a single backbone layer for a given variant."""

    if variant == Variant.A:
        # Baseline CfC — no pulse, no self-attend
        return CfC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

    elif variant == Variant.B:
        return NoiseCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            noise_scale_init=alpha_init,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

    elif variant == Variant.C:
        return PulseCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=False,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

    elif variant == Variant.D:
        return PulseCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=False,
            enable_self_attend=True,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

    elif variant == Variant.E:
        return PulseCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=True,
            enable_idle=False,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

    elif variant == Variant.F:
        return PulseCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            enable_pulse=True,
            enable_self_attend=True,
            enable_idle=True,
            idle_ticks_per_gap=idle_ticks_per_gap,
            alpha_init=alpha_init,
            beta_init=beta_init,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
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
