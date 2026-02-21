"""Variant B: LTC + Noise — random perturbation control for ablation study."""

from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import LTCCell
from ncps.wirings import FullyConnected


class NoiseLTCCell(nn.Module):
    """LTC cell with random noise perturbation (no structure).

    Critical control: if structured pulse (Variant C) performs similarly to
    random noise, then the pulse structure doesn't matter.
    """

    def __init__(
        self,
        wiring: FullyConnected,
        in_features: int | None = None,
        noise_scale_init: float = 0.01,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        if in_features is not None:
            wiring.build(in_features)

        self._wiring = wiring
        self.ltc_cell = LTCCell(
            wiring=wiring,
            in_features=in_features,
            ode_unfolds=ode_unfolds,
            implicit_param_constraints=True,
        )
        # Learnable noise scale — matches alpha in PulseLTCCell
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale_init))

    @property
    def state_size(self) -> int:
        return self._wiring.units

    @property
    def output_size(self) -> int:
        return self._wiring.output_dim

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        elapsed_time: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, new_state = self.ltc_cell(inputs, state, elapsed_time)

        # Add random noise perturbation (only during training)
        if self.training:
            noise = torch.randn_like(new_state) * self.noise_scale
            new_state = new_state + noise

        # Re-compute output from modified state
        output = new_state
        if self.output_size < self.state_size:
            output = output[:, : self.output_size]

        return output, new_state


class NoiseLTC(nn.Module):
    """RNN wrapper for NoiseLTCCell — Variant B."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        noise_scale_init: float = 0.01,
        return_sequences: bool = True,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
        self.cell = NoiseLTCCell(
            wiring=wiring,
            in_features=input_size,
            noise_scale_init=noise_scale_init,
            ode_unfolds=ode_unfolds,
        )

    @property
    def state_size(self) -> int:
        return self.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        gap_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        else:
            h = hx

        outputs = []
        for step in range(seq_len):
            out, h = self.cell(x[:, step], h)
            if self.return_sequences:
                outputs.append(out)

        if self.return_sequences:
            readout = torch.stack(outputs, dim=1)
        else:
            readout = out

        return readout, h
