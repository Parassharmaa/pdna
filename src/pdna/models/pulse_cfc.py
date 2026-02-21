"""PulseCfC: CfC (Closed-form Continuous-time) augmented with oscillatory pulse and self-attention.

CfC is the closed-form solution of LTC dynamics — same continuous-time properties,
but ~20x faster because it eliminates the iterative ODE solver.

The pulse and self-attend are applied as sequence-parallel transformations on
the hidden state output, making the entire model GPU-friendly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from ncps.torch import CfC


class PulseModule(nn.Module):
    """Oscillatory pulse generator applied in parallel across all timesteps.

    pulse(t, h) = A * sin(omega * t + phi(h))

    Applied to a full sequence (batch, seq_len, hidden_size) at once.
    """

    def __init__(self, hidden_size: int, alpha_init: float = 0.01) -> None:
        super().__init__()
        self.amplitude = nn.Parameter(torch.randn(hidden_size) * 0.1)
        # Diverse omega initialization: log-uniform across frequencies
        self.omega = nn.Parameter(
            torch.exp(torch.linspace(math.log(0.1), math.log(10.0), hidden_size))
        )
        self.phase_net = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, h: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Apply pulse to hidden states.

        Args:
            h: Hidden states (batch, seq_len, hidden_size).
            time_steps: Time values (seq_len,) or (batch, seq_len).

        Returns:
            Pulse signal (batch, seq_len, hidden_size).
        """
        phi = self.phase_net(h)  # (batch, seq_len, hidden_size)
        # Expand time_steps to (1, seq_len, 1) for broadcasting
        if time_steps.dim() == 1:
            t = time_steps.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        else:
            t = time_steps.unsqueeze(-1)  # (batch, seq_len, 1)
        oscillation = self.amplitude * torch.sin(self.omega * t + phi)
        return self.alpha * oscillation


class SelfAttendModule(nn.Module):
    """Self-attention on own state: W_self * sigma(h).

    Applied in parallel across all timesteps.
    """

    def __init__(self, hidden_size: int, beta_init: float = 0.01) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to hidden states.

        Args:
            h: Hidden states (batch, seq_len, hidden_size).

        Returns:
            Self-attend signal (batch, seq_len, hidden_size).
        """
        return self.beta * self.proj(torch.sigmoid(h))


class PulseCfC(nn.Module):
    """CfC backbone augmented with pulse and self-attend, fully GPU-parallel.

    Architecture:
        1. CfC processes the full sequence (parallel, fast)
        2. Pulse signal added to hidden outputs (parallel across time)
        3. Self-attend signal added to hidden outputs (parallel across time)
        4. For idle ticks: separate pass with zero input through CfC
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        enable_pulse: bool = True,
        enable_self_attend: bool = True,
        enable_idle: bool = False,
        idle_ticks_per_gap: int = 10,
        alpha_init: float = 0.01,
        beta_init: float = 0.01,
        return_sequences: bool = True,
        backbone_units: int = 128,
        backbone_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.enable_pulse = enable_pulse
        self.enable_self_attend = enable_self_attend
        self.enable_idle = enable_idle
        self.idle_ticks_per_gap = idle_ticks_per_gap
        self.return_sequences = return_sequences

        # CfC backbone — closed-form continuous-time, processes full sequence
        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )

        # Pulse module
        if enable_pulse:
            self.pulse = PulseModule(hidden_size, alpha_init)

        # Self-attend module
        if enable_self_attend:
            self.self_attend = SelfAttendModule(hidden_size, beta_init)

    @property
    def state_size(self) -> int:
        return self.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        gap_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input (batch, seq_len, input_size).
            hx: Initial hidden state (batch, hidden_size) or None.
            gap_mask: Optional boolean mask (batch, seq_len) where True = gap.

        Returns:
            Tuple of (output, final_hidden_state).
        """
        batch_size, seq_len, _ = x.shape

        # Zero out input at gap positions (for all variants except idle)
        if gap_mask is not None and not self.enable_idle:
            x = x * (~gap_mask).unsqueeze(-1).float()

        # CfC forward pass — processes full sequence in parallel
        h_seq, h_final = self.cfc(x, hx)  # h_seq: (batch, seq_len, hidden)

        # Time steps for pulse (simple linear time)
        time_steps = torch.arange(seq_len, dtype=x.dtype, device=x.device)

        # Add pulse signal (parallel across all timesteps)
        if self.enable_pulse:
            h_seq = h_seq + self.pulse(h_seq, time_steps)

        # Add self-attend signal (parallel across all timesteps)
        if self.enable_self_attend:
            h_seq = h_seq + self.self_attend(h_seq)

        # Handle idle ticks at gap positions
        if self.enable_idle and gap_mask is not None and gap_mask.any():
            h_seq = self._apply_idle_ticks(h_seq, gap_mask, time_steps)

        # Output
        if self.return_sequences:
            return h_seq, h_seq[:, -1, :]
        else:
            return h_seq[:, -1, :], h_seq[:, -1, :]

    def _apply_idle_ticks(
        self,
        h_seq: torch.Tensor,
        gap_mask: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Run extra idle ticks at gap positions to evolve state without input.

        Uses the CfC cell directly for idle evolution.
        """
        batch_size, seq_len, hidden = h_seq.shape
        zero_input = torch.zeros(batch_size, self.input_size, device=h_seq.device, dtype=h_seq.dtype)

        # For each gap position, run idle ticks to evolve the hidden state
        # We process contiguous gap regions efficiently
        for step in range(seq_len):
            if gap_mask[:, step].any():
                # Get hidden state at this position
                h = h_seq[:, step, :].clone()
                # Run idle ticks through CfC cell
                idle_input = zero_input.unsqueeze(1).expand(-1, self.idle_ticks_per_gap, -1)
                idle_out, h_evolved = self.cfc(idle_input, h)
                # Apply pulse and self-attend to evolved state
                if self.enable_pulse:
                    t_idle = time_steps[step:step+1] + torch.arange(self.idle_ticks_per_gap, device=h.device).float()
                    h_evolved_seq = idle_out
                    pulse_sig = self.pulse(h_evolved_seq, t_idle)
                    h_evolved = h_evolved + pulse_sig[:, -1, :]
                if self.enable_self_attend:
                    sa = self.self_attend(h_evolved.unsqueeze(1))
                    h_evolved = h_evolved + sa.squeeze(1)
                # Replace gap position with evolved state
                h_seq = h_seq.clone()
                h_seq[:, step, :] = h_evolved

        return h_seq


class NoiseCfC(nn.Module):
    """CfC with random noise perturbation — Variant B control."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        noise_scale_init: float = 0.01,
        return_sequences: bool = True,
        backbone_units: int = 128,
        backbone_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
        )
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale_init))

    @property
    def state_size(self) -> int:
        return self.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        gap_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gap_mask is not None:
            x = x * (~gap_mask).unsqueeze(-1).float()

        h_seq, h_final = self.cfc(x, hx)

        if self.training:
            noise = torch.randn_like(h_seq) * self.noise_scale
            h_seq = h_seq + noise

        if self.return_sequences:
            return h_seq, h_seq[:, -1, :]
        else:
            return h_seq[:, -1, :], h_seq[:, -1, :]
