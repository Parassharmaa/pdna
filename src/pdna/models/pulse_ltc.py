"""PulseLTCCell and PulseLTC: LTC augmented with oscillatory pulse and self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from ncps.torch import LTCCell
from ncps.wirings import FullyConnected


class PulseLTCCell(nn.Module):
    """LTC cell augmented with oscillatory pulse and self-attention.

    Implements the PDNA core equation:
        tau(x) * dh/dt = -h + f(h, x; theta) + alpha * pulse(t, h) + beta * self_attend(h)

    where:
        pulse(t, h) = A * sin(omega * t + phi(h))
        self_attend(h) = W_self * sigma(h)
    """

    def __init__(
        self,
        wiring: FullyConnected,
        in_features: int | None = None,
        enable_pulse: bool = True,
        enable_self_attend: bool = True,
        alpha_init: float = 0.01,
        beta_init: float = 0.01,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        if in_features is not None:
            wiring.build(in_features)

        self._wiring = wiring
        self._enable_pulse = enable_pulse
        self._enable_self_attend = enable_self_attend
        state_size = wiring.units

        # Underlying LTC cell
        self.ltc_cell = LTCCell(
            wiring=wiring,
            in_features=in_features,
            ode_unfolds=ode_unfolds,
            implicit_param_constraints=True,
        )

        # Pulse parameters: A * sin(omega * t + phi(h))
        if enable_pulse:
            self.amplitude = nn.Parameter(torch.randn(state_size) * 0.1)
            # Diverse omega initialization: log-uniform across frequencies
            self.omega = nn.Parameter(
                torch.exp(torch.linspace(math.log(0.1), math.log(10.0), state_size))
            )
            # State-dependent phase shift
            self.phase_net = nn.Linear(state_size, state_size)
            # Pulse strength (init small for stability)
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Self-attention on own state: W_self * sigma(h)
        if enable_self_attend:
            self.self_attend_proj = nn.Linear(state_size, state_size, bias=False)
            # Self-attend strength (init small for stability)
            self.beta = nn.Parameter(torch.tensor(beta_init))

    @property
    def state_size(self) -> int:
        return self._wiring.units

    @property
    def output_size(self) -> int:
        return self._wiring.output_dim

    def pulse(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute oscillatory pulse: A * sin(omega * t + phi(h))."""
        phi = self.phase_net(h)  # (batch, state_size)
        oscillation = self.amplitude * torch.sin(self.omega * t + phi)
        return oscillation

    def self_attend(self, h: torch.Tensor) -> torch.Tensor:
        """Compute self-attention: W_self * sigmoid(h)."""
        return self.self_attend_proj(torch.sigmoid(h))

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        t: float | torch.Tensor = 0.0,
        elapsed_time: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward pass.

        Args:
            inputs: Input tensor (batch, input_size).
            state: Hidden state (batch, state_size).
            t: Current global time (scalar or tensor).
            elapsed_time: Time elapsed since last step.

        Returns:
            Tuple of (output, new_state).
        """
        # Core LTC dynamics
        output, new_state = self.ltc_cell(inputs, state, elapsed_time)

        # Add pulse perturbation
        if self._enable_pulse:
            t_tensor = torch.as_tensor(t, dtype=new_state.dtype, device=new_state.device)
            pulse_signal = self.pulse(t_tensor, new_state)
            new_state = new_state + self.alpha * pulse_signal

        # Add self-attention perturbation
        if self._enable_self_attend:
            sa_signal = self.self_attend(new_state)
            new_state = new_state + self.beta * sa_signal

        # Re-compute output from modified state (slice motor neurons)
        output = new_state
        if self.output_size < self.state_size:
            output = output[:, : self.output_size]

        return output, new_state


class PulseLTC(nn.Module):
    """RNN wrapper for PulseLTCCell that processes sequences with global time tracking."""

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
        dt: float = 1.0,
        return_sequences: bool = True,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_idle = enable_idle
        self.idle_ticks_per_gap = idle_ticks_per_gap
        self.dt = dt
        self.return_sequences = return_sequences

        wiring = FullyConnected(units=hidden_size, output_dim=hidden_size)
        self.cell = PulseLTCCell(
            wiring=wiring,
            in_features=input_size,
            enable_pulse=enable_pulse,
            enable_self_attend=enable_self_attend,
            alpha_init=alpha_init,
            beta_init=beta_init,
            ode_unfolds=ode_unfolds,
        )

    @property
    def state_size(self) -> int:
        return self.hidden_size

    def run_idle_ticks(
        self,
        h: torch.Tensor,
        num_ticks: int,
        t_start: float,
    ) -> tuple[torch.Tensor, float]:
        """Evolve state using only pulse + self-attend (zero input) for idle ticks.

        Args:
            h: Hidden state (batch, hidden_size).
            num_ticks: Number of idle ticks to run.
            t_start: Starting global time.

        Returns:
            Tuple of (evolved_hidden_state, new_time).
        """
        zero_input = torch.zeros(h.shape[0], self.cell.ltc_cell.sensory_size, device=h.device, dtype=h.dtype)
        t = t_start
        for _ in range(num_ticks):
            _, h = self.cell(zero_input, h, t=t, elapsed_time=self.dt)
            t += self.dt
        return h, t

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        gap_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input sequence.

        Args:
            x: Input (batch, seq_len, input_size).
            hx: Initial hidden state (batch, hidden_size) or None.
            gap_mask: Optional boolean mask (batch, seq_len) where True = gap position.
                      When enable_idle=True, idle ticks are run at gap positions.

        Returns:
            Tuple of (output_sequence_or_last, final_hidden_state).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        if hx is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        else:
            h = hx

        t = 0.0
        outputs = []

        for step in range(seq_len):
            inp = x[:, step]

            # Check if this step is a gap position
            is_gap = gap_mask is not None and gap_mask[:, step].any()

            if is_gap and self.enable_idle:
                # Run idle ticks during gap (zero-input, pulse+self-attend still active)
                h, t = self.run_idle_ticks(h, self.idle_ticks_per_gap, t)
                # Still produce an output for this timestep
                out = h[:, : self.cell.output_size]
            else:
                out, h = self.cell(inp, h, t=t, elapsed_time=self.dt)
                t += self.dt

            if self.return_sequences:
                outputs.append(out)

        if self.return_sequences:
            readout = torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
        else:
            readout = out

        return readout, h
