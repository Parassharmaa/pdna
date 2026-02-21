"""Unit tests for PulseLTCCell, PulseLTC, and all 6 ablation variants."""

import pytest
import torch

from pdna.models.noise_ltc import NoiseLTC
from pdna.models.pulse_ltc import PulseLTC, PulseLTCCell
from pdna.models.variants import Variant, build_variant
from ncps.wirings import FullyConnected

BATCH = 4
SEQ_LEN = 20
INPUT_SIZE = 16
HIDDEN_SIZE = 64  # Smaller for fast tests
OUTPUT_SIZE = 10


@pytest.fixture
def pulse_cell():
    wiring = FullyConnected(units=HIDDEN_SIZE, output_dim=HIDDEN_SIZE)
    return PulseLTCCell(
        wiring=wiring,
        in_features=INPUT_SIZE,
        enable_pulse=True,
        enable_self_attend=True,
    )


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


class TestPulseLTCCell:
    def test_forward_shape(self, pulse_cell):
        inp = torch.randn(BATCH, INPUT_SIZE)
        state = torch.zeros(BATCH, HIDDEN_SIZE)
        out, new_state = pulse_cell(inp, state, t=0.0)
        assert out.shape == (BATCH, HIDDEN_SIZE)
        assert new_state.shape == (BATCH, HIDDEN_SIZE)

    def test_pulse_nonzero(self, pulse_cell):
        state = torch.randn(BATCH, HIDDEN_SIZE)
        t = torch.tensor(1.0)
        pulse_val = pulse_cell.pulse(t, state)
        assert pulse_val.shape == (BATCH, HIDDEN_SIZE)
        assert (pulse_val != 0).any(), "Pulse output should be non-zero"

    def test_self_attend(self, pulse_cell):
        state = torch.randn(BATCH, HIDDEN_SIZE)
        sa_val = pulse_cell.self_attend(state)
        assert sa_val.shape == (BATCH, HIDDEN_SIZE)

    def test_gradient_flow(self, pulse_cell):
        """Gradients should flow through pulse and self-attend components."""
        inp = torch.randn(BATCH, INPUT_SIZE)
        state = torch.zeros(BATCH, HIDDEN_SIZE)
        out, new_state = pulse_cell(inp, state, t=1.0)
        loss = new_state.sum()
        loss.backward()

        assert pulse_cell.alpha.grad is not None
        assert pulse_cell.alpha.grad != 0, "Alpha gradient should be non-zero"
        assert pulse_cell.beta.grad is not None
        assert pulse_cell.beta.grad != 0, "Beta gradient should be non-zero"
        assert pulse_cell.phase_net.weight.grad is not None
        assert pulse_cell.self_attend_proj.weight.grad is not None

    def test_time_varying_output(self, pulse_cell):
        """Output should change with different time values (pulse is time-dependent)."""
        inp = torch.randn(BATCH, INPUT_SIZE)
        state = torch.randn(BATCH, HIDDEN_SIZE)
        _, state_t0 = pulse_cell(inp, state.clone(), t=0.0)
        _, state_t5 = pulse_cell(inp, state.clone(), t=5.0)
        assert not torch.allclose(state_t0, state_t5), "States at different times should differ"

    def test_pulse_disabled(self):
        wiring = FullyConnected(units=HIDDEN_SIZE, output_dim=HIDDEN_SIZE)
        cell = PulseLTCCell(wiring=wiring, in_features=INPUT_SIZE, enable_pulse=False, enable_self_attend=False)
        assert not hasattr(cell, "alpha")
        assert not hasattr(cell, "beta")
        inp = torch.randn(BATCH, INPUT_SIZE)
        state = torch.zeros(BATCH, HIDDEN_SIZE)
        out, new_state = cell(inp, state)
        assert out.shape == (BATCH, HIDDEN_SIZE)


class TestPulseLTC:
    def test_forward_shape(self, sample_input):
        model = PulseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        out, hx = model(sample_input)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)
        assert hx.shape == (BATCH, HIDDEN_SIZE)

    def test_idle_ticks_bounded(self, sample_input):
        """Idle ticks should not cause state divergence."""
        model = PulseLTC(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            enable_idle=True,
            idle_ticks_per_gap=20,
        )
        gap_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        gap_mask[:, 5:15] = True  # 10 gap steps

        out, hx = model(sample_input, gap_mask=gap_mask)
        # State norm should stay bounded (not explode)
        assert hx.norm(dim=-1).max() < 1000, "State diverged during idle ticks"

    def test_idle_vs_no_idle(self, sample_input):
        """Idle model should produce different outputs than non-idle for gapped input."""
        torch.manual_seed(42)
        model_idle = PulseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_idle=True, idle_ticks_per_gap=5)
        torch.manual_seed(42)
        model_no_idle = PulseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_idle=False)

        # Copy weights
        model_no_idle.load_state_dict(model_idle.state_dict(), strict=False)

        gap_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        gap_mask[:, 8:12] = True

        out_idle, _ = model_idle(sample_input, gap_mask=gap_mask)
        out_no_idle, _ = model_no_idle(sample_input, gap_mask=gap_mask)
        assert not torch.allclose(out_idle, out_no_idle), "Idle and non-idle should differ"

    def test_time_advances_during_idle(self):
        model = PulseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_idle=True, idle_ticks_per_gap=5)
        h = torch.zeros(BATCH, HIDDEN_SIZE)
        h_evolved, t_new = model.run_idle_ticks(h, num_ticks=5, t_start=0.0)
        assert t_new == 5.0, f"Expected time 5.0, got {t_new}"


class TestNoiseLTC:
    def test_noise_vs_pulse_differ(self, sample_input):
        """Noise control (B) should produce different outputs than pulse (C)."""
        torch.manual_seed(42)
        noise_model = NoiseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        torch.manual_seed(42)
        pulse_model = PulseLTC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_pulse=True, enable_self_attend=False)

        noise_model.train()
        pulse_model.train()
        out_noise, _ = noise_model(sample_input)
        out_pulse, _ = pulse_model(sample_input)
        assert not torch.allclose(out_noise, out_pulse), "Noise and pulse should differ"


class TestAllVariants:
    @pytest.mark.parametrize("variant", list(Variant))
    def test_variant_builds_and_runs(self, variant, sample_input):
        model = build_variant(variant, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
        gap_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        gap_mask[:, 8:12] = True
        logits, hx = model(sample_input, gap_mask=gap_mask)
        assert logits.shape == (BATCH, OUTPUT_SIZE)
        assert len(hx) == 1

    @pytest.mark.parametrize("variant", list(Variant))
    def test_variant_backward_pass(self, variant, sample_input):
        model = build_variant(variant, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
        logits, _ = model(sample_input)
        loss = logits.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"No gradients for variant {variant}"

    @pytest.mark.parametrize("variant", list(Variant))
    def test_variant_multi_layer(self, variant, sample_input):
        model = build_variant(
            variant, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layers=2
        )
        logits, hx = model(sample_input)
        assert logits.shape == (BATCH, OUTPUT_SIZE)
        assert len(hx) == 2
