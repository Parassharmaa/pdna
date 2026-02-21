"""Unit tests for PulseCfC, NoiseCfC, and all 6 CfC-based ablation variants."""

import pytest
import torch

from pdna.models.pulse_cfc import NoiseCfC, PulseCfC, PulseModule, SelfAttendModule
from pdna.models.variants import Variant, build_variant

BATCH = 4
SEQ_LEN = 20
INPUT_SIZE = 16
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


@pytest.fixture
def gap_mask():
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, 8:12] = True
    return mask


class TestPulseModule:
    def test_output_shape(self):
        mod = PulseModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        t = torch.arange(SEQ_LEN, dtype=torch.float)
        out = mod(h, t)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_nonzero_output(self):
        mod = PulseModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        t = torch.arange(SEQ_LEN, dtype=torch.float)
        out = mod(h, t)
        assert (out != 0).any(), "Pulse output should be non-zero"

    def test_time_varying(self):
        mod = PulseModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        t1 = torch.arange(SEQ_LEN, dtype=torch.float)
        t2 = torch.arange(SEQ_LEN, dtype=torch.float) + 100
        out1 = mod(h, t1)
        out2 = mod(h, t2)
        assert not torch.allclose(out1, out2), "Pulse should vary with time"

    def test_gradient_flow(self):
        mod = PulseModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        t = torch.arange(SEQ_LEN, dtype=torch.float)
        out = mod(h, t)
        out.sum().backward()
        assert mod.alpha.grad is not None and mod.alpha.grad != 0
        assert mod.amplitude.grad is not None
        assert mod.omega.grad is not None
        assert mod.phase_net.weight.grad is not None


class TestSelfAttendModule:
    def test_output_shape(self):
        mod = SelfAttendModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        out = mod(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_gradient_flow(self):
        mod = SelfAttendModule(HIDDEN_SIZE)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
        out = mod(h)
        out.sum().backward()
        assert mod.beta.grad is not None and mod.beta.grad != 0
        assert mod.proj.weight.grad is not None


class TestPulseCfC:
    def test_forward_shape(self, sample_input):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        out, h = model(sample_input)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)
        assert h.shape == (BATCH, HIDDEN_SIZE)

    def test_forward_no_sequences(self, sample_input):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, return_sequences=False)
        out, h = model(sample_input)
        assert out.shape == (BATCH, HIDDEN_SIZE)
        assert h.shape == (BATCH, HIDDEN_SIZE)

    def test_gap_mask_zeros_input(self, sample_input, gap_mask):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        out, h = model(sample_input, gap_mask=gap_mask)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_idle_bounded(self, sample_input, gap_mask):
        model = PulseCfC(
            input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
            enable_idle=True, idle_ticks_per_gap=5,
        )
        out, h = model(sample_input, gap_mask=gap_mask)
        assert h.norm(dim=-1).max() < 1000, "State diverged during idle ticks"

    def test_pulse_only(self, sample_input):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_pulse=True, enable_self_attend=False)
        out, h = model(sample_input)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_self_attend_only(self, sample_input):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_pulse=False, enable_self_attend=True)
        out, h = model(sample_input)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_gradient_flow_full(self, sample_input):
        model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        out, h = model(sample_input)
        out.sum().backward()
        # Check pulse and self-attend gradients
        assert model.pulse.alpha.grad is not None
        assert model.self_attend.beta.grad is not None
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 5, "Expected many gradients"


class TestNoiseCfC:
    def test_forward_shape(self, sample_input):
        model = NoiseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        out, h = model(sample_input)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)

    def test_noise_in_train(self, sample_input):
        torch.manual_seed(42)
        model = NoiseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        model.train()
        out1, _ = model(sample_input)
        out2, _ = model(sample_input)
        # With noise, outputs should differ between calls
        assert not torch.allclose(out1, out2), "Training outputs should differ (noise)"

    def test_no_noise_in_eval(self, sample_input):
        model = NoiseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        model.eval()
        out1, _ = model(sample_input)
        out2, _ = model(sample_input)
        assert torch.allclose(out1, out2), "Eval outputs should be deterministic"


class TestAllVariants:
    @pytest.mark.parametrize("variant", list(Variant))
    def test_variant_builds_and_runs(self, variant, sample_input, gap_mask):
        model = build_variant(variant, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
        logits, hx = model(sample_input, gap_mask=gap_mask)
        assert logits.shape == (BATCH, OUTPUT_SIZE)
        assert len(hx) == 1

    @pytest.mark.parametrize("variant", list(Variant))
    def test_variant_backward_pass(self, variant, sample_input):
        model = build_variant(variant, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
        logits, _ = model(sample_input)
        loss = logits.sum()
        loss.backward()
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

    def test_pulse_vs_noise_differ(self, sample_input):
        torch.manual_seed(42)
        noise_model = NoiseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        torch.manual_seed(42)
        pulse_model = PulseCfC(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, enable_pulse=True, enable_self_attend=False)
        noise_model.train()
        pulse_model.train()
        out_n, _ = noise_model(sample_input)
        out_p, _ = pulse_model(sample_input)
        assert not torch.allclose(out_n, out_p), "Noise and pulse should differ"
