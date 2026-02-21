# PDNA: Pulse-Driven Neural Architecture

**A neural network with an intrinsic, self-sustaining oscillatory dynamic ("pulse") that maintains useful internal state across temporal gaps in input.**

Built on [CfC (Closed-form Continuous-time)](https://arxiv.org/abs/2106.13898) recurrent networks, augmented with learnable oscillatory dynamics and state-dependent self-attention.

## The Idea

Current sequence models are fundamentally **stateless between inference steps** — when input stops, internal state freezes or decays. PDNA adds a continuous internal "heartbeat" that keeps the hidden state evolving even when input is absent:

```
τ(x) · dh/dt = -h + f(h, x; θ) + α · pulse(t, h) + β · self_attend(h)

pulse(t, h) = A · sin(ω · t + φ(h))      # learnable oscillation
self_attend(h) = W_self · σ(h)            # recurrent self-attention
```

The pulse gives the model (1) an internal sense of elapsed time, (2) active state evolution during input gaps, and (3) a temporal scaffold bridging discontinuities.

## Key Results

**6-variant ablation study** on sequence classification (sMNIST), evaluated under increasing input gap severity:

| Variant | Test Acc | Degradation (0%→30% gap) |
|---------|----------|--------------------------|
| A. Baseline CfC | 97.74% | 72.91% |
| B. CfC + Noise | 97.45% | 73.62% |
| C. CfC + Pulse | 97.95% | 70.26% |
| D. CfC + SelfAttend | 98.05% | 67.84% |
| **E. Full PDNA** | **98.03%** | **68.79%** |

<p align="center">
  <img src="reports/figures/degradation_curves.png" alt="Degradation curves across gap levels" width="85%">
</p>

*Performance under increasing input gaps — the core PDNA hypothesis test. Models with structured internal dynamics (pulse) degrade less when input is interrupted.*

**Highlights:**
- **+0.29%** accuracy over baseline on sMNIST
- **+7.62%** advantage on multi-gap robustness (PDNA 92.68% vs Baseline 85.06%)
- **Structured > Random** confirmed: noise control (B) performs worse than pulse (C)
- Negligible compute overhead: +38% parameters, but **no wall-time increase**

<p align="center">
  <img src="reports/figures/degradation_bars.png" alt="Degradation comparison" width="55%">
</p>

## Architecture Variants

The ablation study isolates each component's contribution:

| Variant | Pulse | Self-Attend | Idle Ticks | Purpose |
|---------|:-----:|:-----------:|:----------:|---------|
| A. Baseline CfC | | | | Control |
| B. CfC + Noise | random | | | Is structure needed, or just any perturbation? |
| C. CfC + Pulse | ✓ | | | Oscillation alone |
| D. CfC + SelfAttend | | ✓ | | Self-attention alone |
| E. Full PDNA | ✓ | ✓ | | Combined architecture |
| F. Full + Idle | ✓ | ✓ | ✓ | Active gap processing |

## Project Structure

```
src/pdna/
├── models/
│   ├── pulse_cfc.py       # Core PulseCfC implementation
│   ├── pulse_ltc.py       # Original PulseLTC cell
│   ├── baseline_ltc.py    # Baseline model (Variant A)
│   ├── noise_ltc.py       # Noise control (Variant B)
│   └── variants.py        # Factory for all 6 variants
├── data/
│   ├── gapped.py          # Gapped-LRA benchmark creator
│   ├── listops.py         # ListOps task
│   └── ...                # Other LRA tasks
├── training/
│   ├── trainer.py         # Training loop with early stopping
│   └── config.py          # YAML-driven experiment config
└── analysis/
    ├── results.py         # Statistical analysis (t-tests, Cohen's d)
    └── visualize.py       # Matplotlib visualization pipeline

configs/default.yaml       # Hyperparameter configuration
scripts/run_fast_ablation.py  # Main experiment script
reports/technical_report.md   # Full results writeup
```

## Getting Started

```bash
# Clone
git clone https://github.com/Parassharmaa/pdna.git
cd pdna

# Install (requires uv — https://docs.astral.sh/uv/)
uv sync

# Run tests
uv run pytest

# Run the ablation study (requires GPU)
uv run python scripts/run_fast_ablation.py

# Generate report from results
uv run python scripts/generate_report.py
```

### Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
model:
  hidden_size: 128
  num_layers: 1
  alpha_init: 0.01      # pulse strength
  beta_init: 0.01       # self-attend strength
  idle_ticks_per_gap: 10

training:
  lr: 1.0e-3
  batch_size: 32
  max_epochs: 50
  early_stopping_patience: 10
  seeds: [42, 123, 456]
```

## Gap Robustness Evaluation

The **Gapped-LRA** benchmark tests temporal robustness by zeroing out portions of the input sequence at test time:

| Gap Level | Description |
|-----------|-------------|
| 0% | Standard evaluation (no gaps) |
| 5% | 5% of timesteps zeroed — mild interruption |
| 15% | 15% zeroed — moderate interruption |
| 30% | 30% zeroed — severe interruption |
| Multi-gap | Multiple scattered gaps throughout sequence |

**Degradation** = (Gap 0% accuracy) − (Gap 30% accuracy). Lower is more robust.

## Detailed Results

See [`reports/technical_report.md`](reports/technical_report.md) for:
- Full ablation tables with confidence intervals
- Statistical significance tests (paired t-test, Cohen's d)
- Per-gap-level accuracy breakdown
- Compute overhead analysis
- Training convergence curves

<details>
<summary>Additional figures</summary>

### Training Convergence
<img src="reports/figures/training_curves.png" alt="Training curves" width="85%">

### Ablation Heatmap
<img src="reports/figures/ablation_heatmap.png" alt="Ablation heatmap" width="55%">

</details>

## Current Status & Limitations

**Success level: Moderate (Promising)**

- Results are directionally positive but not yet statistically significant (n=2 seeds)
- The Adding Problem did not learn above chance with CfC at this scale
- Idle ticks (Variant F) produce identical results to Variant E due to CfC's parallel sequence processing — a sequential architecture would be needed for idle ticks to propagate
- Larger-scale experiments (more seeds, more tasks, longer sequences) are the natural next step

## Hypotheses Tested

| Hypothesis | Result |
|-----------|--------|
| **H1** (State Persistence): Pulse model retains more info across gaps | Supported — PDNA degrades 0.94x vs baseline on sMNIST |
| **H2** (Structured > Random): Pulse beats noise control | Supported — Noise (B) < Baseline < Pulse (C) on sMNIST |
| **H3** (Temporal Encoding): Learned oscillation encodes elapsed time | Partially supported — diverse ω learned, but needs more analysis |

## Tech Stack

- Python 3.10+, managed with [uv](https://docs.astral.sh/uv/)
- [PyTorch](https://pytorch.org/) 2.x
- [ncps](https://github.com/mlech26l/ncps) (Neural Circuit Policies) for CfC base
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for ODE solving

## License

MIT
