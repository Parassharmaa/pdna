# Pulse-Driven Neural Architecture (PDNA) — POC Plan

## Project Codename: "Heartbeat"

**Objective:** Build a minimal proof-of-concept that demonstrates a neural network with an intrinsic oscillatory "pulse" — a self-sustaining internal temporal dynamic that maintains and evolves state *independently of input*, and show that this persistent state improves performance on tasks requiring temporal continuity, anticipation, and state maintenance.

---

## 1. ARCHITECTURAL DESIGN (Minimal POC Version)

### 1.1 Core Idea

We take a Liquid Time-Constant (LTC) network — which already models continuous-time neural dynamics via ODEs — and augment it with an **intrinsic oscillatory component** (the "pulse"). The standard LTC neuron evolves its state only when driven by input. Our modification adds a self-sustaining oscillation that keeps the state evolving even during input gaps.

### 1.2 Mathematical Formulation

**Standard LTC neuron:**
```
τ(x) · dh/dt = -h + f(h, x; θ)
```
Where `h` is hidden state, `x` is input, `τ(x)` is an input-dependent time constant, and `f` is a learned nonlinear function.

**Pulse-augmented LTC neuron (our modification):**
```
τ(x) · dh/dt = -h + f(h, x; θ) + α · pulse(t, h) + β · self_attend(h)
```

Where:
- `pulse(t, h)` = intrinsic oscillatory term (the heartbeat)
- `self_attend(h)` = state self-referencing term (the model "thinking about" its own state)
- `α, β` = learnable scaling parameters

**Pulse function (simplest version for POC):**
```
pulse(t, h) = A · sin(ω · t + φ(h))
```

Where:
- `A` = learnable amplitude vector (per neuron)
- `ω` = learnable base frequency vector (per neuron — different neurons oscillate at different rates)
- `φ(h)` = state-dependent phase shift (makes the oscillation responsive to current state)

**Self-attend function (simplest version for POC):**
```
self_attend(h) = W_self · σ(h)
```

Where `W_self` is a learnable weight matrix and `σ` is a nonlinearity. This lets the network's state evolution be influenced by its own current state, even without input.

### 1.3 Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         PULSE-AUGMENTED LTC          │
                    │                                       │
  Input x(t) ──────►  ┌───────────┐                        │
  (may be absent)   │  │  Standard  │                        │
                    │  │  LTC Term  ├──┐                     │
                    │  │  f(h,x;θ)  │  │                     │
                    │  └───────────┘  │   ┌──────────┐      │
                    │                  ├──►│  State    │      │
                    │  ┌───────────┐  │   │  Update   ├──► h(t+dt)
                    │  │  Pulse     │  │   │  (ODE     │      │
                    │  │  Generator ├──┤   │  Solver)  │      │
                    │  │  ω,A,φ(h)  │  │   └──────────┘      │
                    │  └───────────┘  │                      │
                    │                  │                      │
                    │  ┌───────────┐  │                      │
                    │  │  Self-     │  │                      │
                    │  │  Attend    ├──┘                      │
                    │  │  W_self·h  │                         │
                    │  └───────────┘                         │
                    │                                       │
                    └─────────────────────────────────────┘
```

### 1.4 What We're NOT Building in V1

- Multi-frequency hierarchical processing (future work)
- Synchronization-based attention between neuron groups (future work)
- Spike-gated output mechanism (future work)
- Memory consolidation during idle periods (future work)
- Neuromorphic hardware deployment (future work)

---

## 2. IMPLEMENTATION PLAN

### 2.1 Tech Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch 2.x
- **Base Library:** `ncps` (Neural Circuit Policies) — provides LTC and CfC implementations in PyTorch
- **ODE Solver:** `torchdiffeq` (for continuous-time integration)
- **Data:** PyTorch datasets / custom synthetic datasets
- **Tracking:** Weights & Biases (optional) or TensorBoard
- **Hardware:** Single GPU (the POC should be trainable on a single A100/V100 or even a good consumer GPU)

### 2.2 Implementation Phases

#### PHASE 1: Foundation (Days 1–3)
**Goal:** Get the base LTC network running on a benchmark task.

- [ ] Install `ncps` library and verify LTC/CfC models work
- [ ] Set up a simple sequence modeling task (see Section 3 for dataset choices)
- [ ] Train baseline LTC model, record performance metrics
- [ ] This becomes our control group for comparison

**Key code:**
```python
from ncps.torch import LTC
from ncps.wirings import AutoNCP

wiring = AutoNCP(units=64, output_size=output_dim)
baseline_model = LTC(input_size, wiring)
```

#### PHASE 2: Pulse Module (Days 4–7)
**Goal:** Implement the pulse-augmented LTC cell.

- [ ] Create `PulseLTCCell` class extending the base LTC cell
- [ ] Implement the oscillatory pulse generator with learnable parameters (A, ω per neuron)
- [ ] Implement state-dependent phase shift φ(h)
- [ ] Implement the self-attend term
- [ ] Add a global time counter that advances independently of input steps
- [ ] Integrate using Euler method first (simpler), upgrade to RK4/dopri5 later if needed

**Core implementation sketch:**
```python
class PulseLTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Standard LTC parameters
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))  # time constants

        # === PULSE PARAMETERS (the new stuff) ===
        self.amplitude = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.frequency = nn.Parameter(torch.rand(hidden_size) * 2 * math.pi)
        self.phase_net = nn.Linear(hidden_size, hidden_size)  # state-dependent phase

        # Self-attend parameters
        self.W_self = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # pulse strength
        self.beta = nn.Parameter(torch.tensor(0.1))   # self-attend strength

    def pulse(self, t, h):
        phase = self.phase_net(h)
        return self.amplitude * torch.sin(self.frequency * t + phase)

    def self_attend(self, h):
        return self.W_self(torch.tanh(h))

    def forward(self, x, h, t):
        # Standard LTC dynamics
        f_input = self.W_in(x) + self.W_rec(h)

        # Pulse-augmented dynamics
        dh = (-h + torch.tanh(f_input)
              + self.alpha * self.pulse(t, h)
              + self.beta * self.self_attend(h))
        dh = dh / (1 + torch.abs(self.tau))

        h_new = h + dh * dt  # Euler step (dt is the integration step size)
        return h_new
```

#### PHASE 3: Idle-Time Simulation (Days 8–10)
**Goal:** Implement the mechanism that allows the model to "tick" during input gaps.

This is what makes this a true pulse POC — the model continues evolving its state even when no input is provided.

- [ ] Create a `run_idle_ticks(h, num_ticks)` method that advances the state using only the pulse and self-attend terms (no input)
- [ ] Implement a training wrapper that inserts idle periods between input sequences
- [ ] The idle ticks should use zero-input (or a learned "idle embedding")

```python
def run_idle_ticks(self, h, num_ticks, t_start, dt=0.1):
    """Evolve state for num_ticks steps with no external input."""
    x_idle = torch.zeros(h.shape[0], self.input_size, device=h.device)
    t = t_start
    for _ in range(num_ticks):
        h = self.forward(x_idle, h, t)
        t += dt
    return h, t
```

#### PHASE 4: Training & Validation (Days 11–16)
**Goal:** Train both baseline and pulse-augmented models, run validation experiments.

- [ ] Train baseline LTC on chosen tasks
- [ ] Train Pulse-LTC on same tasks
- [ ] Run all validation experiments (see Section 4)
- [ ] Analyze pulse behavior (frequency analysis, state evolution visualization)
- [ ] Ablation studies (pulse only, self-attend only, both)

#### PHASE 5: Analysis & Documentation (Days 17–20)
**Goal:** Analyze results, create visualizations, write up findings.

- [ ] Generate comparison charts (baseline vs pulse-augmented)
- [ ] Visualize learned oscillation parameters (which neurons learned what frequencies)
- [ ] Visualize state evolution during idle periods
- [ ] Phase-space plots of hidden state dynamics
- [ ] Write up findings as a technical report

---

## 3. DATASETS & TASKS

We need tasks where persistent state and temporal continuity actually *matter*. Three categories, ordered by priority:

### 3.1 PRIMARY TASK: Interrupted Sequence Prediction

**Custom synthetic dataset.** This is the most direct test of the pulse concept.

**Design:** Generate sequences where there are deliberate gaps (idle periods) between segments. The model must maintain relevant state across the gap.

Example task: "Delayed Copy with Distraction"
- Phase 1: Model sees a pattern (e.g., [A, B, C, D])
- Phase 2: Gap of N timesteps (no meaningful input, just noise/zeros)
- Phase 3: Model must reproduce the pattern from Phase 1

**Why this tests the pulse:** A standard LTC's state will decay during the gap. The pulse should help maintain the state.

**Variations to test:**
- Vary gap length (5, 10, 20, 50, 100 timesteps)
- Vary pattern complexity
- Add distractor sequences during the gap

### 3.2 SECONDARY TASK: Periodic Signal Prediction with Phase Gaps

**Semi-synthetic.** Model must predict a periodic signal (sine waves, sawtooth, etc.) but training sequences have random gaps where the signal is missing. The model must maintain phase tracking across gaps.

**Why this tests the pulse:** The internal oscillation should naturally help the model maintain a sense of where in the cycle the signal is, even when observations are missing.

**Dataset generation:**
```python
def generate_gapped_periodic(seq_len=200, gap_prob=0.3, freq=0.1):
    t = np.arange(seq_len)
    signal = np.sin(2 * np.pi * freq * t)
    mask = np.random.random(seq_len) > gap_prob  # 1 = observed, 0 = gap
    observed = signal * mask
    return observed, signal, mask  # input, target, mask
```

### 3.3 TERTIARY TASK: Real-World Sequential Decision Making

**Walker2D or CartPole via OpenAI Gym** — use the LTC as a policy network in an RL setting, but introduce observation blackout periods where the agent receives no sensor data for several timesteps. The pulse should help maintain situational awareness.

This is a stretch goal — only if Phases 1-4 go smoothly.

### 3.4 BONUS TASK: Sequential MNIST with Delays

Feed MNIST images as sequences of pixel rows, but insert blank rows (gaps) at random positions. The model must classify despite incomplete observations with temporal gaps.

---

## 4. VALIDATION FRAMEWORK

### 4.1 Core Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Gap Robustness Score** | Performance degradation vs gap length | Accuracy/MSE at gap lengths [0, 5, 10, 20, 50, 100] |
| **State Persistence Index** | How much state information survives idle periods | Cosine similarity of hidden state before gap vs after gap (compare pulse vs baseline) |
| **Phase Coherence** | Whether pulse maintains temporal alignment | Correlation between predicted and actual phase after gaps in periodic tasks |
| **Convergence Speed** | Training efficiency | Epochs to reach target performance |
| **Idle State Entropy** | Richness of state evolution during idle periods | Shannon entropy of hidden state activations during idle ticks |

### 4.2 Validation Experiments

#### Experiment 1: Gap Length Sweep (Primary)
- Train both models on interrupted sequence task
- Evaluate at gap lengths: 0, 5, 10, 20, 50, 100 timesteps
- **Expected result:** Pulse model degrades slower than baseline as gap length increases
- **Success criterion:** Pulse model retains >80% accuracy at gap=50 where baseline drops below 50%

#### Experiment 2: State Trajectory Analysis
- Record hidden state vectors at every timestep (including idle ticks) for both models
- Visualize using t-SNE/UMAP
- **Expected result:** Pulse model states during idle periods form structured trajectories (orbits); baseline states collapse toward a fixed point
- **Success criterion:** Qualitative — pulse model states should show non-trivial dynamics during idle periods

#### Experiment 3: Ablation Study
- Train 4 variants: (a) Baseline LTC, (b) LTC + pulse only, (c) LTC + self-attend only, (d) Full pulse-augmented LTC
- Compare on interrupted sequence task
- **Expected result:** Full model > pulse-only > self-attend-only > baseline
- **Success criterion:** Each component contributes measurably

#### Experiment 4: Learned Frequency Analysis
- After training, extract the learned frequency parameters (ω) for each neuron
- Plot frequency distribution
- **Expected result:** Neurons should learn a diversity of frequencies (not all converge to same value), possibly correlating with task-relevant time scales
- **Success criterion:** Non-degenerate frequency distribution (std/mean > 0.3)

#### Experiment 5: Spontaneous Recall Test
- Train model on multiple distinct patterns
- During idle period, observe if hidden state spontaneously moves toward representations of previously seen patterns
- **Expected result:** Evidence of pattern replay during idle periods (analogous to hippocampal replay)
- **Success criterion:** Cosine similarity between idle-period states and pattern-associated states exceeds random chance (p < 0.05)

### 4.3 Baselines for Comparison

1. **Standard LTC** — no pulse, no self-attend (direct control)
2. **CfC (Closed-form Continuous)** — the efficient approximation of LTC
3. **Standard LSTM** — classical recurrent baseline
4. **GRU** — another classical baseline
5. **LTC + noise injection** — to rule out that "any perturbation during idle time" helps (the pulse should be *structured*, not just noise)

---

## 5. PROJECT TIMELINE

```
Week 1 (Days 1-7):     Foundation + Pulse Module Implementation
  ├── Days 1-3:  Setup, baseline LTC training, dataset generation
  └── Days 4-7:  PulseLTCCell implementation, unit tests

Week 2 (Days 8-14):    Idle-Time Mechanism + Initial Training
  ├── Days 8-10:  Idle tick simulation, training wrapper
  └── Days 11-14: Train all models, initial results

Week 3 (Days 15-20):   Validation + Analysis + Writeup
  ├── Days 15-16: Full validation experiments
  ├── Days 17-18: Visualization, frequency analysis
  └── Days 19-20: Documentation, findings report
```

---

## 6. RISKS & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pulse term destabilizes training (gradients explode) | High | High | Initialize α small (0.01), use gradient clipping, add pulse magnitude regularization |
| Pulse doesn't help (model ignores it) | Medium | High | Try different α initialization, curriculum learning (start with long gaps), make pulse contribution mandatory via architectural constraints |
| ODE solver is too slow | Medium | Medium | Start with Euler, only upgrade if needed; use CfC-style closed-form approximation of the pulse term |
| Frequency parameters collapse (all neurons learn same ω) | Medium | Medium | Initialize with diverse frequencies, add diversity regularization loss |
| Idle state diverges | Medium | High | Add state norm regularization, clip hidden state magnitude |

---

## 7. RESOURCE REQUIREMENTS

- **Compute:** Single GPU (8-16 GB VRAM sufficient for POC scale)
- **Training time estimate:** ~2-4 hours per model configuration (small network, synthetic data)
- **Storage:** Minimal (<10GB for all data and checkpoints)
- **Dependencies:** PyTorch, ncps, torchdiffeq, numpy, matplotlib, scikit-learn (for t-SNE)

---

## 8. SUCCESS CRITERIA (POC-Level)

The POC is considered successful if ANY of the following are demonstrated:

1. **Primary:** Pulse-augmented model shows statistically significant improvement over baseline LTC on the interrupted sequence task at gap length ≥ 20 timesteps

2. **Secondary:** Hidden state analysis shows structured, non-degenerate dynamics during idle periods (qualitative evidence of "alive" behavior)

3. **Tertiary:** Learned frequency parameters are diverse and correlate with task-relevant time scales

A single clear positive result on any of these would justify further investment in the architecture.

---

## 9. FUTURE DIRECTIONS (Post-POC)

If the POC succeeds, the next steps would be:

1. **Multi-frequency hierarchy** — Different neuron groups at different time scales (borrowing from Nested Learning / H.O.P.E.)
2. **Synchronization-based binding** — Use phase coherence between neuron groups as an attention mechanism (borrowing from CTMs)
3. **Scale to language** — Integrate pulse mechanism into a small transformer as a persistent state module
4. **Memory consolidation** — Train the idle-time dynamics to actively reorganize/compress stored information
5. **Neuromorphic deployment** — Map the architecture to spiking neural network hardware (Intel Loihi) for energy-efficient always-on operation

---

## 10. KEY REFERENCES

- Hasani et al. (2021). "Liquid Time-Constant Networks." AAAI.
- Hasani et al. (2022). "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence.
- Darlow et al. (2025). "Continuous Thought Machines." Sakana AI / NeurIPS 2025 Spotlight.
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." ICLR 2024.
- Behrouz et al. (2025). "Nested Learning: The Illusion of Deep Learning Architectures." NeurIPS 2025.
- Neural Circuit Policies library: github.com/mlech26l/ncps
- Liquid-S4: github.com/raminmh/liquid-s4
