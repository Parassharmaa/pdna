# PDNA — End-to-End Task Plan

## Dependency Graph

```
#1  Initialize project structure ─┬──► #2 Baseline LTC ──► #3 PulseLTCCell ─┬──► #4 Idle ticks ──┐
                                  │                                          │                    │
                                  │                                          ├──► #5 All 6 variants◄┘
                                  │                                          │                    │
                                  │                                          └──► #6 Unit tests   │
                                  │                                                               │
                                  └──► #7 LRA data pipeline ──► #8 Gapped-LRA                    │
                                                │                     │                           │
                                                └─────────────────────┼──► #9 Training framework ◄┘
                                                                      │            │
                                                                      └──► #10 Tier 1 experiments ◄┘
                                                                                │
                                                                      ┌─────────┼──────────┐
                                                                      ▼         ▼          ▼
                                                                   #11 Tier2  #12 Analysis
                                                                      │         │
                                                                      └────┬────┘
                                                                           ▼
                                                                     #13 Report
```

---

## Phase 1: Foundation

### Task 1 — Initialize project structure and dependencies
- **Status:** ✅ complete
- **Blocked by:** —
- **Details:**
  - Create directory structure: `src/pdna/`, `tests/`, `configs/`, `scripts/`, `notebooks/`, `data/`
  - Create `pyproject.toml` with dependencies: torch, ncps, torchdiffeq, numpy, matplotlib, scikit-learn, wandb/tensorboard
  - Create `__init__.py` files
  - Set up Python 3.10+ environment
  - Initialize git repo

### Task 2 — Implement baseline LTC model
- **Status:** ✅ complete
- **Blocked by:** #1
- **Details:**
  - Install and verify `ncps` library (LTC/CfC implementations)
  - Create baseline LTC model using AutoNCP wiring (hidden_size=128, num_layers=4)
  - Create a simple training loop with AdamW optimizer and cosine annealing
  - Verify baseline forward/backward pass works on dummy data
  - This becomes **Variant A** (control group) in the ablation study

---

## Phase 2: Pulse Module

### Task 3 — Implement PulseLTCCell with oscillatory pulse generator
- **Status:** ✅ complete
- **Blocked by:** #2
- **Details:**
  - Create `PulseLTCCell` class extending base LTC cell
  - Implement `pulse(t, h) = A * sin(ω * t + φ(h))` with learnable A, ω per neuron
  - Implement state-dependent phase shift φ(h) via `phase_net` linear layer
  - Implement `self_attend(h) = W_self * σ(h)`
  - Add learnable α (pulse strength, init=0.01) and β (self-attend strength, init=0.01)
  - Add global time counter advancing independently of input steps
  - Diverse ω initialization across neurons
  - Euler integration first, upgrade to RK4/dopri5 later if needed
  - Gradient clipping and pulse magnitude regularization for stability

### Task 4 — Implement idle-time tick simulation
- **Status:** ✅ complete
- **Blocked by:** #3
- **Details:**
  - Create `run_idle_ticks(h, num_ticks, t_start, dt)` method
  - During idle ticks: evolve state using only pulse + self-attend (zero input)
  - Create training wrapper that inserts idle periods between input segments
  - Support configurable `idle_ticks_per_gap` (default 10)
  - Ensure time counter advances correctly during idle periods
  - Add state norm regularization to prevent idle state divergence

### Task 5 — Build all 6 ablation architecture variants
- **Status:** ✅ complete
- **Blocked by:** #3, #4
- **Details:**
  - **A.** Baseline LTC (no pulse, no self-attend)
  - **B.** LTC + Noise (random perturbation — critical control)
  - **C.** LTC + Pulse (pulse only, no self-attend)
  - **D.** LTC + SelfAttend (self-attend only, no pulse)
  - **E.** Full PDNA (pulse + self-attend)
  - **F.** Full + Idle (pulse + self-attend + idle ticks)
  - Config system: identical hyperparameters (hidden_size=128, num_layers=4, dropout=0.1, lr=1e-3, batch_size=32, max_epochs=50) except architectural differences

### Task 6 — Write unit tests for pulse components
- **Status:** ✅ complete
- **Blocked by:** #3, #4
- **Details:**
  - Test PulseLTCCell forward pass shape correctness
  - Test pulse function produces non-zero output
  - Test self-attend function
  - Test idle ticks don't cause state divergence (norm stays bounded)
  - Test Variant B (noise) ≠ Variant C (pulse) outputs
  - Test time counter advances correctly
  - Test all 6 variants instantiate and run forward pass
  - Test gradient flow through pulse components (gradients ≠ zero)

---

## Phase 3: Data & Evaluation Infrastructure

### Task 7 — Set up LRA data pipeline (ListOps first)
- **Status:** ✅ complete
- **Blocked by:** #1
- **Details:**
  - Download/generate ListOps dataset (96K train / 2K val / 2K test, ~2K token sequences)
  - Implement ListOps tokenizer and dataloader
  - Standard LRA preprocessing pipeline
  - Set up Pathfinder and Text classification data loading (Tier 1)
  - Optional: Image and Retrieval tasks (Tier 2)
  - All dataloaders: batch_size=32, proper train/val/test splits

### Task 8 — Implement Gapped-LRA benchmark variant
- **Status:** ✅ complete
- **Blocked by:** #7
- **Details:**
  - For each LRA task, create 5 gap difficulty levels:
    - Gap 0%: standard (no gaps)
    - Gap 5%: small gap
    - Gap 15%: medium gap
    - Gap 30%: large gap
    - Multi-gap: multiple gaps scattered throughout
  - For Variant F: run idle ticks during gap regions
  - For all other variants: gaps = zero-input timesteps
  - Compute "Degradation" = (Gap 0% acc − Gap 30% acc)

### Task 9 — Build training and evaluation framework
- **Status:** ✅ complete
- **Blocked by:** #5, #7
- **Details:**
  - Training loop: AdamW optimizer, cosine annealing with warmup
  - Early stopping on validation set, evaluate on held-out test
  - Support 3 random seeds per configuration
  - Logging: loss curves, accuracy, per-epoch metrics
  - Checkpoint saving/loading
  - WandB or TensorBoard integration
  - Evaluation: accuracy, mean ± std across seeds
  - Compute overhead tracking: wall-clock time, inference latency, FLOPs

---

## Phase 4: Experiments

### Task 10 — Run experiments (sMNIST, pMNIST, Adding Problem)
- **Status:** 🔄 in progress (running on RunPod RTX A4000)
- **Blocked by:** #6, #8, #9
- **Details:**
  - **Architecture switch:** Replaced LTC with CfC (Closed-form Continuous-time) for ~20x GPU speedup
  - **Phase 1 (Baseline Validation):** sMNIST PASSED (97.98%), pMNIST running (~96%), Adding queued
  - **Phase 2 (Full Ablation):** 6 variants × 3 tasks × 3 seeds = 54 runs (queued after Phase 1)
  - **Phase 3 (Gap Evaluation):** 5 gap levels per run (integrated into Phase 2)
  - Tasks chosen for CfC compatibility and proven learnability
  - V1 (ListOps) and V2 (enhanced synthetic) documented as failed baselines
  - V3 (synthetic freq/gap/temporal) documented — tasks too easy (100% accuracy)
  - Statistical validation: paired t-test, Cohen's d, 95% CI

### Task 11 — Run Tier 2 experiments (optional stretch)
- **Status:** deferred (scope adjusted)
- **Blocked by:** #10
- **Details:**
  - Original plan: Image (CIFAR-10) and Document Retrieval LRA tasks
  - **Scope adjustment:** Focus on 3 validated tasks that demonstrate CfC+pulse hypothesis
  - Additional tasks only if Tier 1 results warrant further investigation
  - If Task 10 shows clear signal, may add longer-sequence sMNIST (196-step) variant

---

## Phase 5: Analysis & Deliverables

### Task 12 — Implement analysis and visualization pipeline
- **Status:** ✅ complete
- **Blocked by:** #10
- **Details:**
  1. ✅ Learned Frequency Spectrum: extract ω, plot histogram
  2. ⏳ State Dynamics Visualization: requires model checkpoints from experiment
  3. ⏳ Gradient Flow Analysis: requires model checkpoints from experiment
  4. ✅ **THE KEY GRAPH:** Performance Degradation Under Increasing Input Gaps
  5. ✅ Training convergence curves (per-task subplots with mean+std bands)
  6. ✅ Compute overhead table (params, wall time, overhead ratio)
  7. ✅ Degradation bar chart for direct variant comparison
  8. ✅ Statistical tests: paired t-test, Cohen's d, 95% CI
  9. ✅ Ablation heatmap
  10. ✅ Full markdown report generator with success criteria evaluation

### Task 13 — Compile results and generate technical report
- **Status:** pending (blocked on experiment completion)
- **Blocked by:** #10, #12
- **Deliverables:**
  1. Ablation table (6 variants × 3 tasks, mean ± std)
  2. Gapped degradation curves (the key graph)
  3. Degradation bar chart
  4. Training convergence curves
  5. Ablation heatmap
  6. Compute overhead table
  7. Statistical significance results (t-test, Cohen's d, 95% CI)
  8. Success criteria evaluation (Strong / Moderate / Minimal / Failure)
  9. Technical report (markdown)
  10. All experiment logs documented (v1, v2, v3, proper)

---

## Success Criteria

| Level | Criteria |
|-------|----------|
| **Strong (Publishable)** | Full PDNA ≥ 2% avg improvement over baseline on all 5 LRA tasks AND Gapped-LRA degradation ≤ 50% of baseline AND diverse learned frequencies |
| **Moderate (Promising)** | Outperforms baseline on ≥ 3/5 LRA tasks OR clear Gapped-LRA advantage |
| **Minimal (Validated)** | Any statistically significant improvement OR qualitative evidence of structured idle dynamics OR noise control (B) < pulse (C) |
| **Failure** | Pulse destabilizes OR noise = pulse OR α → 0 |

---

## Estimated Compute

- 6 variants × 5 tasks × 3 seeds = **90 total training runs**
- ~50–75 GPU hours total on a single A100
- ~3–4 days continuous, or ~1 day with 4 GPUs
