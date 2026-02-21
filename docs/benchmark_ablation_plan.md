# Pulse-Driven Neural Architecture — Benchmark & Ablation Study Plan

## BENCHMARK SELECTION: Long Range Arena (LRA)

### Why LRA Is the Right Benchmark for This POC

After evaluating multiple options, **Long Range Arena (LRA)** is the optimal benchmark for demonstrating the pulse concept. Here's the reasoning:

**What LRA is:** A standardized benchmark suite of 5 tasks with sequences ranging from 1K to 16K tokens, designed specifically to test long-range dependency modeling. It's the established benchmark for evaluating SSMs, efficient Transformers, and recurrent architectures — exactly the family our pulse architecture belongs to.

**Why it fits our POC perfectly:**

1. **Established baselines exist for every architecture we compare against.** S4, Mamba, LTC, Liquid-S4, LRU, LSTM, Transformer — all have published LRA scores. We don't need to generate baselines from scratch; we slot our results directly into existing leaderboard tables.

2. **The tasks inherently test long-range state maintenance** — which is exactly what the pulse is designed to improve. If the pulse helps a model maintain coherent state over long sequences, LRA will show it.

3. **Small model sizes are standard.** LRA was designed for models with ~100K-2M parameters. Our POC doesn't need a 7B model — a small recurrent network is the expected architecture class.

4. **We can create a "gapped" variant** of LRA tasks to directly test the pulse. Take any LRA task, insert idle gaps in the input sequence, and measure degradation. This gives us both standard comparison AND pulse-specific evaluation.

5. **Credibility.** Any reviewer or reader immediately recognizes LRA results. It's the lingua franca of sequence model evaluation.

---

## THE 5 LRA TASKS (Ordered by Priority for Our POC)

### Tier 1: Must-Run (Core Ablation)

#### 1. ListOps (Sequence Length: ~2K tokens)
- **Task:** Parse nested mathematical expressions like `[MAX 4 3 [MIN 2 3] 1 0 [MEDIAN 1 5 8 9 2]]` and compute the answer
- **Classification:** 10-way
- **Why critical for us:** Requires maintaining hierarchical structure over the full sequence — a state maintenance task. The pulse should help preserve structural information across long nesting depths.
- **Published baselines:** Transformer ~36%, S4 ~59%, Mega ~63%

#### 2. Pathfinder (Sequence Length: ~1K tokens, image flattened to 1D)
- **Task:** Binary classification — are two dots connected by a dashed path in a synthetic image?
- **Why critical for us:** Requires spatial reasoning over long sequences and incremental evidence accumulation. The pulse's ability to maintain partial path traces should help.
- **Published baselines:** Transformer ~71%, S4 ~94%, Mega ~96%

#### 3. Text Classification (Sequence Length: ~4K bytes)
- **Task:** Byte-level IMDb sentiment classification
- **Why critical for us:** Real-world task. Byte-level means the model must compose characters → words → semantics over thousands of steps. Tests whether the pulse helps with hierarchical composition over long contexts.
- **Published baselines:** Transformer ~64%, S4 ~86%, Mega ~90%

### Tier 2: Should-Run (Completeness)

#### 4. Image Classification (Sequence Length: ~1K, CIFAR-10 pixels as sequence)
- **Task:** Classify CIFAR-10 images fed as a flattened sequence of grayscale pixels
- **Published baselines:** Transformer ~42%, S4 ~88%, Mega ~90%

#### 5. Document Retrieval (Sequence Length: ~4K bytes)
- **Task:** Byte-level document similarity matching
- **Published baselines:** Transformer ~57%, S4 ~87%, Mega ~91%

---

## THE ABLATION STUDY DESIGN

### Architecture Variants (6 Models)

We train and evaluate 6 model variants to isolate the contribution of each component:

```
┌─────────────────────────────────────────────────────────────────┐
│  VARIANT          │ LTC Base │ Pulse │ Self-Attend │ Idle Ticks │
├───────────────────┼──────────┼───────┼─────────────┼────────────┤
│  A. Baseline LTC  │    ✓     │       │             │            │
│  B. LTC + Noise   │    ✓     │ rand  │             │            │
│  C. LTC + Pulse   │    ✓     │   ✓   │             │            │
│  D. LTC + SelfAtt │    ✓     │       │      ✓      │            │
│  E. Full PDNA     │    ✓     │   ✓   │      ✓      │            │
│  F. Full + Idle   │    ✓     │   ✓   │      ✓      │     ✓      │
└───────────────────┴──────────┴───────┴─────────────┴────────────┘
```

**Why each variant matters:**
- **A → C:** Does the pulse alone help? (vs no pulse)
- **B → C:** Is structured oscillation better than random perturbation? (critical control — rules out "any noise helps")
- **A → D:** Does self-attend alone help? (model attending to own state)
- **C vs D vs E:** Do pulse and self-attend contribute independently or synergistically?
- **E → F:** Does running idle ticks between sequence segments improve performance?

### Ablation Experiment Matrix

```
               ListOps  Pathfinder  Text   Image  Retrieval  AVG
─────────────────────────────────────────────────────────────────
A. Base LTC      ?         ?         ?      ?        ?        ?
B. LTC+Noise     ?         ?         ?      ?        ?        ?
C. LTC+Pulse     ?         ?         ?      ?        ?        ?
D. LTC+SelfAtt   ?         ?         ?      ?        ?        ?
E. Full PDNA     ?         ?         ?      ?        ?        ?
F. Full+Idle     ?         ?         ?      ?        ?        ?
─────────────────────────────────────────────────────────────────
Ref: S4         58.35     94.20     86.82  88.65    87.09    83.02
Ref: Mamba        -         -         -      -        -        -
Ref: Liquid-S4    -       95.2+       -      -        -        -
Ref: Transformer 36.37    71.40     64.27  42.44    57.46    54.39
```

---

## GAPPED-LRA: Our Custom Pulse-Specific Benchmark

Beyond standard LRA, we create a **modified version** that directly tests state persistence across temporal gaps. This is the most novel part of our evaluation.

### Design

For each LRA task, we create 5 difficulty levels by inserting zero-valued gap segments into the input sequence:

```
Gap Level 0:  [──────── full sequence ────────]     (standard LRA)
Gap Level 1:  [── seg1 ──][gap 5%][── seg2 ──]      (small gap)
Gap Level 2:  [── seg1 ──][gap 15%][── seg2 ──]     (medium gap)
Gap Level 3:  [── seg1 ──][gap 30%][── seg2 ──]     (large gap)
Gap Level 4:  [seg1][gap][seg2][gap][seg3][gap][seg4] (multiple gaps)
```

**Gap implementation:**
- Replace gap regions with zeros (no information)
- For Variant F (Full+Idle), the model runs idle ticks during gap regions
- For all other variants, gaps are just zero-input timesteps

### Gapped-LRA Ablation Table

```
              Gap 0%   Gap 5%   Gap 15%  Gap 30%  Multi-Gap  Degradation
──────────────────────────────────────────────────────────────────────────
A. Base LTC     ?        ?        ?        ?        ?         Δ = ?
C. LTC+Pulse    ?        ?        ?        ?        ?         Δ = ?
E. Full PDNA    ?        ?        ?        ?        ?         Δ = ?
F. Full+Idle    ?        ?        ?        ?        ?         Δ = ?
──────────────────────────────────────────────────────────────────────────
```

**"Degradation" column** = (Gap 0% accuracy - Gap 30% accuracy). Lower is better. This is the key metric: **how gracefully does performance degrade as gaps grow?**

**Hypothesis:** The pulse model's degradation should be significantly lower than the baseline's, because the oscillatory dynamics maintain state information during gaps rather than letting it decay.

---

## ADDITIONAL ANALYSIS METRICS (Beyond Accuracy)

### 1. Learned Frequency Spectrum
After training, extract ω (frequency) parameters for each neuron and plot:
- Histogram of learned frequencies
- Correlation between neuron frequency and task performance contribution
- Whether different tasks induce different frequency distributions

### 2. State Dynamics Visualization
For a sample of test sequences:
- Plot hidden state trajectories (2D projection via PCA/t-SNE)
- Compare state trajectories during gaps: Baseline (should decay) vs Pulse (should orbit)
- Measure state entropy during gap periods

### 3. Gradient Flow Analysis
- Track gradient magnitudes through the pulse component during training
- Verify the pulse term is actually being used (gradients not zero)
- Track α (pulse strength) and β (self-attend strength) over training

### 4. Compute Overhead Measurement
- Wall-clock training time per epoch for each variant
- Inference latency per sample
- Additional parameter count from pulse components
- FLOPs comparison

---

## IMPLEMENTATION DETAILS FOR LRA

### Model Configuration (Matching LRA Standards)

```python
config = {
    "hidden_size": 128,          # Standard for LRA
    "num_layers": 4,             # Stack of pulse-LTC cells
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 50,            # Standard LRA training budget

    # Pulse-specific
    "pulse_alpha_init": 0.01,    # Start small, let it grow
    "pulse_num_frequencies": 128, # One per hidden unit
    "self_attend_beta_init": 0.01,
    "idle_ticks_per_gap": 10,    # For Variant F
}
```

### Training Protocol

1. All 6 variants use **identical hyperparameters** except for their architectural differences
2. Each variant trained 3x with different seeds for statistical significance
3. Report mean ± std across seeds
4. Use same optimizer (AdamW), same schedule (cosine annealing with warmup)
5. Early stopping on validation set, evaluate on held-out test set

### Dataset Sizes (Standard LRA)
- ListOps: 96K train / 2K val / 2K test
- Text: 25K train / 25K test (IMDb)
- Pathfinder: 160K train / 20K val / 20K test
- Image: 45K train / 5K val / 10K test (CIFAR-10)
- Retrieval: ~147K train / ~18K val / ~17K test

---

## STATISTICAL VALIDATION

### Significance Testing
- Paired t-test between each variant pair across 3 seeds
- Report p-values for key comparisons (A vs E, B vs C, E vs F)
- Use Bonferroni correction for multiple comparisons

### Effect Size
- Report Cohen's d for key comparisons
- Focus on practical significance, not just statistical significance

### Confidence Intervals
- 95% CI for all reported accuracies
- Bootstrap CIs for degradation metrics on Gapped-LRA

---

## SUCCESS CRITERIA (Updated with LRA)

### Strong Success (Publishable Result):
- Full PDNA (E) outperforms Baseline LTC (A) by ≥ 2% average across all 5 LRA tasks
- AND Gapped-LRA degradation for PDNA is ≤ 50% of baseline degradation
- AND learned frequencies are diverse (non-degenerate)

### Moderate Success (Promising Direction):
- Full PDNA outperforms baseline on ≥ 3 of 5 LRA tasks
- OR Gapped-LRA shows clear pulse advantage even if standard LRA is mixed

### Minimal Success (Concept Validated):
- Any statistically significant improvement on any LRA task
- OR clear qualitative evidence of structured state dynamics during idle periods
- OR the noise control (B) performs worse than pulse (C), proving structured oscillation matters

### Failure Indicators:
- Pulse variant performs worse than baseline (pulse destabilizes)
- Noise control (B) performs equally to pulse (C) — structured oscillation doesn't matter
- α converges to 0 — model learns to ignore the pulse entirely

---

## ESTIMATED TIMELINE & COMPUTE

### Per-Variant Training Time (Estimated)
- ListOps: ~2-3 hours on single A100
- Pathfinder: ~1-2 hours
- Text: ~3-4 hours
- Image: ~1-2 hours
- Retrieval: ~3-4 hours

### Total Compute Budget
- 6 variants × 5 tasks × 3 seeds = 90 training runs
- ~10-15 hours per task × 5 tasks = ~50-75 GPU hours total
- Feasible on a single A100 in about 3-4 days of continuous training
- Or ~1 day with 4 GPUs running in parallel

---

## DELIVERABLES

1. **Ablation table** (the main result — 6 variants × 5 tasks)
2. **Gapped-LRA degradation curves** (accuracy vs gap percentage)
3. **Frequency spectrum plots** (what did the neurons learn?)
4. **State trajectory visualizations** (alive vs dead dynamics)
5. **Training curves** (convergence comparison)
6. **Compute overhead table** (cost of the pulse)
7. **Technical report** documenting findings
