# PDNA — Pulse-Driven Neural Architecture

## Project Overview
Codename "Heartbeat". A neural network with an intrinsic oscillatory "pulse" that maintains and evolves state independently of input. Built on Liquid Time-Constant (LTC) networks augmented with learnable oscillatory dynamics.

## Tech Stack
- Python 3.10+ managed with `uv`
- PyTorch 2.x
- `ncps` (Neural Circuit Policies) for LTC/CfC base
- `torchdiffeq` for ODE solving
- `hatchling` build backend

## Project Structure
```
src/pdna/          # main package
tests/             # pytest tests
configs/           # model/experiment configs
scripts/           # training & eval scripts
notebooks/         # analysis notebooks
data/              # datasets (gitignored)
docs/              # project plans and documentation
```

## Commands
- `uv sync` — install dependencies
- `uv run pytest` — run tests
- `uv run pytest tests/test_foo.py -v` — run specific test
- `uv run python scripts/<script>.py` — run a script

## Conventions
- Source code lives in `src/pdna/`
- All model variants (A–F) share identical hyperparameters except architectural differences
- Use layered git commits — one logical change per commit
- Config-driven experiments: hyperparameters in `configs/`, not hardcoded
- Tests mirror source structure: `src/pdna/models/foo.py` → `tests/test_foo.py`

## Key Architecture
- **Variant A:** Baseline LTC (control)
- **Variant B:** LTC + Noise (random perturbation control)
- **Variant C:** LTC + Pulse (structured oscillation only)
- **Variant D:** LTC + SelfAttend (self-attention on own state only)
- **Variant E:** Full PDNA (pulse + self-attend)
- **Variant F:** Full + Idle (pulse + self-attend + idle ticks)

## Core Equation
```
τ(x) · dh/dt = -h + f(h, x; θ) + α · pulse(t, h) + β · self_attend(h)
pulse(t, h) = A · sin(ω · t + φ(h))
self_attend(h) = W_self · σ(h)
```

## Task Plan
See `docs/task_plan.md` for the full 13-task execution plan with dependencies.
