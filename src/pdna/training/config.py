"""Experiment configuration management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    ode_unfolds: int = 6
    alpha_init: float = 0.01
    beta_init: float = 0.01
    idle_ticks_per_gap: int = 10


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 50
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    early_stopping_patience: int = 10


@dataclass
class DataConfig:
    max_seq_len: int = 2048


@dataclass
class LoggingConfig:
    backend: str = "tensorboard"
    project: str = "pdna-ablation"
    log_every_n_steps: int = 50
    log_dir: str = "runs"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load experiment config from YAML file, falling back to defaults."""
    config = ExperimentConfig()
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        return config

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)
    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)
    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)
    if "logging" in raw:
        for k, v in raw["logging"].items():
            if hasattr(config.logging, k):
                setattr(config.logging, k, v)

    return config
