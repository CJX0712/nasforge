"""Environment-driven configuration with schema validation.

Every knob can be overridden by an ``NASFORGE_*`` environment variable. Overrides
are validated eagerly so a malformed value fails at start-up rather than silently
degrading a benchmark run (a silent fallback would invalidate the determinism
gate).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any

from nasforge.core.errors import ConfigError
from nasforge.core.seed import DEFAULT_SEED, get_seed

__all__ = ["EnvPrefix", "SearchConfig", "TrainConfig", "EvalConfig", "DataConfig", "Config"]


class EnvPrefix:
    """Namespace constants for environment overrides."""

    ROOT = "NASFORGE_"


# Coercion registry: type -> parser. Applied to the raw string from os.environ.
def _as_int(raw: str, name: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{name}={raw!r} is not an integer",
            hint=f"set {name} to an integer literal",
        ) from exc


def _as_float(raw: str, name: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{name}={raw!r} is not a float",
            hint=f"set {name} to a decimal literal",
        ) from exc


def _as_bool(raw: str, name: str) -> bool:
    token = raw.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(
        f"{name}={raw!r} is not a boolean",
        hint="use one of 1/0/true/false/yes/no/on/off",
    )


def _apply_env(instance: Any) -> Any:
    """Return a copy of ``instance`` with ENV_NASFORGE_* overrides applied."""
    updates: dict[str, Any] = {}
    for f in fields(instance):
        env_key = f"{EnvPrefix.ROOT}{f.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None or raw.strip() == "":
            continue
        if f.type in ("int", int):
            updates[f.name] = _as_int(raw, env_key)
        elif f.type in ("float", float):
            updates[f.name] = _as_float(raw, env_key)
        elif f.type in ("bool", bool):
            updates[f.name] = _as_bool(raw, env_key)
        else:
            updates[f.name] = raw.strip()
    return replace(instance, **updates) if updates else instance


@dataclass
class SearchConfig:
    """Search algorithm budget and hyper-parameters."""

    #: Number of search iterations (architectures evaluated) per run.
    budget: int = 24
    #: DARTS architecture learning rate (before GRL sign flip).
    arch_lr: float = 0.1
    #: Weight learning rate used during the unrolled search.
    weight_lr: float = 0.05
    #: L2 penalty applied to the architecture logits.
    arch_weight_decay: float = 1e-3
    #: DARTS unroll steps.
    unrolled_steps: int = 2
    #: Use the second-order Taylor approximation instead of the first-order one.
    second_order: bool = True
    #: Regularized evolution population size (aging population).
    population_size: int = 8
    #: Tournament sample size for mutation.
    sample_size: int = 4
    #: Number of independent search runs (random initialisation per run).
    n_runs: int = 1


@dataclass
class TrainConfig:
    """Weight optimisation settings."""

    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    #: Cosine annealing floor, as a fraction of the peak learning rate.
    lr_min_ratio: float = 0.05


@dataclass
class EvalConfig:
    """Evaluation protocol settings."""

    #: Epochs used when a discrete genotype is scored independently.
    eval_epochs: int = 12
    #: Holdout fraction for the validation split.
    val_fraction: float = 0.25
    #: Metric direction: "higher_is_better" or "lower_is_better".
    metric: str = "higher_is_better"
    #: Metric name reported in benchmark.json.
    metric_name: str = "val_accuracy"


@dataclass
class DataConfig:
    """Synthetic dataset generation settings."""

    task: str = "blob"
    n_train: int = 1024
    n_val: int = 256
    n_classes: int = 4
    n_channels: int = 3
    height: int = 8
    width: int = 8


@dataclass
class Config:
    """Top level configuration bundle."""

    seed: int = field(default_factory=lambda: get_seed(DEFAULT_SEED))
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validated(self) -> Config:
        """Validate cross-field constraints.

        Returns
        -------
        Config
            ``self`` when valid.

        Raises
        ------
        ConfigError
            On any violated constraint.
        """
        if self.search.budget < 1:
            raise ConfigError("search.budget must be >= 1", hint="raise the search budget")
        if self.search.unrolled_steps < 1:
            raise ConfigError("search.unrolled_steps must be >= 1", hint="use at least 1 step")
        if not 0.0 < self.eval.val_fraction < 1.0:
            raise ConfigError(
                "eval.val_fraction must be in (0, 1)",
                hint="e.g. 0.25 for a 75/25 train/val split",
            )
        if self.eval.metric not in {"higher_is_better", "lower_is_better"}:
            raise ConfigError(
                f"eval.metric={self.eval.metric!r} is unsupported",
                hint="use higher_is_better or lower_is_better",
            )
        if self.data.n_train < 8:
            raise ConfigError("data.n_train must be >= 8", hint="raise the sample count")
        if self.data.n_classes < 2:
            raise ConfigError("data.n_classes must be >= 2", hint="use at least 2 classes")
        if self.train.epochs < 1 or self.eval.eval_epochs < 1:
            raise ConfigError("train.epochs and eval.eval_epochs must be >= 1")
        if not 0.0 < self.train.lr_min_ratio <= 1.0:
            raise ConfigError("train.lr_min_ratio must be in (0, 1]")
        return self

    @classmethod
    def from_env(cls) -> Config:
        """Build a configuration from defaults plus ``NASFORGE_*`` overrides."""
        cfg = cls()
        cfg = replace(
            cfg,
            search=_apply_env(cfg.search),
            train=_apply_env(cfg.train),
            eval=_apply_env(cfg.eval),
            data=_apply_env(cfg.data),
        )
        return cfg.validated()
