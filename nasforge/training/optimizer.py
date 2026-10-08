"""A tiny SGD optimiser with momentum and decoupled weight decay.

The optimiser stores no global RNG; callers pass a fresh ``Generator`` when they
want stochastic minibatching. State (velocity) is keyed by the flat parameter
vector identity, so it is safe to reuse one optimiser across many parameter
dicts as long as shapes are stable.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict

__all__ = ["SGD"]


class SGD:
    """SGD with momentum and optional decoupled L2 (weight decay)."""

    def __init__(
        self,
        lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._vel: dict[int, dict[str, np.ndarray]] = {}

    def _key(self, params: dict[str, ArrayDict]) -> int:
        return id(params)

    def step(
        self,
        params: dict[str, ArrayDict],
        grads: dict[str, ArrayDict],
    ) -> None:
        key = self._key(params)
        if key not in self._vel:
            self._vel[key] = {
                name: {k: np.zeros_like(v) for k, v in p.items()}
                for name, p in params.items()
            }
        vel = self._vel[key]
        for name, p in params.items():
            if name not in grads:
                continue
            for k, v in p.items():
                g = grads[name][k]
                if self.weight_decay:
                    g = g + self.weight_decay * v
                vel[name][k] = self.momentum * vel[name][k] + g
                p[k] = v - self.lr * vel[name][k]

    def zero_grad_state(self) -> None:
        self._vel.clear()
