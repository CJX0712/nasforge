"""Identity-style operator: skip connection.

``skip_connect`` passes the input through unchanged. It adds no parameters and its
input VJP is the identity, so ``vjp_x_second`` is zero. Under the sum-aggregation
cell (every edge output is summed at the node), ``skip_connect`` is the only
identity-like primitive needed; we deliberately drop the redundant ``none`` /
zero operator from the DARTS catalogue.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict

__all__ = ["SkipConnect"]


class SkipConnect:
    """Residual / identity connection."""

    name = "skip_connect"

    def init_params(self, rng: np.random.Generator, in_c: int, out_c: int) -> ArrayDict:
        del rng, in_c, out_c
        return {}

    def forward(self, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del p
        return x

    def vjp_x(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del x, p
        return dy

    def vjp_w(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> ArrayDict:
        del dy, x, p
        return {}

    def vjp_x_second(
        self, dy: np.ndarray, x: np.ndarray, p: ArrayDict, dx_seed: np.ndarray
    ) -> np.ndarray:
        del dy, p, dx_seed
        return np.zeros_like(x)

    def param_count(self, p: ArrayDict) -> int:
        del p
        return 0
