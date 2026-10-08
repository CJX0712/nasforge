"""Pooling operators (avg / max) with analytic VJPs and same-padding.

Both use ``stride=1`` with symmetric padding so that ``H_out == H_in``. The
forward is a sliding-window reduction; the backward distributes the output
gradient back through the (average / max) mask.

``vjp_x`` does not depend on ``x`` for ``AvgPool`` (it is a fixed upsample), and
for ``MaxPool`` the mask is treated as a local constant, so ``vjp_x_second`` is
identically zero for both. The second-order signal in DarFuse therefore comes
from the GRL / mixed-weight cross term, never from operator-level curvature.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict

__all__ = ["AvgPool", "MaxPool"]


class AvgPool:
    """2D average pooling, kernel ``k`` x ``k``, stride 1, same padding."""

    name = "avg_pool"
    kernel = 3

    def init_params(self, rng: np.random.Generator, in_c: int, out_c: int) -> ArrayDict:
        # Pooling is parameter-free.
        del rng, in_c, out_c
        return {}

    def forward(self, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del p
        kh = kw = self.kernel
        pad = kh // 2
        b, c, h, w = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.zeros((b, c, h, w), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                out += xp[:, :, i : i + h, j : j + w]
        return out / (kh * kw)

    def vjp_x(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del x, p
        kh = kw = self.kernel
        pad = kh // 2
        b, c, h, w = dy.shape
        full = dy / (kh * kw)
        g = np.zeros((b, c, h + 2 * pad, w + 2 * pad), dtype=dy.dtype)
        for i in range(kh):
            for j in range(kw):
                g[:, :, i : i + h, j : j + w] += full
        return g[:, :, pad : pad + h, pad : pad + w]

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


class MaxPool:
    """2D max pooling, kernel ``k`` x ``k``, stride 1, same padding.

    Ties resolve to the lowest linear index (argmax returns the first max), which
    is what keeps the backward deterministic under the seeded RNG contract.
    """

    name = "max_pool"
    kernel = 3

    def init_params(self, rng: np.random.Generator, in_c: int, out_c: int) -> ArrayDict:
        del rng, in_c, out_c
        return {}

    def forward(self, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del p
        kh = kw = self.kernel
        pad = kh // 2
        b, c, h, w = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.full((b, c, h, w), -np.inf, dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                out = np.maximum(out, xp[:, :, i : i + h, j : j + w])
        return out

    def vjp_x(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        del p
        kh = kw = self.kernel
        pad = kh // 2
        b, c, h, w = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        # Recompute the argmax mask (ties -> first max via np.argmax).
        window = np.empty((b, c, h, w, kh * kw), dtype=x.dtype)
        idx = 0
        for i in range(kh):
            for j in range(kw):
                window[:, :, :, :, idx] = xp[:, :, i : i + h, j : j + w]
                idx += 1
        flat = window.reshape(b * c * h * w, kh * kw)
        amax = flat.argmax(axis=1)  # lowest linear index on ties
        mask = np.zeros_like(flat)
        mask[np.arange(flat.shape[0]), amax] = 1.0
        mask = mask.reshape(b, c, h, w, kh, kw)
        g = np.zeros((b, c, h + 2 * pad, w + 2 * pad), dtype=dy.dtype)
        for i in range(kh):
            for j in range(kw):
                g[:, :, i : i + h, j : j + w] += mask[:, :, :, :, i, j] * dy
        return g[:, :, pad : pad + h, pad : pad + w]

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
