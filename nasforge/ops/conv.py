"""Convolution operators with a manual im2col forward and col2im backward.

All three kernels use ``stride=1`` with symmetric padding so that spatial size is
preserved (``H_out == H_in``). This is what lets a cell sum the outputs of every
operator on equal-sized tensors during the DARTS continuous relaxation.

``vjp_x`` for any conv does NOT depend on ``x`` (it is ``convT(dy, W)``), so
``vjp_x_second`` is identically zero -- the second-order signal in DarFuse comes
from the G2 cross term, never from operator-level second derivatives.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict
from nasforge.ops.init import he_uniform

__all__ = ["Conv1x1", "Conv3x3", "Conv5x5"]


def _im2col(x: np.ndarray, kh: int, kw: int, pad: int) -> np.ndarray:
    """Slide a ``kh x kw`` window over ``x`` into column form.

    Returns
    -------
    np.ndarray
        Shape ``(B, Cin*kh*kw, Hout*Wout)``.
    """
    b, cin, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    hout = h + 2 * pad - kh + 1
    wout = w + 2 * pad - kw + 1
    col = np.empty((b, cin * kh * kw, hout * wout), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i : i + hout, j : j + wout].reshape(b, cin, hout * wout)
            col[:, (i * kw + j) * cin : (i * kw + j + 1) * cin, :] = patch
    return col


def _col2im(col: np.ndarray, x_shape: tuple[int, int, int, int], kh: int, kw: int, pad: int) -> np.ndarray:
    """Accumulate a gradient column back into ``grad_x`` with the same windows."""
    b, cin, h, w = x_shape
    hout = h + 2 * pad - kh + 1
    wout = w + 2 * pad - kw + 1
    xp = np.zeros((b, cin, h + 2 * pad, w + 2 * pad), dtype=col.dtype)
    for i in range(kh):
        for j in range(kw):
            slc = col[:, (i * kw + j) * cin : (i * kw + j + 1) * cin, :].reshape(b, cin, hout, wout)
            xp[:, :, i : i + hout, j : j + wout] += slc
    return xp[:, :, pad : pad + h, pad : pad + w]


class _Conv:
    """Shared conv implementation, parameterised by kernel size."""

    name = "conv"
    kernel = 3

    def init_params(self, rng: np.random.Generator, in_c: int, out_c: int) -> ArrayDict:
        kh = kw = self.kernel
        fan_in = in_c * kh * kw
        bound = he_uniform(rng, fan_in, out_c)
        w = rng.uniform(-bound, bound, size=(out_c, in_c, kh, kw))
        b = np.zeros(out_c, dtype=np.float64)
        return {"w": w, "b": b}

    def forward(self, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        kh = kw = self.kernel
        pad = kh // 2
        col = _im2col(x, kh, kw, pad)
        w_flat = p["w"].reshape(p["w"].shape[0], -1)
        flat = col.transpose(0, 2, 1)
        out = flat @ w_flat.T + p["b"]
        b = x.shape[0]
        cout = p["w"].shape[0]
        hout = x.shape[2] + 2 * pad - kh + 1
        wout = x.shape[3] + 2 * pad - kw + 1
        return out.transpose(0, 2, 1).reshape(b, cout, hout, wout)

    def vjp_x(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        kh = kw = self.kernel
        pad = kh // 2
        b, cout, hout, wout = dy.shape
        dy_flat = dy.transpose(0, 2, 3, 1).reshape(b * hout * wout, cout)
        w_flat = p["w"].reshape(cout, -1)
        gcol = dy_flat @ w_flat
        gcol = gcol.reshape(b, hout * wout, -1).transpose(0, 2, 1)
        return _col2im(gcol, x.shape, kh, kw, pad)

    def vjp_w(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> ArrayDict:
        kh = kw = self.kernel
        pad = kh // 2
        col = _im2col(x, kh, kw, pad)
        b, cout, hout, wout = dy.shape
        col_t = col.transpose(0, 2, 1)  # (B, Hout*Wout, Cin*kh*kw)
        dy_t = dy.transpose(0, 2, 3, 1).reshape(b * hout * wout, cout)
        d_w = (col_t.reshape(-1, col_t.shape[-1]).T @ dy_t).T  # (Cout, Cin*kh*kw)
        d_w = d_w.reshape(p["w"].shape)
        d_b = dy.sum(axis=(0, 2, 3))
        return {"w": d_w, "b": d_b}

    def vjp_x_second(
        self, dy: np.ndarray, x: np.ndarray, p: ArrayDict, dx_seed: np.ndarray
    ) -> np.ndarray:
        del dy, p, dx_seed
        return np.zeros_like(x)

    def param_count(self, p: ArrayDict) -> int:
        return int(p["w"].size + p["b"].size)


class Conv1x1(_Conv):
    """1x1 convolution (pointwise)."""

    name = "conv_1x1"
    kernel = 1


class Conv3x3(_Conv):
    """3x3 convolution with same padding."""

    name = "conv_3x3"
    kernel = 3


class Conv5x5(_Conv):
    """5x5 convolution with same padding."""

    name = "conv_5x5"
    kernel = 5
