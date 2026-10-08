"""Operator forward / VJP gradient-check (G1).

Every operator's analytic input/weight VJP is compared against a central
finite-difference of the scalar loss ``0.5 * ||out||^2``. This is the foundational
invariant: if the hand-written VJPs are wrong, the whole differentiable search is
unsound.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.ops.registry import make_op


def _gradcheck(op, rng, x, p, eps=1e-5, n_check=10):
    out = op.forward(x, p)
    d_out = out
    dx = op.vjp_x(d_out, x, p)
    dw = op.vjp_w(d_out, x, p)

    rng_fd = np.random.default_rng(123)
    max_rel = 0.0

    # Finite difference on the input x (subsampled for speed).
    for _ in range(n_check):
        idx = tuple(int(rng_fd.integers(0, s)) for s in x.shape)
        orig = x[idx]
        x[idx] = orig + eps
        fp = 0.5 * float(np.sum(op.forward(x, p) ** 2))
        x[idx] = orig - eps
        fm = 0.5 * float(np.sum(op.forward(x, p) ** 2))
        x[idx] = orig
        fd = (fp - fm) / (2 * eps)
        max_rel = max(max_rel, abs(dx[idx] - fd) / (abs(fd) + 1e-12))

    # Finite difference on each weight tensor.
    for name, arr in p.items():
        if arr.size == 0:
            continue
        for _ in range(n_check):
            idx = tuple(int(rng_fd.integers(0, s)) for s in arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            fp = 0.5 * float(np.sum(op.forward(x, p) ** 2))
            arr[idx] = orig - eps
            fm = 0.5 * float(np.sum(op.forward(x, p) ** 2))
            arr[idx] = orig
            fd = (fp - fm) / (2 * eps)
            max_rel = max(max_rel, abs(dw[name][idx] - fd) / (abs(fd) + 1e-12))

    return max_rel


def test_conv_ops_vjp():
    rng = build_rng(1, "gradcheck")
    x = rng.standard_normal((2, 3, 7, 7))
    for name in ("conv_1x1", "conv_3x3", "conv_5x5"):
        op = make_op(name)
        p = op.init_params(rng, 3, 4)
        rel = _gradcheck(op, rng, x, p)
        assert rel < 1e-3, f"{name} max relative VJP error {rel:.2e}"


def test_paramfree_ops_vjp():
    rng = build_rng(2, "gradcheck2")
    x = rng.standard_normal((2, 3, 7, 7))
    for name in ("skip_connect", "avg_pool", "max_pool"):
        op = make_op(name)
        p = op.init_params(rng, 3, 4)  # {} for parameter-free ops
        rel = _gradcheck(op, rng, x, p)
        assert rel < 1e-3, f"{name} max relative VJP error {rel:.2e}"


def test_registry_single_source_of_truth():
    from nasforge.ops.registry import OP_INDEX, OP_REGISTRY, PRIMITIVE_ORDER

    assert len(PRIMITIVE_ORDER) == 6
    assert set(OP_REGISTRY) == set(PRIMITIVE_ORDER)
    assert OP_INDEX == {n: i for i, n in enumerate(PRIMITIVE_ORDER)}
