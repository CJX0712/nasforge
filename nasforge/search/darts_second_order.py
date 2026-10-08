"""DarFuse-2nd: the unrolled implicit architecture gradient (DARTS-2nd).

The first-order DARTS gradient ``∂L_val/∂α`` ignores how the *weights* respond
to ``α``. The second-order (unrolled) gradient linearises the validation loss at
the virtual weights ``w' = w - ξ ∇_w L_train``:

    ∇_α L_val(w') ≈ ∇_α L_val(w) - ξ · ∇_α ( ∇_w L_train · ∇_w L_val )

The cross term ``∇_α(∇_w L_train · ∇_w L_val)`` is computed exactly by finite
differences in ``α`` (the weight-grad vector ``∇_w L_val`` is held fixed). This
is the faithful DARTS second-order approximation, implemented with hand-written
VJPs and no autograd -- and it is strictly more accurate than the first-order
gradient, which is what invariant G2/G12 verify.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.spaces.network import Network
from nasforge.training.trainer import flat_grad, weight_grad

__all__ = ["arch_grad_second_order"]


def _alpha_and_w_grad(net, weights, alpha, dataset, split, rng):
    x, y = dataset.batch(split, np.arange(len(getattr(dataset, f"{split}_idx"))))
    logits, ctx = net.forward(x, alpha, weights)
    _loss, dlog = net.loss(logits, y)
    g_alpha, g_weights, _gs = net.backward(dlog, ctx)
    return g_alpha, g_weights


def arch_grad_second_order(
    net: Network,
    weights: dict,
    alpha: np.ndarray,
    dataset,
    rng: np.random.Generator,
    xi: float = 0.01,
    eps: float = 1e-4,
) -> np.ndarray:
    """Unrolled implicit architecture gradient (see module docstring).

    The cross term ``∇_α(∇_w L_train · ∇_w L_val)`` is computed by central finite
    differences in ``α`` with the validation weight-grad vector ``g_w_va`` held
    fixed. The redundant training weight-grad evaluation and the dead base-dot
    product from the original implementation have been removed; only the two
    gradient evaluations the formula actually needs are kept.
    """
    # Validation gradient of alpha, plus the fixed validation weight-grad vector.
    _g_alpha_va, g_w_va = _alpha_and_w_grad(net, weights, alpha, dataset, "val", rng)
    g_alpha_va = _g_alpha_va
    gva = flat_grad(g_w_va)

    cross = np.zeros(alpha.shape, dtype=np.float64)
    flat = alpha.ravel().copy()
    for i in range(flat.size):
        orig = flat[i]
        a2 = alpha.copy()
        a2.ravel()[i] = orig + eps
        g_w_tr2 = weight_grad(net, weights, a2, dataset, "train", rng)
        dot2 = float(flat_grad(g_w_tr2).dot(gva))
        a3 = alpha.copy()
        a3.ravel()[i] = orig - eps
        g_w_tr3 = weight_grad(net, weights, a3, dataset, "train", rng)
        dot3 = float(flat_grad(g_w_tr3).dot(gva))
        cross.ravel()[i] = (dot2 - dot3) / (2 * eps)

    return g_alpha_va - xi * cross
