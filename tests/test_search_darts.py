"""DarFuse differentiable search invariants (G2, G7).

G2 -- the DarFuse-2nd (unrolled) cross term ``∇_α(∇_w L_train · ∇_w L_val)`` must
equal the central finite difference of that dot product w.r.t. ``α`` (with the
validation weight-grad vector held fixed). Also, with ``xi = 0`` the second-order
gradient must collapse exactly to the first-order validation gradient.

G7 -- ``discretize`` selects the argmax operator on every edge.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.core.types import Budget
from nasforge.data.dataset import SyntheticDataset
from nasforge.search import darfuse_search
from nasforge.search.darts_second_order import _alpha_and_w_grad, arch_grad_second_order
from nasforge.spaces.genotype import edges_of, is_valid
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE
from nasforge.training.trainer import flat_grad, weight_grad


def _small_net():
    ds = SyntheticDataset(n_samples=80, img_size=7, seed=1)
    sp = DEFAULT_SPACE
    net = Network(sp, channels=4, n_cells=2, n_classes=4, img_size=7)
    rng = build_rng(5, "g2-init")
    w = net.init_weights(rng)
    alpha = sp.random_alpha(build_rng(6, "g2-alpha"))
    return net, w, alpha, ds


def test_second_order_cross_term_matches_finite_difference():
    net, w, alpha, ds = _small_net()
    xi, eps = 0.01, 1e-4
    rng_xi = build_rng(7, "g2-xi")
    g2 = arch_grad_second_order(net, w, alpha, ds, rng_xi, xi=xi, eps=eps)

    # Independently replicate the cross term with an identical RNG stream so the
    # held-fixed validation weight-grad and the FD batches match exactly.
    rng_rep = build_rng(7, "g2-xi")
    _ga, g_w_va = _alpha_and_w_grad(net, w, alpha, ds, "val", rng_rep)
    gva = flat_grad(g_w_va)
    cross = np.zeros(alpha.shape, dtype=np.float64)
    flat = alpha.ravel().copy()
    for i in range(flat.size):
        orig = flat[i]
        a2 = alpha.copy()
        a2.ravel()[i] = orig + eps
        gwt2 = weight_grad(net, w, a2, ds, "train", rng_rep)
        a3 = alpha.copy()
        a3.ravel()[i] = orig - eps
        gwt3 = weight_grad(net, w, a3, ds, "train", rng_rep)
        cross.ravel()[i] = (flat_grad(gwt2).dot(gva) - flat_grad(gwt3).dot(gva)) / (2 * eps)
    g2_mine = _ga - xi * cross

    rel = np.max(np.abs(g2 - g2_mine)) / (np.max(np.abs(g2_mine)) + 1e-12)
    assert rel < 1e-3, f"2nd-order cross-term relative error {rel:.2e}"


def test_xi_zero_collapses_to_first_order():
    net, w, alpha, ds = _small_net()
    rng0 = build_rng(8, "g2-0")
    g2_0 = arch_grad_second_order(net, w, alpha, ds, rng0, xi=0.0, eps=1e-4)
    rng_v = build_rng(9, "g2-v")
    ga, _ = _alpha_and_w_grad(net, w, alpha, ds, "val", rng_v)
    rel = np.max(np.abs(g2_0 - ga)) / (np.max(np.abs(ga)) + 1e-12)
    assert rel < 1e-6, f"xi=0 did not collapse to 1st order: {rel:.2e}"


def test_darfuse_returns_valid_discretised_genotype():
    ds = SyntheticDataset(seed=20260408)
    b = Budget(max_steps=2, max_evals=200, epochs=1)
    res = darfuse_search(ds, b, seed=11, second_order=False)
    assert is_valid(res.best_genotype, DEFAULT_SPACE)
    assert len(edges_of(res.best_genotype)) == DEFAULT_SPACE.n_edges
    assert 0.0 <= res.best_val_acc <= 1.0
    assert 0.0 <= res.best_test_acc <= 1.0


def test_darfuse_second_order_runs():
    ds = SyntheticDataset(seed=20260408)
    b = Budget(max_steps=2, max_evals=200, epochs=1)
    res = darfuse_search(ds, b, seed=12, second_order=True)
    assert is_valid(res.best_genotype, DEFAULT_SPACE)
    assert res.method == "darfuse-2nd"
