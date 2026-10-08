"""DarFuse: one-shot differentiable architecture search (DARTS family).

First-order (DarFuse-1st) and second-order (DarFuse-2nd) variants share the same
supernet training loop. The architecture logits ``alpha`` are updated from the
*validation* loss gradient, with three DarFuse refinements layered on top of the
baseline DARTS gradient:

* **GRL** -- the first-order *training* architecture gradient is reversed
  (``g_alpha = g_val - grl_scale * g_train``) so the search is pushed towards
  generalisation rather than train overfitting.
* **architecture weight decay** -- an L2 penalty on ``alpha`` for stability.
* **weight inheritance** -- the found discrete genotype is fine-tuned from the
  supernet weights instead of from scratch (one-shot evaluation).

DarFuse-2nd replaces the first-order validation gradient with the *unrolled
implicit* gradient (see :mod:`nasforge.search.darts_second_order`), which is the
more accurate Taylor expansion the task brief asks for.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import time

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.core.types import SearchResult
from nasforge.search.darts_second_order import arch_grad_second_order
from nasforge.search.grl import grl
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import (
    DEFAULT_SPACE,
    genotype_to_onehot,
)
from nasforge.training.optimizer import SGD
from nasforge.training.trainer import (
    evaluate,
    run_epoch_weights,
    train_discrete,
)

__all__ = ["search"]


def _alpha_grad(net, weights, alpha, dataset, split, rng):
    x, y = dataset.batch(split, np.arange(len(getattr(dataset, f"{split}_idx"))))
    logits, ctx = net.forward(x, alpha, weights)
    _loss, dlog = net.loss(logits, y)
    g_alpha, _gw, _gs = net.backward(dlog, ctx)
    return g_alpha


def search(
    dataset,
    budget,
    seed: int,
    *,
    second_order: bool = False,
    alpha_lr: float = 0.005,
    w_lr: float = 0.05,
    alpha_wd: float = 1e-3,
    grl_scale: float = 1.0,
    xi: float = 0.01,
    epochs_per_step: int | None = None,
    n_classes: int | None = None,
    img_size: int | None = None,
) -> SearchResult:
    space = DEFAULT_SPACE
    n_classes = n_classes or dataset.n_classes()
    img_size = img_size or dataset.in_shape()[1]
    net = Network(space, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    rng = build_rng(seed, stage="darfuse")
    weights = net.init_weights(rng)
    alpha = space.random_alpha(rng)
    w_opt = SGD(lr=w_lr, momentum=0.9, weight_decay=1e-4)
    epochs = epochs_per_step or budget.epochs
    trace: list[float] = []
    t0 = time.perf_counter()

    for step in range(budget.max_steps):
        for _ in range(epochs):
            run_epoch_weights(net, weights, alpha, dataset, w_opt, rng)

        g_val = (
            arch_grad_second_order(net, weights, alpha, dataset, rng, xi=xi)
            if second_order
            else _alpha_grad(net, weights, alpha, dataset, "val", rng)
        )
        g_train = _alpha_grad(net, weights, alpha, dataset, "train", rng)
        # DarFuse: follow validation, reverse training (GRL), decay alpha.
        g_alpha = g_val - grl_scale * grl(g_train)
        g_alpha = g_alpha - alpha_wd * alpha
        alpha = alpha - alpha_lr * g_alpha

        geno = space.discretize(alpha)
        a_one = genotype_to_onehot(geno, space)
        _vl, val_acc = evaluate(net, weights, a_one, dataset, "val")
        trace.append(float(val_acc))

    final_geno = space.discretize(alpha)
    # weight inheritance: fine-tune from the supernet weights
    _w2, val_acc, test_acc = train_discrete(
        final_geno, dataset, space, net, seed=int(seed) + 1, epochs=budget.epochs, init_weights=weights
    )
    elapsed = time.perf_counter() - t0
    return SearchResult(
        method="darfuse-2nd" if second_order else "darfuse-1st",
        best_genotype=final_geno,
        best_val_acc=float(val_acc),
        best_test_acc=float(test_acc),
        n_evaluated=budget.max_steps,
        wall_time_s=elapsed,
        search_trace=tuple(trace),
        param_count=final_geno.param_count(8, n_classes, 8),
        genotype_json=final_geno.to_json(),
        # The trained one-shot supernet weights back the G12 rank-correlation
        # invariant: one-shot evaluation reuses them, independent evaluation does not.
        extra={"supernet_weights": weights},
    )
