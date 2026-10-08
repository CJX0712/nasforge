"""Benchmark training-free / one-shot proxies against ground-truth evaluation.

* **G12 (rank correlation)** -- the zero-cost proxy (NAS-WiZ composite: snip +
  grad_norm + logits_var, penalised by parameter count) should rank architectures
  positively with their independently-trained accuracy.
* **G11 (weight inheritance)** -- the one-shot evaluation (reusing the trained
  supernet weights) is never worse than an independent from-scratch evaluation of
  the same genotype, because inheritance carries more information.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import Budget
from nasforge.evaluate import independent_evaluate, one_shot_evaluate, spearman
from nasforge.search.naswiz_metrics import zero_cost_scores
from nasforge.search.zerocost import _composite
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE

__all__ = ["benchmark_zerocost_vs_independent", "benchmark_one_shot_vs_independent"]


def benchmark_zerocost_vs_independent(
    dataset,
    budget: Budget,
    seed: int,
    n_arch: int = 12,
    *,
    rng_seed: int = 7,
) -> dict:
    """Return proxy scores, independent accuracies, and their Spearman correlation."""
    space = DEFAULT_SPACE
    n_classes = dataset.n_classes()
    img_size = dataset.in_shape()[1]
    net = Network(space, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    rng = np.random.default_rng(rng_seed)
    comp: list[float] = []
    indep: list[float] = []
    for i in range(n_arch):
        g = space.random_genotype(rng)
        sc = zero_cost_scores(net, g, dataset, seed=300 + i)
        comp.append(_composite(sc))
        ie = independent_evaluate(g, dataset, budget, seed=200 + i)
        indep.append(float(ie.best_val_acc))
    return {
        "proxy": comp,
        "independent": indep,
        "spearman": spearman(comp, indep),
        "n": n_arch,
        "std_independent": float(np.std(indep)),
    }


def benchmark_one_shot_vs_independent(
    dataset,
    supernet_weights: dict,
    budget: Budget,
    seed: int,
    n_arch: int = 8,
    *,
    rng_seed: int = 3,
) -> dict:
    """Compare one-shot (shared-weight) vs independent accuracy for random genotypes."""
    space = DEFAULT_SPACE
    rng = np.random.default_rng(rng_seed)
    one: list[float] = []
    indep: list[float] = []
    for i in range(n_arch):
        g = space.random_genotype(rng)
        o = one_shot_evaluate(g, supernet_weights, dataset)
        ie = independent_evaluate(g, dataset, budget, seed=400 + i)
        one.append(float(o["val_acc"]))
        indep.append(float(ie.best_val_acc))
    return {
        "one_shot": one,
        "independent": indep,
        "spearman": spearman(one, indep),
        "n": n_arch,
        "one_ge_indep": sum(1 for a, b in zip(one, indep) if a >= b - 1e-9),
    }
