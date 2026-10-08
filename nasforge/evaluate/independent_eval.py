"""Independent (from-scratch) evaluation of a single discrete genotype.

This is the ground-truth generalisation signal. It trains the discrete
architecture on its own weights (optionally *inheriting* a starting point via
``start_weights``, which is what makes it directly comparable to the one-shot
evaluation in :mod:`nasforge.evaluate.one_shot`).

The rank-correlation invariant (G12) compares a set of one-shot scores against a
set of independent scores to confirm the cheap proxy tracks true generalisation.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import time

from nasforge.core.types import Genotype, SearchResult
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE
from nasforge.training.trainer import train_discrete

__all__ = ["independent_evaluate"]


def independent_evaluate(
    genotype: Genotype,
    dataset,
    budget,
    seed: int,
    start_weights: dict | None = None,
) -> SearchResult:
    """Train ``genotype`` independently and report its validation / test accuracy.

    Parameters
    ----------
    genotype:
        The discrete architecture to evaluate.
    dataset:
        A :class:`~nasforge.data.dataset.SyntheticDataset` (or compatible).
    budget:
        Supplies ``epochs`` for the independent training run.
    seed:
        Base seed for the training RNG stream.
    start_weights:
        If given, training *inherits* these weights (one-shot weight inheritance);
        otherwise weights are initialised afresh.

    Returns
    -------
    SearchResult
        ``method == "independent-eval"`` with ``best_val_acc`` / ``best_test_acc``.
    """
    space = DEFAULT_SPACE
    n_classes = dataset.n_classes()
    img_size = dataset.in_shape()[1]
    net = Network(space, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    t0 = time.perf_counter()
    _w, val_acc, test_acc = train_discrete(
        genotype,
        dataset,
        space,
        net,
        seed=seed,
        epochs=budget.epochs,
        init_weights=start_weights,
    )
    elapsed = time.perf_counter() - t0
    return SearchResult(
        method="independent-eval",
        best_genotype=genotype,
        best_val_acc=float(val_acc),
        best_test_acc=float(test_acc),
        n_evaluated=1,
        wall_time_s=elapsed,
        param_count=genotype.param_count(8, n_classes, 8),
        genotype_json=genotype.to_json(),
    )
