"""Random search baseline (randomly sample and independently evaluate).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import time

from nasforge.core.seed import build_rng
from nasforge.core.types import SearchResult
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE
from nasforge.training.trainer import train_discrete

__all__ = ["search"]


def search(
    dataset,
    budget,
    seed: int,
    *,
    n_classes: int | None = None,
    img_size: int | None = None,
) -> SearchResult:
    space = DEFAULT_SPACE
    n_classes = n_classes or dataset.n_classes()
    img_size = img_size or dataset.in_shape()[1]
    net = Network(space, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    rng = build_rng(seed, stage="random-search")
    t0 = time.perf_counter()
    trace: list[float] = []
    best = None
    for _ in range(budget.max_steps):
        g = space.random_genotype(rng)
        _w, val_acc, test_acc = train_discrete(
            g, dataset, space, net, seed=int(rng.integers(1, 2**31)), epochs=budget.epochs
        )
        trace.append(float(val_acc))
        if best is None or val_acc > best[1]:
            best = (g, float(val_acc), float(test_acc))
    elapsed = time.perf_counter() - t0
    return SearchResult(
        method="random-search",
        best_genotype=best[0],
        best_val_acc=best[1],
        best_test_acc=best[2],
        n_evaluated=budget.max_steps,
        wall_time_s=elapsed,
        search_trace=tuple(trace),
        param_count=best[0].param_count(8, n_classes, 8),
        genotype_json=best[0].to_json(),
    )
