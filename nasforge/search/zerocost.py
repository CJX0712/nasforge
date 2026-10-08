"""Zero-Cost proxy search: rank random architectures by a training-free proxy.

Samples ``budget.max_steps`` random discrete genotypes, scores each with the
zero-cost proxies (no training), then independently evaluates the top-scoring one
to report its true generalisation. The proxy scores are stored in
``one_shot_scores`` for the rank-correlation invariant (G12).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import time

from nasforge.core.seed import build_rng
from nasforge.core.types import SearchResult
from nasforge.search.naswiz_metrics import zero_cost_scores
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE
from nasforge.training.trainer import train_discrete

__all__ = ["search"]


def _composite(scores: dict[str, float]) -> float:
    """Higher = better. Combine the signal proxies (lower params is better)."""
    return scores["snip"] + scores["grad_norm"] + scores["logits_var"] - 1e-4 * scores["n_params"]


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
    rng = build_rng(seed, stage="zerocost-search")
    t0 = time.perf_counter()

    candidates = []
    for _ in range(budget.max_steps):
        g = space.random_genotype(rng)
        sc = zero_cost_scores(net, g, dataset, seed=int(rng.integers(1, 2**31)))
        candidates.append((g, sc))
    candidates.sort(key=lambda c: _composite(c[1]), reverse=True)
    best_g, _ = candidates[0]

    _w, val_acc, test_acc = train_discrete(
        best_g, dataset, space, net, seed=int(rng.integers(1, 2**31)), epochs=budget.epochs
    )
    elapsed = time.perf_counter() - t0
    one_shot = {f"cand{i}_{k}": v for i, (_, sc) in enumerate(candidates) for k, v in sc.items()}
    return SearchResult(
        method="zerocost-proxy",
        best_genotype=best_g,
        best_val_acc=float(val_acc),
        best_test_acc=float(test_acc),
        n_evaluated=budget.max_steps,
        wall_time_s=elapsed,
        one_shot_scores=one_shot,
        param_count=best_g.param_count(8, n_classes, 8),
        genotype_json=best_g.to_json(),
    )
