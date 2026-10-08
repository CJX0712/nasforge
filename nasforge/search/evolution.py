"""Regularized Evolution (Real et al., AAAI'2019) for NasForge.

A tournament-based evolutionary search: at each step the best of ``sample``
randomly drawn population members is mutated, the child is *independently*
evaluated, and the oldest member is removed (aging keeps the population fresh).
This is the faithful, training-based NAS baseline the DarFuse one-shot methods
are compared against.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import time

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.core.types import Genotype, SearchResult
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE, PRIMITIVE_ORDER
from nasforge.training.trainer import train_discrete

__all__ = ["search"]


def _mutate(g: Genotype, rng: np.random.Generator, space) -> Genotype:
    """Flip one random edge's operator to a different primitive."""
    node_tuples = []
    for node in g.nodes:
        new_edges = []
        for src, op in node:
            if rng.random() < 1.0 / max(1, len(g.nodes) * 2):
                others = [o for o in PRIMITIVE_ORDER if o != op]
                new_edges.append((src, others[int(rng.integers(0, len(others)))]))
            else:
                new_edges.append((src, op))
        node_tuples.append(tuple(new_edges))
    return Genotype(nodes=tuple(node_tuples), macro=g.macro)


def search(
    dataset,
    budget,
    seed: int,
    *,
    population: int = 20,
    sample: int = 10,
    n_classes: int | None = None,
    img_size: int | None = None,
) -> SearchResult:
    space = DEFAULT_SPACE
    n_classes = n_classes or dataset.n_classes()
    img_size = img_size or dataset.in_shape()[1]
    net = Network(space, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    rng = build_rng(seed, stage="evolution")
    t0 = time.perf_counter()

    pop: list[tuple[Genotype, float, float]] = []
    for _ in range(population):
        g = space.random_genotype(rng)
        _w, val_acc, test_acc = train_discrete(
            g, dataset, space, net, seed=int(rng.integers(1, 2**31)), epochs=budget.epochs
        )
        pop.append((g, float(val_acc), float(test_acc)))

    trace: list[float] = []
    for _ in range(budget.max_steps):
        drawn = rng.choice(len(pop), size=sample, replace=False)
        best_i = int(drawn[int(np.argmax([pop[i][1] for i in drawn]))])
        child = _mutate(pop[best_i][0], rng, space)
        _w, val_acc, test_acc = train_discrete(
            child, dataset, space, net, seed=int(rng.integers(1, 2**31)), epochs=budget.epochs
        )
        pop.append((child, float(val_acc), float(test_acc)))
        if len(pop) > population:
            pop.pop(0)
        trace.append(float(val_acc))

    best = max(pop, key=lambda m: m[1])
    elapsed = time.perf_counter() - t0
    return SearchResult(
        method="regularized-evolution",
        best_genotype=best[0],
        best_val_acc=best[1],
        best_test_acc=best[2],
        n_evaluated=population + budget.max_steps,
        wall_time_s=elapsed,
        search_trace=tuple(trace),
        param_count=best[0].param_count(8, n_classes, 8),
        genotype_json=best[0].to_json(),
    )
