"""Canonicalisation of discrete genotypes under input-node permutation.

A cell has two input nodes (the previous two cells). Swapping them produces an
*architecturally equivalent* network, so two genotypes related by this swap are
isomorphic. :func:`canonicalize` returns the lexicographically smallest of the
original and swapped versions, giving a unique representative (used by the
isomorphism-pruning invariant G6/G7).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Genotype
from nasforge.spaces.search_space import SearchSpace

__all__ = ["canonicalize", "canonical_key"]


def _swap_inputs(genotype: Genotype) -> Genotype:
    """Swap the two input node indices (0 <-> 1) on every edge."""
    node_tuples = []
    for node in genotype.nodes:
        new_edges = []
        for src, op in node:
            s = 1 - src if src in (0, 1) else src
            new_edges.append((s, op))
        node_tuples.append(tuple(sorted(new_edges, key=lambda e: e[0])))
    return Genotype(nodes=tuple(node_tuples), macro=genotype.macro)


def canonical_key(genotype: Genotype) -> tuple:
    """A hashable, order-independent key for the genotype's edge set."""
    return tuple(
        tuple(sorted((src, op) for src, op in node)) for node in genotype.nodes
    )


def canonicalize(genotype: Genotype, space: SearchSpace | None = None) -> Genotype:
    """Return the canonical (isomorphism-class) representative of ``genotype``."""
    del space
    swapped = _swap_inputs(genotype)
    return genotype if canonical_key(genotype) <= canonical_key(swapped) else swapped
