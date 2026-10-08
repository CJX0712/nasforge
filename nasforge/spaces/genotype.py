"""Genotype helpers (edge extraction / validation).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Genotype
from nasforge.spaces.search_space import SearchSpace

__all__ = ["edges_of", "is_valid", "op_histogram"]


def edges_of(genotype: Genotype) -> list[tuple[int, int, str]]:
    """Flatten a genotype into ``(src, dst, op)`` triples."""
    out = []
    for j, node in enumerate(genotype.nodes):
        dst = j + 2  # n_input == 2
        for src, op in node:
            out.append((src, dst, op))
    return out


def is_valid(genotype: Genotype, space: SearchSpace) -> bool:
    """Check every edge references a known operator and a valid acyclic src<dst."""
    op_set = set(space.op_order)
    for src, dst, op in edges_of(genotype):
        if op not in op_set:
            return False
        if not (src < dst):
            return False
    return True


def op_histogram(genotype: Genotype, space: SearchSpace) -> dict[str, int]:
    """Count how many edges use each operator."""
    hist = dict.fromkeys(space.op_order, 0)
    for _s, _d, op in edges_of(genotype):
        hist[op] = hist.get(op, 0) + 1
    return hist
