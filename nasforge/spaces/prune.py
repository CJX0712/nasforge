"""Isomorphism pruning: deduplicate genotypes by their canonical form.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Genotype
from nasforge.spaces.canonical import canonical_key, canonicalize
from nasforge.spaces.search_space import SearchSpace

__all__ = ["unique_canonical", "prune_duplicates"]


def unique_canonical(genotypes: list[Genotype], space: SearchSpace | None = None) -> list[Genotype]:
    """Return one representative per isomorphism class (input-swap invariant)."""
    seen: dict[tuple, Genotype] = {}
    for g in genotypes:
        c = canonicalize(g, space)
        k = canonical_key(c)
        if k not in seen:
            seen[k] = c
    return list(seen.values())


def prune_duplicates(genotypes: list[Genotype], space: SearchSpace | None = None) -> tuple[list[Genotype], int]:
    """Return ``(unique, n_removed)``."""
    uniq = unique_canonical(genotypes, space)
    return uniq, len(genotypes) - len(uniq)
