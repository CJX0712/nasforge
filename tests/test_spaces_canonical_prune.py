"""Canonicalisation, isomorphism pruning, and genotype validation (G3, G4, G6).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from nasforge.core.types import Genotype
from nasforge.spaces.canonical import canonical_key, canonicalize
from nasforge.spaces.genotype import edges_of, is_valid, op_histogram
from nasforge.spaces.prune import prune_duplicates, unique_canonical
from nasforge.spaces.search_space import DEFAULT_SPACE


def _make_genotype(edges):
    """Build a Genotype from ``(src, dst, op)`` triples grouped by destination node."""
    d = defaultdict(list)
    for s, dd, o in edges:
        d[dd].append((s, o))
    nodes = []
    for dst in sorted(d):
        nodes.append(tuple(sorted(d[dst], key=lambda e: e[0])))
    return Genotype(nodes=tuple(nodes), macro=("normal",))


SAMPLE_EDGES = [
    (1, 2, "conv_3x3"),
    (0, 2, "skip_connect"),
    (2, 3, "conv_1x1"),
    (1, 3, "avg_pool"),
    (3, 4, "max_pool"),
    (2, 4, "conv_5x5"),
]


def test_is_valid_accepts_sample():
    g = _make_genotype(SAMPLE_EDGES)
    assert is_valid(g, DEFAULT_SPACE)


def test_op_histogram_counts_all_edges():
    g = DEFAULT_SPACE.random_genotype(np.random.default_rng(1))
    hist = op_histogram(g, DEFAULT_SPACE)
    assert sum(hist.values()) == DEFAULT_SPACE.n_edges


def test_canonicalize_is_idempotent_under_input_swap():
    g = _make_genotype(SAMPLE_EDGES)
    c = canonicalize(g)
    c2 = canonicalize(c)
    assert canonical_key(c) == canonical_key(c2)


def test_prune_removes_isomorphic_duplicate():
    g = _make_genotype(SAMPLE_EDGES)
    candidates = [g, canonicalize(g)]
    uniq, n_removed = prune_duplicates(candidates)
    assert n_removed == 1
    assert len(uniq) == 1


def test_unique_canonical_collapses_isomorphic_class():
    g = _make_genotype(SAMPLE_EDGES)
    uniq = unique_canonical([g, canonicalize(g), g])
    assert len(uniq) == 1
    keys = {canonical_key(c) for c in uniq}
    assert len(keys) == 1


def test_edges_of_roundtrips_nodes():
    g = _make_genotype(SAMPLE_EDGES)
    assert {(s, d, o) for s, d, o in edges_of(g)} == set(SAMPLE_EDGES)
