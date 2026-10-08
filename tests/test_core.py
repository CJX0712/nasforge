"""Core types and search-space well-formedness (G5).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np
import pytest

from nasforge.core.errors import SearchError
from nasforge.core.types import Budget, Genotype
from nasforge.spaces.genotype import edges_of, is_valid
from nasforge.spaces.search_space import DEFAULT_SPACE, SearchSpace


def test_budget_demo_is_time_bounded():
    b = Budget.demo()
    assert b.max_steps == 6 and b.epochs == 2


def test_budget_research_is_larger():
    b = Budget.research()
    assert b.max_steps == 200 and b.epochs == 12


def test_default_space_topology():
    sp = DEFAULT_SPACE
    assert sp.n_input == 2
    assert sp.n_nodes == 3
    assert sp.n_edges == 6  # 3 internal nodes * 2 edges each
    assert sp.alpha_shape() == (6, 6)


def test_default_space_edges_acyclic():
    for src, dst in DEFAULT_SPACE.edges:
        assert src < dst
        assert DEFAULT_SPACE.n_input <= dst < DEFAULT_SPACE.n_input + DEFAULT_SPACE.n_nodes


def test_back_edge_space_rejected():
    with pytest.raises(SearchError):
        SearchSpace(edges=((0, 1), (1, 0)))  # (1,0) violates src < dst


def test_duplicate_op_order_rejected():
    with pytest.raises(SearchError):
        SearchSpace(
            op_order=(
                "skip_connect",
                "skip_connect",
                "conv_3x3",
                "conv_5x5",
                "avg_pool",
                "max_pool",
            )
        )


def test_discretize_picks_argmax_op_per_edge():
    sp = DEFAULT_SPACE
    alpha = np.zeros(sp.alpha_shape())
    alpha[0, 2] = 5.0  # edge 0 -> conv_3x3
    alpha[1, 0] = 5.0  # edge 1 -> skip_connect
    g = sp.discretize(alpha)
    assert is_valid(g, sp)
    edges = {(s, d): o for s, d, o in edges_of(g)}
    assert edges[(1, 2)] == "conv_3x3"  # edge 0 is (1,2)
    assert edges[(0, 2)] == "skip_connect"  # edge 1 is (0,2)
    assert len(edges_of(g)) == sp.n_edges


def test_genotype_json_roundtrip():
    g = DEFAULT_SPACE.random_genotype(np.random.default_rng(0))
    assert Genotype.from_json(g.to_json()).nodes == g.nodes
