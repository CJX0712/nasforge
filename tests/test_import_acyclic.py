"""Acyclic search space import-safety (G5).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import pytest

from nasforge.core.errors import SearchError
from nasforge.spaces.search_space import DEFAULT_SPACE, SearchSpace


def test_all_default_edges_acyclic():
    for src, dst in DEFAULT_SPACE.edges:
        assert src < dst


def test_back_edge_rejected_on_import():
    with pytest.raises(SearchError):
        SearchSpace(edges=((0, 1), (2, 1)))  # dst < src


def test_dst_must_be_internal_node():
    with pytest.raises(SearchError):
        SearchSpace(edges=((0, 5), (1, 5)))  # dst 5 is out of range


def test_space_import_does_not_raise():
    # Importing the space must not construct an invalid (cyclic) default.
    assert DEFAULT_SPACE.n_edges == 6
