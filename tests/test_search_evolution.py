"""Regularized Evolution baseline invariant (faithful training-based NAS).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Budget
from nasforge.data.dataset import SyntheticDataset
from nasforge.search import evolution_search
from nasforge.spaces.genotype import is_valid
from nasforge.spaces.search_space import DEFAULT_SPACE


def test_evolution_returns_valid_genotype():
    ds = SyntheticDataset(seed=20260408)
    # Tiny budget so the test stays fast: 3 initial + 2 mutated children.
    b = Budget(max_steps=2, max_evals=200, epochs=1)
    r = evolution_search(ds, b, seed=3, population=3, sample=2)
    assert r.method == "regularized-evolution"
    assert is_valid(r.best_genotype, DEFAULT_SPACE)
    assert 0.0 <= r.best_val_acc <= 1.0
    assert 0.0 <= r.best_test_acc <= 1.0
    assert r.n_evaluated >= 3
