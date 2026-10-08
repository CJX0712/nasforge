"""Cross-cutting invariants: determinism contract and reproducibility (G8).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.data.dataset import SyntheticDataset


def test_dataset_byte_reproducible_same_seed():
    a = SyntheticDataset(seed=20260408)
    b = SyntheticDataset(seed=20260408)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.val_idx, b.val_idx)


def test_dataset_different_seed_differs():
    a = SyntheticDataset(seed=1)
    b = SyntheticDataset(seed=2)
    assert not np.array_equal(a.X, b.X)


def test_build_rng_deterministic_per_stage():
    r1 = build_rng(123, "x")
    r2 = build_rng(123, "x")
    assert np.array_equal(r1.standard_normal(5), r2.standard_normal(5))


def test_build_rng_different_stage_is_separated():
    r1 = build_rng(123, "x")
    r3 = build_rng(123, "y")
    assert not np.array_equal(r1.standard_normal(5), r3.standard_normal(5))


def test_search_seed_reproducibility():
    from nasforge.core.types import Budget
    from nasforge.search import darfuse_search

    ds = SyntheticDataset(seed=20260408)
    b = Budget(max_steps=2, max_evals=200, epochs=1)
    r1 = darfuse_search(ds, b, seed=42, second_order=False)
    r2 = darfuse_search(ds, b, seed=42, second_order=False)
    assert r1.genotype_json == r2.genotype_json
    assert r1.best_val_acc == r2.best_test_acc or r1.best_val_acc == r2.best_val_acc
