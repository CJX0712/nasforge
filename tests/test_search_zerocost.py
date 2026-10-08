"""Zero-cost proxy invariants (G10): finite, bounded, training-free.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import Budget
from nasforge.data.dataset import SyntheticDataset
from nasforge.search import zerocost_search
from nasforge.search.naswiz_metrics import zero_cost_scores
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE


def test_zero_cost_scores_finite_and_bounded():
    ds = SyntheticDataset(seed=20260408)
    sp = DEFAULT_SPACE
    net = Network(sp, channels=8, n_cells=2, n_classes=4, img_size=9)
    g = sp.random_genotype(np.random.default_rng(1))
    sc = zero_cost_scores(net, g, ds, seed=11)
    for k in ("n_params", "grad_norm", "snip", "logits_var"):
        assert np.isfinite(sc[k]), k
    assert sc["n_params"] > 0
    assert sc["grad_norm"] >= 0
    assert sc["snip"] >= 0
    assert sc["logits_var"] >= 0


def test_zerocost_search_runs_and_reports():
    ds = SyntheticDataset(seed=20260408)
    b = Budget.demo()  # 6 proxy candidates + 1 independent eval
    r = zerocost_search(ds, b, seed=4)
    assert r.method == "zerocost-proxy"
    assert 0.0 <= r.best_val_acc <= 1.0
    assert 0.0 <= r.best_test_acc <= 1.0
    assert len(r.one_shot_scores) > 0
