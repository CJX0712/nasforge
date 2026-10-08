"""One-shot (weight-inherited) evaluation of a discretised supernet.

After the one-shot search finds ``alpha``, we discretise to a genotype and reuse
the *trained supernet weights* to score it without retraining. This is the
"warm" evaluation that the rank-correlation invariant (G12) compares against the
fully independent evaluation.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Genotype
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE, genotype_to_onehot
from nasforge.training.trainer import evaluate

__all__ = ["one_shot_evaluate"]


def one_shot_evaluate(
    genotype: Genotype,
    weights: dict,
    dataset,
    *,
    n_classes: int | None = None,
    img_size: int | None = None,
) -> dict:
    """Score ``genotype`` with the already-trained supernet ``weights``."""
    n_classes = n_classes or dataset.n_classes()
    img_size = img_size or dataset.in_shape()[1]
    net = Network(DEFAULT_SPACE, channels=8, n_cells=2, n_classes=n_classes, img_size=img_size)
    alpha = genotype_to_onehot(genotype, DEFAULT_SPACE)
    _vl, val_acc = evaluate(net, weights, alpha, dataset, "val")
    _tl, test_acc = evaluate(net, weights, alpha, dataset, "test")
    return {"val_acc": float(val_acc), "test_acc": float(test_acc)}
