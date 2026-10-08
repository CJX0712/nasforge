"""Training utilities shared by the search and evaluation layers.

The same mixed-cell forward/backward trains both the one-shot supernet (with a
learnable ``alpha``) and a discrete architecture (with a fixed one-hot ``alpha``
produced by :func:`nasforge.spaces.search_space.genotype_to_onehot`).

Everything is seeded: minibatch order comes from a caller-supplied RNG, and no
global state is touched.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.core.types import ArrayDict, Genotype
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import SearchSpace, genotype_to_onehot
from nasforge.training.optimizer import SGD

__all__ = ["run_epoch_weights", "evaluate", "train_discrete"]


def run_epoch_weights(
    net: Network,
    weights: dict[str, ArrayDict],
    alpha: np.ndarray,
    dataset,
    opt: SGD,
    rng: np.random.Generator,
    batch_size: int = 16,
) -> tuple[float, float]:
    """One SGD epoch updating ``weights`` (alpha is held fixed)."""
    n = len(dataset.train_idx)
    order = rng.permutation(n)
    tot_loss = 0.0
    tot_acc = 0.0
    nb = 0
    for s in range(0, n, batch_size):
        rb = order[s : s + batch_size]
        x, y = dataset.batch("train", rb)
        logits, ctx = net.forward(x, alpha, weights)
        loss, dlog = net.loss(logits, y)
        _g_alpha, g_weights, _g_stem = net.backward(dlog, ctx)
        opt.step(weights, g_weights)
        tot_loss += loss * len(rb)
        tot_acc += net.accuracy(logits, y) * len(rb)
        nb += len(rb)
    return tot_loss / nb, tot_acc / nb


def evaluate(
    net: Network,
    weights: dict[str, ArrayDict],
    alpha: np.ndarray,
    dataset,
    split: str,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Mean loss and accuracy over ``split`` (no parameter update)."""
    """Mean loss and accuracy over ``split`` (no parameter update)."""
    idx = getattr(dataset, f"{split}_idx")
    n = len(idx)
    tot_loss = 0.0
    tot_acc = 0.0
    nb = 0
    for s in range(0, n, batch_size):
        rb = np.arange(s, min(s + batch_size, n))
        x, y = dataset.batch(split, rb)
        logits, _ = net.forward(x, alpha, weights)
        loss, _ = net.loss(logits, y)
        tot_loss += loss * len(rb)
        tot_acc += net.accuracy(logits, y) * len(rb)
        nb += len(rb)
    return tot_loss / nb, tot_acc / nb


def train_discrete(
    genotype: Genotype,
    dataset,
    space: SearchSpace,
    net: Network,
    seed: int,
    epochs: int,
    lr: float = 0.05,
    init_weights: dict[str, ArrayDict] | None = None,
) -> tuple[dict[str, ArrayDict], float, float]:
    """Train a discrete genotype from scratch (or from inherited weights).

    If ``init_weights`` is supplied, training *inherits* the one-shot supernet
    weights (the DarFuse weight-inheritance trick); otherwise weights are
    initialised afresh. Returns ``(weights, val_acc, test_acc)``.
    """
    rng = build_rng(seed, stage="train-discrete")
    if init_weights is None:
        weights = net.init_weights(rng)
    else:
        weights = {k: {kk: v.copy() for kk, v in p.items()} for k, p in init_weights.items()}
    alpha = genotype_to_onehot(genotype, space)
    opt = SGD(lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        run_epoch_weights(net, weights, alpha, dataset, opt, rng)
    _vl, val_acc = evaluate(net, weights, alpha, dataset, "val")
    _tl, test_acc = evaluate(net, weights, alpha, dataset, "test")
    return weights, float(val_acc), float(test_acc)


def weight_grad(
    net: Network,
    weights: dict[str, ArrayDict],
    alpha: np.ndarray,
    dataset,
    split: str,
    rng: np.random.Generator,
    batch_size: int = 32,
) -> dict[str, ArrayDict]:
    """Average weight gradient of the loss over ``split`` (no update)."""
    idx = getattr(dataset, f"{split}_idx")
    n = len(idx)
    order = rng.permutation(n)
    acc: dict[str, ArrayDict] = {
        name: {k: np.zeros_like(v) for k, v in p.items()} for name, p in weights.items()
    }
    nb = 0
    for s in range(0, n, batch_size):
        rb = order[s : s + batch_size]
        x, y = dataset.batch(split, rb)
        logits, ctx = net.forward(x, alpha, weights)
        _loss, dlog = net.loss(logits, y)
        _g_alpha, g_weights, _g_stem = net.backward(dlog, ctx)
        for name, p in g_weights.items():
            for k, v in p.items():
                acc[name][k] = acc[name][k] + v
        nb += len(rb)
    for name, p in acc.items():
        for k in p:
            acc[name][k] = p[k] / nb
    return acc


def flat_grad(g: dict[str, ArrayDict]) -> np.ndarray:
    parts = []
    for name in sorted(g.keys()):
        for k in sorted(g[name].keys()):
            parts.append(g[name][k].ravel())
    return np.concatenate(parts) if parts else np.zeros(0)
