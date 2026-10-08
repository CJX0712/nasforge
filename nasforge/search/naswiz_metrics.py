"""Zero-cost proxies (NAS-WiZ / Zero-Cost-NAS, Abdelfattah et al., ICLR'21).

These score a *discrete* architecture without training it: they measure the
gradient signal and expressivity of the network at a single random initialisation
on one minibatch. All are finite and bounded, so they are safe to use as a
training-free ranking signal. The metrics implemented are:

* ``n_params``   -- parameter count (cheap complexity prior).
* ``grad_norm``  -- Frobenius norm of the weight-gradient of the loss (signal).
* ``snip``       -- |Σ_p grad_p · p| (SNIP importance, single-shot).
* ``logits_var`` -- variance of the logits across samples (expressivity proxy).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.spaces.network import Network
from nasforge.spaces.search_space import DEFAULT_SPACE, genotype_to_onehot
from nasforge.training.trainer import flat_grad, weight_grad

__all__ = ["zero_cost_scores"]


def zero_cost_scores(
    net: Network,
    genotype,
    dataset,
    seed: int,
    n_classes: int | None = None,
    img_size: int | None = None,
) -> dict[str, float]:
    """Return the zero-cost proxy dictionary for one discrete genotype."""
    space = DEFAULT_SPACE
    n_classes = n_classes or dataset.n_classes()
    rng = build_rng(seed, stage="zerocost-init")
    weights = net.init_weights(rng)
    alpha = genotype_to_onehot(genotype, space)
    x, _ = dataset.batch("train", np.arange(min(32, len(dataset.train_idx))))
    logits, _ = net.forward(x, alpha, weights)
    g_weights = weight_grad(net, weights, alpha, dataset, "train", rng, batch_size=len(x))
    fg = flat_grad(g_weights)

    params = sum(v.size for p in weights.values() for v in p.values())
    grad_norm = float(np.sqrt(np.sum(fg * fg)))
    # SNIP: |sum_p grad_p * p| (flattened dot product of grad and weights)
    wp = np.concatenate([weights[name][k].ravel() for name in sorted(weights) for k in sorted(weights[name])])
    snip = float(abs(np.sum(fg * wp)))
    logits_var = float(np.var(logits))
    return {
        "n_params": float(params),
        "grad_norm": grad_norm,
        "snip": snip,
        "logits_var": logits_var,
    }
