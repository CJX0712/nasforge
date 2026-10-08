"""Evaluation metrics for NasForge.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "spearman", "n_params"]


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y))


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (1.0 = identical ranking)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    if denom == 0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def n_params(genotype, channels: int, num_classes: int, stem_channels: int) -> int:
    return genotype.param_count(channels, num_classes, stem_channels)
