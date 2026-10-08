"""Weight initialisers. He (uniform) for conv layers, zeros for bias.

All randomness comes from the caller-supplied :class:`numpy.random.Generator`,
never from a global seed.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

__all__ = ["he_uniform", "zeros"]


def he_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> float:
    """He-uniform bound ``sqrt(6 / fan_in)`` for a uniform ``[-b, b]`` init."""
    del fan_out
    return float(np.sqrt(6.0 / max(1, fan_in)))


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    """A zero array of the given shape (used for biases)."""
    return np.zeros(shape, dtype=np.float64)
