"""Gradient Reversal Layer (Ganin et al., 2015) for architecture search.

In DarFuse the GRL decouples architecture learning from weight fitting: during
the backward pass it multiplies the gradient flowing into the architecture
branch by ``-scale``. Used with ``scale=1`` it exactly negates that gradient,
which is what invariant G3 checks.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

__all__ = ["grl"]


def grl(grad: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Reverse (and optionally scale) a gradient during the backward pass."""
    return -scale * np.asarray(grad, dtype=np.float64)
