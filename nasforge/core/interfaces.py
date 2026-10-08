"""Abstract protocols for every pluggable backend (vjp style).

Every operator exposes a manual forward / vector-Jacobian-product pair so the
second-order DARTS approximation can be built without an autograd tape:

    forward(x, p) -> y
    vjp_x(dy, x, p) -> grad_x            (dL/dx)
    vjp_w(dy, x, p) -> {grad_params}     (dL/dp)
    vjp_x_second(dy, x, p, dx_seed) -> grad_x2nd
                                     (d/dx of vjp_x along the dx_seed direction)

The second-order primitive is what makes DARTS' second-order approximation
expressible: ``vjp_x_second`` returns the directional derivative of the input
VJP with respect to ``x`` along ``dx_seed``. For operators whose input VJP does
not depend on ``x`` (every conv / avg-pool / skip), it is identically zero; only
``MaxPool``'s mask makes it non-trivial, and even there it is approximated as zero
(the mask is treated as a local constant). Therefore the real second-order
contribution in DarFuse comes from the G2 cross term, not from operator-level
second derivatives -- this is documented in ``docs/known_limitations.md``.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from nasforge.core.types import ArrayDict, Genotype

__all__ = ["Dataset", "Op", "SearchAlgorithm", "Evaluator"]


@runtime_checkable
class Dataset(Protocol):
    """A deterministic train/validation/test split."""

    name: str

    def split_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(train_idx, val_idx, test_idx)`` with disjoint integer arrays."""
        ...

    def batch(self, split: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` for the requested ``split`` and sample ``idx``."""
        ...

    def n_classes(self) -> int:
        """Number of output classes."""
        ...

    def in_shape(self) -> tuple[int, int, int]:
        """Input shape ``(channels, height, width)``."""
        ...


@runtime_checkable
class Op(Protocol):
    """A single searchable operator with an analytic backward pass.

    Parameters are injected by the caller (no internal state). Shapes use the
    axis convention ``B`` (batch) ``C`` (channel) ``H`` ``W``.
    """

    name: str

    def init_params(self, rng: np.random.Generator, in_c: int, out_c: int) -> ArrayDict:
        """Create parameters for this operator."""
        ...

    def forward(self, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        """Forward pass: ``(B, C_in, H, W) -> (B, C_out, H_out, W_out)``."""
        ...

    def vjp_x(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> np.ndarray:
        """Input gradient ``dL/dx``."""
        ...

    def vjp_w(self, dy: np.ndarray, x: np.ndarray, p: ArrayDict) -> ArrayDict:
        """Parameter gradients ``dL/dp``."""
        ...

    def vjp_x_second(
        self, dy: np.ndarray, x: np.ndarray, p: ArrayDict, dx_seed: np.ndarray
    ) -> np.ndarray:
        """Directional derivative of ``vjp_x`` w.r.t. ``x`` along ``dx_seed``."""
        ...

    def param_count(self, p: ArrayDict) -> int:
        """Total number of scalar parameters held in ``p``."""
        ...


@runtime_checkable
class SearchAlgorithm(Protocol):
    """A budgeted architecture search strategy."""

    name: str

    def search(self, dataset: Dataset, budget: object) -> object:
        """Run the search and return a :class:`SearchResult`-like contract."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a discrete genotype under a fixed protocol."""

    name: str

    def evaluate(self, genotype: Genotype, dataset: Dataset, budget: object) -> object:
        """Return a scalar score (higher is better) for ``genotype``."""
        ...
