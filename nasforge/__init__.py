"""NasForge - world-class Neural Architecture Search, pure NumPy, CPU-only, deterministic.

Flagship algorithm **DarFuse** = continuous relaxation (softmax ``alpha``) +
second-order (unrolled) architecture gradient + Gradient Reversal Layer (GRL) +
architecture weight decay + one-shot weight inheritance, benchmarked against
random search, regularized evolution (Real et al., AAAI'19) and zero-cost proxies
(NAS-WiZ / Zero-Cost-NAS, Abdelfattah et al., ICLR'21).

Everything runs offline on a single CPU core with hand-written forward/backward
VJPs (no PyTorch, no autograd) and a single deterministic RNG contract.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Budget, Genotype, SearchResult
from nasforge.data.dataset import SyntheticDataset
from nasforge.evaluate import (
    accuracy,
    independent_evaluate,
    n_params,
    one_shot_evaluate,
    spearman,
)
from nasforge.ops.registry import OP_INDEX, OP_REGISTRY, PRIMITIVE_ORDER, make_op
from nasforge.search import (
    darfuse_search,
    evolution_search,
    random_search,
    zerocost_search,
)
from nasforge.spaces.search_space import DEFAULT_SPACE, SearchSpace

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"

__all__ = [
    "__version__",
    "__author__",
    # core types
    "Budget",
    "Genotype",
    "SearchResult",
    # data
    "SyntheticDataset",
    # search algorithms
    "darfuse_search",
    "evolution_search",
    "random_search",
    "zerocost_search",
    # evaluation
    "independent_evaluate",
    "one_shot_evaluate",
    "accuracy",
    "spearman",
    "n_params",
    # spaces / operators
    "SearchSpace",
    "DEFAULT_SPACE",
    "PRIMITIVE_ORDER",
    "OP_REGISTRY",
    "OP_INDEX",
    "make_op",
]
