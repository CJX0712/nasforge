"""Operator registry and the canonical operator catalogue.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.ops.conv import Conv1x1, Conv3x3, Conv5x5
from nasforge.ops.identity_ops import SkipConnect
from nasforge.ops.pooling import AvgPool, MaxPool
from nasforge.ops.registry import OP_INDEX, OP_REGISTRY, PRIMITIVE_ORDER, make_op

__all__ = [
    "Conv1x1",
    "Conv3x3",
    "Conv5x5",
    "AvgPool",
    "MaxPool",
    "SkipConnect",
    "OP_REGISTRY",
    "OP_INDEX",
    "PRIMITIVE_ORDER",
    "make_op",
]
