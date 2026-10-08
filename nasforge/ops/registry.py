"""The canonical operator catalogue -- single source of truth.

Every searchable primitive is registered here exactly once. ``PRIMITIVE_ORDER``
fixes the index of each operator (the "edge sort key" in :mod:`nasforge.core.types`
relies on it for canonical ordering), ``OP_REGISTRY`` maps name -> class, and
``make_op`` is the only sanctioned constructor.

At import time we assert that each registry key equals the class' own ``name``
attribute, so the two can never silently drift apart -- this is the guard that
keeps the architecture encoding self-consistent.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.ops.conv import Conv1x1, Conv3x3, Conv5x5
from nasforge.ops.identity_ops import SkipConnect
from nasforge.ops.pooling import AvgPool, MaxPool

__all__ = ["PRIMITIVE_ORDER", "OP_REGISTRY", "OP_INDEX", "make_op"]


# Fixed, documented ordering. The integer position of an op here is its "edge
# sort key" used during canonicalisation (see core/types._edge_sort_key).
PRIMITIVE_ORDER: tuple[str, ...] = (
    "skip_connect",
    "conv_1x1",
    "conv_3x3",
    "conv_5x5",
    "avg_pool",
    "max_pool",
)


_OP_CLASSES = {
    "skip_connect": SkipConnect,
    "conv_1x1": Conv1x1,
    "conv_3x3": Conv3x3,
    "conv_5x5": Conv5x5,
    "avg_pool": AvgPool,
    "max_pool": MaxPool,
}


# Self-consistency guard: registry key must equal class.name.
for _name, _cls in _OP_CLASSES.items():
    if _cls.name != _name:
        raise RuntimeError(
            f"operator registry drift: key {_name!r} != class.name {_cls.name!r}"
        )


OP_REGISTRY: dict[str, type] = dict(_OP_CLASSES)
OP_INDEX: dict[str, int] = {name: i for i, name in enumerate(PRIMITIVE_ORDER)}


def make_op(name: str):
    """Construct a fresh operator instance by its canonical name.

    Raises
    ------
    KeyError
        If ``name`` is not in the catalogue.
    """
    try:
        return OP_REGISTRY[name]()
    except KeyError as exc:  # pragma: no cover - defensive
        from nasforge.core.errors import OperatorError

        raise OperatorError(
            f"unknown operator {name!r}; known: {PRIMITIVE_ORDER}",
            code="E310",
        ) from exc
