"""Core data structures for NasForge.

Layout (the whole project is flat, top level package ``nasforge``):

    core/  ops/  spaces/  data/  training/  search/  evaluate/  pipeline/

The search-space struct (:class:`ArchConfig`) lives in ``spaces/search_space.py``
because it owns the operator catalogue; this module holds the pure-value types that
interfaces depend on and that must not import any upper layer.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ArrayDict", "Genotype", "CellSpec", "SearchResult", "Budget"]


# A flat parameter container shared between every operator and the trainer.
ArrayDict = dict[str, np.ndarray]


def _edge_sort_key(item: tuple[int, str]) -> tuple[int, int]:
    """Sort key for canonical ordering: by operator index, then source node.

    ``OP_INDEX`` is supplied lazily to avoid a circular import with ``spaces``.
    """
    src, op = item
    from nasforge.ops.registry import OP_INDEX

    return (OP_INDEX.get(op, len(OP_INDEX)), src)


@dataclass(frozen=True, slots=True)
class Genotype:
    """A discrete, canonical, hashable architecture.

    ``nodes`` is one tuple per internal cell node. Each entry is a tuple of
    ``(src, op_name)`` pairs for that node's incoming edges, already sorted by
    (:meth:`_edge_sort_key`) and de-duplicated. ``macro`` records the cell roles.
    """

    nodes: tuple[tuple[tuple[int, str], ...], ...]
    macro: tuple[str, ...] = ("normal",)

    def to_json(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {"nodes": [list(map(list, n)) for n in self.nodes], "macro": list(self.macro)}

    @staticmethod
    def from_json(d: dict) -> Genotype:
        """Reconstruct a :class:`Genotype` from :meth:`to_json` output."""
        raw_nodes = tuple(tuple((int(s), str(o)) for s, o in n) for n in d["nodes"])
        macro = tuple(d.get("macro", ["normal"]))
        return Genotype(nodes=raw_nodes, macro=macro)

    def param_count(self, channels: int, num_classes: int, stem_channels: int) -> int:
        """Count parameters of the discrete network this genotype describes.

        Parameters
        ----------
        channels:
            Per-cell channel width.
        num_classes:
            Final classifier width.
        stem_channels:
            Stem conv output width.
        """
        count = 0
        # stem conv: 3 -> stem_channels, 3x3 weights + bias
        count += 3 * stem_channels * 3 * 3 + stem_channels
        # each cell has one conv op per edge (skip_connect / pooling add nothing)
        for node in self.nodes:
            for _src, op in node:
                if op.startswith("conv_"):
                    kh = 3 if "1x1" not in op else 1
                    count += channels * channels * kh * kh + channels
        # final classifier: channels -> num_classes
        count += channels * num_classes + num_classes
        return count


@dataclass(frozen=True, slots=True)
class CellSpec:
    """Topology of a single cell: which operator feeds each internal node."""

    node_ops: tuple[tuple[tuple[int, str], ...], ...]

    def n_internal(self) -> int:
        """Number of internal (non-input) nodes."""
        return len(self.node_ops)

    def op_of(self, src: int, dst: int) -> str:
        """Return the operator connecting ``src`` into internal node ``dst``."""
        for s, o in self.node_ops[dst]:
            if s == src:
                return o
        raise KeyError(f"no edge ({src} -> {dst})")


@dataclass(frozen=True, slots=True)
class Budget:
    """Computational budget for one search / evaluation run."""

    max_steps: int = 8
    max_evals: int = 200
    epochs: int = 3
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    @classmethod
    def demo(cls) -> Budget:
        """Default, time-bounded profile: ``make demo`` must finish in <= 60s."""
        return cls(max_steps=6, max_evals=200, epochs=2)

    @classmethod
    def research(cls) -> Budget:
        """Full search profile, only reachable via ``--profile research``."""
        return cls(max_steps=200, max_evals=200, epochs=12)


@dataclass(slots=True)
class SearchResult:
    """The single exchange contract between a search algorithm and the pipeline."""

    method: str
    best_genotype: Genotype
    best_val_acc: float
    best_test_acc: float
    n_evaluated: int
    wall_time_s: float
    search_trace: tuple[float, ...] = ()
    one_shot_scores: Mapping[str, float] = field(default_factory=dict)
    rank_corr_one_shot_vs_independent: float | None = None
    param_count: int = 0
    genotype_json: dict = field(default_factory=dict)
    extra: Mapping[str, object] = field(default_factory=dict)
