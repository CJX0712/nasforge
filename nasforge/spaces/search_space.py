"""The DARTS-style search space for NasForge.

A *cell* is a small directed acyclic graph over ``n_nodes`` internal nodes. Each
internal node receives exactly ``edges_per_node`` edges from earlier nodes; the
two cell inputs (the outputs of the previous two cells) are nodes ``0`` and ``1``.
Every edge is a *mixed* operator -- a softmax over the 6 primitives -- whose
continuous weights ``alpha`` are what DarFuse optimises. A *discrete* genotype
picks one primitive per edge (the argmax), which is what gets evaluated on its
own.

The space is intentionally small so that the whole pipeline (one-shot supernet
training + discretisation + independent evaluation) runs in well under a minute
on a single CPU core -- this is the ``demo`` budget. All randomness comes from
:func:`nasforge.core.seed.build_rng`.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nasforge.core.seed import build_rng
from nasforge.core.types import Genotype
from nasforge.ops.registry import OP_INDEX, PRIMITIVE_ORDER

__all__ = ["SearchSpace", "DEFAULT_SPACE"]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Fixed topology of the searchable cell."""

    n_input: int = 2
    n_nodes: int = 3
    edges_per_node: int = 2
    op_order: tuple[str, ...] = PRIMITIVE_ORDER
    # Edges as ``(src, dst)`` pairs; dst in ``[n_input, n_input+n_nodes)``.
    edges: tuple[tuple[int, int], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.edges:
            # Fixed DAG: node j pulls from the two immediately preceding nodes.
            edges = []
            for j in range(self.n_input, self.n_input + self.n_nodes):
                for k in range(self.edges_per_node):
                    src = j - 1 - k
                    edges.append((src, j))
            object.__setattr__(self, "edges", tuple(edges))
        if len(self.op_order) != len(set(self.op_order)):
            from nasforge.core.errors import SearchError

            raise SearchError("duplicate operator in op_order", code="E400")
        # Sanity: every edge is acyclic (src < dst) and dst is an internal node.
        for src, dst in self.edges:
            if not (src < dst):
                from nasforge.core.errors import SearchError

                raise SearchError(f"acyclic violation: ({src},{dst})", code="E400")
            if not (self.n_input <= dst < self.n_input + self.n_nodes):
                from nasforge.core.errors import SearchError

                raise SearchError(f"bad dst {dst}", code="E400")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_ops(self) -> int:
        return len(self.op_order)

    def alpha_shape(self) -> tuple[int, int]:
        return (self.n_edges, self.n_ops)

    def random_alpha(self, rng: np.random.Generator) -> np.ndarray:
        """A random but valid (logit) alpha, drawn from the seeded RNG."""
        return rng.standard_normal(self.alpha_shape()) * 0.1

    def random_genotype(self, rng: np.random.Generator) -> Genotype:
        """A uniformly random discrete genotype (one op per edge)."""
        nodes: list[tuple[tuple[int, str], ...]] = []
        for dst in range(self.n_input, self.n_input + self.n_nodes):
            incoming = [(s, o) for (s, o) in self._edges_into(dst)]
            nodes.append(tuple(incoming))
        # assign random op per edge
        rng2 = build_rng(int(rng.integers(0, 2**31 - 1)), stage="random-genotype")
        chosen = [PRIMITIVE_ORDER[int(rng2.integers(0, self.n_ops))] for _ in self.edges]
        # rebuild per-node tuples with chosen ops
        per_node: list[list[tuple[int, str]]] = [[] for _ in range(self.n_nodes)]
        for (s, d), op in zip(self.edges, chosen):
            per_node[d - self.n_input].append((s, op))
        node_tuples = tuple(tuple(sorted(p, key=lambda e: e[0])) for p in per_node)
        return Genotype(nodes=node_tuples, macro=("normal",))

    def _edges_into(self, dst: int) -> list[tuple[int, int]]:
        return [(s, d) for (s, d) in self.edges if d == dst]

    def discretize(self, alpha: np.ndarray) -> Genotype:
        """Pick the argmax op per edge and group by destination node."""
        if alpha.shape != self.alpha_shape():
            from nasforge.core.errors import SearchError

            raise SearchError(
                f"alpha shape {alpha.shape} != {self.alpha_shape()}", code="E400"
            )
        per_node: list[list[tuple[int, str]]] = [[] for _ in range(self.n_nodes)]
        for (s, d), a_row in zip(self.edges, alpha):
            op = self.op_order[int(np.argmax(a_row))]
            per_node[d - self.n_input].append((s, op))
        node_tuples = tuple(
            tuple(sorted(p, key=lambda e: e[0])) for p in per_node
        )
        return Genotype(nodes=node_tuples, macro=("normal",))


DEFAULT_SPACE = SearchSpace()


def op_index(name: str) -> int:
    """Operator name -> canonical index (re-export for convenience)."""
    return OP_INDEX[name]


def genotype_to_onehot(genotype: Genotype, space: SearchSpace) -> np.ndarray:
    """Encode a discrete genotype as a one-hot alpha (argmax-selects each op).

    Running the mixed cell with this alpha is mathematically identical to the
    discrete cell, so the same forward/backward can train a discrete arch.
    """
    alpha = np.zeros(space.alpha_shape(), dtype=np.float64)
    # build (src,dst)->op from genotype
    op_map: dict[tuple[int, int], str] = {}
    for j, node in enumerate(genotype.nodes):
        dst = j + space.n_input
        for src, op in node:
            op_map[(src, dst)] = op
    for e, (src, dst) in enumerate(space.edges):
        op = op_map[(src, dst)]
        alpha[e, OP_INDEX[op]] = 1.0
    return alpha
