"""Mixed (continuous) and discrete (one-shot) cell implementations.

``MixedCell`` evaluates the DARTS continuous relaxation: every edge is a softmax
mixture over the 6 primitives with *shared* operator weights (the standard
one-shot weight-sharing assumption). Its ``forward`` returns an autodiff context
and ``backward`` returns gradients for the incoming node activations, the
architecture logits ``alpha`` and the shared operator weights -- entirely from
the operators' analytic VJPs, no autograd tape.

The forward walks the cell's nodes in topological order (each internal node is
summed from its incoming *mixed* edges, then optionally ReLU'd); the backward is
the exact reverse, redistributing each node's gradient to the edges that feed it
and, for internal source nodes, folding the contribution back into their own
pre-activation gradient before continuing down the DAG.

``forward_discrete`` evaluates a single discrete genotype (one op per edge) which
is what the independent evaluator trains from scratch.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict, Genotype
from nasforge.ops.registry import make_op
from nasforge.spaces.search_space import SearchSpace

__all__ = ["MixedCell"]


def _instancenorm_forward(x: np.ndarray, eps: float = 1e-5):
    """Per-sample, per-channel normalisation (stateless). Returns (xn, ctx)."""
    mean = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    std = np.sqrt(var + eps)
    xn = (x - mean) / std
    return xn, {"mean": mean, "std": std, "xn": xn}


def _instancenorm_backward(d_xn: np.ndarray, ctx: dict) -> np.ndarray:
    """Reverse of :func:`_instancenorm_forward`."""
    xn = ctx["xn"]
    dmean = d_xn.mean(axis=(2, 3), keepdims=True)
    dxn_xn = (d_xn * xn).mean(axis=(2, 3), keepdims=True)
    return (d_xn - dmean - xn * dxn_xn) / ctx["std"]


class MixedCell:
    """A searchable cell with continuous (mixed) edges."""

    def __init__(self, space: SearchSpace) -> None:
        self.space = space
        self.n_input = space.n_input

    # ------------------------------------------------------------------ #
    # Continuous relaxation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(alpha: np.ndarray) -> np.ndarray:
        a = alpha - alpha.max(axis=1, keepdims=True)
        ex = np.exp(a)
        return ex / ex.sum(axis=1, keepdims=True)

    def forward(
        self,
        inputs: list[np.ndarray],
        alpha: np.ndarray,
        weights: dict[str, ArrayDict],
        relu: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Forward of the mixed cell. Returns ``(cell_out, ctx)``.

        ``inputs`` is the list of external input node activations (length
        ``n_input``). Internal nodes are computed and appended internally.
        """
        edges = self.space.edges
        n_ops = self.space.n_ops
        b, c, h, w = inputs[0].shape
        s = self._softmax(alpha)  # (E, n_ops)

        node_h: list[np.ndarray] = list(inputs)
        masks: list[np.ndarray | None] = [None] * self.n_input
        in_ctxs: list[dict | None] = [None] * self.n_input
        z: dict[tuple[int, int], np.ndarray] = {}

        for j in range(self.n_input, self.n_input + self.space.n_nodes):
            acc = np.zeros((b, c, h, w), dtype=np.float64)
            for e, (src, d) in enumerate(edges):
                if d != j:
                    continue
                x = node_h[src]
                m = np.zeros((b, c, h, w), dtype=np.float64)
                for o in range(n_ops):
                    op_name = self.space.op_order[o]
                    zo = make_op(op_name).forward(x, weights[op_name])
                    z[(e, o)] = zo
                    m = m + s[e, o] * zo
                acc = acc + m
            xn, in_ctx = _instancenorm_forward(acc)
            mask = None
            if relu:
                mask = xn > 0
                out = np.where(mask, xn, 0.0)
            else:
                out = xn
            node_h.append(out)
            masks.append(mask)
            in_ctxs.append(in_ctx)

        cell_out = np.zeros((b, c, h, w), dtype=np.float64)
        for j in range(self.n_input, self.n_input + self.space.n_nodes):
            cell_out = cell_out + node_h[j]

        ctx = {
            "z": z,
            "s": s,
            "inputs": inputs,
            "weights": weights,
            "alpha": alpha,
            "relu": relu,
            "masks": masks,
            "in_ctxs": in_ctxs,
            "h": node_h,
            "shape": (b, c, h, w),
        }
        return cell_out, ctx

    def backward(
        self, d_out: np.ndarray, ctx: dict
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, ArrayDict]]:
        """Reverse mode of :meth:`forward`.

        Returns
        -------
        g_inputs : list[np.ndarray]
            Gradient for each external input node (length ``n_input``).
        g_alpha : np.ndarray
            Gradient for the architecture logits (shape == alpha_shape).
        g_weights : dict[str, ArrayDict]
            Accumulated gradient for each shared operator's parameters.
        """
        edges = self.space.edges
        n_ops = self.space.n_ops
        b, c, h, w = ctx["shape"]
        s = ctx["s"]
        z = ctx["z"]
        weights = ctx["weights"]
        relu = ctx["relu"]
        masks = ctx["masks"]
        in_ctxs = ctx["in_ctxs"]
        node_h = ctx["h"]
        n_total = len(node_h)
        n_input = self.n_input

        # Pre-activation gradient accumulator for every node index.
        d_node = [np.zeros_like(node_h[i]) for i in range(n_total)]
        # The cell output is the sum of the internal node outputs (post-ReLU).
        for j in range(n_input, n_input + self.space.n_nodes):
            dacc = d_out
            if relu and masks[j] is not None:
                dacc = dacc * masks[j].astype(np.float64)
            d_node[j] = d_node[j] + _instancenorm_backward(dacc, in_ctxs[j])

        g_inputs = [np.zeros_like(node_h[i]) for i in range(n_input)]
        g_weights: dict[str, ArrayDict] = {
            name: {k: np.zeros_like(v) for k, v in weights[name].items()}
            for name in weights
        }
        g_alpha = np.zeros(self.space.alpha_shape(), dtype=np.float64)
        d_s = np.zeros((len(edges), n_ops), dtype=np.float64)
        d_mixed = np.zeros((len(edges), b, c, h, w), dtype=np.float64)

        # Reverse topological order over internal nodes.
        for j in reversed(range(n_input, n_input + self.space.n_nodes)):
            dacc = d_node[j]
            for e, (src, d) in enumerate(edges):
                if d != j:
                    continue
                d_mixed[e] = d_mixed[e] + dacc
                x = node_h[src]
                for o in range(n_ops):
                    op_name = self.space.op_order[o]
                    coeff = s[e, o]
                    d_z = coeff * d_mixed[e]
                    d_s[e, o] = d_s[e, o] + np.sum(d_mixed[e] * z[(e, o)])
                    op = make_op(op_name)
                    gx = op.vjp_x(d_z, x, weights[op_name])
                    gw = op.vjp_w(d_z, x, weights[op_name])
                    if src < n_input:
                        g_inputs[src] = g_inputs[src] + gx
                    else:
                        # fold back into the source node's pre-activation grad
                        d_node[src] = d_node[src] + _instancenorm_backward(gx, in_ctxs[src])
                    for k, v in gw.items():
                        if k in g_weights[op_name]:
                            g_weights[op_name][k] = g_weights[op_name][k] + v

        for e in range(len(edges)):
            dot = float(np.sum(s[e] * d_s[e]))
            g_alpha[e] = s[e] * (d_s[e] - dot)
        return g_inputs, g_alpha, g_weights

    # ------------------------------------------------------------------ #
    # Discrete evaluation
    # ------------------------------------------------------------------ #
    def forward_discrete(
        self,
        inputs: list[np.ndarray],
        genotype: Genotype,
        weights: dict[str, ArrayDict],
        relu: bool = True,
    ) -> np.ndarray:
        """Forward of a single discrete genotype (one op per edge)."""
        b, c, h, w = inputs[0].shape
        op_map: dict[tuple[int, int], str] = {}
        for j, node in enumerate(genotype.nodes):
            dst = j + self.n_input
            for src, op in node:
                op_map[(src, dst)] = op

        node_h: list[np.ndarray] = list(inputs)
        masks: list[np.ndarray | None] = [None] * self.n_input
        for j in range(self.n_input, self.n_input + self.space.n_nodes):
            acc = np.zeros((b, c, h, w), dtype=np.float64)
            for (src, d), op in op_map.items():
                if d == j:
                    acc = acc + make_op(op).forward(node_h[src], weights[op])
            xn, _in_ctx = _instancenorm_forward(acc)
            mask = None
            if relu:
                mask = xn > 0
                out = np.where(mask, xn, 0.0)
            else:
                out = xn
            node_h.append(out)
            masks.append(mask)

        out = np.zeros((b, c, h, w), dtype=np.float64)
        for j in range(self.n_input, self.n_input + self.space.n_nodes):
            out = out + node_h[j]
        return out
