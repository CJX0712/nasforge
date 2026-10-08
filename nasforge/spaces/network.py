"""The one-shot supernet: stem + stacked mixed cells + classifier head.

The network owns the *shared* operator weights (one-shot weight sharing) plus a
single architecture ``alpha`` reused by every cell. ``forward`` / ``backward``
expose gradients for ``alpha`` and the weights; ``flat_weights`` / ``set_weights``
give a single vector view used by the implicit-gradient (HVP) machinery of the
DarFuse second-order search.

Loss is mean softmax cross-entropy; ``backward`` returns the gradient of the
logits so the caller can chain any loss it likes.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.types import ArrayDict
from nasforge.ops.init import he_uniform
from nasforge.ops.registry import make_op
from nasforge.spaces.cell import MixedCell
from nasforge.spaces.search_space import SearchSpace

__all__ = ["Network"]

# Deterministic flatten order for the weight vector.
_W_KEYS = ("stem", "conv_1x1", "conv_3x3", "conv_5x5", "head")


class Network:
    """One-shot supernet for NasForge."""

    def __init__(
        self,
        space: SearchSpace,
        in_channels: int = 3,
        channels: int = 8,
        n_cells: int = 2,
        n_classes: int = 4,
        img_size: int = 9,
    ) -> None:
        self.space = space
        self.in_channels = in_channels
        self.C = channels
        self.n_cells = n_cells
        self.n_classes = n_classes
        self.img_size = img_size
        self.stem = make_op("conv_3x3")
        self.cell = MixedCell(space)

    # ------------------------------------------------------------------ #
    # Weight management
    # ------------------------------------------------------------------ #
    def init_weights(self, rng: np.random.Generator) -> dict[str, ArrayDict]:
        """Initialise every (shared) operator weight from the seeded RNG."""
        c = self.C
        w: dict[str, ArrayDict] = {}
        # stem: in_channels -> C (3x3)
        fb = he_uniform(rng, self.in_channels * 9, c)
        w["stem"] = {
            "w": rng.uniform(-fb, fb, size=(c, self.in_channels, 3, 3)),
            "b": np.zeros(c, dtype=np.float64),
        }
        for name, k in (("conv_1x1", 1), ("conv_3x3", 3), ("conv_5x5", 5)):
            fb = he_uniform(rng, c * k * k, c)
            w[name] = {
                "w": rng.uniform(-fb, fb, size=(c, c, k, k)),
                "b": np.zeros(c, dtype=np.float64),
            }
        # head: C -> n_classes
        fb = he_uniform(rng, c, self.n_classes)
        w["head"] = {
            "w": rng.uniform(-fb, fb, size=(self.n_classes, c)),
            "b": np.zeros(self.n_classes, dtype=np.float64),
        }
        # param-free operators still need an entry so the mixed cell can index them
        for name in ("skip_connect", "avg_pool", "max_pool"):
            w[name] = {}
        return w

    def flat_weights(self, weights: dict[str, ArrayDict]) -> np.ndarray:
        parts = []
        for k in _W_KEYS:
            if k not in weights:
                continue
            for sub in sorted(weights[k].keys()):
                parts.append(weights[k][sub].ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def set_weights(self, weights: dict[str, ArrayDict], flat: np.ndarray) -> None:
        i = 0
        for k in _W_KEYS:
            if k not in weights:
                continue
            for sub in sorted(weights[k].keys()):
                n = weights[k][sub].size
                weights[k][sub] = flat[i : i + n].reshape(weights[k][sub].shape)
                i += n

    # ------------------------------------------------------------------ #
    # Forward / backward
    # ------------------------------------------------------------------ #
    def forward(
        self, x: np.ndarray, alpha: np.ndarray, weights: dict[str, ArrayDict]
    ) -> tuple[np.ndarray, dict]:
        stem_out = self.stem.forward(x, weights["stem"])
        node_bank = [stem_out]
        windows: list[list[int]] = []
        records: list[tuple[list[np.ndarray], dict]] = []
        prev = [stem_out, stem_out]
        for i in range(self.n_cells):
            inp = prev
            src = [0, 0] if i == 0 else [i - 1, i]
            out, cctx = self.cell.forward(inp, alpha, weights)
            records.append((inp, cctx))
            windows.append(src)
            node_bank.append(out)
            prev = [prev[1], out]
        final = node_bank[-1]
        pooled = final.mean(axis=(2, 3))  # (B, C)
        logits = pooled @ weights["head"]["w"].T + weights["head"]["b"]
        ctx = {
            "weights": weights,
            "alpha": alpha,
            "x": x,
            "stem_out": stem_out,
            "node_bank": node_bank,
            "windows": windows,
            "records": records,
            "final": final,
            "pooled": pooled,
        }
        return logits, ctx

    def backward(
        self, d_logits: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, dict[str, ArrayDict], np.ndarray]:
        weights = ctx["weights"]
        alpha = ctx["alpha"]
        pooled = ctx["pooled"]
        g_weights: dict[str, ArrayDict] = {
            k: {kk: np.zeros_like(v) for kk, v in weights[k].items()} for k in weights
        }
        # head
        g_pooled = d_logits @ weights["head"]["w"]  # (B, C)
        g_weights["head"]["w"] = d_logits.T @ pooled
        g_weights["head"]["b"] = d_logits.sum(axis=0)
        hw, hh = ctx["final"].shape[2], ctx["final"].shape[3]
        d_final = (g_pooled / (hw * hh))[:, :, None, None]

        g_bank = [np.zeros_like(t) for t in ctx["node_bank"]]
        g_alpha = np.zeros(alpha.shape, dtype=np.float64)
        records = ctx["records"]
        windows = ctx["windows"]
        d_out = d_final
        for i in reversed(range(self.n_cells)):
            _, cctx = records[i]
            g_nodes, ga, gw = self.cell.backward(d_out, cctx)
            g_alpha = g_alpha + ga
            for name, grads in gw.items():
                for k, v in grads.items():
                    g_weights[name][k] = g_weights[name][k] + v
            src0, src1 = windows[i]
            g_bank[src0] = g_bank[src0] + g_nodes[0]
            g_bank[src1] = g_bank[src1] + g_nodes[1]
            # gradient entering the previous cell's output (node i) drives cell i-1
            if i > 0:
                d_out = g_bank[i]
            else:
                d_out = None
        g_stem = g_bank[0]
        # stem weight gradient (stem input is raw data x; we only need the
        # parameter gradient, not the gradient w.r.t x).
        g_weights["stem"] = self.stem.vjp_w(g_stem, ctx["x"], weights["stem"])
        return g_alpha, g_weights, g_stem

    def loss(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean softmax cross-entropy and its gradient w.r.t. logits."""
        b = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        sm = ex / ex.sum(axis=1, keepdims=True)
        # one-hot
        oh = np.zeros_like(sm)
        oh[np.arange(b), y] = 1.0
        loss = float(np.mean(-np.log(sm[np.arange(b), y] + 1e-12)))
        d_logits = (sm - oh) / b
        return loss, d_logits

    # ------------------------------------------------------------------ #
    # Discrete (independent) evaluation helpers
    # ------------------------------------------------------------------ #
    def forward_discrete(
        self,
        x: np.ndarray,
        genotype,
        weights: dict[str, ArrayDict],
    ) -> np.ndarray:
        """Forward a discrete genotype with its own (already-trained) weights."""

        stem_out = self.stem.forward(x, weights["stem"])
        node_bank = [stem_out]
        prev = [stem_out, stem_out]
        for i in range(self.n_cells):
            inp = prev
            out = self.cell.forward_discrete(inp, genotype, weights)
            node_bank.append(out)
            prev = [prev[1], out]
        final = node_bank[-1]
        pooled = final.mean(axis=(2, 3))
        return pooled @ weights["head"]["w"].T + weights["head"]["b"]

    def accuracy(self, logits: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.argmax(logits, axis=1) == y))
