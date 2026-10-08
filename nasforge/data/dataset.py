"""Deterministic synthetic image-classification dataset.

We do not download anything (the environment runs offline / on CPU). Instead we
synthesise a small, *learnable* multi-class image task: each class is a smoothed
random pattern plus label noise, so a tiny conv network can reach > 0.5 accuracy
and architectures are meaningfully rankable. Everything is drawn from the seeded
RNG so the dataset is byte-for-byte reproducible from a single seed.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import numpy as np

from nasforge.core.seed import build_rng

__all__ = ["SyntheticDataset"]


class SyntheticDataset:
    """A reproducible toy image-classification dataset."""

    def __init__(
        self,
        n_samples: int = 320,
        n_classes: int = 4,
        img_size: int = 9,
        seed: int = 20260408,
    ) -> None:
        self.n_samples = n_samples
        self._n_classes = n_classes
        self._img = (3, img_size, img_size)
        rng = build_rng(seed, stage="dataset")
        c, h, w = self._img
        # class prototype filters (smooth, low-frequency) so convs can separate
        proto = rng.standard_normal((n_classes, c, h, w))
        proto = _smooth(proto)
        X = np.zeros((n_samples, c, h, w), dtype=np.float64)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            cls = int(rng.integers(0, n_classes))
            noise = rng.standard_normal((c, h, w)) * 0.25
            X[i] = proto[cls] + noise
            y[i] = cls
        self.X = X
        self.y = y
        # deterministic disjoint split
        perm = build_rng(seed, stage="split").permutation(n_samples)
        n_tr = int(0.7 * n_samples)
        n_va = int(0.15 * n_samples)
        self.train_idx = perm[:n_tr]
        self.val_idx = perm[n_tr : n_tr + n_va]
        self.test_idx = perm[n_tr + n_va :]

    @staticmethod
    def _smooth(x: np.ndarray) -> np.ndarray:
        # very light box blur to make patterns low-frequency
        from scipy.ndimage import uniform_filter

        return uniform_filter(x, size=2, mode="reflect", axes=(-2, -1))

    def split_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.train_idx, self.val_idx, self.test_idx

    def batch(self, split: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            s = self.train_idx
        elif split == "val":
            s = self.val_idx
        elif split == "test":
            s = self.test_idx
        else:
            raise ValueError(split)
        sel = np.asarray(s)[idx]
        return self.X[sel], self.y[sel]

    def n_classes(self) -> int:
        return self._n_classes

    def in_shape(self) -> tuple[int, int, int]:
        return self._img


def _smooth(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    return uniform_filter(x, size=2, mode="reflect", axes=(-2, -1))
