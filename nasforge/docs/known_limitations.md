# Known Limitations — NasForge

Author: **晨星 (CJX0712)**. This document is an honest account of what NasForge
does **not** do. It should be read alongside
[model_card.md](model_card.md).

---

## 1. CPU-only (no GPU)

NasForge is implemented in pure NumPy and runs **only on CPU**. There is no
device abstraction, no CUDA/ROCm path, and no batching across accelerators. All
performance claims ("≈ 45 s on a single CPU core") assume a single-threaded BLAS
configuration, which NasForge enforces at import time for float-bit stability.
Scaling to large datasets or deeper search is therefore bounded by CPU throughput.

---

## 2. Tiny synthetic toy dataset

The bundled dataset (`SyntheticDataset`, `n_samples=320`, `img_size=9`,
`n_classes=4`) is **not** ImageNet-scale. It is a deliberately small,
*learnable* proxy: smoothed per-class prototypes plus label-consistent noise.
Its purpose is to make architectures **rankable** and the pipeline **fast and
deterministic**, not to benchmark real-world vision performance. Reaching
val/test accuracy ≈ 1.000 on this task is expected and is a correctness signal,
not a measure of generalisation to real images.

---

## 3. DarFuse-2nd is expensive on CPU

The second-order (unrolled) architecture gradient computes the cross term
`∇_α(∇_w L_train · ∇_w L_val)` by **finite differences in `alpha`**. With
`n_edges=6` and `n_ops=6` there are **36 `alpha` indices**, and each index
requires two extra weight-gradient evaluations (`α ± ε`). On CPU this is far
slower than DarFuse-1st:

- DarFuse-1st demo: ≈ 45 s (≤ 60 s budget met).
- DarFuse-2nd demo: significantly longer; the ≤ 60 s guarantee applies to
  DarFuse-1st only.

Use DarFuse-2nd for accuracy/research, not for the time-bounded demo.

---

## 4. One-shot weight-sharing approximation

DarFuse shares a single set of operator weights across all edges of the same
primitive and reuses them for every cell. The discrete genotype found by argmax
is therefore evaluated under the *one-shot* assumption that shared weights are a
good proxy for independently trained ones. Weight inheritance (fine-tuning the
discrete cell from the supernet weights) reduces but does **not** eliminate this
approximation error. The `research` profile and independent evaluation exist to
quantify the gap.

---

## 5. Small, fixed search-space topology

The search space is a **fixed** DAG: 2 inputs, 3 internal nodes, 2 edges per
node (6 edges total), and a fixed 6-primitive operator catalogue. NasForge does
**not** search the macro topology (number of cells, channel widths, number of
nodes), skip connections between non-adjacent nodes, or multi-branch reductions.
This keeps the pipeline fast and deterministic but limits the diversity of
discoverable architectures compared with modern hierarchical or once-for-all
search spaces.

---

## 6. Synthetic data may not transfer to real vision tasks

Because the system is developed and validated end-to-end on the synthetic
dataset, there is **no guarantee** that the discovered cells, the DarFuse
gradient behaviour, or the zero-cost proxy rankings transfer to natural-image
benchmarks (CIFAR, ImageNet). The synthetic task is designed for separability and
determinism; real vision tasks have different statistics, larger spatial
resolution, and harder class boundaries. Treat NasForge as a research/teaching
artifact and a reproducible baseline, not a drop-in production NAS engine.

---

## Summary table

| # | Limitation | Impact |
| --- | --- | --- |
| 1 | CPU-only | No GPU speedup; throughput-bound. |
| 2 | Tiny synthetic toy dataset | Not a real-vision benchmark. |
| 3 | DarFuse-2nd expensive on CPU | 36 α-indices × finite-difference; slow. |
| 4 | One-shot weight sharing | Discrete eval ≈ proxy of true arch. |
| 5 | Small fixed topology | Limited architecture diversity. |
| 6 | Synthetic → real transfer unproven | No guarantee on natural images. |
