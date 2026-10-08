# Model Card — NasForge

Author: **晨星 (CJX0712)**. Following the model-card convention, this document
records the intended use, training data, hyperparameters, evaluation results, and
risk/limitation profile of the NasForge architecture-search system and the
architectures it discovers.

---

## 1. Model details

- **System**: NasForge — a differentiable / evolutionary / zero-cost Neural
  Architecture Search pipeline.
- **Implementation**: Pure NumPy (no PyTorch, no autograd). Forward and backward
  passes are hand-written with VJPs; CPU-only, offline, deterministic.
- **Search target**: A single DARTS-style cell (fixed DAG: 2 inputs, 3 internal
  nodes, 2 edges/node = 6 edges) over a 6-primitive operator catalogue.
- **Version**: `0.1.0`.

---

## 2. Intended use

- **Primary**: reproducible research on differentiable NAS; teaching the DarFuse
  (GRL + architecture weight decay + weight inheritance) and DARTS-2nd gradient
  math in an auditable, dependency-light codebase.
- **Secondary**: a fast, deterministic architecture-search baseline on tiny
  vision tasks and a sandbox for zero-cost-proxy ranking experiments.
- **Out of scope**: production-scale architecture discovery, training on
  ImageNet-scale data, GPU acceleration, or deployment of the discovered cells to
  safety-critical systems.

---

## 3. Training data

A **deterministic synthetic image-classification dataset**
(`nasforge/data/dataset.py::SyntheticDataset`):

- `n_samples=320`, `n_classes=4`, `img_size=9` → input shape `(3, 9, 9)`.
- Each class is a smooth (low-frequency, box-blurred) random prototype plus
  Gaussian label-consistent noise (`std=0.25`), so a tiny conv net can exceed
  0.5 accuracy and architectures are meaningfully rankable.
- Deterministic train/val/test split: 70% / 15% / 15%, derived from a
  CRC32-seeded permutation (no overlap between splits).
- Fully reproducible from a single seed; **no network download**. It is a
  *learnability proxy*, not a real-world vision benchmark.

---

## 4. Hyperparameters

| Symbol | Value | Scope |
| --- | --- | --- |
| `w_lr` | 0.05 | Supernet weight learning rate (SGD). |
| `momentum` | 0.9 | SGD momentum. |
| `weight_decay` | 1e-4 | SGD weight decay. |
| `alpha_lr` | 0.005 | Architecture parameter learning rate. |
| `alpha_wd` | 1e-3 | Architecture weight decay. |
| `grl_scale` | 1.0 | Gradient Reversal Layer scale (reverses `g_train`). |
| `xi` | 0.01 | DARTS-2nd unroll step. |
| `epochs` | 2 (demo) / 12 (research) | Supernet epochs per search step. |
| `max_steps` | 6 (demo) / 200 (research) | Search steps. |
| `channels` | 8 | Per-cell channel width. |
| `n_cells` | 2 | Stacked mixed cells. |

Optimizer: SGD with momentum and weight decay for weights; plain SGD step for
`alpha`. All RNG streams come from `build_rng(seed, stage)`.

---

## 5. Evaluation results

On the demo budget (`Budget.demo()`, 6 steps, 2 epochs, single CPU core):

| Method | Val acc | Test acc | Wall time | Notes |
| --- | --- | --- | --- | --- |
| DarFuse-1st | ≈ 1.000 | ≈ 1.000 | ≈ 45 s | ≤ 60 s target met. |
| DarFuse-2nd | ≈ 1.000 | ≈ 1.000 | higher | More accurate, slower (invariant G2/G12). |
| Regularized evolution | high | high | higher | Training-based NAS baseline. |
| Random search | high | high | higher | Uniform sampling. |
| Zero-cost proxy | high | high | lowest | Training-free ranking + 1 train. |

Accuracy is reported by `nasforge.evaluate` (`independent_evaluate` and
`one_shot_evaluate`). The DarFuse-1st demo reaches ≈ 1.000 because the synthetic
task is small and easily separable; this is a correctness/learnability signal,
not evidence of ImageNet-level capability.

---

## 6. Factors & risks

- **Architecture ranking quality** depends on the one-shot weight-sharing
  approximation; the discovered cell is validated by *independent* re-training
  (weight inheritance), which mitigates but does not eliminate the gap.
- **Zero-cost proxies** rank architectures without training; their correlation
  with true accuracy (`spearman`) is measured by invariant G12 but is not
  guaranteed on non-synthetic data.
- **Determinism** is a deliberate design goal and is gated (G8); deviations
  indicate an environment that violates single-thread BLAS or seed hygiene.

---

## 7. Limitations

See [known_limitations.md](known_limitations.md) for the full list, including
CPU-only execution, the tiny synthetic toy dataset, the CPU cost of DarFuse-2nd,
the one-shot approximation, the small fixed topology, and lack of transfer to
real vision tasks.
