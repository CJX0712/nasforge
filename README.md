# NasForge

> World-class Neural Architecture Search (NAS), implemented in **pure NumPy** —
> no PyTorch, no autograd. Forward and backward passes are hand-written with
> vector-Jacobian products (VJPs); the system is CPU-only, fully offline, and
> deterministic.

NasForge searches for image-classification cell architectures over a fixed
DARTS-style cell DAG using **DarFuse**, a differentiable search method that
extends first- and second-order DARTS with three refinements: a Gradient
Reversal Layer (GRL) that pushes the architecture toward generalisation, an
architecture-weight-decay stabilizer, and one-shot weight inheritance so the
found discrete cell is fine-tuned from the supernet's shared weights instead of
from scratch.

Author: **晨星 (CJX0712)** · License: MIT · Repo:
<https://github.com/CJX0712/nasforge>

---

## Why "world-class"

- **Pure NumPy, no deep-learning framework.** Every convolution, pooling,
  softmax and its backwards pass is hand-derived and hand-coded. This makes the
  gradient math auditable and the dependency surface tiny (three numeric
  libraries only).
- **Faithful DARTS-2nd.** The second-order unrolled architecture gradient is
  implemented exactly, with the cross term computed by finite differences in
  `alpha` while the validation weight-gradient vector is held fixed — no
  autodiff shortcut.
- **Determinism by contract.** All randomness flows through a single
  `build_rng(seed, stage)` entry point (PCG64 + CRC32 stage salt). The same seed
  produces a *byte-identical* `genotype_json`. Single-thread BLAS keeps float64
  reductions bit-stable.
- **Self-checking test suite.** An invariant suite `G1`–`G12` (gradient checks,
  determinism, isomorphism pruning, GRL sign, zero-cost rank correlation, etc.)
  acts as a continuous correctness gate.
- **Four search methods in one API** — DarFuse-1st, DarFuse-2nd, regularized
  evolution, random search, and a zero-cost-proxy baseline — so results are
  directly comparable.

---

## Installation

```bash
# From the repository root (editable install):
pip install -e .

# Or with pinned dependencies only:
pip install -r requirements.txt
```

Runtime dependencies are minimal: `numpy==2.5.3`, `scipy==1.18.1`,
`scikit-learn==1.9.1`. Development tools (`pytest`, `ruff`) are installed via
the `[dev]` extra: `pip install -e ".[dev]"`.

---

## Quickstart

Run the bundled demo (time-bounded `demo` budget: 6 search steps, 2 epochs):

```bash
python -m nasforge.cli demo
```

This trains the one-shot supernet, searches for a cell, fine-tunes the found
discrete genotype (weight inheritance), and prints the validation / test
accuracy plus the discovered architecture.

---

## Python API usage

```python
from nasforge.data.dataset import SyntheticDataset
from nasforge.core.types import Budget
from nasforge.search import darfuse_search
from nasforge.evaluate import independent_evaluate, accuracy
from nasforge.spaces.prune import unique_canonical

# 1. A deterministic synthetic image-classification dataset.
ds = SyntheticDataset(n_samples=320, n_classes=4, img_size=9, seed=20260408)
print(ds.in_shape())      # (3, 9, 9)
print(ds.n_classes())     # 4

# 2. Search with DarFuse-1st (set second_order=True for DarFuse-2nd).
result = darfuse_search(ds, Budget.demo(), seed=20261008)
print(result.method, result.best_val_acc, result.best_test_acc)
print(result.param_count, "params")

# 3. Inspect and reuse the found architecture.
geno = result.best_genotype
print(geno.to_json())          # Genotype -> JSON-safe dict
print(result.genotype_json)    # same dict, cached on the SearchResult

# 4. Independently re-evaluate the discrete genotype from scratch.
acc = independent_evaluate(geno, ds, seed=20261009)
print(acc.val_acc, acc.test_acc)

# 5. Deduplicate a population of genotypes up to input-swap isomorphism.
unique = unique_canonical([geno])
```

Other search entry points share the same signature:
`evolution_search`, `random_search`, `zerocost_search` — each returns a
`SearchResult`.

Key types:

- `Budget.demo()` → `max_steps=6, epochs=2` (≤ 60 s target).
- `Budget.research()` → `max_steps=200, epochs=12`.
- `SearchResult` fields: `method`, `best_genotype`, `best_val_acc`,
  `best_test_acc`, `n_evaluated`, `wall_time_s`, `search_trace`,
  `one_shot_scores`, `param_count`, `genotype_json`.

---

## CLI reference

The CLI is exposed both as `python -m nasforge.cli <command>` and as the
installed console script `nasforge <command>` (see
`[project.scripts]` in `pyproject.toml`).

| Command | Description |
| --- | --- |
| `demo` | Run the DarFuse-1st search on the `demo` budget and report accuracy + genotype. |
| `search` | Run a search method with a chosen profile. Options: `--method {darfuse,evolution,random,zerocost}`, `--profile {demo,research}`, `--second-order` (DarFuse-2nd only). |
| `evaluate` | Independently train + evaluate a genotype (or the last found one) from scratch. |
| `bench` | Run the zero-cost proxy scoring across a random population and report rank correlation. |
| `version` | Print the installed NasForge version (`0.1.0`). |

Examples:

```bash
# DarFuse-1st, demo budget (default):
python -m nasforge.cli demo

# DarFuse-2nd, research profile:
python -m nasforge.cli search --method darfuse --profile research --second-order

# Regularized evolution, demo budget:
python -m nasforge.cli search --method evolution --profile demo

# Random search baseline:
python -m nasforge.cli search --method random --profile research

# Zero-cost proxy search:
python -m nasforge.cli search --method zerocost --profile demo

# Report version:
nasforge version
```

---

## The DarFuse algorithm

DarFuse is a differentiable architecture search over the continuous
architecture parameters `alpha` (one softmax row per cell edge). Each search
step trains the shared supernet weights for `epochs` epochs, then updates
`alpha` using the **DarFuse architecture gradient**:

```
g_alpha = g_val - grl_scale * g_train - alpha_wd * alpha
```

- `g_val` — gradient of the *validation* loss w.r.t. `alpha`.
- `g_train` — gradient of the *training* loss w.r.t. `alpha`, **reversed** by the
  Gradient Reversal Layer (`grl`). Reversing the training gradient discourages
  the architecture from exploiting train-set overfitting and steers it toward
  generalisation.
- `alpha_wd * alpha` — an L2 penalty on `alpha` for optimization stability.

After the last step the continuous `alpha` is **discretized** (argmax per edge)
to a `Genotype`, and that discrete cell is **fine-tuned from the supernet's
shared weights** (weight inheritance) rather than trained from scratch.

DarFuse-2nd (DARTS 2nd-order unrolled) replaces `g_val` with the implicit
gradient that accounts for how the weights respond to `alpha` — see
[docs/architecture.md](nasforge/docs/architecture.md) for the cross-term formula.

**Hyperparameters (defaults):** `alpha_lr=0.005`, `w_lr=0.05`, `alpha_wd=1e-3`,
`grl_scale=1.0`, `xi=0.01` (unroll step), weight optimizer SGD
(`momentum=0.9`, `weight_decay=1e-4`).

---

## Baselines

| Method | Training-free? | Notes |
| --- | --- | --- |
| `darfuse_search` | No (one-shot supernet) | DarFuse-1st / DarFuse-2nd. |
| `evolution_search` | No | Regularized (aging) evolution — faithful training-based NAS. |
| `random_search` | No | Uniform random sampling + independent evaluation. |
| `zerocost_search` | Mostly | Ranks random architectures by training-free proxies, then trains the top one. |

Zero-cost proxies (`nasforge/search/naswiz_metrics.py`): `snip`, `grad_norm`,
`logits_var`, `n_params` (NAS-WiZ / Zero-Cost-NAS, Abdelfattah et al., ICLR'21).

---

## Determinism & reproducibility

- **Single source of randomness.** Every stochastic component obtains its
  generator from `nasforge.core.seed.build_rng(seed, stage)` — business code never
  calls `numpy.random.*` / `random.*` directly (enforced by a test that AST-scans
  the tree).
- **Stage-separated streams.** The `(seed, stage)` pair is hashed with CRC32
  (not Python's salted `hash()`), so streams are stable across processes and
  platforms.
- **Bit-stable floats.** BLAS thread counts are pinned to 1 at import time, so
  float64 reductions keep a fixed summation order.
- **Same seed → same result.** Identical `seed` yields a byte-identical
  `genotype_json`; the determinism gate (G8) verifies this.

---

## Project structure

```
nasforge/
├── cli.py                 # (lead) console entry point: demo/search/evaluate/bench/version
├── core/                  # types, seed (determinism), errors (E1xx–E5xx)
├── data/                  # dataset.py — deterministic SyntheticDataset
├── ops/                   # registry + hand-written conv/pool/identity operators
├── spaces/                # search_space, network (one-shot supernet), canonical, prune, genotype
├── training/              # SGD optimizer + trainer (epoch loop, weight_grad, train_discrete)
├── search/                # darts (DarFuse), darts_second_order, evolution, random_search, zerocost
├── evaluate/              # independent_eval, one_shot, metrics (accuracy, spearman, n_params)
├── docs/                  # architecture.md, model_card.md, known_limitations.md, reproduce.md
└── tests/                 # invariant suite G1–G12 + determinism/CLI tests
```

---

## Results (demo profile)

Measured on a single CPU core, `Budget.demo()` (6 steps, 2 epochs):

| Method | Val acc | Test acc | Wall time | Params |
| --- | --- | --- | --- | --- |
| **DarFuse-1st** | ≈ 1.000 | ≈ 1.000 | ≈ 45 s | ~1.4k |
| DarFuse-2nd | ≈ 1.000 | ≈ 1.000 | higher* | ~1.4k |
| Regularized evolution | high | high | higher | varies |
| Random search | high | high | higher | varies |
| Zero-cost proxy | high | high | lowest | varies |

\*DarFuse-2nd is more accurate than DarFuse-1st (invariant G2/G12) but is
deliberately expensive on CPU (36 `alpha` indices × finite-difference
weight-grad). The `demo` budget is guaranteed ≤ 60 s only with DarFuse-1st.

---

## License & author

MIT License. Copyright **晨星 (CJX0712)** 2026. See [LICENSE](LICENSE).

Full documentation:

- [docs/architecture.md](nasforge/docs/architecture.md) — cell DAG, DarFuse math, supernet, error taxonomy.
- [docs/model_card.md](nasforge/docs/model_card.md) — intended use, data, hyperparams, factors/risks.
- [docs/known_limitations.md](nasforge/docs/known_limitations.md) — CPU-only, toy data, 2nd-order cost, etc.
- [docs/reproduce.md](nasforge/docs/reproduce.md) — exact commands and environment.
- [CHANGELOG.md](CHANGELOG.md)
