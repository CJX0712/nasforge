# Reproducing NasForge

Author: **晨星 (CJX0712)**. This page lists the exact commands and environment
needed to reproduce every result in the README and model card.

---

## 1. Environment

| Component | Version |
| --- | --- |
| Python | ≥ 3.12 |
| numpy | == 2.5.3 |
| scipy | == 1.18.1 |
| scikit-learn | == 1.9.1 |

Pinned versions are also in `requirements.txt` and `requirements.lock.txt`. The
`[dev]` extra adds `pytest`, `pytest-cov`, and `ruff` for the test/lint gate.

---

## 2. Installation

```bash
# Editable install with dev tools (recommended):
pip install -e ".[dev]"

# Or runtime dependencies only:
pip install -r requirements.txt
```

---

## 3. Core commands

Run the time-bounded demo (DarFuse-1st, `demo` budget, ≤ 60 s target):

```bash
python -m nasforge.cli demo
```

Run a search with a chosen method and profile:

```bash
# DarFuse-2nd on the research profile:
python -m nasforge.cli search --method darfuse --profile research --second-order

# Regularized evolution, demo budget:
python -m nasforge.cli search --method evolution --profile demo

# Random search, research profile:
python -m nasforge.cli search --method random --profile research

# Zero-cost proxy search, demo budget:
python -m nasforge.cli search --method zerocost --profile demo
```

Run the full test suite:

```bash
python -m pytest -q
```

Run the local CI gate (`make gate`): secret scan + UTF-8/BOM/NUL check across
`nasforge/*.py`:

```bash
make gate
```

---

## 4. Determinism notes

- **All randomness** flows through `nasforge.core.seed.build_rng(seed, stage)`.
  Business modules never call `numpy.random.*` or `random.*` directly (an AST
  scan test enforces this).
- **Single-thread BLAS.** NasForge pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to
  `1` at import time so float64 reductions have a fixed summation order — this is
  what keeps results bit-stable across machines.
- **Same seed → byte-identical genotype.** Identical `seed` yields a
  byte-identical `genotype_json`. The determinism gate (G8) verifies this; set
  the seed via the API `seed=` argument or the `NASFORGE_SEED` environment
  variable.
- **Stage separation.** The `(seed, stage)` pair is hashed with CRC32 (not
  Python's salted `hash()`), so independent stages never share a random stream
  and results are reproducible across processes and platforms.

---

## 5. Makefile shortcuts

```bash
make setup   # pip install -e ".[dev]"
make lint    # ruff check nasforge
make test    # python -m pytest -q
make demo    # python -m nasforge.cli demo
make gate    # secret scan + UTF-8/BOM/NUL check
make ci      # lint + test + demo + gate
make build   # python -m build
make docker  # docker build -t nasforge:0.1.0 .
```

---

## 6. Expected demo outcome

On a single CPU core, `python -m nasforge.cli demo` (DarFuse-1st, 6 steps, 2
epochs) should complete in **≈ 45 s** (≤ 60 s) and report val/test accuracy
**≈ 1.000** on the synthetic dataset, along with the discovered discrete
genotype. DarFuse-2nd reaches the same accuracy but takes longer (see
[known_limitations.md](known_limitations.md)).
