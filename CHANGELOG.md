# Changelog

All notable changes to NasForge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
semantic versioning.

Author: **晨星 (CJX0712)**.

---

## [0.1.0] — 2026-10-08

Initial release.

### Added

- **DarFuse differentiable NAS** — first-order (DarFuse-1st) and second-order
  unrolled (DarFuse-2nd, DARTS-2nd) architecture search over a fixed DARTS-style
  cell.
- **DarFuse refinements** layered on baseline DARTS:
  - Gradient Reversal Layer (GRL) decoupling architecture learning from weight
    fitting (`g_alpha = g_val - grl_scale * g_train - alpha_wd * alpha`).
  - Architecture weight decay for stability.
  - **Weight inheritance** — the found discrete genotype is fine-tuned from the
    one-shot supernet's shared weights.
- **Baselines** in a single API:
  - Regularized (aging) evolution (`evolution_search`).
  - Random search (`random_search`).
  - Zero-cost-proxy search (`zerocost_search`) with NAS-WiZ proxies: `snip`,
    `grad_norm`, `logits_var`, `n_params`.
- **Deterministic synthetic dataset** (`SyntheticDataset`, 320 samples, 4
  classes, 9×9, fully reproducible from one seed; no network download).
- **Pure-NumPy implementation, single-file-free** — every forward/backward pass
  is hand-written with VJPs; no PyTorch/autograd, CPU-only, fully offline.
- **Determinism infrastructure** — single `build_rng(seed, stage)` entry point
  (PCG64 + CRC32 stage salt) and single-thread BLAS pinning for bit-stable
  floats.
- **Full invariant test suite G1–G12** — gradient checks, determinism,
  isomorphism pruning, GRL sign, zero-cost rank correlation, and more — plus CLI
  and determinism contract tests.
- **CLI** (`nasforge.cli`) with subcommands: `demo`, `search`
  (`--method`, `--profile`, `--second-order`), `evaluate`, `bench`, `version`.
- Project scaffolding: `pyproject.toml`, `Makefile`, `Dockerfile`, requirements
  files, documentation (`README.md`, `docs/architecture.md`,
  `docs/model_card.md`, `docs/known_limitations.md`, `docs/reproduce.md`),
  `CHANGELOG.md`, and `LICENSE` (MIT, copyright 晨星 (CJX0712) 2026).

---

## License

Released under the MIT License. Copyright **晨星 (CJX0712)** 2026.
