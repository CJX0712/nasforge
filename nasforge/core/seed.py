"""Global determinism: the single source of randomness for the whole project.

Contract (enforced by ``tests/test_determinism_contract.py`` via AST scan):

* Business modules MUST NOT call ``numpy.random.*`` or ``random.*`` directly.
* Every stochastic component obtains its generator from :func:`build_rng`.
* ``build_rng`` derives the stream from ``(seed, stage)`` through a CRC32 of the
  stage name, so the mapping is stable across processes and platforms. The
  builtin ``hash()`` is deliberately avoided because PYTHONHASHSEED randomisation
  would make it non-reproducible.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import os
import random
import zlib

# Pin thread counts before numpy imports anything that could race. Single-thread
# BLAS keeps the reduction order (and therefore the float64 result) bit-stable
# across machines, which is what makes the determinism gate (G8) hold.
for _env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env, "1")

import numpy as np

from nasforge.core.errors import SeedError

__all__ = ["ENV_SEED", "build_rng", "stage_offset", "set_all", "get_seed"]

ENV_SEED = "NASFORGE_SEED"
DEFAULT_SEED = 20261008

# Offset applied to the per-stage CRC32 so independent stages that hash to
# nearby values still start from well separated seeds.
_STAGE_SALT = 0x9E3779B9


def _validate_seed(seed: int) -> int:
    """Validate and normalise a seed.

    Parameters
    ----------
    seed:
        Any integer. Negative values are remapped into ``[0, 2**32)``.

    Returns
    -------
    int
        Seed in ``[0, 2**32)``.

    Raises
    ------
    SeedError
        If ``seed`` is not an integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise SeedError(
            f"seed must be an integer, got {type(seed).__name__}",
            hint="pass seed=20261008 or set NASFORGE_SEED",
        )
    value = int(seed) % (2**32)
    if value < 0:
        value += 2**32
    return value


def stage_offset(stage: str) -> int:
    """Return a deterministic 31-bit offset for a named random stage.

    Parameters
    ----------
    stage:
        Stage identifier, e.g. ``"data"`` or ``"search"``.

    Returns
    -------
    int
        Offset in ``[0, 2**31)``.

    Notes
    -----
    ``zlib.crc32`` is used instead of :func:`hash` because it is stable across
    processes, platforms and Python versions.
    """
    if not isinstance(stage, str) or not stage:
        raise SeedError(
            "stage must be a non-empty string",
            hint='e.g. build_rng(seed, "data")',
        )
    mixed = zlib.crc32(stage.encode("utf-8"), _STAGE_SALT)
    return mixed % (2**31 - 1)


def build_rng(seed: int, stage: str) -> np.random.Generator:
    """Create the one and only generator used by NasForge.

    Parameters
    ----------
    seed:
        Base integer seed.
    stage:
        Independent stream name. Two different stages never share a stream.

    Returns
    -------
    numpy.random.Generator
        PCG64 generator positioned at ``crc32(seed) + crc32(stage)``.
    """
    base = _validate_seed(seed)
    ss = np.random.SeedSequence(entropy=base, spawn_key=(stage_offset(stage),))
    return np.random.Generator(np.random.PCG64(ss))


def get_seed(default: int = DEFAULT_SEED) -> int:
    """Read the base seed from the environment, falling back to ``default``.

    ``NASFORGE_SEED`` accepts any integer; malformed values raise
    :class:`SeedError` rather than silently falling back, so misconfiguration is
    never hidden.
    """
    raw = os.environ.get(ENV_SEED)
    if raw is None or raw.strip() == "":
        return _validate_seed(default)
    try:
        return _validate_seed(int(raw.strip()))
    except ValueError as exc:
        raise SeedError(
            f"{ENV_SEED}={raw!r} is not an integer",
            hint=f"unset {ENV_SEED} or set it to an integer",
        ) from exc


def set_all(seed: int, stage: str = "global") -> dict[str, object]:
    """Seed every global RNG backend that a third-party library might touch.

    Parameters
    ----------
    seed:
        Base seed.
    stage:
        Stream name for the returned global generator.

    Returns
    -------
    dict
        ``{"seed": int, "stage": str, "generator": Generator}``.

    Notes
    -----
    Python's :mod:`random` and :mod:`os.environ`-based ``PYTHONHASHSEED`` cannot be
    changed after interpreter start, but seeding :mod:`random` still protects
    against third-party code that uses it.
    """
    base = _validate_seed(seed)
    random.seed(base)
    generator = build_rng(base, stage)
    np.random.seed(base % (2**32))
    return {"seed": base, "stage": stage, "generator": generator}
