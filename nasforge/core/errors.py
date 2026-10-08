"""Error codes and exception hierarchy for NasForge.

Code segments (see docs/architecture.md):

    E1xx  configuration / validation
    E2xx  data layer
    E3xx  search space / operators
    E4xx  search algorithms
    E5xx  evaluation / training

All exceptions derive from :class:`NasForgeError` so callers can catch a single
base class. Author: 晨星 (CJX0712).
"""

from __future__ import annotations

__all__ = [
    "NasForgeError",
    "ConfigError",
    "SeedError",
    "DataError",
    "SpaceError",
    "OperatorError",
    "SearchError",
    "DiscretizationError",
    "EvaluationError",
    "OneShotError",
    "TrainingError",
    "EXIT_OK",
    "EXIT_CONFIG",
    "EXIT_DATA",
    "EXIT_SEARCH",
    "EXIT_EVAL",
]

# CLI exit codes -- stable contract, asserted by tests/test_cli.py
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_SEARCH = 4
EXIT_EVAL = 5


class NasForgeError(Exception):
    """Base class for every NasForge error.

    Attributes
    ----------
    code:
        Numeric error code, e.g. ``310``.
    message:
        Human readable description (ASCII-safe).
    hint:
        Actionable remediation hint.
    """

    code: int = 0

    def __init__(self, message: str, *, code=None, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        # A caller may override the class-level code with either an int (400) or a
        # string ("E400"); normalise both to the bare integer used in "[E{code}]".
        if code is not None:
            norm = str(code).lstrip("E")
            self.code = int(norm) if norm.isdigit() else code
        detail = f"[E{self.code}] {message}"
        if hint:
            detail = f"{detail} | hint: {hint}"
        super().__init__(detail)


# --------------------------------------------------------------------------- E1xx
class ConfigError(NasForgeError):
    """Invalid configuration value or environment override."""

    code = 100


class SeedError(NasForgeError):
    """Illegal seed or seed-stage combination."""

    code = 110


# --------------------------------------------------------------------------- E2xx
class DataError(NasForgeError):
    """Dataset generation or loading failure."""

    code = 200


# --------------------------------------------------------------------------- E3xx
class SpaceError(NasForgeError):
    """Malformed search space specification."""

    code = 300


class OperatorError(NasForgeError):
    """Operator configuration or shape mismatch."""

    code = 310


# --------------------------------------------------------------------------- E4xx
class SearchError(NasForgeError):
    """Search algorithm failed to produce a valid candidate."""

    code = 400


class DiscretizationError(NasForgeError):
    """Architecture discretization produced an invalid genotype."""

    code = 410


# --------------------------------------------------------------------------- E5xx
class EvaluationError(NasForgeError):
    """Evaluation protocol violation."""

    code = 500


class OneShotError(NasForgeError):
    """One-shot weight inheritance protocol violation."""

    code = 510


class TrainingError(NasForgeError):
    """Optimiser or schedule failure."""

    code = 520
