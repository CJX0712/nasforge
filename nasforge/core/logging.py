"""Structured logging plus the benchmark schema constants.

This module owns ``SCHEMA_VERSION`` and the single source of truth for the
``benchmark.json`` field names so both the pipeline and the tests assert against
the same key set.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import json
import logging
import sys

__all__ = ["SCHEMA_VERSION", "RESULT_KEYS", "get_logger", "validate_benchmark"]

SCHEMA_VERSION = "1.0"

RESULT_KEYS = (
    "darfuse",
    "re",
    "random",
    "zerocost",
    "handcraft_conv3x3",
    "handcraft_avgpool",
)

_RESULT_FIELDS = (
    "best_val_acc",
    "best_test_acc",
    "n_evaluated",
    "wall_time_s",
    "param_count",
    "rank_corr_one_shot_vs_independent",
    "genotype",
    "stopped_early",
)


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger (no print in this project)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def validate_benchmark(payload: dict) -> None:
    """Raise ``ValueError`` if ``payload`` is missing required benchmark fields.

    Parameters
    ----------
    payload:
        The dict that will be serialised to ``benchmark.json``.

    Raises
    ------
    ValueError
        When a structural field or a required result entry is absent.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("benchmark.json missing 'results' mapping")
    for key in RESULT_KEYS:
        if key not in results:
            raise ValueError(f"benchmark.results missing key '{key}'")
    for key in RESULT_KEYS:
        entry = results[key]
        for field_name in _RESULT_FIELDS:
            if field_name not in entry:
                raise ValueError(f"benchmark.results['{key}'] missing '{field_name}'")


def dump_benchmark(payload: dict) -> str:
    """Serialise a validated benchmark payload to a stable, sorted JSON string."""
    validate_benchmark(payload)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
