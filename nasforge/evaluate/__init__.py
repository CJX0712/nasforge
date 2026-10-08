"""Evaluation layer for NasForge.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.evaluate.independent_eval import independent_evaluate
from nasforge.evaluate.metrics import accuracy, n_params, spearman
from nasforge.evaluate.one_shot import one_shot_evaluate

__all__ = [
    "independent_evaluate",
    "one_shot_evaluate",
    "accuracy",
    "spearman",
    "n_params",
]
