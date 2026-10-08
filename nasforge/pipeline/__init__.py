"""End-to-end pipeline: search orchestration, baselines, and benchmarking.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.pipeline.baselines import compare_table, run_all_baselines
from nasforge.pipeline.benchmark import (
    benchmark_one_shot_vs_independent,
    benchmark_zerocost_vs_independent,
)
from nasforge.pipeline.pipeline import SearchReport, format_report, run_search

__all__ = [
    "run_search",
    "SearchReport",
    "format_report",
    "run_all_baselines",
    "compare_table",
    "benchmark_zerocost_vs_independent",
    "benchmark_one_shot_vs_independent",
]
