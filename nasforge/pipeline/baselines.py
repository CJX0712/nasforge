"""Run every search strategy under a shared budget and compare them.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import Budget, SearchResult
from nasforge.pipeline.pipeline import run_search

__all__ = ["run_all_baselines", "compare_table"]


def run_all_baselines(
    dataset,
    budget: Budget,
    seed: int,
    methods=("darfuse", "random", "evolution", "zerocost"),
    **kwargs,
) -> dict[str, SearchResult]:
    """Run each method under ``budget`` and return ``{method: SearchResult}``."""
    out: dict[str, SearchResult] = {}
    for m in methods:
        out[m] = run_search(m, dataset, budget, seed, **kwargs)
    return out


def compare_table(results: dict[str, SearchResult]) -> str:
    """Render a fixed-width comparison table of several :class:`SearchResult`."""
    header = f"{'method':<12} {'val':>7} {'test':>7} {'params':>7} {'t(s)':>7}"
    lines = [header, "-" * len(header)]
    for m, r in results.items():
        lines.append(
            f"{m:<12} {r.best_val_acc:7.4f} {r.best_test_acc:7.4f} "
            f"{r.param_count:7d} {r.wall_time_s:7.1f}"
        )
    return "\n".join(lines)
