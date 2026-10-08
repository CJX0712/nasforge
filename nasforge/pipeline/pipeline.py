"""Search orchestration for NasForge.

Wraps the four search algorithms behind a single dispatch and renders a
human-readable report. The :class:`~nasforge.core.types.SearchResult` contract is
the only object that crosses this boundary, so every algorithm (and every future
one) plugs in through ``run_search``.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.core.types import SearchResult
from nasforge.search import (
    darfuse_search,
    evolution_search,
    random_search,
    zerocost_search,
)

__all__ = ["run_search", "format_report", "SearchReport"]

_METHODS = {
    "darfuse": darfuse_search,
    "evolution": evolution_search,
    "random": random_search,
    "zerocost": zerocost_search,
}


def run_search(method, dataset, budget, seed, **kwargs) -> SearchResult:
    """Dispatch to a search algorithm by name.

    Parameters
    ----------
    method:
        One of ``darfuse``, ``evolution``, ``random``, ``zerocost``.
    dataset, budget, seed:
        Forwarded to the algorithm.
    kwargs:
        Algorithm-specific options (e.g. ``second_order`` for darfuse).
    """
    if method not in _METHODS:
        from nasforge.core.errors import SearchError

        raise SearchError(
            f"unknown search method {method!r}; known: {sorted(_METHODS)}",
            code="E400",
        )
    if method == "darfuse":
        kwargs.setdefault("second_order", False)
    return _METHODS[method](dataset, budget, seed, **kwargs)


def format_report(result: SearchResult, name: str | None = None) -> str:
    """One-line summary of a :class:`SearchResult`."""
    name = name or result.method
    return (
        f"[{name}] val={result.best_val_acc:.4f} test={result.best_test_acc:.4f} "
        f"params={result.param_count} steps={result.n_evaluated} "
        f"t={result.wall_time_s:.1f}s"
    )


class SearchReport:
    """A small wrapper that pretty-prints a :class:`SearchResult`."""

    def __init__(self, result: SearchResult, name: str | None = None) -> None:
        self.result = result
        self.name = name or result.method

    def __str__(self) -> str:
        return format_report(self.result, self.name)

    def as_dict(self) -> dict:
        return {
            "method": self.result.method,
            "best_val_acc": self.result.best_val_acc,
            "best_test_acc": self.result.best_test_acc,
            "param_count": self.result.param_count,
            "n_evaluated": self.result.n_evaluated,
            "wall_time_s": self.result.wall_time_s,
        }
