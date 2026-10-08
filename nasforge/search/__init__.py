"""Search algorithms for NasForge.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

from nasforge.search.darts import search as darfuse_search
from nasforge.search.evolution import search as evolution_search
from nasforge.search.random_search import search as random_search
from nasforge.search.zerocost import search as zerocost_search

__all__ = [
    "darfuse_search",
    "evolution_search",
    "random_search",
    "zerocost_search",
]
