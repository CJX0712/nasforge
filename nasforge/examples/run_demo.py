"""Run the DarFuse-1st demo and print the discovered architecture.

Usage:  python -m nasforge.examples.run_demo

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nasforge.core.types import Budget
from nasforge.data.dataset import SyntheticDataset
from nasforge.search import darfuse_search
from nasforge.spaces.genotype import op_histogram
from nasforge.spaces.search_space import DEFAULT_SPACE


def main() -> None:
    ds = SyntheticDataset(seed=20260408)
    res = darfuse_search(ds, Budget.demo(), seed=20260408, second_order=False)
    print(f"DarFuse-1st demo: val={res.best_val_acc:.4f} test={res.best_test_acc:.4f} t={res.wall_time_s:.1f}s")
    print("Discovered genotype:", json.dumps(res.genotype_json, ensure_ascii=False))
    print("Operator histogram:", op_histogram(res.best_genotype, DEFAULT_SPACE))


if __name__ == "__main__":
    main()
