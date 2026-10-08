"""Run all four search strategies under the demo budget and compare them.

Usage:  python -m nasforge.examples.run_baselines

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nasforge.core.types import Budget
from nasforge.data.dataset import SyntheticDataset
from nasforge.pipeline.baselines import compare_table, run_all_baselines


def main() -> None:
    ds = SyntheticDataset(seed=20260408)
    results = run_all_baselines(ds, Budget.demo(), seed=20260408)
    print(compare_table(results))


if __name__ == "__main__":
    main()
