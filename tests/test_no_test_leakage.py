"""Forbidden direct RNG usage scan (determinism contract, G8-adjacent).

Business modules MUST obtain randomness from ``nasforge.core.seed.build_rng`` and
must never call ``numpy.random.*`` / ``random.*`` directly (except the sanctioned
seed module itself). Tests are exempt. This AST-light source scan enforces that.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "nasforge"

FORBIDDEN = [
    "np.random.standard_normal",
    "np.random.randint",
    "np.random.random",
    "np.random.permutation",
    "np.random.choice",
    "np.random.uniform",
    "np.random.normal",
    "np.random.seed",
    "np.random.rand",
    "np.random.randn",
    "np.random.multivariate_normal",
    "random.randint",
    "random.random",
    "random.choice",
    "random.seed",
    "random.uniform",
    "random.gauss",
]

# The seed module is the single sanctioned source of randomness.
EXEMPT = {"core/seed.py", "_gradcheck.py"}


def test_no_direct_random_in_business_modules():
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("tests/"):
            continue
        if rel in EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN:
            if pat in text:
                violations.append((rel, pat))
    assert not violations, f"forbidden direct RNG usage: {violations}"
