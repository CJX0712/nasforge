"""CLI smoke test and the demo timing gate (G9: demo must finish in <= 60s).

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args, timeout=180):
    return subprocess.run(
        [PY, "-m", "nasforge.cli", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_cli_version():
    r = _run(["version"])
    assert r.returncode == 0, r.stderr
    assert "nasforge" in r.stdout.lower()


def test_cli_demo_within_budget():
    r = _run(["demo"], timeout=180)
    assert r.returncode == 0, r.stderr
    assert "val=" in r.stdout
    # G9 gate: the demo report prints t=<seconds>s; it must be under 60s.
    m = re.search(r"t=([0-9]*\.[0-9]+)s", r.stdout)
    assert m is not None, "could not parse wall-time from demo report"
    assert float(m.group(1)) <= 60.0, f"demo took {m.group(1)}s (> 60s budget)"


def test_cli_search_random_quick():
    r = _run(["search", "--method", "random", "--steps", "2"], timeout=180)
    assert r.returncode == 0, r.stderr
    assert "val=" in r.stdout
