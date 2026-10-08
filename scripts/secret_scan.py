#!/usr/bin/env python
"""Scan the NasForge repo for accidentally committed secrets.

Importable: `scan_path(root) -> list[str]` returns the list of "<file>:<line>"
findings. Runnable: `python scripts/secret_scan.py` exits 0 when clean and 1
when a secret is detected.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# The scanner's own source contains the literal pattern strings (e.g. "ghp_",
# "sk-"), so it must never flag itself. Excluded by default.
_SELF = Path(__file__).resolve()

# --- Patterns to flag (case-insensitive) -------------------------------------
PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ghp_",
        r"gho_",
        r"ghu_",
        r"ghs_",
        r"ghr_",
        r"github_pat_",
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"xox[baprs]-",
        r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        r"AIza[0-9A-Za-z_-]{35}",
        r"eyJ[A-Za-z0-9_-]{10,}\.eyJ",
        r"OPENAI_API_KEY\s*=\s*sk-",
        r"ANTHROPIC_API_KEY\s*=\s*sk-ant",
    ]
]

# --- Legitimate strings that MUST NEVER be flagged ----------------------------
ALLOWED = (
    "CJX0712@users.noreply.github.com",  # author email
    "晨星",                                 # author name (Chinese, required)
    "github.com/CJX0712/nasforge",         # repo URL
)

SCAN_EXTS = {
    ".py", ".txt", ".md", ".yml", ".yaml", ".toml",
    ".cfg", ".ini", ".sh", ".json", ".rst", ".Dockerfile",
}


def _iter_text_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        if p.name == "Dockerfile" or p.suffix.lower() in SCAN_EXTS:
            yield p


def scan_path(root: str | Path, exclude: set[Path] | None = None) -> list[str]:
    """Return a list of "<file>:<line>" strings where a secret pattern matched.

    `exclude` is a set of absolute paths to skip (defaults to this scanner's own
    file, whose pattern literals are not real secrets).
    """
    root = Path(root)
    excluded = exclude if exclude is not None else {_SELF}
    findings: list[str] = []
    for path in _iter_text_files(root):
        if path.resolve() in excluded:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root)
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Never flag legitimate author email / name / repo URL.
            if any(allowed in line for allowed in ALLOWED):
                continue
            for pat in PATTERNS:
                if pat.search(line):
                    findings.append(f"{rel}:{lineno}")
                    break
    return findings


def main() -> int:
    root = Path(__file__).resolve().parent.parent  # scripts/ -> project root
    findings = scan_path(root)
    if findings:
        for f in findings:
            print(f"[FAIL] secret in {f}")
        count = len(findings)
        print(f"[FAIL] {count} potential secret(s) found.", file=sys.stderr)
        return 1
    print("[OK] no secrets found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
