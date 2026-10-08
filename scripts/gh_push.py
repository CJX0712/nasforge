#!/usr/bin/env python
"""Robust 3-tier GitHub push helper for NasForge.

Importable: `push(repo=..., remote=..., branch=..., commit_message=..., dry_run=...) -> int`.
Runnable: `python scripts/gh_push.py [--repo ...] [--branch ...] [--dry-run] ...`

Tiers (attempted in order, first success wins):
  T1: plain `git push -u <remote> <branch>` (relies on gh credential helper / auth).
  T2: inject a token obtained via `gh auth token` into the remote URL, push, then
      reset the remote URL back to the public https URL (token is never persisted).
  T3: SSH push via `git@github.com:<repo>.git`.

The raw token is never printed; any URL containing credentials is masked in logs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

AUTHOR_NAME = "晨星"
AUTHOR_EMAIL = "CJX0712@users.noreply.github.com"
REPO_DESC = (
    "NasForge — world-class Neural Architecture Search "
    "(DarFuse DARTS 2nd-order + GRL + one-shot weight inheritance + zero-cost proxy), "
    "pure NumPy, CPU-only, deterministic."
)
DEFAULT_COMMIT = "NasForge v0.1.0 — world-class NAS (DarFuse + baselines)"

# Mask any credentials embedded in a URL before logging.
_TOKEN_RE = re.compile(r"https://[^@/]+@")


def mask(s: str) -> str:
    return _TOKEN_RE.sub("https://***@", s)


class _Result:
    def __init__(self, rc: int, out: str, err: str):
        self.returncode = rc
        self.out = out
        self.err = err


def run(cmd, *, capture: bool = False, check: bool = True, dry_run: bool = False,
        readonly: bool = False) -> _Result:
    printable = " ".join(mask(str(c)) for c in cmd)
    if dry_run and not readonly:
        print(f"[dry-run] would run: {printable}")
        return _Result(0, "", "")
    if dry_run and readonly:
        print(f"[dry-run] check: {printable}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] command not found: {mask(str(cmd[0]))} ({e})", file=sys.stderr)
        return _Result(127, "", str(e))
    if check and proc.returncode != 0:
        print(f"[ERROR] command failed ({proc.returncode}): {printable}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
    return _Result(proc.returncode, proc.stdout or "", proc.stderr or "")


def ensure_git_config(dry_run: bool) -> None:
    for key, val in (("user.name", AUTHOR_NAME), ("user.email", AUTHOR_EMAIL)):
        res = run(["git", "config", key], capture=True, check=False,
                  dry_run=dry_run, readonly=True)
        if not res.out.strip():
            run(["git", "config", key, val], dry_run=dry_run)


def ensure_git_repo(dry_run: bool) -> bool:
    if Path(".git").is_dir():
        return True
    print("[INFO] not a git repo — initializing.")
    run(["git", "init"], dry_run=dry_run)
    return False


def ensure_remote(remote: str, repo: str, dry_run: bool) -> None:
    url = f"https://github.com/{repo}.git"
    res = run(["git", "remote", "get-url", remote], capture=True, check=False,
              dry_run=dry_run, readonly=True)
    if res.returncode == 0 and res.out.strip():
        return
    print(f"[INFO] adding remote '{remote}' -> {mask(url)}")
    run(["git", "remote", "add", remote, url], dry_run=dry_run)


def ensure_gh_repo(repo: str, dry_run: bool) -> None:
    print(f"[INFO] ensuring GitHub repo '{repo}' exists (ignored if already present).")
    run(
        ["gh", "repo", "create", repo, "--public", "--description", REPO_DESC, "--confirm"],
        dry_run=dry_run,
    )


def commit_all(branch: str, message: str, dry_run: bool) -> bool:
    run(["git", "add", "-A"], dry_run=dry_run)
    res = run(["git", "status", "--porcelain"], capture=True, check=False,
              dry_run=dry_run, readonly=True)
    if not res.out.strip():
        print("[INFO] nothing to commit.")
        return False
    run(["git", "commit", "-m", message], dry_run=dry_run)
    return True


def _push_tier1(remote: str, branch: str, repo: str = "", dry_run: bool = False) -> bool:
    print("[INFO] Tier 1: plain push (credential helper).")
    res = run(["git", "push", "-u", remote, branch], dry_run=dry_run)
    return res.returncode == 0


def _push_tier2(remote: str, branch: str, repo: str, dry_run: bool) -> bool:
    print("[INFO] Tier 2: push with gh-obtained token.")
    tok_res = run(["gh", "auth", "token"], capture=True, check=False, dry_run=dry_run)
    token = tok_res.out.strip()
    if not token:
        print("[WARN] Tier 2: no token from `gh auth token`; skipping.", file=sys.stderr)
        return False
    auth_url = f"https://{token}@github.com/{repo}.git"
    pub_url = f"https://github.com/{repo}.git"
    try:
        run(["git", "remote", "set-url", remote, auth_url], dry_run=dry_run)
        res = run(["git", "push", "-u", remote, branch], dry_run=dry_run)
        return res.returncode == 0
    finally:
        # Never persist the token in the remote URL.
        run(["git", "remote", "set-url", remote, pub_url], dry_run=dry_run, readonly=False)


def _push_tier3(remote: str, branch: str, repo: str, dry_run: bool) -> bool:
    print("[INFO] Tier 3: SSH push.")
    ssh_url = f"git@github.com:{repo}.git"
    run(["git", "remote", "set-url", remote, ssh_url], dry_run=dry_run)
    res = run(["git", "push", "-u", remote, branch], dry_run=dry_run)
    return res.returncode == 0


def push(*, repo: str = "CJX0712/nasforge", remote: str = "origin",
         branch: str = "main", commit_message: str = DEFAULT_COMMIT,
         dry_run: bool = False) -> int:
    ensure_git_config(dry_run)
    was_repo = ensure_git_repo(dry_run)
    ensure_gh_repo(repo, dry_run)
    ensure_remote(remote, repo, dry_run)
    commit_all(branch, commit_message, dry_run)

    if dry_run:
        print(
            "[dry-run] plan: "
            f"repo={'existing' if was_repo else 'would-init'}, "
            f"remote={remote}, branch={branch}, "
            f"commit_message={commit_message!r}, "
            "tiers=[T1 plain -> T2 token -> T3 ssh]"
        )
        return 0

    for tier in (_push_tier1, _push_tier2, _push_tier3):
        if tier(remote, branch, repo, dry_run=False):
            print(f"[OK] push succeeded via {tier.__name__}.")
            return 0
    print("[FAIL] all push tiers failed.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NasForge GitHub push helper (3-tier).")
    parser.add_argument("--repo", default="CJX0712/nasforge", help="owner/name")
    parser.add_argument("--remote", default="origin", help="git remote name")
    parser.add_argument("--branch", default="main", help="branch to push")
    parser.add_argument("--commit-message", default=DEFAULT_COMMIT, help="commit message")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without making network changes")
    args = parser.parse_args(argv)
    return push(
        repo=args.repo,
        remote=args.remote,
        branch=args.branch,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
