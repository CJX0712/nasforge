"""Command-line interface for NasForge.

Subcommands:

* ``demo``      -- run the DarFuse-1st demo search (``Budget.demo``, must finish <= 60s).
* ``search``    -- run any search method (``--method``, ``--profile``, ``--second-order``).
* ``evaluate``  -- independently evaluate a genotype (random if no ``--genotype`` file).
* ``bench``     -- benchmark the zero-cost proxy against independent evaluation (G12).
* ``version``   -- print the version.

The console-script entry point is ``nasforge = nasforge.cli:main`` (see
``pyproject.toml``). The CLI never writes secrets and always exits with a
:mod:`nasforge.core.errors` exit code on failure.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from nasforge.core.seed import get_seed
from nasforge.core.types import Budget, Genotype
from nasforge.data.dataset import SyntheticDataset
from nasforge.pipeline.pipeline import format_report, run_search
from nasforge.search import darfuse_search

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nasforge",
        description="NasForge - world-class Neural Architecture Search (pure NumPy, CPU-only, deterministic).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Run the DarFuse-1st demo search (Budget.demo, <= 60s).")

    sp = sub.add_parser("search", help="Run a search method.")
    sp.add_argument("--method", choices=["darfuse", "evolution", "random", "zerocost"], default="darfuse")
    sp.add_argument("--profile", choices=["demo", "research"], default="demo")
    sp.add_argument("--second-order", action="store_true", help="DarFuse-2nd (unrolled implicit gradient).")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--steps", type=int, default=None, help="Override Budget.max_steps.")

    ev = sub.add_parser("evaluate", help="Independently evaluate a genotype (random if no --genotype).")
    ev.add_argument("--genotype", type=str, default=None, help="Path to a genotype JSON file.")
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--profile", choices=["demo", "research"], default="demo")

    sub.add_parser("bench", help="Benchmark the zero-cost proxy vs independent evaluation (G12).")

    sub.add_parser("version", help="Print the version string.")
    return p


def _dataset() -> SyntheticDataset:
    return SyntheticDataset(seed=20260408)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    seed = get_seed() if getattr(args, "seed", None) is None else args.seed
    try:
        if args.cmd == "demo":
            res = darfuse_search(_dataset(), Budget.demo(), seed=seed, second_order=False)
            print(format_report(res))
            print("genotype:", json.dumps(res.genotype_json, ensure_ascii=False))
            return 0

        if args.cmd == "search":
            budget = Budget.research() if args.profile == "research" else Budget.demo()
            if args.steps is not None:
                from dataclasses import replace

                budget = replace(budget, max_steps=args.steps)
            if args.method == "darfuse":
                res = run_search(args.method, _dataset(), budget, seed, second_order=args.second_order)
            else:
                res = run_search(args.method, _dataset(), budget, seed)
            print(format_report(res))
            print("genotype:", json.dumps(res.genotype_json, ensure_ascii=False))
            return 0

        if args.cmd == "evaluate":
            budget = Budget.research() if args.profile == "research" else Budget.demo()
            if args.genotype:
                with open(args.genotype, encoding="utf-8") as f:
                    g = Genotype.from_json(json.load(f))
            else:
                from nasforge.spaces.search_space import DEFAULT_SPACE

                g = DEFAULT_SPACE.random_genotype(np.random.default_rng(seed))
            from nasforge.evaluate import independent_evaluate

            r = independent_evaluate(g, _dataset(), budget, seed=seed)
            print(format_report(r, name="independent-eval"))
            return 0

        if args.cmd == "bench":
            from nasforge.pipeline.benchmark import benchmark_zerocost_vs_independent

            out = benchmark_zerocost_vs_independent(_dataset(), Budget.demo(), seed=seed, n_arch=12)
            print("G12 zero-cost proxy vs independent evaluation:")
            print("  spearman(correlation) =", round(out["spearman"], 4))
            print("  std(independent acc)  =", round(out["std_independent"], 4))
            return 0

        if args.cmd == "version":
            from nasforge import __author__, __version__

            print(f"nasforge {__version__} ({__author__})")
            return 0

    except Exception as exc:  # pragma: no cover - top-level guard
        from nasforge.core.errors import NasForgeError

        if isinstance(exc, NasForgeError):
            print(f"ERROR [E{exc.code}] {exc.message}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
            return 1
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
