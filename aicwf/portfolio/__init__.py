"""Constrained portfolio construction utilities (projection, optimiser, orders)."""

from __future__ import annotations

__all__ = ["run_portfolio"]


def run_portfolio(ctx):  # pragma: no cover - thin re-export
    from aicwf.portfolio.stage import run_portfolio as _run

    return _run(ctx)
