"""Backtest engine, cost model, metrics, robustness suite and stage handlers."""

from __future__ import annotations

__all__ = ["run_backtest", "run_robustness"]


def run_backtest(ctx):  # pragma: no cover - thin re-export
    from aicwf.backtest.stage import run_backtest as _run

    return _run(ctx)


def run_robustness(ctx):  # pragma: no cover - thin re-export
    from aicwf.backtest.stage import run_robustness as _run

    return _run(ctx)
