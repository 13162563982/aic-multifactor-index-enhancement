"""Performance metrics for the backtest, matching the independent verifier.

The independent verification subsystem recomputes the NAV path from the trade
ledger and the frozen prices and then derives the headline metrics; a run only
passes if the declared numbers agree with that recomputation within a relative
tolerance of ``1e-6``.  This module therefore implements exactly the same
definitions (documented in ``aicwf.verify.independent_backtest``):

* the NAV series spans **every frozen trading day up to ``as_of``** (capital sits
  in cash before the first rebalance), so the annualisation count matches the
  verifier's rebuild;
* ``annualized_return = (nav_T / nav_0) ** (252 / n_returns) - 1``;
* ``annualized_volatility = std(daily_returns, ddof=1) * sqrt(252)``;
* ``sharpe = annualized_return / annualized_volatility`` (risk-free rate 0);
* ``max_drawdown = min(nav / cummax(nav) - 1)``;
* ``annualized_excess_return = annualized_return - benchmark annualized return``;
* ``tracking_error = std(r - r_benchmark, ddof=1) * sqrt(252)``;
* ``information_ratio = annualized_excess_return / tracking_error``.

No wall-clock value is produced anywhere.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def benchmark_nav_from_frame(benchmark: pd.DataFrame) -> pd.Series:
    """Daily benchmark NAV (starts at 1.0) from a benchmark artifact."""

    frame = pd.DataFrame(benchmark).copy()
    frame["date"] = [str(value)[:10] for value in frame["date"]]
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if "close" in frame.columns and frame["close"].notna().all():
        close = frame["close"].astype(float)
        base = float(close.iloc[0])
        values = close / base
    else:  # pragma: no cover - data-prep contract guarantees close
        ret = frame["ret"].astype(float).fillna(0.0)
        values = (1.0 + ret).cumprod()
    values.index = pd.Index(frame["date"].astype(str).tolist(), name="date")
    return values.astype(float)


def _annualise(growth: float, n_returns: int, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if n_returns <= 0 or not math.isfinite(growth):
        return float("nan")
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n_returns) - 1.0)


def _finite(value: float) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def compute_performance_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Headline metrics with the verifier's exact definitions."""

    series = pd.Series(nav).dropna().astype(float).sort_index()
    if len(series) == 0:
        return {"n_returns": 0, "n_days": 0}
    bench = None
    if benchmark_nav is not None:
        aligned = pd.Series(benchmark_nav).astype(float).reindex(series.index).ffill()
        mask = series.notna() & aligned.notna()
        series = series[mask]
        aligned = aligned[mask]
        if len(series) < 2:  # pragma: no cover - degenerate input
            return {"n_returns": 0, "n_days": int(len(series))}
        bench = aligned
    returns = series.pct_change().dropna()
    n_returns = int(len(returns))
    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    total_return = end_value / start_value - 1.0 if start_value else float("nan")
    ann_return = _annualise(end_value / start_value, n_returns, periods_per_year)
    volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year)) if n_returns > 1 else float("nan")
    sharpe = ann_return / volatility if volatility and math.isfinite(volatility) and volatility > 0 else float("nan")
    running_max = series.cummax()
    drawdown = float((series / running_max - 1.0).min())
    metrics: dict[str, Any] = {
        "start_date": str(series.index[0]),
        "end_date": str(series.index[-1]),
        "n_days": int(len(series)),
        "n_returns": n_returns,
        "cumulative_return": _finite(total_return),
        "annualized_return": _finite(ann_return),
        "annualized_volatility": _finite(volatility),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(drawdown),
    }
    if bench is not None:
        benchmark_returns = bench.pct_change().dropna()
        bench_total = float(bench.iloc[-1]) / float(bench.iloc[0]) - 1.0
        bench_ann = _annualise(
            float(bench.iloc[-1]) / float(bench.iloc[0]),
            len(benchmark_returns),
            periods_per_year,
        )
        bench_vol = (
            float(benchmark_returns.std(ddof=1) * math.sqrt(periods_per_year))
            if len(benchmark_returns) > 1
            else float("nan")
        )
        bench_sharpe = (
            bench_ann / bench_vol if bench_vol and math.isfinite(bench_vol) and bench_vol > 0 else float("nan")
        )
        bench_drawdown = float((bench / bench.cummax() - 1.0).min())
        common = returns.align(benchmark_returns, join="inner")
        excess_daily = common[0] - common[1]
        tracking_error = (
            float(excess_daily.std(ddof=1) * math.sqrt(periods_per_year)) if len(excess_daily) > 1 else float("nan")
        )
        ann_excess = ann_return - bench_ann if math.isfinite(ann_return) and math.isfinite(bench_ann) else float("nan")
        information_ratio = (
            ann_excess / tracking_error
            if tracking_error and math.isfinite(tracking_error) and tracking_error > 0
            else float("nan")
        )
        metrics.update(
            {
                "benchmark_cumulative_return": _finite(bench_total),
                "annualized_excess_return": _finite(ann_excess),
                "tracking_error": _finite(tracking_error),
                "information_ratio": _finite(information_ratio),
                "excess_cumulative_return": _finite(
                    (float(series.iloc[-1]) / float(series.iloc[0]))
                    / (float(bench.iloc[-1]) / float(bench.iloc[0]))
                    - 1.0
                ),
                "benchmark_annualized_return": _finite(bench_ann),
                "benchmark_annualized_volatility": _finite(bench_vol),
                "benchmark_sharpe": _finite(bench_sharpe),
                "benchmark_max_drawdown": _finite(bench_drawdown),
            }
        )
    return metrics


def compute_benchmark_metrics(
    benchmark_nav: pd.Series, *, periods_per_year: int = PERIODS_PER_YEAR
) -> dict[str, Any]:
    """Benchmark-only metrics with the same definitions as the verifier."""

    base = compute_performance_metrics(benchmark_nav, None, periods_per_year=periods_per_year)
    keys = (
        "start_date",
        "end_date",
        "n_days",
        "n_returns",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    )
    return {key: base.get(key) for key in keys if key in base}


def by_year_table(nav: pd.Series, benchmark_nav: pd.Series) -> dict[str, dict[str, Any]]:
    """Calendar-year return/excess/vol/drawdown table (reporting aid)."""

    series = pd.Series(nav).dropna().astype(float).sort_index()
    bench = pd.Series(benchmark_nav).astype(float).reindex(series.index).ffill()
    frame = pd.DataFrame({"nav": series, "benchmark_nav": bench})
    frame["year"] = [str(index)[:4] for index in frame.index]
    table: dict[str, dict[str, Any]] = {}
    for year, group in frame.groupby("year", sort=True):
        returns = group["nav"].pct_change().dropna()
        bench_returns = group["benchmark_nav"].pct_change().dropna()
        if returns.empty:
            continue
        strategy_return = float((1.0 + returns).prod() - 1.0)
        benchmark_return = float((1.0 + bench_returns).prod() - 1.0) if not bench_returns.empty else float("nan")
        drawdown = float((group["nav"] / group["nav"].cummax() - 1.0).min())
        common = returns.align(bench_returns, join="inner")
        excess_daily = common[0] - common[1]
        table[str(year)] = {
            "return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess_return": strategy_return - benchmark_return if math.isfinite(benchmark_return) else None,
            "volatility": float(returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if len(returns) > 1 else None,
            "tracking_error": float(excess_daily.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
            if len(excess_daily) > 1
            else None,
            "max_drawdown": drawdown,
            "n_days": int(len(group)),
        }
    return table


def nav_series_from_path(nav: Mapping[str, float] | pd.Series) -> pd.Series:
    series = pd.Series(nav).astype(float)
    series.index = pd.Index([str(index)[:10] for index in series.index])
    return series.sort_index()


__all__ = [
    "PERIODS_PER_YEAR",
    "benchmark_nav_from_frame",
    "by_year_table",
    "compute_benchmark_metrics",
    "compute_performance_metrics",
    "nav_series_from_path",
]
