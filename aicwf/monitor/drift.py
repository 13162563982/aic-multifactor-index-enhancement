"""Monitoring analytics: performance attribution, factor IC drift, exposures, freshness.

All functions are pure (no wall clock, no randomness) so the monitor outputs are
deterministic JSON.  They consume the previous run's artifacts plus the latest
frozen snapshot and return plain dictionaries that :mod:`aicwf.monitor.run`
serialises.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import date_str, parse_date

PERIODS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _iso_series(series: pd.Series) -> pd.Series:
    return pd.Series([date_str(value) for value in series.tolist()], index=series.index, dtype="object")


def _price_pivot(prices: pd.DataFrame, value_column: str = "close") -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = _iso_series(frame["date"])
    frame["code"] = frame["code"].astype(str)
    pivot = frame.pivot_table(index="date", columns="code", values=value_column, aggfunc="last")
    return pivot.sort_index().ffill()


def _benchmark_close(benchmark: pd.DataFrame) -> pd.Series | None:
    if benchmark is None or len(benchmark) == 0:
        return None
    frame = benchmark.copy()
    frame["date"] = _iso_series(frame["date"])
    frame = frame.sort_values("date", kind="mergesort")
    if "close" in frame.columns:
        values = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    elif "ret" in frame.columns:
        values = (1.0 + pd.to_numeric(frame["ret"], errors="coerce").fillna(0.0)).cumprod()
    else:  # pragma: no cover - schema check catches this
        return None
    return pd.Series(values.to_numpy(), index=pd.Index(frame["date"].tolist(), name="date"), name="close")


def _safe(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Factor RankIC drift
# ---------------------------------------------------------------------------


def factor_rank_ic(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    as_of: Any,
    horizon_days: int = 20,
    recent_periods: int = 3,
    baseline_periods: int = 12,
) -> dict[str, Any]:
    """Rank IC of the composite score vs forward returns, plus recent-vs-baseline drift."""

    empty = {
        "series": [],
        "recent_mean": None,
        "baseline_mean": None,
        "decay": None,
        "n_periods": 0,
        "horizon_days": int(horizon_days),
        "method": "spearman(composite_score_t, forward_return_t+%dd)" % horizon_days,
        "warning": None,
    }
    if factor_panel is None or len(factor_panel) == 0:
        empty["warning"] = "factor_panel missing or empty"
        return empty
    if "composite_score" not in factor_panel.columns:
        empty["warning"] = "factor_panel has no composite_score column"
        return empty
    close = _price_pivot(prices)
    if close.empty:
        empty["warning"] = "price history missing or empty"
        return empty
    cutoff = date_str(as_of)
    dates = close.index.tolist()
    factor_frame = factor_panel.copy()
    factor_frame["date"] = _iso_series(factor_frame["date"])
    factor_frame["code"] = factor_frame["code"].astype(str)
    if "eligible" in factor_frame.columns:
        factor_frame = factor_frame[factor_frame["eligible"].astype(bool)]
    position = {day: index for index, day in enumerate(dates)}
    series: list[dict[str, Any]] = []
    for day, group in factor_frame.groupby("date", sort=True):
        if day > cutoff or day not in position:
            continue
        index = position[day]
        forward_index = index + int(horizon_days)
        if forward_index >= len(dates):
            continue
        forward_date = dates[forward_index]
        if forward_date > cutoff:
            continue
        scores = group.drop_duplicates("code").set_index("code")["composite_score"].astype(float)
        start = close.iloc[index]
        end = close.iloc[forward_index]
        forward = (end / start - 1.0).dropna()
        joined = pd.DataFrame({"score": scores, "forward": forward}).dropna()
        if len(joined) < 3:
            continue
        correlation = joined["score"].rank().corr(joined["forward"].rank())
        if correlation is None or not math.isfinite(float(correlation)):
            continue
        series.append(
            {
                "date": str(day),
                "forward_date": forward_date,
                "rank_ic": float(correlation),
                "n_names": int(len(joined)),
            }
        )
    if not series:
        empty["warning"] = (
            f"no factor date has a full {horizon_days}-trading-day forward return on or before {cutoff}"
        )
        return empty
    values = [item["rank_ic"] for item in series]
    recent = values[-int(recent_periods):]
    baseline = values[-(int(recent_periods) + int(baseline_periods)): -int(recent_periods)]
    if not baseline:
        baseline = values[:-int(recent_periods)] if len(values) > int(recent_periods) else values
    recent_mean = float(np.mean(recent)) if recent else None
    baseline_mean = float(np.mean(baseline)) if baseline else None
    decay = None
    if recent_mean is not None and baseline_mean is not None:
        decay = float(baseline_mean - recent_mean)
    return {
        "series": series[-24:],
        "recent_mean": _safe(recent_mean),
        "baseline_mean": _safe(baseline_mean),
        "decay": _safe(decay),
        "n_periods": len(series),
        "n_recent": len(recent),
        "n_baseline": len(baseline),
        "horizon_days": int(horizon_days),
        "method": f"spearman(composite_score_t, forward_return_t+{horizon_days}d)",
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Realized performance since the last rebalance
# ---------------------------------------------------------------------------


def mark_to_market_performance(
    *,
    holdings: pd.DataFrame | None,
    equity_curve: pd.DataFrame | None,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    as_of: Any,
    initial_capital: float = 1.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Mark the last published holdings to the latest frozen closes.

    Returns realized return, benchmark return, realized tracking error, tracking
    difference and the active drawdown since the last rebalance / run date.
    Falls back to the equity curve when holdings are unavailable.
    """

    cutoff = date_str(as_of)
    close = _price_pivot(prices)
    if close.empty:
        return {"method": "unavailable", "reason": "price history empty"}
    close = close.loc[[day for day in close.index if day <= cutoff]]
    if close.empty:
        return {"method": "unavailable", "reason": f"no price history on or before {cutoff}"}
    benchmark_close = _benchmark_close(benchmark)
    if benchmark_close is not None:
        benchmark_close = benchmark_close.loc[[day for day in benchmark_close.index if day <= cutoff]]

    base_date: str | None = None
    nav_path: pd.Series | None = None
    cash = 0.0
    positions: dict[str, float] = {}
    if holdings is not None and len(holdings):
        frame = holdings.copy()
        frame["date"] = _iso_series(frame["date"])
        frame["code"] = frame["code"].astype(str)
        dates = [day for day in sorted(frame["date"].unique()) if day <= cutoff]
        if dates:
            base_date = dates[-1]
            latest = frame[frame["date"] == base_date]
            positions = {
                str(row.code): float(row.shares) for row in latest.itertuples(index=False) if float(row.shares) > 0
            }
            holdings_value = sum(
                float(row.shares) * float(row.price) for row in latest.itertuples(index=False)
            )
            portfolio_value = None
            if equity_curve is not None and len(equity_curve):
                curve = equity_curve.copy()
                curve["date"] = _iso_series(curve["date"])
                rows = curve[curve["date"] == base_date]
                if len(rows):
                    portfolio_value = float(rows.iloc[0]["nav"]) * float(initial_capital)
                    cash = portfolio_value - holdings_value
            if portfolio_value is None:
                portfolio_value = holdings_value
                cash = 0.0
            window_dates = [day for day in close.index if day >= base_date]
            values: list[float] = []
            nav_series: list[float] = []
            for day in window_dates:
                value = cash
                for code, quantity in positions.items():
                    price = close.at[day, code] if code in close.columns else float("nan")
                    if math.isfinite(price):
                        value += quantity * float(price)
                values.append(value)
                nav_series.append(value / float(initial_capital) if initial_capital else value)
            if values and values[0] > 0:
                nav_path = pd.Series(nav_series, index=window_dates, dtype=float)
    if nav_path is None:
        # Fallback: use the published equity curve, last 21 trading days.
        if equity_curve is None or len(equity_curve) == 0:
            return {"method": "unavailable", "reason": "holdings and equity curve are unavailable"}
        curve = equity_curve.copy()
        curve["date"] = _iso_series(curve["date"])
        curve = curve[curve["date"] <= cutoff].sort_values("date", kind="mergesort")
        if len(curve) < 2:
            return {"method": "unavailable", "reason": "equity curve has fewer than two observations"}
        curve = curve.tail(22)
        base_date = str(curve.iloc[0]["date"])
        nav_path = pd.Series(curve["nav"].astype(float).to_numpy(), index=curve["date"].tolist(), dtype=float)
    if nav_path is None or len(nav_path) < 1 or float(nav_path.iloc[0]) <= 0:
        return {"method": "unavailable", "reason": "could not build a NAV path"}

    window_returns = nav_path.pct_change().dropna()
    portfolio_return = float(nav_path.iloc[-1] / nav_path.iloc[0] - 1.0)
    result: dict[str, Any] = {
        "method": "mark_to_market" if positions else "equity_curve",
        "as_of": cutoff,
        "last_rebalance_date": base_date,
        "window_start": str(nav_path.index[0]),
        "window_end": str(nav_path.index[-1]),
        "n_days": int(len(nav_path) - 1),
        "n_positions": len(positions),
        "portfolio_return": portfolio_return,
        "realized_te": None,
        "tracking_difference": None,
        "annualized_excess_return": None,
        "benchmark_return": None,
        "excess_return": None,
        "active_drawdown": None,
    }
    realized_te: float | None = None
    active_drawdown: float | None = None
    ann_excess: float | None = None
    if benchmark_close is not None and len(benchmark_close):
        bench = benchmark_close.astype(float).sort_index()
        bench = bench.reindex([day for day in bench.index if day >= str(nav_path.index[0])])
        bench = bench.dropna()
        if len(bench) >= 1:
            bench = bench.reindex(nav_path.index).ffill().dropna()
            nav_aligned = nav_path.reindex(bench.index)
            benchmark_return = float(bench.iloc[-1] / bench.iloc[0] - 1.0) if len(bench) else None
            cumulative_excess = (
                float((nav_aligned.iloc[-1] / nav_aligned.iloc[0]) / (bench.iloc[-1] / bench.iloc[0]) - 1.0)
                if len(nav_aligned) >= 1
                else None
            )
            if len(bench) > 2 and len(nav_aligned) > 2:
                nav_returns = nav_aligned.pct_change().dropna()
                bench_returns = bench.pct_change().dropna()
                common = nav_returns.align(bench_returns, join="inner")
                excess_daily = common[0] - common[1]
                realized_te = (
                    float(excess_daily.std(ddof=1) * math.sqrt(periods_per_year))
                    if len(excess_daily) > 1
                    else None
                )
                active = nav_aligned / bench
                active_drawdown = float((active / active.cummax() - 1.0).min())
                n_returns = int(len(nav_returns))
                if n_returns > 0 and benchmark_return is not None:
                    ann_portfolio = _annualise(float(nav_aligned.iloc[-1] / nav_aligned.iloc[0]), n_returns)
                    ann_bench = _annualise(float(bench.iloc[-1] / bench.iloc[0]), len(bench_returns))
                    ann_excess = ann_portfolio - ann_bench
                else:
                    ann_excess = None
            else:
                realized_te = None
                active_drawdown = None
                ann_excess = None
            result.update(
                {
                    "benchmark_return": _safe(benchmark_return),
                    "excess_return": _safe(cumulative_excess),
                    "tracking_difference": _safe(cumulative_excess),
                    "realized_te": _safe(realized_te),
                    "annualized_excess_return": _safe(ann_excess),
                    "active_drawdown": _safe(active_drawdown),
                    "n_days": int(len(bench) - 1),
                }
            )
        else:  # pragma: no cover - benchmark reindex edge case
            pass
    else:
        realized_te = (
            float(window_returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(window_returns) > 1 else None
        )
        result["realized_te"] = _safe(realized_te)
    return result


def _annualise(growth: float, n_returns: int, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if n_returns <= 0 or not math.isfinite(growth):
        return float("nan")
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n_returns) - 1.0)


# ---------------------------------------------------------------------------
# Exposure drift
# ---------------------------------------------------------------------------


def exposure_drift(
    holdings: pd.DataFrame | None,
    *,
    as_of: Any,
    industry_limit: float,
) -> dict[str, Any]:
    """Latest industry/name active exposures and their drift from the limit."""

    empty: dict[str, Any] = {
        "date": None,
        "industry_active": {},
        "max_abs_industry_active": None,
        "industry_limit": float(industry_limit),
        "industry_breaches": [],
        "max_abs_name_active": None,
        "n_names": 0,
        "breach": False,
    }
    if holdings is None or len(holdings) == 0:
        empty["warning"] = "holdings unavailable"
        return empty
    frame = holdings.copy()
    frame["date"] = _iso_series(frame["date"])
    frame["code"] = frame["code"].astype(str)
    cutoff = date_str(as_of)
    dates = [day for day in sorted(frame["date"].unique()) if day <= cutoff]
    if not dates:
        empty["warning"] = "no holdings on or before as_of"
        return empty
    latest = frame[frame["date"] == dates[-1]].copy()
    industry_active = (
        latest.groupby("industry")["active_weight"].sum().sort_index().to_dict()
        if "industry" in latest.columns and "active_weight" in latest.columns
        else {}
    )
    industries = {str(key): float(value) for key, value in industry_active.items()}
    breaches = [
        {"industry": industry, "active_weight": value}
        for industry, value in industries.items()
        if abs(value) > float(industry_limit) + 1e-9
    ]
    max_industry = max((abs(value) for value in industries.values()), default=None)
    max_name = (
        float(latest["active_weight"].astype(float).abs().max())
        if "active_weight" in latest.columns and len(latest)
        else None
    )
    return {
        "date": dates[-1],
        "industry_active": dict(sorted(industries.items())),
        "max_abs_industry_active": _safe(max_industry),
        "industry_limit": float(industry_limit),
        "industry_breaches": breaches,
        "max_abs_name_active": _safe(max_name),
        "n_names": int(len(latest)),
        "breach": bool(breaches),
    }


# ---------------------------------------------------------------------------
# Data freshness and coverage
# ---------------------------------------------------------------------------


def data_freshness(
    *,
    as_of: Any,
    snapshot_prices: pd.DataFrame | None,
    snapshot_manifest: Mapping[str, Any] | None = None,
    staleness_limit_days: int = 3,
    calendar: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """How stale the frozen data is relative to the monitoring date."""

    cutoff = date_str(as_of)
    latest_price_date: str | None = None
    if snapshot_prices is not None and len(snapshot_prices):
        latest_price_date = max(_iso_series(snapshot_prices["date"]).tolist())
    manifest_as_of = None
    if snapshot_manifest:
        as_of_value = snapshot_manifest.get("as_of")
        if as_of_value:
            manifest_as_of = date_str(as_of_value)
    reference = latest_price_date or manifest_as_of
    calendar_days = None
    trading_days = None
    if reference:
        calendar_days = (parse_date(cutoff) - parse_date(reference)).days
        approximated = max(0, int(round(calendar_days * 5 / 7)))
        if calendar is not None and len(calendar):
            dates = _iso_series(calendar["date"]).tolist()
            if dates and max(dates) >= cutoff:
                trading_days = sum(1 for day in dates if reference < day <= cutoff)
            else:
                # The snapshot calendar itself stops before the monitoring date, so a
                # pure count would understate the staleness; approximate with weekdays.
                trading_days = approximated
        else:
            trading_days = approximated
    stale = False
    severity = "ok"
    if reference is None:
        stale = True
        severity = "critical"
    elif (trading_days or 0) > int(staleness_limit_days):
        stale = True
        severity = "warn"
    return {
        "as_of": cutoff,
        "latest_price_date": latest_price_date,
        "snapshot_manifest_as_of": manifest_as_of,
        "calendar_days_stale": calendar_days,
        "trading_days_stale": trading_days,
        "staleness_limit_days": int(staleness_limit_days),
        "stale": stale,
        "severity": severity,
    }


def coverage(
    *,
    as_of: Any,
    panel: pd.DataFrame | None,
    snapshot_universe: pd.DataFrame | None,
    factor_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Share of universe names with a usable latest price and factor score."""

    cutoff = date_str(as_of)
    result: dict[str, Any] = {
        "as_of": cutoff,
        "universe_size": None,
        "latest_panel_date": None,
        "price_coverage": None,
        "factor_coverage": None,
        "min_coverage": None,
    }
    universe_codes: set[str] = set()
    if snapshot_universe is not None and len(snapshot_universe):
        result["universe_size"] = int(len(snapshot_universe))
        universe_codes = {str(code) for code in snapshot_universe["code"].astype(str)}
    if panel is not None and len(panel):
        frame = panel.copy()
        frame["date"] = _iso_series(frame["date"])
        frame = frame[frame["date"] <= cutoff]
        if len(frame):
            latest_date = max(frame["date"].tolist())
            result["latest_panel_date"] = latest_date
            latest = frame[frame["date"] == latest_date]
            codes = {str(code) for code in latest["code"].astype(str)}
            denominator = len(universe_codes) if universe_codes else max(1, len(codes))
            result["price_coverage"] = float(len(codes & universe_codes) / denominator) if universe_codes else 1.0
    if factor_panel is not None and len(factor_panel):
        frame = factor_panel.copy()
        frame["date"] = _iso_series(frame["date"])
        frame = frame[frame["date"] <= cutoff]
        if len(frame):
            latest_date = max(frame["date"].tolist())
            latest = frame[frame["date"] == latest_date]
            if "eligible" in latest.columns:
                latest = latest[latest["eligible"].astype(bool)]
            denominator = len(universe_codes) if universe_codes else max(1, len(latest))
            result["factor_coverage"] = float(len(set(latest["code"].astype(str))) / denominator)
    return result


def detect_new_data(
    snapshot_hash: str | None,
    previous_snapshot_hash: str | None,
) -> dict[str, Any]:
    """New-data trigger: compare the latest snapshot hash with the previous monitor run.

    The very first monitoring run has no previous snapshot, so it is a scheduled
    run rather than a new-data trigger.
    """

    changed = bool(snapshot_hash) and bool(previous_snapshot_hash) and str(snapshot_hash) != str(previous_snapshot_hash)
    return {
        "type": "new_data" if changed else "scheduled",
        "new_data": changed,
        "snapshot_hash": snapshot_hash,
        "previous_snapshot_hash": previous_snapshot_hash,
    }
