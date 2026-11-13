"""Independent recomputation of the backtest from raw artifacts.

This module deliberately imports **nothing** from ``aicwf.backtest``,
``aicwf.portfolio`` or ``aicwf.factors``: it must be able to disagree with the
implementation it verifies.  Inputs are only the frozen market data and the
trade ledger the backtest wrote:

* ``ledger.parquet``   -- one row per trade (date, code, side, shares, price, costs)
* ``prices``           -- daily qfq closes (from the snapshot / prepared panel)
* ``benchmark.parquet``-- benchmark daily closes or returns

From those it rebuilds positions, cash, the daily NAV path, benchmark NAV and
all headline performance metrics.  The definitions used here are documented in
``compute_performance_metrics`` so that discrepancies are attributable.

All functions are pure: same inputs -> identical outputs (no wall-clock, no
randomness), which is what makes the determinism evidence meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import date_str, parse_date

PERIODS_PER_YEAR = 252

#: Headline metrics the verifier recomputes and compares against metrics.json.
COMPARABLE_METRICS: tuple[str, ...] = (
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
)

#: Benchmark-only headline metrics.
COMPARABLE_BENCHMARK_METRICS: tuple[str, ...] = (
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
)

LEDGER_COLUMNS: tuple[str, ...] = (
    "date",
    "code",
    "side",
    "shares",
    "price",
    "gross_amount",
    "commission",
    "stamp_duty",
    "slippage",
    "total_cost",
    "cash_flow",
    "reason",
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _iso(value: Any) -> str:
    return parse_date(value).isoformat()


def normalise_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a defensively normalised copy of the ledger (dates ISO, numbers float)."""

    frame = ledger.copy()
    frame["date"] = [_iso(v) for v in frame["date"]]
    frame["code"] = frame["code"].astype(str)
    frame["side"] = frame["side"].astype(str).str.strip().str.lower()
    for column in (
        "shares",
        "price",
        "gross_amount",
        "commission",
        "stamp_duty",
        "slippage",
        "total_cost",
        "cash_flow",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        else:  # tolerate older ledgers by deriving the obvious column
            if column == "gross_amount":
                frame[column] = frame["shares"] * frame["price"]
            elif column == "total_cost":
                frame[column] = (
                    frame.get("commission", 0.0)
                    + frame.get("stamp_duty", 0.0)
                    + frame.get("slippage", 0.0)
                )
            else:
                frame[column] = 0.0
    if "reason" not in frame.columns:
        frame["reason"] = ""
    frame["reason"] = frame["reason"].fillna("").astype(str)
    return frame.sort_values(["date", "code", "side"], kind="mergesort").reset_index(drop=True)


def close_matrix(prices: pd.DataFrame, *, value_column: str = "close") -> pd.DataFrame:
    """Pivot daily prices to a date x code matrix, forward-filled within each code."""

    if prices is None or len(prices) == 0:
        return pd.DataFrame()
    frame = prices.copy()
    frame["date"] = [_iso(v) for v in frame["date"]]
    frame["code"] = frame["code"].astype(str)
    matrix = frame.pivot_table(index="date", columns="code", values=value_column, aggfunc="last")
    matrix = matrix.sort_index()
    return matrix.ffill()


def benchmark_nav(benchmark: pd.DataFrame) -> pd.Series | None:
    """Daily benchmark NAV (starts at 1.0) from either a ``ret`` or a ``close`` column."""

    if benchmark is None or len(benchmark) == 0:
        return None
    frame = benchmark.copy()
    frame["date"] = [_iso(v) for v in frame["date"]]
    frame = frame.sort_values("date", kind="mergesort")
    if "close" in frame.columns and frame["close"].notna().all():
        close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
        base = float(close.iloc[0]) if len(close) else float("nan")
        if not math.isfinite(base) or base == 0.0:
            return None
        series = close / base
    elif "ret" in frame.columns:
        ret = pd.to_numeric(frame["ret"], errors="coerce").fillna(0.0).astype(float)
        series = (1.0 + ret).cumprod()
    else:  # pragma: no cover - schema check catches this first
        return None
    series.index = pd.Index(frame["date"].tolist(), name="date")
    return series.astype(float)


# ---------------------------------------------------------------------------
# Portfolio rebuild
# ---------------------------------------------------------------------------


def _trade_cash_flow(row: Any) -> float:
    """Declared cash flow if present, otherwise derived from components."""

    value = getattr(row, "cash_flow", None)
    if value is not None and math.isfinite(float(value)):
        return float(value)
    gross = float(row.gross_amount)
    cost = float(row.total_cost)
    return -(gross + cost) if row.side == "buy" else gross - cost


@dataclass
class RebuildResult:
    """Result of rebuilding a portfolio from the ledger."""

    path: pd.DataFrame  # date, cash, holdings_value, total_value, nav, n_positions
    nav: pd.Series  # index = ISO date string
    positions: pd.DataFrame  # date, code, shares, price (one row per held name/day)
    trades: pd.DataFrame  # normalised ledger rows actually applied
    initial_capital: float
    issues: list[str] = field(default_factory=list)

    def nav_hash(self) -> str:
        from aicwf.util.hashing import hash_json

        return hash_json(
            [
                [row_date, round(float(value), 10)]
                for row_date, value in zip(self.nav.index.tolist(), self.nav.tolist())
            ]
        )


def rebuild_from_ledger(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: pd.DataFrame | None = None,
    initial_capital: float = 1.0,
    as_of: Any | None = None,
) -> RebuildResult:
    """Rebuild cash, positions and the daily NAV path from the trade ledger alone.

    Convention: ``initial_capital`` in currency value; the reported NAV is
    normalised so that the first valuation equals ``initial_capital``.  Trades
    dated ``d`` are applied at the close of ``d`` using the ledger's own price,
    then the portfolio is marked at the frozen close of ``d``.
    """

    cap = float(initial_capital)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital!r}")
    trades = normalise_ledger(ledger)
    matrix = close_matrix(prices)
    issues: list[str] = []
    if matrix.empty:
        raise ValueError("cannot rebuild a portfolio without prices")

    cutoff = _iso(as_of) if as_of is not None else None
    date_set: set[str] = set(matrix.index.tolist())
    if benchmark is not None and len(benchmark):
        bench_dates = [_iso(v) for v in benchmark["date"]]
        date_set.update(bench_dates)
    if cutoff is not None:
        date_set = {d for d in date_set if d <= cutoff}

    applied_dates = [d for d in trades["date"].tolist() if cutoff is None or d <= cutoff]
    late = sorted({d for d in trades["date"].tolist() if cutoff is not None and d > cutoff})
    if late:
        issues.append(f"{len(late)} trade(s) dated after as_of {cutoff} (e.g. {late[0]})")
    date_set.update(applied_dates)
    orphan_dates = sorted(set(applied_dates) - set(matrix.index.tolist()))
    if orphan_dates:
        issues.append(
            f"{len(orphan_dates)} ledger trade date(s) not present in the price panel "
            f"(e.g. {orphan_dates[0]})"
        )
    dates = sorted(date_set)
    if not dates:
        raise ValueError("no valuation dates available after applying the as_of cutoff")

    trades_by_date: dict[str, list[Any]] = {}
    for row in trades.itertuples(index=False):
        trades_by_date.setdefault(str(row.date), []).append(row)

    shares: dict[str, float] = {}
    last_price: dict[str, float] = {}
    cash = cap
    rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    missing_marks = 0

    for day in dates:
        for trade in trades_by_date.get(day, []):
            quantity = float(trade.shares)
            if quantity <= 0 or not math.isfinite(quantity):
                issues.append(f"non-positive shares on {day} {trade.code} {trade.side}")
                continue
            flow = _trade_cash_flow(trade)
            if math.isnan(flow) or math.isinf(flow):
                issues.append(f"non-finite cash flow on {day} {trade.code} {trade.side}")
                continue
            if trade.side == "buy":
                shares[trade.code] = shares.get(trade.code, 0.0) + quantity
            elif trade.side == "sell":
                shares[trade.code] = shares.get(trade.code, 0.0) - quantity
            else:
                issues.append(f"unknown side {trade.side!r} on {day} {trade.code}")
                continue
            cash += flow
            applied.append(
                {
                    "date": day,
                    "code": trade.code,
                    "side": trade.side,
                    "shares": quantity,
                    "price": float(trade.price),
                    "gross_amount": float(trade.gross_amount),
                    "cash_flow": float(flow),
                }
            )

        holdings_value = 0.0
        n_positions = 0
        for code, quantity in shares.items():
            if abs(quantity) < 1e-12:
                continue
            close = matrix.at[day, code] if code in matrix.columns else float("nan")
            if (not math.isfinite(close)) or close <= 0:
                close = last_price.get(code, float("nan"))
                if not math.isfinite(close):
                    missing_marks += 1
                    continue
                issues.append(f"missing/zero frozen price for {code} on {day}; carried forward")
            else:
                last_price[code] = float(close)
            holdings_value += quantity * float(close)
            n_positions += 1
            position_rows.append(
                {"date": day, "code": code, "shares": quantity, "price": float(close)}
            )
        total_value = cash + holdings_value
        rows.append(
            {
                "date": day,
                "cash": cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
                "nav": total_value / cap,
                "n_positions": n_positions,
            }
        )

    path = pd.DataFrame(rows)
    path = path.sort_values("date", kind="mergesort").reset_index(drop=True)
    nav = pd.Series(
        path["nav"].astype(float).to_numpy(),
        index=pd.Index(path["date"].tolist(), name="date"),
        name="nav",
    )
    if missing_marks:
        issues.append(f"{missing_marks} position-day marks were unavailable")
    return RebuildResult(
        path=path,
        nav=nav,
        positions=pd.DataFrame(position_rows),
        trades=pd.DataFrame(applied),
        initial_capital=cap,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Performance metrics (independent definitions)
# ---------------------------------------------------------------------------


def _finite(value: float) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _annualise(total_growth: float, n_returns: int, periods_per_year: int) -> float:
    if n_returns <= 0 or not math.isfinite(total_growth):
        return float("nan")
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (periods_per_year / n_returns) - 1.0)


def compute_performance_metrics(
    nav: pd.Series,
    benchmark_nav_series: pd.Series | None = None,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float | int | str | None]:
    """Recompute headline performance metrics with documented formulas.

    * ``annualized_return`` = ``(nav_T/nav_0)**(periods_per_year / n_returns) - 1``
    * ``annualized_volatility`` = ``std(daily_returns, ddof=1) * sqrt(periods_per_year)``
    * ``sharpe`` = ``annualized_return / annualized_volatility`` (risk-free rate 0)
    * ``max_drawdown`` = ``min(nav / cummax(nav) - 1)`` (negative number)
    * ``annualized_excess_return`` = portfolio annualized return - benchmark annualized return
    * ``tracking_error`` = ``std(r - r_benchmark, ddof=1) * sqrt(periods_per_year)``
    * ``information_ratio`` = ``annualized_excess_return / tracking_error``

    ``n_returns`` is the number of daily return observations.  The benchmark is
    reindexed onto the NAV dates and forward-filled, so both return series have
    identical lengths.
    """

    series = nav.dropna().astype(float).sort_index()
    if len(series) == 0:
        return {"n_returns": 0, "n_days": 0}
    bench = benchmark_nav_series
    if bench is not None:
        aligned = bench.astype(float).reindex(series.index)
        aligned = aligned.ffill()
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
    vol = float(returns.std(ddof=1) * math.sqrt(periods_per_year)) if n_returns > 1 else float("nan")
    sharpe = ann_return / vol if vol and math.isfinite(vol) and vol > 0 else float("nan")
    running_max = series.cummax()
    drawdown = float((series / running_max - 1.0).min())

    metrics: dict[str, float | int | str | None] = {
        "start_date": str(series.index[0]),
        "end_date": str(series.index[-1]),
        "n_days": int(len(series)),
        "n_returns": n_returns,
        "cumulative_return": _finite(total_return),
        "annualized_return": _finite(ann_return),
        "annualized_volatility": _finite(vol),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(drawdown),
    }

    if bench is not None:
        benchmark_returns = bench.pct_change().dropna()
        bench_total = float(bench.iloc[-1]) / float(bench.iloc[0]) - 1.0
        bench_ann = _annualise(float(bench.iloc[-1]) / float(bench.iloc[0]), len(benchmark_returns), periods_per_year)
        bench_vol = (
            float(benchmark_returns.std(ddof=1) * math.sqrt(periods_per_year))
            if len(benchmark_returns) > 1
            else float("nan")
        )
        bench_sharpe = bench_ann / bench_vol if bench_vol and math.isfinite(bench_vol) and bench_vol > 0 else float("nan")
        bench_dd = float((bench / bench.cummax() - 1.0).min())
        common = returns.align(benchmark_returns, join="inner")
        excess_daily = common[0] - common[1]
        tracking_error = (
            float(excess_daily.std(ddof=1) * math.sqrt(periods_per_year)) if len(excess_daily) > 1 else float("nan")
        )
        ann_excess = ann_return - bench_ann if math.isfinite(ann_return) and math.isfinite(bench_ann) else float("nan")
        info_ratio = ann_excess / tracking_error if tracking_error and math.isfinite(tracking_error) and tracking_error > 0 else float("nan")
        metrics.update(
            {
                "benchmark_cumulative_return": _finite(bench_total),
                "annualized_excess_return": _finite(ann_excess),
                "tracking_error": _finite(tracking_error),
                "information_ratio": _finite(info_ratio),
                "excess_cumulative_return": _finite(
                    (float(series.iloc[-1]) / float(series.iloc[0]))
                    / (float(bench.iloc[-1]) / float(bench.iloc[0]))
                    - 1.0
                ),
            }
        )
    return metrics


def compute_benchmark_metrics(
    benchmark_nav_series: pd.Series, *, periods_per_year: int = PERIODS_PER_YEAR
) -> dict[str, float | int | str | None]:
    """Benchmark-only headline metrics using the same definitions."""

    base = compute_performance_metrics(benchmark_nav_series, None, periods_per_year=periods_per_year)
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


def equity_curve_frame(
    nav: pd.Series, benchmark_nav_series: pd.Series | None
) -> pd.DataFrame:
    """Derive the daily equity-curve identities (used for reconciliation)."""

    frame = pd.DataFrame({"nav": nav.astype(float)})
    frame["daily_return"] = frame["nav"].pct_change()
    if benchmark_nav_series is not None:
        bench = benchmark_nav_series.astype(float).reindex(frame.index).ffill()
        frame["benchmark_nav"] = bench
        frame["benchmark_return"] = bench.pct_change()
        frame["excess_return"] = frame["daily_return"] - frame["benchmark_return"]
        frame["active_nav"] = frame["nav"] / bench
    frame["drawdown"] = frame["nav"] / frame["nav"].cummax() - 1.0
    return frame


# ---------------------------------------------------------------------------
# Turnover and metric comparison helpers
# ---------------------------------------------------------------------------


def compute_turnover(
    ledger: pd.DataFrame,
    path: pd.DataFrame,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float | int]:
    """Recompute turnover from the ledger and the rebuilt capital path.

    Two definitions are reported so a difference in convention is visible:

    * ``average_one_way_per_rebalance``: mean over trade dates of
      ``sum(|gross_amount|) / 2 / total_value``
    * ``annualized_two_way``: ``12 * mean_over_dates(sum(|gross_amount|) / total_value)``
    """

    trades = normalise_ledger(ledger)
    if trades.empty or path.empty:
        return {
            "trade_dates": 0,
            "average_one_way_per_rebalance": 0.0,
            "annualized_two_way": 0.0,
            "max_two_way_per_rebalance": 0.0,
        }
    value_by_date = dict(zip(path["date"].tolist(), path["total_value"].astype(float).tolist()))
    gross_by_date = trades.groupby("date")["gross_amount"].apply(lambda s: float(s.abs().sum()))
    two_way: list[float] = []
    for day, gross in gross_by_date.items():
        capital = value_by_date.get(str(day))
        if not capital or capital <= 0 or not math.isfinite(gross):
            continue
        two_way.append(gross / capital)
    if not two_way:
        return {
            "trade_dates": 0,
            "average_one_way_per_rebalance": 0.0,
            "annualized_two_way": 0.0,
            "max_two_way_per_rebalance": 0.0,
        }
    events_per_year = 12.0
    array = np.asarray(two_way, dtype=float)
    return {
        "trade_dates": int(len(two_way)),
        "average_one_way_per_rebalance": float(array.mean() / 2.0),
        "annualized_two_way": float(events_per_year * array.mean()),
        "max_two_way_per_rebalance": float(array.max()),
    }


def compare_metrics(
    declared: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    keys: tuple[str, ...] = COMPARABLE_METRICS,
    rel_tol: float = 1e-6,
    artifact: str = "metrics.json",
    section: str = "metrics",
) -> list[dict[str, Any]]:
    """Compare declared vs recomputed metrics; returns discrepancy records."""

    discrepancies: list[dict[str, Any]] = []
    for key in keys:
        if key not in declared or key not in recomputed:
            continue
        left = declared.get(key)
        right = recomputed.get(key)
        if left is None or right is None:
            continue
        try:
            left_f = float(left)
            right_f = float(right)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(left_f) and math.isfinite(right_f)):
            continue
        tolerance = rel_tol * max(1.0, abs(right_f))
        delta = left_f - right_f
        if abs(delta) > tolerance:
            discrepancies.append(
                {
                    "id": f"{section}.{key}",
                    "artifact": artifact,
                    "field": f"{section}.{key}",
                    "declared": left_f,
                    "recomputed": right_f,
                    "delta": delta,
                    "tolerance": tolerance,
                    "severity": "fail",
                    "detail": (
                        f"{section}.{key} declared {left_f:.10g} but independent "
                        f"recomputation gives {right_f:.10g} (delta {delta:+.3g}, "
                        f"tolerance {tolerance:.3g})"
                    ),
                }
            )
    return discrepancies


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/NaN/inf values into strict-JSON safe objects."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return date_str(value)
    return value
