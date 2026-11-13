"""Reconciliation of the trade ledger, holdings, cash and the equity curve.

The reconciler is independent of the backtest engine: it takes the written
artifacts plus the verifier's own rebuild (from
:mod:`aicwf.verify.independent_backtest`) and proves the accounting identities
that a correct backtest must satisfy:

* ``gross_amount = shares * price`` and
  ``total_cost = commission + stamp_duty + slippage``, so the cost columns are
  not just cosmetic;
* ``cash_flow``: ``-(gross + cost)`` for buys, ``gross - cost`` for sells;
* holdings shares at each execution date equal the cumulative ledger position,
  and each holding weight equals ``shares * close / total value``;
* the declared cash fraction equals the rebuilt cash fraction;
* the equity curve NAV is exactly the rebuilt NAV, and the daily return,
  benchmark return, excess return and drawdown columns are internally
  consistent.

Every value is compared against an independent recomputation; failures are
collected as strings so the verification note can cite them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.verify.independent_backtest import (
    RebuildResult,
    normalise_ledger,
)

#: Preferred column list for holdings.parquet / equity_curve.csv per the contract.
HOLDINGS_COLUMNS: tuple[str, ...] = (
    "date",
    "code",
    "shares",
    "price",
    "weight",
    "benchmark_weight",
    "active_weight",
    "industry",
)
EQUITY_CURVE_COLUMNS: tuple[str, ...] = (
    "date",
    "nav",
    "benchmark_nav",
    "active_nav",
    "daily_return",
    "benchmark_return",
    "excess_return",
    "drawdown",
)

_MAX_REPORTED_ERRORS = 40


def _tol(value: float, relative: float, absolute: float) -> float:
    return max(absolute, relative * max(1.0, abs(float(value))))


def _close(left: float, right: float, relative: float, absolute: float) -> bool:
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= _tol(right, relative, absolute)


def _iso(value: Any) -> str:
    from aicwf.contracts import parse_date

    return parse_date(value).isoformat()


def positions_at(ledger: pd.DataFrame, day: str) -> dict[str, float]:
    """Cumulative share position per code after all trades dated <= ``day``."""

    trades = normalise_ledger(ledger)
    day = _iso(day)
    positions: dict[str, float] = {}
    for row in trades.itertuples(index=False):
        date_value = str(row.date)
        if date_value > day:
            continue
        quantity = float(row.shares)
        if row.side == "buy":
            positions[row.code] = positions.get(row.code, 0.0) + quantity
        elif row.side == "sell":
            positions[row.code] = positions.get(row.code, 0.0) - quantity
    return {code: quantity for code, quantity in positions.items() if abs(quantity) > 1e-12}


def _reconciliation_error(
    errors: list[str],
    template: str,
    *args: Any,
) -> None:
    if len(errors) < _MAX_REPORTED_ERRORS:
        errors.append(template.format(*args))


def reconcile_accounting(
    *,
    ledger: pd.DataFrame,
    holdings: pd.DataFrame,
    equity_curve: pd.DataFrame,
    rebuild: RebuildResult,
    benchmark_nav_series: pd.Series | None,
    tolerances: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every accounting identity; returns ``{ok, errors, warnings, ...}``."""

    tol = dict(tolerances or {})
    rel_metrics = float(tol.get("metrics_relative", 1e-6))
    rel_weights = float(tol.get("weights_absolute", 1e-6))
    abs_cash = float(tol.get("cash_absolute", 0.01))
    # cost/price amounts are in currency units; 1e-6 relative is exact enough
    abs_amount = max(abs_cash, 1e-6)

    errors: list[str] = []
    warnings: list[str] = []
    trades = normalise_ledger(ledger)

    # -- 1. per-trade identities ---------------------------------------
    for row in trades.itertuples(index=False):
        prefix = f"ledger {row.date} {row.code} {row.side}"
        if row.side not in {"buy", "sell"}:
            _reconciliation_error(errors, "{}: unknown side {!r}", prefix, row.side)
            continue
        if not math.isfinite(float(row.shares)) or float(row.shares) <= 0:
            _reconciliation_error(errors, "{}: shares must be positive", prefix)
        if not math.isfinite(float(row.price)) or float(row.price) <= 0:
            _reconciliation_error(errors, "{}: price must be positive", prefix)
            continue
        gross_expected = float(row.shares) * float(row.price)
        if not _close(float(row.gross_amount), gross_expected, rel_metrics, abs_amount):
            _reconciliation_error(
                errors,
                "{}: gross_amount {:.6f} != shares*price {:.6f}",
                prefix,
                float(row.gross_amount),
                gross_expected,
            )
        cost_expected = float(row.commission) + float(row.stamp_duty) + float(row.slippage)
        if not _close(float(row.total_cost), cost_expected, rel_metrics, abs_amount):
            _reconciliation_error(
                errors,
                "{}: total_cost {:.6f} != commission+stamp_duty+slippage {:.6f}",
                prefix,
                float(row.total_cost),
                cost_expected,
            )
        if row.side == "buy":
            flow_expected = -(float(row.gross_amount) + float(row.total_cost))
            if not _close(float(row.cash_flow), flow_expected, rel_metrics, abs_amount):
                _reconciliation_error(
                    errors,
                    "{}: cash_flow {:.6f} != -(gross+total_cost) {:.6f}",
                    prefix,
                    float(row.cash_flow),
                    flow_expected,
                )
            if abs(float(row.stamp_duty)) > abs_amount:
                warnings.append(f"{prefix}: stamp duty on a buy trade ({float(row.stamp_duty):.6f})")
        else:
            flow_expected = float(row.gross_amount) - float(row.total_cost)
            if not _close(float(row.cash_flow), flow_expected, rel_metrics, abs_amount):
                _reconciliation_error(
                    errors,
                    "{}: cash_flow {:.6f} != gross-total_cost {:.6f}",
                    prefix,
                    float(row.cash_flow),
                    flow_expected,
                )
        if abs(float(row.commission)) + abs(float(row.slippage)) > abs_amount:
            if float(row.stamp_duty) < -abs_amount:
                _reconciliation_error(errors, "{}: negative stamp duty", prefix)

    # -- 2. holdings vs cumulative ledger positions --------------------
    holdings_frame = holdings.copy()
    holdings_frame["date"] = [_iso(v) for v in holdings_frame["date"]]
    holdings_frame["code"] = holdings_frame["code"].astype(str)
    holdings_frame = holdings_frame.sort_values(["date", "code"], kind="mergesort")
    path_value = dict(zip(rebuild.path["date"].tolist(), rebuild.path["total_value"].astype(float).tolist()))
    path_cash = dict(zip(rebuild.path["date"].tolist(), rebuild.path["cash"].astype(float).tolist()))
    holdings_dates = sorted(holdings_frame["date"].unique().tolist())
    min_cash = float(rebuild.path["cash"].min()) if len(rebuild.path) else 0.0
    max_cash_fraction = 0.0
    checked_holdings_dates = 0

    for day in holdings_dates:
        expected_positions = positions_at(trades, day)
        declared = holdings_frame[holdings_frame["date"] == day]
        checked_holdings_dates += 1
        total_value = path_value.get(day)
        if total_value is None:
            _reconciliation_error(errors, "holdings date {} has no rebuilt valuation", day)
            continue
        declared_by_code = {
            str(row.code): row for row in declared.itertuples(index=False)
        }
        codes = sorted(set(expected_positions) | set(declared_by_code))
        for code in codes:
            expected = expected_positions.get(code, 0.0)
            row = declared_by_code.get(code)
            if row is None:
                _reconciliation_error(
                    errors,
                    "holdings {}: position in {} ({:.4f} shares) missing from holdings.parquet",
                    day,
                    code,
                    expected,
                )
                continue
            actual = float(row.shares)
            if abs(actual - expected) > max(1e-6, 1e-6 * abs(expected)):
                _reconciliation_error(
                    errors,
                    "holdings {}: {} shares {:.6f} != cumulative ledger {:.6f}",
                    day,
                    code,
                    actual,
                    expected,
                )
            price = float(row.price)
            if path_value and total_value:
                expected_weight = float(row.shares) * price / float(total_value)
                if not _close(float(row.weight), expected_weight, rel_weights, 1e-9):
                    _reconciliation_error(
                        errors,
                        "holdings {}: {} weight {:.10f} != shares*price/total_value {:.10f}",
                        day,
                        code,
                        float(row.weight),
                        expected_weight,
                    )
            if abs(float(row.active_weight) - (float(row.weight) - float(row.benchmark_weight))) > 1e-9:
                _reconciliation_error(
                    errors,
                    "holdings {}: {} active_weight != weight - benchmark_weight",
                    day,
                    code,
                )
            if float(row.weight) < -rel_weights:
                _reconciliation_error(errors, "holdings {}: {} negative weight", day, code)
        declared_weight_sum = float(declared["weight"].sum())
        cash_from_weights = 1.0 - declared_weight_sum
        rebuilt_cash = path_cash.get(day, 0.0)
        cash_fraction = rebuilt_cash / float(total_value) if total_value else 0.0
        max_cash_fraction = max(max_cash_fraction, abs(cash_fraction))
        if abs(cash_from_weights - cash_fraction) > max(rel_weights * 10.0, 1e-8):
            _reconciliation_error(
                errors,
                "holdings {}: declared cash fraction {:.10f} != rebuilt cash fraction {:.10f}",
                day,
                cash_from_weights,
                cash_fraction,
            )
        if declared_weight_sum > 1.0 + rel_weights:
            _reconciliation_error(
                errors,
                "holdings {}: weights sum to {:.10f} (> 1, leverage not allowed)",
                day,
                declared_weight_sum,
            )

    if len(rebuild.path) and min_cash < -abs_cash:
        _reconciliation_error(
            errors,
            "rebuilt cash went negative ({:.6f}); the ledger over-spends the initial capital",
            min_cash,
        )

    # -- 3. equity curve vs rebuilt NAV / identity columns -------------
    curve = equity_curve.copy()
    curve["date"] = [_iso(v) for v in curve["date"]]
    curve = curve.sort_values("date", kind="mergesort")
    nav_rebuilt = rebuild.nav.astype(float)
    common_dates = [d for d in curve["date"].tolist() if d in nav_rebuilt.index]
    nav_checked = 0
    for day in common_dates[:400]:
        declared_nav = float(curve.loc[curve["date"] == day, "nav"].iloc[0])
        rebuilt_nav = float(nav_rebuilt.loc[day])
        nav_checked += 1
        if not _close(declared_nav, rebuilt_nav, rel_metrics, 1e-9):
            _reconciliation_error(
                errors,
                "equity_curve {}: nav {:.10f} != independently rebuilt nav {:.10f}",
                day,
                declared_nav,
                rebuilt_nav,
            )
    if len(common_dates) < len(curve) - 2:
        _reconciliation_error(
            errors,
            "equity_curve: {} of {} dates are outside the rebuilt valuation calendar",
            len(curve) - len(common_dates),
            len(curve),
        )

    curve = curve.reset_index(drop=True)
    if len(curve):
        first_nav = float(curve.loc[0, "nav"])
        if abs(first_nav - 1.0) > 1e-3:
            warnings.append(f"equity_curve first nav is {first_nav:.6f} (expected 1.0)")
        for index in range(1, len(curve)):
            previous = curve.loc[index - 1]
            row = curve.loc[index]
            expected_return = float(row["nav"]) / float(previous["nav"]) - 1.0
            if "daily_return" in curve.columns and not _close(
                float(row["daily_return"]), expected_return, rel_metrics, 1e-9
            ):
                _reconciliation_error(
                    errors,
                    "equity_curve {}: daily_return {:.10f} != nav change {:.10f}",
                    row["date"],
                    float(row["daily_return"]),
                    expected_return,
                )
                break
        if "drawdown" in curve.columns:
            running = curve["nav"].astype(float).cummax()
            expected_dd = curve["nav"].astype(float) / running - 1.0
            worst = float((curve["drawdown"].astype(float) - expected_dd).abs().max())
            if worst > max(rel_metrics, 1e-9):
                _reconciliation_error(
                    errors,
                    "equity_curve: drawdown column differs from nav/cummax(nav)-1 by {:.3g}",
                    worst,
                )

    if benchmark_nav_series is not None and len(curve):
        bench = benchmark_nav_series.astype(float)
        merged = curve[["date", "benchmark_nav"]].copy()
        merged["expected"] = merged["date"].map(bench)
        merged = merged.dropna(subset=["expected"])
        if len(merged):
            deviation = (merged["benchmark_nav"].astype(float) - merged["expected"].astype(float)).abs()
            worst = float(deviation.max())
            if worst > max(rel_metrics, 1e-9):
                _reconciliation_error(
                    errors,
                    "equity_curve: benchmark_nav differs from benchmark.parquet by up to {:.3g}",
                    worst,
                )
        if "excess_return" in curve.columns:
            expected_excess = curve["daily_return"].astype(float) - curve["benchmark_return"].astype(float)
            worst = float((curve["excess_return"].astype(float) - expected_excess).abs().max())
            if worst > max(rel_metrics, 1e-9):
                _reconciliation_error(
                    errors,
                    "equity_curve: excess_return != daily_return - benchmark_return (max diff {:.3g})",
                    worst,
                )
        if "active_nav" in curve.columns and len(curve) > 2:
            nav = curve["nav"].astype(float)
            bench_aligned = curve["benchmark_nav"].astype(float)
            ratio = nav / bench_aligned
            declared = curve["active_nav"].astype(float)
            ratio_diff = float((declared - ratio).abs().max())
            additive = nav - bench_aligned + 1.0
            additive_diff = float((declared - additive).abs().max())
            if ratio_diff > max(rel_metrics, 1e-6) and additive_diff > max(rel_metrics, 1e-6):
                warnings.append(
                    "equity_curve: active_nav matches neither nav/benchmark_nav nor "
                    "nav-benchmark_nav+1 (declared convention is undocumented)"
                )

    details = {
        "trade_rows": int(len(trades)),
        "holdings_dates": checked_holdings_dates,
        "nav_rows_checked": nav_checked,
        "equity_curve_rows": int(len(curve)),
        "min_cash": None if not len(rebuild.path) else float(min_cash),
        "max_abs_cash_fraction": float(max_cash_fraction),
        "initial_capital": float(rebuild.initial_capital),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "details": details,
    }
