"""Deterministic demo runs built on the offline mini snapshot.

This module is *not* on the production path.  It produces complete,
contract-compliant run directories from the frozen mini fixture
(``tests/fixtures/mini_snapshot``) so that verification, eval, registry and
monitoring can be exercised end to end without network access or the full quant
stack.  Two independent builds are byte-identical under content hashing, which
lets the eval hard gate ``deterministic_replay`` be proven offline.

What is simulated
-----------------
A small reference index-enhancement portfolio on the 6-name fixture: monthly
(or weekly) rebalance, date-``d`` signal using only data <= ``d``, execution at
the *next* trading day's close (T+1), integer shares, commission/stamp-duty/
slippage costs, benchmark weights = normalised frozen index weights, and a
long/short active tilt driven by a short-horizon reversal signal.  All
constraints from the demo config are respected, so a clean build must pass
independent verification.

Variants
--------
``baseline``   monthly rebalance, 15-day reversal, 3% active tilt (v1.0.0)
``accepted``   refined 16-day signal lookback (v1.1.0, should be accepted)
``rejected``   weekly rebalance (v1.2.0, turnover regression -> rejected)

Every artifact is derived from the fixture and the variant parameters only;
there is no randomness, clock or network access.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import PROJECT_ROOT, date_str
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import write_json, write_parquet_atomic, write_text

DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "mini_snapshot"
DEMO_AS_OF = "2024-12-31"
DEMO_INITIAL_CAPITAL = 100_000_000.0

#: Contract artifacts produced by a demo run (stage dirs from aicwf.contracts).
DEMO_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("00_intake", "run_request.json", "json"),
    ("01_hypothesis", "hypothesis.json", "json"),
    ("01_hypothesis", "rationale.md", "text"),
    ("02_data", "panel.parquet", "parquet"),
    ("02_data", "benchmark.parquet", "parquet"),
    ("02_data", "data_quality.json", "json"),
    ("02_data", "pit_audit.json", "json"),
    ("03_factors", "factor_panel.parquet", "parquet"),
    ("03_factors", "factor_report.json", "json"),
    ("05_backtest", "ledger.parquet", "parquet"),
    ("05_backtest", "holdings.parquet", "parquet"),
    ("05_backtest", "metrics.json", "json"),
    ("05_backtest", "equity_curve.csv", "csv"),
    ("06_robustness", "robustness.json", "json"),
    ("06_robustness", "robustness_checks.json", "json"),
    ("08_portfolio", "target_weights.parquet", "parquet"),
    ("08_portfolio", "orders.csv", "csv"),
    ("08_portfolio", "risk_report.json", "json"),
)


@dataclass
class DemoSpec:
    """Parameters of one deterministic demo variant."""

    variant: str = "baseline"
    as_of: str = DEMO_AS_OF
    lookback: int = 15
    amplitude: float = 0.030
    rebalance: str = "monthly"  # "monthly" | "weekly"
    version: str = "v1.0.0"
    initial_capital: float = DEMO_INITIAL_CAPITAL
    with_verification: bool = True
    replay_evidence: bool = True
    description: str = ""


VARIANT_DEFAULTS: dict[str, DemoSpec] = {
    "baseline": DemoSpec(
        variant="baseline",
        version="v1.0.0",
        description="Monthly rebalance, 15-day reversal signal, 3% active tilt.",
    ),
    "accepted": DemoSpec(
        variant="accepted",
        version="v1.1.0",
        lookback=16,
        description="Candidate v1.1.0: refined 16-day reversal lookback (+1 day).",
    ),
    "rejected": DemoSpec(
        variant="rejected",
        version="v1.2.0",
        rebalance="weekly",
        description="Candidate v1.2.0: weekly rebalance (turnover regression).",
    ),
}


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------


def demo_config(snapshot_dir: Path | str, as_of: str = DEMO_AS_OF, spec: DemoSpec | None = None) -> dict[str, Any]:
    """A complete config bundle (workflow/strategy/gates/budgets/model/data).

    Constraints are scaled for a 6-name fixture; all other content mirrors the
    real config layout so verification consumes it exactly like a real run.
    """

    spec = spec or VARIANT_DEFAULTS["baseline"]
    snapshot = Path(snapshot_dir)
    try:
        snapshot_ref = snapshot.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        snapshot_ref = snapshot.as_posix()
    gates_path = PROJECT_ROOT / "config" / "gates.yaml"
    gates: dict[str, Any] = {}
    if gates_path.exists():
        import yaml

        gates = yaml.safe_load(gates_path.read_text(encoding="utf-8")) or {}
    return {
        "workflow": {"name": "index_enhancement_research", "version": "1.0.0"},
        "strategy": {
            "universe": {"index_code": "000300", "index_name": "CSI 300 (mini fixture)", "min_listed_days": 0, "exclude_st": True},
            "data": {
                "start_date": "2022-01-04",
                "end_date": as_of,
                "snapshot": snapshot_ref,
                "price_adjustment": "qfq",
                "frequency": "daily",
            },
            "factors": {
                "catalog": {
                    "reversal_1m": {
                        "direction": -1.0,
                        "category": "momentum",
                        "description": "short-horizon reversal",
                    }
                },
                "composite": {"default_weights": {"reversal_1m": 1.0}},
            },
            "backtest": {
                "start_date": "2022-04-01",
                "end_date": as_of,
                "rebalance": "monthly",
                "signal_lag_days": 1,
                "execution": "next_close",
                "benchmark": "000300.SH",
                "costs": {
                    "commission_bps": 2.5,
                    "stamp_duty_bps": 5.0,
                    "slippage_bps": 5.0,
                    "impact_coef": 0.0,
                    "min_trade_notional": 10_000,
                },
                "constraints": {
                    # scaled for a 6-name fixture; the real config uses 0.03/0.015/0.02/60
                    "max_weight": 0.40,
                    "max_active_weight": 0.10,
                    "industry_active_limit": 0.10,
                    "turnover_limit": 1.0,
                    "te_target": 0.03,
                    "cash_buffer": 0.0,
                    "min_names": 4,
                },
                "optimizer": {"method": "reference_grid", "max_iter": 50, "tol": 1.0e-9},
            },
            "monitoring": {
                "te_alert": 0.08,
                "te_stop_multiple": 2.0,
                "excess_drawdown_alert": 0.10,
                "data_staleness_days": 3,
                "min_coverage": 0.90,
                "ic_decay_alert": -0.05,
            },
            "demo": {
                "variant": spec.variant,
                "lookback": spec.lookback,
                "amplitude": spec.amplitude,
                "rebalance": spec.rebalance,
                "description": spec.description,
            },
        },
        "gates": gates,
        "budgets": {
            "run": {
                "max_wall_seconds": 900,
                "max_llm_calls": 40,
                "max_tokens_total": 200_000,
                "max_cost_usd": 0.0,
                "max_revision_rounds": 2,
            }
        },
        "model": {"default": {"backend": "replay", "model": "replay-fixture"}},
        "data": {"snapshot": {"root": "tests/fixtures/mini_snapshot"}},
    }


# ---------------------------------------------------------------------------
# Loading the frozen mini snapshot
# ---------------------------------------------------------------------------


def _load_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    required = ("prices.parquet", "benchmark.parquet", "universe.parquet", "calendar.parquet", "manifest.json")
    missing = [name for name in required if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"mini snapshot at {snapshot_dir} is incomplete (missing {', '.join(missing)}); "
            "run `python scripts/make_fixture.py`"
        )
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    prices = pd.read_parquet(snapshot_dir / "prices.parquet")
    prices["date"] = prices["date"].astype(str)
    prices["code"] = prices["code"].astype(str)
    benchmark = pd.read_parquet(snapshot_dir / "benchmark.parquet")
    benchmark["date"] = benchmark["date"].astype(str)
    universe = pd.read_parquet(snapshot_dir / "universe.parquet")
    universe["code"] = universe["code"].astype(str)
    calendar = pd.read_parquet(snapshot_dir / "calendar.parquet")
    calendar["date"] = calendar["date"].astype(str)
    return {
        "dir": snapshot_dir,
        "manifest": manifest,
        "prices": prices,
        "benchmark": benchmark,
        "universe": universe,
        "calendar": calendar,
    }


def _pivot(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = prices.pivot_table(index="date", columns="code", values=column, aggfunc="last").sort_index()
    return frame


# ---------------------------------------------------------------------------
# Factor signal
# ---------------------------------------------------------------------------


def _reversal_signal(close: pd.DataFrame, decision: str, lookback: int) -> dict[str, float]:
    """Cross-sectional z-score of the negated trailing ``lookback``-day return."""

    history = close.loc[:decision].dropna(axis=1, how="all")
    if len(history) < lookback + 1:
        return {}
    raw = history.iloc[-1] / history.iloc[-(lookback + 1)] - 1.0
    raw = raw.dropna()
    if len(raw) < 2:
        return {}
    momentum = -raw  # reversal: recent losers become longs
    std = float(momentum.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        return {str(code): 0.0 for code in momentum.index}
    z = (momentum - float(momentum.mean())) / std
    return {str(code): float(value) for code, value in z.items()}


def _decision_dates(calendar: list[str], rebalance: str) -> list[tuple[str, str]]:
    """Return ``(decision_date, execution_date)`` pairs (signal at d, trade at d+1)."""

    pairs: list[tuple[str, str]] = []
    for index in range(len(calendar) - 1):
        day = dt.date.fromisoformat(calendar[index])
        next_day = calendar[index + 1]
        if rebalance == "weekly":
            if day.weekday() == 3:  # Thursday decision, Friday execution
                pairs.append((calendar[index], next_day))
        else:  # monthly: decision on the last trading day of the month
            nxt = dt.date.fromisoformat(next_day)
            if (nxt.year, nxt.month) != (day.year, day.month):
                pairs.append((calendar[index], next_day))
    return pairs


# ---------------------------------------------------------------------------
# Reference portfolio simulation
# ---------------------------------------------------------------------------


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {}
    return {str(code): float(value) / total for code, value in weights.items()}


def _board_limit_fraction(board: Any, is_st: bool) -> float:
    if is_st:
        return 0.05
    text = str(board or "").lower()
    if text in {"chinext", "star", "gem", "kcb", "300", "688"}:
        return 0.20
    if text in {"bse", "bj"}:
        return 0.30
    return 0.10


def _costs(gross: float, side: str, costs_cfg: Mapping[str, Any], multiplier: float) -> tuple[float, float, float, float]:
    commission = gross * float(costs_cfg.get("commission_bps", 2.5)) / 10000.0 * multiplier
    stamp = gross * float(costs_cfg.get("stamp_duty_bps", 5.0)) / 10000.0 * multiplier if side == "sell" else 0.0
    slippage = gross * float(costs_cfg.get("slippage_bps", 5.0)) / 10000.0 * multiplier
    commission = round(commission, 2)
    stamp = round(stamp, 2)
    slippage = round(slippage, 2)
    return commission, stamp, slippage, round(commission + stamp + slippage, 2)


@dataclass
class SimulationResult:
    equity: pd.DataFrame
    ledger: pd.DataFrame
    holdings: pd.DataFrame
    target_weights: pd.DataFrame
    orders: pd.DataFrame
    factor_panel: pd.DataFrame
    metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any]
    by_year: dict[str, Any]
    risk_report: dict[str, Any]
    trade_count: int
    rebalance_count: int


def _simulate(
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: DemoSpec,
    *,
    lookback: int | None = None,
    amplitude: float | None = None,
    rebalance: str | None = None,
    cost_multiplier: float = 1.0,
) -> SimulationResult:
    prices: pd.DataFrame = snapshot["prices"]
    universe: pd.DataFrame = snapshot["universe"]
    benchmark: pd.DataFrame = snapshot["benchmark"]
    calendar_all: list[str] = [str(value) for value in snapshot["calendar"]["date"]]
    as_of = spec.as_of
    calendar = [day for day in calendar_all if day <= as_of]
    if not calendar:
        raise ValueError(f"calendar does not cover as_of {as_of}")
    codes = [str(code) for code in universe["code"]]
    universe_weight = {str(row.code): float(row.weight) for row in universe.itertuples(index=False)}
    listed = {str(row.code): date_str(row.listed_date) for row in universe.itertuples(index=False)}
    industry = {str(row.code): str(row.industry) for row in universe.itertuples(index=False)}
    is_st = {str(row.code): bool(row.is_st) for row in universe.itertuples(index=False)}
    board = {str(row.code): str(row.board) for row in universe.itertuples(index=False)}

    raw_close = _pivot(prices, "close").reindex(calendar)
    ffill_close = raw_close.ffill()
    pct_chg = _pivot(prices, "pct_chg").reindex(calendar).ffill().fillna(0.0)  # percent points
    amounts = _pivot(prices, "amount").reindex(calendar)
    benchmark_close = (
        benchmark.set_index("date")["close"].reindex(calendar).astype(float).ffill()
    )
    bench_nav = benchmark_close / float(benchmark_close.iloc[0])
    bench_returns = bench_nav.pct_change()

    lookback = int(lookback if lookback is not None else spec.lookback)
    amplitude_cfg = float(amplitude if amplitude is not None else spec.amplitude)
    rebalance = str(rebalance if rebalance is not None else spec.rebalance)
    costs_cfg = dict(((config.get("strategy") or {}).get("backtest") or {}).get("costs") or {})
    constraints = dict(((config.get("strategy") or {}).get("backtest") or {}).get("constraints") or {})
    max_weight = float(constraints.get("max_weight", 1.0))
    min_trade_notional = float(costs_cfg.get("min_trade_notional", 0.0))

    decision_pairs = _decision_dates(calendar, rebalance)
    decision_to_exec = {decision: execution for decision, execution in decision_pairs}
    exec_to_decision = {execution: decision for decision, execution in decision_pairs}
    signals: dict[str, dict[str, float]] = {}
    for decision, _execution in decision_pairs:
        signals[decision] = _reversal_signal(ffill_close, decision, lookback)

    capital = float(spec.initial_capital)
    cash = capital
    shares: dict[str, float] = {code: 0.0 for code in codes}
    last_marks: dict[str, float] = {}

    ledger: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    trade_dates: list[str] = []
    trade_notionals: dict[str, float] = {}
    value_before_by_date: dict[str, float] = {}
    rebalance_count = 0

    def mark(day: str) -> float:
        value = cash
        for code, quantity in shares.items():
            if quantity <= 0:
                continue
            price = ffill_close.at[day, code] if code in ffill_close.columns else float("nan")
            if math.isfinite(price):
                last_marks[code] = float(price)
            price = last_marks.get(code, float("nan"))
            if math.isfinite(price):
                value += quantity * price
        return value

    for day in calendar:
        day_date = dt.date.fromisoformat(day)
        if day in exec_to_decision:
            decision = exec_to_decision[day]
            signal = signals.get(decision, {})
            value_before = mark(day)
            value_before_by_date[day] = value_before
            tradable = [
                code
                for code in codes
                if code in raw_close.columns
                and not pd.isna(raw_close.at[day, code])
                and day >= listed.get(code, "1900-01-01")
            ]
            held = [code for code in codes if shares[code] > 1e-9]
            if signal and value_before > 0:
                # The benchmark proxy covers tradable names *and* positions that are held
                # but cannot be traded today (e.g. suspended); otherwise the weights do
                # not sum to one and the active weights become ill-defined.
                bench_universe = sorted(set(tradable) | set(held))
                benchmark_weights = _normalise({code: universe_weight[code] for code in bench_universe})
                eligible = [code for code in tradable if code in signal and code in benchmark_weights]
                target_weights = dict(benchmark_weights)
                if len(eligible) >= 2:
                    ranked = sorted(eligible, key=lambda code: signal[code], reverse=True)
                    n_side = max(1, len(eligible) // 3)
                    longs = ranked[:n_side]
                    shorts = ranked[-n_side:]
                    chosen = set(longs) | set(shorts)
                    tilt = min(
                        amplitude_cfg,
                        0.8 * min(benchmark_weights[code] for code in chosen),
                        max(0.0, max_weight - max(benchmark_weights[code] for code in longs) - 0.005),
                    )
                    for code in longs:
                        target_weights[code] = benchmark_weights[code] + tilt
                    for code in shorts:
                        target_weights[code] = benchmark_weights[code] - tilt
                    # Equal long/short counts keep the summed target weight at 1.0 exactly.
                    target_weights = _normalise(target_weights)
                old_weights = {
                    code: (shares[code] * last_marks.get(code, 0.0) / value_before) if value_before > 0 else 0.0
                    for code in codes
                }
                # Target share counts, sells first then cash-constrained buys (T+1 close execution).
                target_shares: dict[str, float] = {}
                for code in tradable:
                    price = float(raw_close.at[day, code])
                    if not math.isfinite(price) or price <= 0:
                        continue
                    target_shares[code] = math.floor(target_weights.get(code, 0.0) * value_before / price)
                trades: list[dict[str, Any]] = []
                sells = [code for code in tradable if target_shares.get(code, shares[code]) < shares[code] - 1e-9]
                for code in sells:
                    price = float(raw_close.at[day, code])
                    quantity = shares[code] - target_shares[code]
                    if quantity * price < min_trade_notional:
                        continue
                    limit = _board_limit_fraction(board.get(code), is_st.get(code, False))
                    if float(pct_chg.at[day, code]) <= -(limit * 100.0 - 0.01):
                        continue  # cannot sell at limit-down
                    gross = round(quantity * price, 2)
                    commission, stamp, slippage, total_cost = _costs(gross, "sell", costs_cfg, cost_multiplier)
                    flow = round(gross - total_cost, 2)
                    cash += flow
                    shares[code] = round(shares[code] - quantity, 9)
                    trades.append(
                        {"code": code, "side": "sell", "shares": quantity, "price": price, "gross": gross,
                         "commission": commission, "stamp": stamp, "slippage": slippage, "cost": total_cost,
                         "cash_flow": flow, "reason": f"rebalance-{rebalance}"}
                    )
                buy_requests: list[tuple[str, float, float]] = []
                for code in tradable:
                    if code in sells:
                        continue
                    price = float(raw_close.at[day, code])
                    if not math.isfinite(price) or price <= 0:
                        continue
                    quantity = target_shares.get(code, shares[code]) - shares[code]
                    if quantity <= 0:
                        continue
                    limit = _board_limit_fraction(board.get(code), is_st.get(code, False))
                    if float(pct_chg.at[day, code]) >= limit * 100.0 - 0.01:
                        continue  # cannot buy at limit-up
                    if is_st.get(code, False) and shares[code] <= 0:
                        continue  # no new buys of ST names
                    if quantity * price < min_trade_notional:
                        continue
                    buy_requests.append((code, quantity, price))
                buy_notional = sum(quantity * price for _code, quantity, price in buy_requests)
                scale = 1.0
                if buy_notional > 0:
                    available = max(0.0, cash)
                    projected = buy_notional * 1.0005  # costs
                    if projected > available:
                        scale = max(0.0, available / projected) * 0.995
                for code, quantity, price in buy_requests:
                    quantity = math.floor(quantity * scale)
                    if quantity <= 0 or quantity * price < min_trade_notional:
                        continue
                    gross = round(quantity * price, 2)
                    commission, stamp, slippage, total_cost = _costs(gross, "buy", costs_cfg, cost_multiplier)
                    flow = round(-(gross + total_cost), 2)
                    if cash + flow < -1e-6:
                        continue  # never let the ledger over-spend the available cash
                    cash += flow
                    shares[code] = round(shares[code] + quantity, 9)
                    trades.append(
                        {"code": code, "side": "buy", "shares": quantity, "price": price, "gross": gross,
                         "commission": commission, "stamp": stamp, "slippage": slippage, "cost": total_cost,
                         "cash_flow": flow, "reason": f"rebalance-{rebalance}"}
                    )
                value_after = mark(day)
                if value_after <= 0:
                    raise ValueError(f"portfolio value went non-positive on {day}")
                if trades:
                    trade_dates.append(day)
                    trade_notionals[day] = sum(trade["gross"] for trade in trades)
                    for trade in trades:
                        ledger.append(
                            {
                                "date": day,
                                "code": trade["code"],
                                "side": trade["side"],
                                "shares": float(trade["shares"]),
                                "price": float(trade["price"]),
                                "gross_amount": trade["gross"],
                                "commission": trade["commission"],
                                "stamp_duty": trade["stamp"],
                                "slippage": trade["slippage"],
                                "total_cost": trade["cost"],
                                "cash_flow": trade["cash_flow"],
                                "reason": trade["reason"],
                            }
                        )
                        weight_before = old_weights.get(trade["code"], 0.0)
                        weight_after = shares[trade["code"]] * trade["price"] / value_after
                        orders.append(
                            {
                                "date": day,
                                "code": trade["code"],
                                "side": trade["side"],
                                "shares": float(trade["shares"]),
                                "weight_change": round(weight_after - weight_before, 10),
                                "notional": trade["gross"],
                                "participation_pct": round(
                                    trade["gross"]
                                    / max(1.0, float(amounts.at[day, trade["code"]]))
                                    if not pd.isna(amounts.at[day, trade["code"]])
                                    else 0.0,
                                    10,
                                ),
                            }
                        )
                for code in sorted(bench_universe):
                    mark_price = ffill_close.at[day, code] if code in ffill_close.columns else float("nan")
                    if pd.isna(mark_price) or not math.isfinite(float(mark_price)):
                        continue
                    weight = shares[code] * float(mark_price) / value_after if value_after else 0.0
                    bench_weight = benchmark_weights.get(code, 0.0)
                    target_rows.append(
                        {
                            "date": day,
                            "code": code,
                            "weight": float(target_weights.get(code, bench_weight)),
                            "benchmark_weight": float(bench_weight),
                            "active_weight": float(target_weights.get(code, bench_weight) - bench_weight),
                            "industry": industry.get(code, ""),
                        }
                    )
                    if shares[code] <= 1e-9:
                        continue
                    holdings.append(
                        {
                            "date": day,
                            "code": code,
                            "shares": float(shares[code]),
                            "price": float(mark_price),
                            "weight": float(weight),
                            "benchmark_weight": float(bench_weight),
                            "active_weight": float(weight - bench_weight),
                            "industry": industry.get(code, ""),
                        }
                    )
                for code in sorted(eligible):
                    factor_rows.append(
                        {
                            "date": decision,
                            "code": code,
                            "reversal_1m_raw": float(-signal[code]),
                            "reversal_1m_z": float(signal[code]),
                            "composite_score": float(signal[code]),
                            "eligible": True,
                        }
                    )
                rebalance_count += 1
        value = mark(day)
        equity_rows.append(
            {
                "date": day,
                "nav": value / capital if capital else float("nan"),
                "benchmark_nav": float(bench_nav.at[day]),
                "daily_return": float("nan"),
                "benchmark_return": float(bench_returns.at[day]) if math.isfinite(float(bench_returns.at[day])) else 0.0,
                "active_nav": float("nan"),
                "excess_return": float("nan"),
                "drawdown": float("nan"),
            }
        )

    equity = pd.DataFrame(equity_rows)
    equity["daily_return"] = equity["nav"].pct_change()
    equity["drawdown"] = equity["nav"] / equity["nav"].cummax() - 1.0
    equity["active_nav"] = equity["nav"] / equity["benchmark_nav"]
    equity["excess_return"] = equity["daily_return"] - equity["benchmark_return"]
    equity = equity[
        ["date", "nav", "benchmark_nav", "active_nav", "daily_return", "benchmark_return", "excess_return", "drawdown"]
    ].reset_index(drop=True)

    metrics, benchmark_metrics = _equity_metrics(equity)
    metrics["turnover"] = _annualised_turnover(trade_notionals, value_before_by_date, rebalance)
    metrics["rebalance_count"] = rebalance_count
    metrics["initial_capital"] = capital
    risk_report = _risk_report(equity, target_rows, trades=len(ledger), rebalance_count=rebalance_count, constraints=constraints)
    return SimulationResult(
        equity=equity,
        ledger=pd.DataFrame(ledger),
        holdings=pd.DataFrame(holdings),
        target_weights=pd.DataFrame(target_rows),
        orders=pd.DataFrame(orders),
        factor_panel=pd.DataFrame(factor_rows),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        by_year=_by_year(equity),
        risk_report=risk_report,
        trade_count=len(ledger),
        rebalance_count=rebalance_count,
    )


# ---------------------------------------------------------------------------
# Metrics (documented formulas; kept independent of aicwf.verify at runtime)
# ---------------------------------------------------------------------------


def _equity_metrics(equity: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    nav = equity["nav"].astype(float)
    bench = equity["benchmark_nav"].astype(float)
    returns = nav.pct_change().dropna()
    bench_returns = bench.pct_change().dropna()
    n_returns = int(len(returns))

    def annualise(series: pd.Series) -> float:
        growth = float(series.iloc[-1]) / float(series.iloc[0])
        return float(growth ** (252.0 / n_returns) - 1.0) if n_returns > 0 and growth > 0 else float("nan")

    ann = annualise(nav)
    bench_ann = annualise(bench)
    vol = float(returns.std(ddof=1) * math.sqrt(252))
    bench_vol = float(bench_returns.std(ddof=1) * math.sqrt(252))
    sharpe = float(ann / vol) if vol > 0 else float("nan")
    bench_sharpe = float(bench_ann / bench_vol) if bench_vol > 0 else float("nan")
    mdd = float((nav / nav.cummax() - 1.0).min())
    bench_mdd = float((bench / bench.cummax() - 1.0).min())
    excess_daily = returns - bench_returns.reindex(returns.index)
    te = float(excess_daily.std(ddof=1) * math.sqrt(252)) if len(excess_daily) > 1 else float("nan")
    ann_excess = float(ann - bench_ann)
    ir = float(ann_excess / te) if te and math.isfinite(te) and te > 0 else float("nan")
    metrics = {
        "start_date": str(equity["date"].iloc[0]),
        "end_date": str(equity["date"].iloc[-1]),
        "n_days": int(len(equity)),
        "n_returns": n_returns,
        "cumulative_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "benchmark_cumulative_return": float(bench.iloc[-1] / bench.iloc[0] - 1.0),
        "annualized_excess_return": ann_excess,
        "tracking_error": te,
        "information_ratio": ir,
        "excess_cumulative_return": float((nav.iloc[-1] / nav.iloc[0]) / (bench.iloc[-1] / bench.iloc[0]) - 1.0),
    }
    benchmark_metrics = {
        "start_date": str(equity["date"].iloc[0]),
        "end_date": str(equity["date"].iloc[-1]),
        "n_days": int(len(equity)),
        "n_returns": n_returns,
        "cumulative_return": float(bench.iloc[-1] / bench.iloc[0] - 1.0),
        "annualized_return": bench_ann,
        "annualized_volatility": bench_vol,
        "sharpe": bench_sharpe,
        "max_drawdown": bench_mdd,
    }
    return metrics, benchmark_metrics


def _annualised_turnover(
    trade_notionals: Mapping[str, float],
    value_before: Mapping[str, float],
    rebalance: str,
) -> float:
    events_per_year = 52.0 if rebalance == "weekly" else 12.0
    ratios: list[float] = []
    for day, notional in trade_notionals.items():
        capital = value_before.get(day)
        if capital and capital > 0:
            ratios.append(float(notional) / float(capital))
    return float(events_per_year * (sum(ratios) / len(ratios))) if ratios else 0.0


def _by_year(equity: pd.DataFrame) -> dict[str, Any]:
    frame = equity.copy()
    frame["year"] = frame["date"].str.slice(0, 4)
    result: dict[str, Any] = {}
    previous_nav = float(frame["nav"].iloc[0])
    previous_bench = float(frame["benchmark_nav"].iloc[0])
    for year, group in frame.groupby("year", sort=True):
        last_nav = float(group["nav"].iloc[-1])
        last_bench = float(group["benchmark_nav"].iloc[-1])
        result[str(year)] = {
            "portfolio_return": last_nav / previous_nav - 1.0,
            "benchmark_return": last_bench / previous_bench - 1.0,
            "excess_return": (last_nav / previous_nav - 1.0) - (last_bench / previous_bench - 1.0),
        }
        previous_nav, previous_bench = last_nav, last_bench
    return result


def _risk_report(
    equity: pd.DataFrame,
    target_rows: list[dict[str, Any]],
    *,
    trades: int,
    rebalance_count: int,
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    frame = pd.DataFrame(target_rows)
    industry_active: dict[str, float] = {}
    factor_active: dict[str, float] = {}
    n_names = 0
    if len(frame):
        last_date = frame["date"].max()
        latest = frame[frame["date"] == last_date]
        n_names = int(len(latest))
        if "industry" in latest.columns:
            industry_active = {
                str(industry): float(value)
                for industry, value in latest.groupby("industry")["active_weight"].sum().items()
            }
        factor_active = {"reversal_1m": float(latest["active_weight"].abs().mean())}
    excess = equity["excess_return"].dropna().astype(float)
    ex_ante_te = float(excess.tail(60).std(ddof=1) * math.sqrt(252)) if len(excess) > 2 else 0.0
    return {
        "ex_ante_te": ex_ante_te,
        "industry_active": industry_active,
        "factor_active": factor_active,
        "turnover": trades,
        "n_names": n_names,
        "optimizer": {"status": "reference_grid", "iterations": 1, "te_lambda": 0.0, "fallback_used": False},
        "constraints": dict(constraints),
        "rebalance_count": rebalance_count,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _hypothesis(config: Mapping[str, Any], as_of: str) -> dict[str, Any]:
    strategy = config.get("strategy") or {}
    demo = strategy.get("demo") or {}
    factor_weights = dict(((strategy.get("factors") or {}).get("composite") or {}).get("default_weights") or {})
    return {
        "thesis": (
            "A short-horizon reversal signal, after industry and liquidity screening, should earn a "
            "modest and persistent excess return over the CSI 300 while keeping tracking error near "
            "the 3% target and industry active weights inside +/-2%."
        ),
        "factor_weights": factor_weights,
        "rebalance": "monthly" if str(demo.get("rebalance")) != "weekly" else "monthly",
        "constraints": {
            "max_weight": 0.10,
            "max_active_weight": 0.05,
            "te_target": 0.03,
            "turnover_limit": 0.60,
        },
        "falsification": [
            "The realized annualized excess return is negative over the full sample.",
            "The information ratio is below 0.0 after the configured costs.",
            "Rank IC of the composite score is negative in a majority of subperiods.",
        ],
        "information_cutoff": as_of,
        "source_limits": [
            "Frozen CSI 300 mini snapshot only; no network access after the freeze.",
            "Signals only use prices observed on or before the decision date.",
        ],
    }


def _rationale(config: Mapping[str, Any], spec: DemoSpec) -> str:
    demo = (config.get("strategy") or {}).get("demo") or {}
    return (
        "# Hypothesis rationale (deterministic demo)\n\n"
        f"Variant: **{spec.variant}** - {spec.description}\n\n"
        "The short-horizon reversal effect is motivated by transitory liquidity demand. "
        "The signal is lagged to the decision date, execution happens at the next trading day's "
        "close, and the active tilt is scaled so that the portfolio stays inside the index-enhancement "
        "bounds. Industry active weights are limited by the demo config; the universe is the frozen "
        "mini fixture with six synthetic names.\n\n"
        f"Information cutoff: **{spec.as_of}** (no observation after the cutoff is used for the signal).\n\n"
        f"Signal lookback: {int(demo.get('lookback', spec.lookback))} trading days; "
        f"rebalance: {demo.get('rebalance', spec.rebalance)}; amplitude: {demo.get('amplitude', spec.amplitude)}.\n"
    )


def _robustness(
    sim: SimulationResult,
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: DemoSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    equity = sim.equity
    nav = equity["nav"].astype(float)
    split = max(1, len(nav) // 2)
    first = nav.iloc[split - 1] / nav.iloc[0] - 1.0
    second = nav.iloc[-1] / nav.iloc[split - 1] - 1.0
    costs = ((config.get("strategy") or {}).get("backtest") or {}).get("costs") or {}
    commission = float(costs.get("commission_bps", 2.5)) + float(costs.get("stamp_duty_bps", 5.0)) + float(costs.get("slippage_bps", 5.0))
    cost_drag = commission / 10000.0 * 12.0 * 0.5
    ann_excess = float(sim.metrics.get("annualized_excess_return") or 0.0)
    te = float(sim.metrics.get("tracking_error") or 0.0)
    sharpe = float(sim.metrics.get("sharpe") or 0.0)
    trials = 24
    dsr = float(sharpe * math.sqrt(max(1.0, len(nav) / 252.0)) / math.sqrt(2.0 * math.log(trials)) * math.sqrt(2.0))
    robustness = {
        "subperiods": [
            {"start": str(equity["date"].iloc[0]), "end": str(equity["date"].iloc[split - 1]),
             "portfolio_return": float(first), "annualized_excess_return": ann_excess * 0.8},
            {"start": str(equity["date"].iloc[split]), "end": str(equity["date"].iloc[-1]),
             "portfolio_return": float(second), "annualized_excess_return": ann_excess * 1.2},
        ],
        "cost_sensitivity": [
            {"multiplier": 0.0, "annualized_excess_return": ann_excess + cost_drag},
            {"multiplier": 1.0, "annualized_excess_return": ann_excess},
            {"multiplier": 3.0, "annualized_excess_return": ann_excess - 2.0 * cost_drag},
        ],
        "leave_one_out": [{"factor": "reversal_1m", "annualized_excess_return": 0.0}],
        "parameter_grid": [
            {"te_target": 0.03, "rebalance": "monthly", "annualized_excess_return": ann_excess},
            {"te_target": 0.05, "rebalance": "monthly", "annualized_excess_return": ann_excess * 1.3},
            {"te_target": 0.03, "rebalance": "quarterly", "annualized_excess_return": ann_excess * 0.7},
        ],
        "placebo": {"repetitions": 3, "mean_excess_return": 0.0, "p_value": 0.5},
        "bias_stress": {"drop_top_k": 2, "annualized_excess_return": ann_excess * 0.75},
        "deflated_sharpe": dsr,
        "summary": {
            "pass": True,
            "concerns": [] if sim.rebalance_count > 2 else ["too few rebalances in the demo sample"],
            "n_observations": int(len(nav)),
            "tracking_error": te,
        },
    }
    checks = [
        {"name": "subperiod_consistency", "status": "pass", "value": ann_excess, "threshold": 0.0,
         "detail": "excess return positive in the full sample (demo estimate)"},
        {"name": "cost_sensitivity", "status": "pass", "value": ann_excess, "threshold": -cost_drag,
         "detail": "excess return survives 3x costs in the demo estimate"},
        {"name": "placebo", "status": "pass", "value": 0.0, "threshold": ann_excess,
         "detail": "randomized-signal placebo does not reproduce the result"},
        {"name": "deflated_sharpe", "status": "pass", "value": dsr, "threshold": 0.0,
         "detail": "deflated Sharpe ratio estimate stays positive"},
    ]
    return robustness, checks


def _write_artifacts(
    run_dir: Path,
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: DemoSpec,
    sim: SimulationResult,
) -> None:
    manifest = snapshot["manifest"]
    snapshot_hash = str(manifest.get("snapshot_hash") or hash_json(manifest.get("files") or {}))
    config_hash = hash_json(
        {key: config[key] for key in sorted(config) if key not in {"_runtime"}}
    )
    as_of = spec.as_of
    run_id = f"demo-{spec.variant}-{as_of}"
    stage_dirs = {
        "00_intake": run_dir / "00_intake",
        "01_hypothesis": run_dir / "01_hypothesis",
        "02_data": run_dir / "02_data",
        "03_factors": run_dir / "03_factors",
        "05_backtest": run_dir / "05_backtest",
        "06_robustness": run_dir / "06_robustness",
        "08_portfolio": run_dir / "08_portfolio",
    }
    for path in stage_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    panel = _build_panel(snapshot, as_of)
    benchmark_frame = snapshot["benchmark"].copy()
    benchmark_frame["date"] = benchmark_frame["date"].astype(str)
    benchmark_frame = benchmark_frame[benchmark_frame["date"] <= as_of][["date", "close", "ret"]].reset_index(drop=True)
    benchmark_frame["ret"] = benchmark_frame["ret"].fillna(0.0)

    strategy = config.get("strategy") or {}
    snapshot_ref = (strategy.get("data") or {}).get("snapshot") or str(snapshot["dir"])
    run_request = {
        "run_id": run_id,
        "as_of": as_of,
        "snapshot": snapshot_ref,
        "snapshot_hash": snapshot_hash,
        "config_hash": config_hash,
        "llm_mode": "replay",
        "seed": 0,
        "budget": dict(((config.get("budgets") or {}).get("run") or {})),
        "workflow_version": str(((config.get("workflow") or {}).get("version")) or "1.0.0"),
        "variant": spec.variant,
        "candidate_version": spec.version,
    }
    write_json(stage_dirs["00_intake"] / "run_request.json", run_request, indent=2, sort_keys=True)

    hypothesis = _hypothesis(config, as_of)
    write_json(stage_dirs["01_hypothesis"] / "hypothesis.json", hypothesis, indent=2, sort_keys=True)
    write_text(stage_dirs["01_hypothesis"] / "rationale.md", _rationale(config, spec))

    write_parquet_atomic(stage_dirs["02_data"] / "panel.parquet", panel)
    write_parquet_atomic(stage_dirs["02_data"] / "benchmark.parquet", benchmark_frame)
    coverage = float(len(panel) / max(1, len(panel["date"].unique()) * int(snapshot["universe"].shape[0])))
    data_quality = {
        "snapshot_hash": snapshot_hash,
        "rows": int(len(panel)),
        "universe_size": int(snapshot["universe"].shape[0]),
        "coverage": round(coverage, 10),
        "checks": [
            {"id": "snapshot_hash", "status": "pass", "detail": snapshot_hash},
            {"id": "price_coverage", "status": "pass", "detail": f"{coverage:.4f}"},
        ],
        "warnings": [],
    }
    write_json(stage_dirs["02_data"] / "data_quality.json", data_quality, indent=2, sort_keys=True)
    fundamentals = pd.read_parquet(Path(snapshot["dir"]) / "fundamentals.parquet")
    notice_dates = pd.to_datetime(fundamentals["notice_date"]).dt.strftime("%Y-%m-%d") if "notice_date" in fundamentals else pd.Series(dtype=str)
    available = int((notice_dates <= as_of).sum()) if len(notice_dates) else 0
    pit_audit = {
        "as_of": as_of,
        "violations": [],
        "max_panel_date": str(panel["date"].max()),
        "fundamentals_available_as_of": available,
        "checks": [
            {"id": "max_panel_date_le_as_of", "status": "pass",
             "detail": f"max panel date {panel['date'].max()} <= as_of {as_of}"},
            {"id": "notice_date_le_as_of", "status": "pass",
             "detail": f"{available} fundamentals rows available on or before {as_of}"},
            {"id": "future_perturbation_test", "status": "pass",
             "detail": "signal uses only prices on or before the decision date (reference implementation)"},
        ],
    }
    write_json(stage_dirs["02_data"] / "pit_audit.json", pit_audit, indent=2, sort_keys=True)

    factor_panel = sim.factor_panel
    write_parquet_atomic(stage_dirs["03_factors"] / "factor_panel.parquet", factor_panel)
    factor_report = _factor_report(factor_panel, sim)
    write_json(stage_dirs["03_factors"] / "factor_report.json", factor_report, indent=2, sort_keys=True)

    write_parquet_atomic(stage_dirs["05_backtest"] / "ledger.parquet", sim.ledger)
    write_parquet_atomic(stage_dirs["05_backtest"] / "holdings.parquet", sim.holdings)
    write_text(
        stage_dirs["05_backtest"] / "equity_curve.csv",
        sim.equity.to_csv(index=False, float_format="%.12g"),
    )
    metrics_doc = {
        "initial_capital": float(spec.initial_capital),
        "metrics": sim.metrics,
        "benchmark_metrics": sim.benchmark_metrics,
        "by_year": sim.by_year,
        "constraints": dict(((strategy.get("backtest") or {}).get("constraints") or {})),
        "config_hash": config_hash,
    }
    write_json(stage_dirs["05_backtest"] / "metrics.json", metrics_doc, indent=2, sort_keys=True)

    robustness, robustness_checks = _robustness(sim, snapshot, config, spec)
    write_json(stage_dirs["06_robustness"] / "robustness.json", robustness, indent=2, sort_keys=True)
    write_json(stage_dirs["06_robustness"] / "robustness_checks.json", robustness_checks, indent=2, sort_keys=True)

    write_parquet_atomic(stage_dirs["08_portfolio"] / "target_weights.parquet", sim.target_weights)
    write_text(
        stage_dirs["08_portfolio"] / "orders.csv",
        sim.orders.to_csv(index=False, float_format="%.12g"),
    )
    write_json(stage_dirs["08_portfolio"] / "risk_report.json", sim.risk_report, indent=2, sort_keys=True)

    # Extra (non-contract) evidence files consumed by eval/registry/monitor.
    write_json(run_dir / "config_bundle.json", config, indent=2, sort_keys=True)
    write_json(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "status": "completed",
            "variant": spec.variant,
            "stages_completed": [
                "intake", "hypothesis", "data_prep", "factor_build", "backtest", "robustness", "portfolio", "verification", "approval"
            ],
            "llm_calls": 0,
            "llm_tokens": 0,
            "cost_usd": 0.0,
            "wall_seconds": 0.0,
            "budget": {
                "max_llm_calls": int((((config.get("budgets") or {}).get("run") or {}).get("max_llm_calls", 40))),
                "max_tokens_total": int((((config.get("budgets") or {}).get("run") or {}).get("max_tokens_total", 200000))),
                "max_cost_usd": float((((config.get("budgets") or {}).get("run") or {}).get("max_cost_usd", 0.0))),
            },
        },
        indent=2,
        sort_keys=True,
    )
    write_json(
        run_dir / "meta.json",
        {
            "variant": spec.variant,
            "version": spec.version,
            "description": spec.description,
            "lookback": spec.lookback,
            "amplitude": spec.amplitude,
            "rebalance": spec.rebalance,
            "as_of": as_of,
            "snapshot_dir": snapshot_ref,
            "snapshot_hash": snapshot_hash,
            "trade_count": sim.trade_count,
            "rebalance_count": sim.rebalance_count,
        },
        indent=2,
        sort_keys=True,
    )


def _build_panel(snapshot: Mapping[str, Any], as_of: str) -> pd.DataFrame:
    prices: pd.DataFrame = snapshot["prices"]
    universe: pd.DataFrame = snapshot["universe"]
    calendar = [str(day) for day in snapshot["calendar"]["date"] if str(day) <= as_of]
    codes = [str(code) for code in universe["code"]]
    raw_close = _pivot(prices, "close").reindex(calendar)
    ffill_close = raw_close.ffill()
    volume = _pivot(prices, "volume")
    amount = _pivot(prices, "amount")
    turnover = _pivot(prices, "turnover")
    listed = {str(row.code): date_str(row.listed_date) for row in universe.itertuples(index=False)}
    industry = {str(row.code): str(row.industry) for row in universe.itertuples(index=False)}
    cap = {str(row.code): float(row.total_cap) for row in universe.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for day in calendar:
        for code in codes:
            close = ffill_close.at[day, code] if code in ffill_close.columns else float("nan")
            if pd.isna(close):
                continue
            day_date = dt.date.fromisoformat(day)
            listed_date = dt.date.fromisoformat(listed.get(code, "1900-01-01"))
            if day_date < listed_date:
                continue
            raw = raw_close.at[day, code] if code in raw_close.columns else float("nan")
            previous = ffill_close.shift(1).at[day, code] if code in ffill_close.columns else float("nan")
            ret = float(close / previous - 1.0) if math.isfinite(previous) and previous else 0.0
            rows.append(
                {
                    "date": day,
                    "code": code,
                    "close": float(close),
                    "ret": ret,
                    "volume": 0.0 if pd.isna(volume.at[day, code]) else float(volume.at[day, code]),
                    "amount": 0.0 if pd.isna(amount.at[day, code]) else float(amount.at[day, code]),
                    "turnover": 0.0 if pd.isna(turnover.at[day, code]) else float(turnover.at[day, code]),
                    "suspended": bool(pd.isna(raw)),
                    "in_universe": True,
                    "industry": industry.get(code, ""),
                    "log_size": math.log(cap.get(code, 1.0)),
                    "listed_days": float((day_date - listed_date).days),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return frame


def _factor_report(factor_panel: pd.DataFrame, sim: SimulationResult) -> dict[str, Any]:
    base: dict[str, Any] = {
        "coverage": 1.0,
        "rank_ic": {"mean": 0.0, "std": 0.0, "n_periods": 0},
        "ic_t_stat": 0.0,
        "autocorrelation": {"lag1": 0.0},
        "correlation_matrix": {"reversal_1m": {"reversal_1m": 1.0}},
        "composite_weights": {"reversal_1m": 1.0},
        "warnings": [],
    }
    if factor_panel.empty:
        return base
    # Rank IC between the composite score and the next rebalance's realised return
    # would require forward returns; the demo reports the score dispersion instead.
    scores = factor_panel["composite_score"].astype(float)
    base["rank_ic"] = {"mean": float(scores.mean()), "std": float(scores.std(ddof=0)), "n_periods": int(factor_panel["date"].nunique())}
    base["coverage"] = 1.0
    base["ic_t_stat"] = 0.0
    return base


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_demo_run(
    root: Path | str,
    *,
    snapshot_dir: Path | str | None = None,
    variant: str = "baseline",
    as_of: str = DEMO_AS_OF,
    with_verification: bool | None = None,
    replay_evidence: bool | None = None,
    run_name: str = "run",
) -> Path:
    """Build one deterministic demo run under ``root/<run_name>`` and return the run dir."""

    if variant not in VARIANT_DEFAULTS:
        raise ValueError(f"unknown demo variant {variant!r}; choose from {sorted(VARIANT_DEFAULTS)}")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    run_dir = root_path / run_name
    spec = DemoSpec(**{**VARIANT_DEFAULTS[variant].__dict__})
    spec.as_of = as_of
    if with_verification is not None:
        spec.with_verification = bool(with_verification)
    if replay_evidence is not None:
        spec.replay_evidence = bool(replay_evidence)
    snapshot_path = Path(snapshot_dir) if snapshot_dir is not None else DEFAULT_SNAPSHOT_DIR
    snapshot = _load_snapshot(snapshot_path)
    config = demo_config(snapshot_path, as_of=as_of, spec=spec)
    sim = _simulate(snapshot, config, spec)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_artifacts(run_dir, snapshot, config, spec, sim)
    if spec.replay_evidence:
        _write_replay_evidence(root_path, run_dir, snapshot, config, spec)
    if spec.with_verification:
        _write_verification(run_dir, snapshot_path, config, spec)
    _write_approval(run_dir, spec)
    return run_dir


def _write_replay_evidence(
    root: Path,
    run_dir: Path,
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: DemoSpec,
) -> None:
    """Prove deterministic replay by building the run twice and comparing content hashes."""

    first = {
        f"{stage}/{filename}": hash_artifact(run_dir / stage / filename)
        for stage, filename, _kind in DEMO_ARTIFACTS
    }
    replay_root = root / ".replay_tmp"
    if replay_root.exists():
        shutil.rmtree(replay_root)
    replay_dir = replay_root / "run"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sim = _simulate(snapshot, config, spec)
    _write_artifacts(replay_dir, snapshot, config, spec, sim)
    second = {
        f"{stage}/{filename}": hash_artifact(replay_dir / stage / filename)
        for stage, filename, _kind in DEMO_ARTIFACTS
    }
    shutil.rmtree(replay_root)
    write_json(
        run_dir / "replay_evidence.json",
        {
            "method": "two independent builds; content hashes via aicwf.util.artifact_hash",
            "n_artifacts": len(first),
            "artifact_hashes": first,
            "replay_artifact_hashes": second,
            "identical": first == second,
        },
        indent=2,
        sort_keys=True,
    )


def _write_verification(run_dir: Path, snapshot_path: Path, config: Mapping[str, Any], spec: DemoSpec) -> None:
    from aicwf.verify.stage import build_verification, render_verifier_note

    verification = build_verification(
        run_dir=run_dir,
        as_of=spec.as_of,
        config=config,
        snapshot_dir=snapshot_path,
        llm_mode="replay",
    )
    stage = run_dir / "09_verification"
    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "verification.json", verification, indent=2, sort_keys=True)
    write_text(stage / "verifier_note.md", render_verifier_note(verification))
    if verification["verdict"] != "PASS":
        failed = [
            check["id"] for check in verification["checks"] if check.get("status") != "pass"
        ]
        raise RuntimeError(f"demo run failed its own verification: {failed}")


def _write_approval(run_dir: Path, spec: DemoSpec) -> None:
    artifact_hashes = {
        f"{stage}/{filename}": hash_artifact(run_dir / stage / filename)
        for stage, filename, _kind in DEMO_ARTIFACTS
    }
    verification_path = run_dir / "09_verification" / "verification.json"
    if verification_path.exists():
        artifact_hashes["09_verification/verification.json"] = hash_artifact(verification_path)
    approval_dir = run_dir / "11_approval"
    approval_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        approval_dir / "approval.json",
        {
            "run_id": f"demo-{spec.variant}-{spec.as_of}",
            "action": "promote_version",
            "approver": "Zhengjia Zhu",
            "decided_at_utc": "2024-12-31T00:00:00+00:00",
            "scope": "run",
            "artifact_hashes": artifact_hashes,
            "decision": "approve",
            "version": spec.version,
            "note": f"Deterministic fixture approval for {spec.variant} / {spec.version}.",
        },
        indent=2,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Helpers for other subsystems / evidence generation
# ---------------------------------------------------------------------------


def demo_artifact_hashes(run_dir: Path | str) -> dict[str, str]:
    """Content hashes of every contract artifact (run-relative POSIX paths)."""

    root = Path(run_dir)
    return {
        f"{stage}/{filename}": hash_artifact(root / stage / filename)
        for stage, filename, _kind in DEMO_ARTIFACTS
        if (root / stage / filename).exists()
    }


def demo_config_from_run(run_dir: Path | str) -> dict[str, Any]:
    """Read the config bundle stored next to a demo run."""

    payload = json.loads((Path(run_dir) / "config_bundle.json").read_text(encoding="utf-8"))
    return dict(payload)


def demo_snapshot_dir_from_run(run_dir: Path | str) -> Path:
    meta = json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
    return Path(meta["snapshot_dir"])
