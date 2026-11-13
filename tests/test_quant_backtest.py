"""Backtest engine, cost model, metrics and accounting-invariant tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_quant_support as sup  # noqa: E402

from aicwf.backtest import costs as cost_model  # noqa: E402
from aicwf.backtest import metrics as metric_model  # noqa: E402
from aicwf.util.artifact_hash import hash_dataframe, hash_json_artifact  # noqa: E402
from aicwf.util.jsonio import write_parquet_atomic  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_synthetic(tmp_path: Path, *, config_overrides=None):
    market = sup.make_market(n_codes=8, n_months=36)
    ctx, market, paths = sup.make_run_context(
        tmp_path, market=market, config_overrides=config_overrides
    )
    from aicwf.backtest.stage import run_backtest
    from aicwf.factors.stage import run_factor_build

    factor_output = run_factor_build(ctx)
    ctx.register("factor_build", factor_output)
    backtest_output = run_backtest(ctx)
    ctx.register("backtest", backtest_output)
    return ctx, market, paths, factor_output, backtest_output


def _ledger_frame(pipeline) -> pd.DataFrame:
    return pd.read_parquet(sup.artifact_path(pipeline, "backtest", "ledger"))


def _holdings_frame(pipeline) -> pd.DataFrame:
    return pd.read_parquet(sup.artifact_path(pipeline, "backtest", "holdings"))


def _metrics_doc(pipeline) -> dict:
    return json.loads(sup.artifact_path(pipeline, "backtest", "metrics").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_trade_cost_model_matches_hand_calculation():
    notional = 1_000_000.0
    adv = 20_000_000.0
    costs = cost_model.resolve_costs({"commission_bps": 2.5, "stamp_duty_bps": 5.0, "slippage_bps": 5.0, "impact_coef": 0.10})
    expected_impact = notional * (0.10 * 100.0 * np.sqrt(notional / adv)) / 1e4
    sell = cost_model.trade_cost(notional, "sell", costs, adv=adv)
    assert sell["commission"] == pytest.approx(250.0)
    assert sell["stamp_duty"] == pytest.approx(500.0)
    assert sell["slippage"] == pytest.approx(500.0 + expected_impact)
    assert sell["total_cost"] == pytest.approx(250.0 + 500.0 + 500.0 + expected_impact)
    assert sell["cash_flow"] == pytest.approx(notional - sell["total_cost"])
    buy = cost_model.trade_cost(notional, "buy", costs, adv=adv)
    assert buy["stamp_duty"] == 0.0
    assert buy["cash_flow"] == pytest.approx(-(notional + buy["total_cost"]))
    no_adv = cost_model.trade_cost(notional, "buy", costs, adv=None)
    assert no_adv["slippage"] == pytest.approx(500.0)


def test_zero_cost_multiplier_removes_all_costs():
    costs = cost_model.scaled_costs({}, 0.0)
    outcome = cost_model.trade_cost(1_000_000.0, "sell", costs, adv=10_000_000.0)
    assert outcome["total_cost"] == pytest.approx(0.0)
    assert outcome["cash_flow"] == pytest.approx(1_000_000.0)


# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------


def test_performance_metrics_match_manual_definitions():
    dates = [f"2024-{month:02d}-{day:02d}" for month in range(1, 7) for day in (1, 8, 15, 22)]
    nav = pd.Series(np.linspace(1.0, 1.10, len(dates)), index=dates)
    benchmark = pd.Series(np.linspace(1.0, 1.04, len(dates)), index=dates)
    metrics = metric_model.compute_performance_metrics(nav, benchmark)

    returns = nav.pct_change().dropna()
    n_returns = len(returns)
    ann_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / n_returns) - 1
    annual_vol = returns.std(ddof=1) * np.sqrt(252)
    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_ann = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (252 / len(benchmark_returns)) - 1
    excess = (returns - benchmark_returns).dropna()
    tracking_error = excess.std(ddof=1) * np.sqrt(252)

    assert metrics["annualized_return"] == pytest.approx(float(ann_return), rel=1e-10)
    assert metrics["annualized_volatility"] == pytest.approx(float(annual_vol), rel=1e-10)
    assert metrics["sharpe"] == pytest.approx(float(ann_return / annual_vol), rel=1e-10)
    assert metrics["max_drawdown"] == pytest.approx(float((nav / nav.cummax() - 1).min()), rel=1e-10)
    assert metrics["annualized_excess_return"] == pytest.approx(float(ann_return - benchmark_ann), rel=1e-9)
    assert metrics["tracking_error"] == pytest.approx(float(tracking_error), rel=1e-9)
    assert metrics["information_ratio"] == pytest.approx(
        float((ann_return - benchmark_ann) / tracking_error), rel=1e-9
    )
    benchmark_metrics = metric_model.compute_benchmark_metrics(benchmark)
    assert benchmark_metrics["annualized_return"] == pytest.approx(float(benchmark_ann), rel=1e-10)


# ---------------------------------------------------------------------------
# Accounting invariants / independent rebuild
# ---------------------------------------------------------------------------


def test_ledger_holdings_and_nav_reconcile_with_independent_rebuild():
    pytest.importorskip("sklearn")
    from aicwf.verify.independent_backtest import (
        benchmark_nav,
        compare_metrics,
        compute_performance_metrics,
        normalise_ledger,
        rebuild_from_ledger,
    )
    from aicwf.verify.reconcile import reconcile_accounting

    pipeline = sup.cached_fixture_pipeline()
    ledger = _ledger_frame(pipeline)
    holdings = _holdings_frame(pipeline)
    equity_curve = pd.read_csv(sup.artifact_path(pipeline, "backtest", "equity_curve"))
    metrics_doc = _metrics_doc(pipeline)
    benchmark = pd.read_parquet(pipeline["paths"]["benchmark"])
    panel = pipeline["market"]["panel"]
    capital = float(metrics_doc["initial_capital"])

    assert len(ledger) > 0
    rebuild = rebuild_from_ledger(ledger, panel, benchmark=benchmark, initial_capital=capital, as_of="2024-12-31")
    assert not rebuild.issues
    # Cash is never negative and the ledger rebuild starts at 1.0 before inception.
    assert float(rebuild.path["cash"].min()) >= -1e-6
    assert float(rebuild.nav.iloc[0]) == pytest.approx(1.0)
    # Declared metrics are reproduced independently from the *inception* NAV path
    # (the verifier truncates its rebuild at the first execution date).
    first_execution = str(ledger["date"].min())
    recomputed = compute_performance_metrics(rebuild.nav.loc[first_execution:], benchmark_nav(benchmark))
    discrepancies = compare_metrics(metrics_doc["metrics"], recomputed, rel_tol=1e-9)
    assert discrepancies == []
    # Full accounting reconciliation (ledger <-> holdings <-> equity curve).
    outcome = reconcile_accounting(
        ledger=ledger,
        holdings=holdings,
        equity_curve=equity_curve,
        rebuild=rebuild,
        benchmark_nav_series=benchmark_nav(benchmark),
    )
    assert outcome["ok"], outcome["errors"][:5]

    # Shares are positive board lots; holdings weights equal shares*price/NAV.
    trades = normalise_ledger(ledger)
    assert (trades["shares"] > 0).all()
    assert (trades["shares"] % 100 == 0).all()
    assert (holdings["shares"] > 0).all()
    assert (holdings["shares"] % 100 == 0).all()
    totals = dict(zip(rebuild.path["date"], rebuild.path["total_value"]))
    for row in holdings.itertuples(index=False):
        expected = float(row.shares) * float(row.price) / float(totals[str(row.date)])
        assert float(row.weight) == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_holdings_respect_bounds_turnover_and_industry_neutrality():
    pipeline = sup.cached_fixture_pipeline()
    holdings = _holdings_frame(pipeline)
    metrics_doc = _metrics_doc(pipeline)
    constraints = metrics_doc["constraints"]
    max_weight = float(constraints["max_weight"])
    max_active = float(constraints["max_active_weight"])
    industry_limit = float(constraints["industry_active_limit"])
    turnover_limit = float(constraints["turnover_limit"])
    assert float(holdings["weight"].max()) <= max_weight + 1e-9
    assert float(holdings["active_weight"].abs().max()) <= max_active + 1e-9
    industry_active = holdings.groupby(["date", "industry"])["active_weight"].sum()
    assert float(industry_active.abs().max()) <= industry_limit + 1e-9
    weight_sums = holdings.groupby("date")["weight"].sum()
    assert float(weight_sums.max()) <= 1.0 + 1e-9
    assert float(weight_sums.min()) >= 0.99
    turnover = [item["turnover"] for item in metrics_doc["rebalances"] if not item.get("first_rebalance")]
    assert len(turnover) > 0
    assert max(turnover) <= turnover_limit + 1e-9


# ---------------------------------------------------------------------------
# Market-rule handling
# ---------------------------------------------------------------------------


def test_t_plus_one_and_board_lot_rules(tmp_path):
    pipeline = sup.run_pipeline(tmp_path / "run", with_robustness=False)
    ledger = _ledger_frame(pipeline)
    days = pipeline["market"]["trading_dates"]
    symbols = set(pipeline["market"]["codes"])

    assert set(ledger["date"]).issubset(set(days))
    assert set(ledger["code"]).issubset(symbols)
    # No same-day round trip of the same name (T+1 settlement).
    same_day = ledger.groupby(["date", "code"])["side"].nunique()
    assert int(same_day.max()) == 1
    # A sell can only consume shares bought on an earlier date.
    positions: dict[str, int] = {}
    bought_on: dict[tuple[str, str], int] = {}
    for row in ledger.sort_values(["date", "code"]).itertuples(index=False):
        key = (str(row.date), str(row.code))
        if row.side == "buy":
            positions[str(row.code)] = positions.get(str(row.code), 0) + int(row.shares)
            bought_on[key] = int(row.shares)
        else:
            assert positions.get(str(row.code), 0) >= int(row.shares)
            positions[str(row.code)] -= int(row.shares)
            assert bought_on.get(key, 0) == 0


def test_suspended_names_are_not_traded(tmp_path):
    ctx, market, paths, factor_output, baseline = _run_synthetic(tmp_path / "base")
    ledger = pd.read_parquet(baseline.artifacts["ledger"])
    first = ledger.iloc[0]
    trade_date, code = str(first["date"]), str(first["code"])
    panel = pd.read_parquet(paths["panel"])
    mask = (panel["date"] == trade_date) & (panel["code"] == code)
    panel.loc[mask, "suspended"] = True
    write_parquet_atomic(paths["panel"], panel, index=False)

    from aicwf.backtest.stage import run_backtest

    rerun = run_backtest(ctx)
    rerun_ledger = pd.read_parquet(rerun.artifacts["ledger"])
    blocked = rerun_ledger[(rerun_ledger["date"] == trade_date) & (rerun_ledger["code"] == code)]
    assert len(blocked) == 0


def test_limit_up_blocks_buys(tmp_path):
    ctx, market, paths, factor_output, baseline = _run_synthetic(tmp_path / "base")
    ledger = pd.read_parquet(baseline.artifacts["ledger"])
    buy = ledger[ledger["side"] == "buy"].iloc[0]
    panel = pd.read_parquet(paths["panel"])
    mask = (panel["date"] == str(buy["date"])) & (panel["code"] == str(buy["code"]))
    panel.loc[mask, "ret"] = 0.30  # limit-up move on the execution day (all boards)
    write_parquet_atomic(paths["panel"], panel, index=False)

    from aicwf.backtest.stage import run_backtest

    rerun = run_backtest(ctx)
    rerun_ledger = pd.read_parquet(rerun.artifacts["ledger"])
    blocked = rerun_ledger[
        (rerun_ledger["date"] == str(buy["date"]))
        & (rerun_ledger["code"] == str(buy["code"]))
        & (rerun_ledger["side"] == "buy")
    ]
    assert len(blocked) == 0
    metrics_doc = json.loads(Path(rerun.artifacts["metrics"]).read_text(encoding="utf-8"))
    assert metrics_doc["warnings"].get("limit_up_blocked", 0) >= 1


def test_limit_down_blocks_sells(tmp_path):
    ctx, market, paths, factor_output, baseline = _run_synthetic(tmp_path / "base")
    ledger = pd.read_parquet(baseline.artifacts["ledger"])
    sells = ledger[ledger["side"] == "sell"]
    assert len(sells) > 0
    sell = sells.iloc[0]
    panel = pd.read_parquet(paths["panel"])
    mask = (panel["date"] == str(sell["date"])) & (panel["code"] == str(sell["code"]))
    panel.loc[mask, "ret"] = -0.30  # limit-down move on the execution day (all boards)
    write_parquet_atomic(paths["panel"], panel, index=False)

    from aicwf.backtest.stage import run_backtest

    rerun = run_backtest(ctx)
    rerun_ledger = pd.read_parquet(rerun.artifacts["ledger"])
    blocked = rerun_ledger[
        (rerun_ledger["date"] == str(sell["date"]))
        & (rerun_ledger["code"] == str(sell["code"]))
        & (rerun_ledger["side"] == "sell")
    ]
    assert len(blocked) == 0
    metrics_doc = json.loads(Path(rerun.artifacts["metrics"]).read_text(encoding="utf-8"))
    assert metrics_doc["warnings"].get("limit_down_blocked", 0) >= 1


# ---------------------------------------------------------------------------
# Index-newcomer regression (constraint breach found on the live real run)
# ---------------------------------------------------------------------------


def _newcomer_market():
    """Synthetic market with one factor-ineligible newcomer carrying a large weight.

    Mirrors the live-run failure mode: a current index member that is not yet
    factor-eligible (listed < 250 trading days) has a large benchmark-proxy weight
    and rallies strongly, so carrying it at the compounded holding level would
    breach the weight / active / industry caps until it becomes eligible.
    """

    market = sup.make_market(n_codes=40, n_months=36)
    universe = market["universe"].copy()
    newcomer = market["codes"][0]
    universe["weight"] = 1.0 / (len(universe) - 1)
    universe.loc[universe["code"] == newcomer, "weight"] = 0.15
    universe["weight"] = universe["weight"] / universe["weight"].sum()
    panel = market["panel"].copy()
    mask = panel["code"] == newcomer
    # Never seasoned: stays factor-ineligible for the whole backtest.
    panel.loc[mask, "listed_days"] = 30.0
    # Persistent strong rally: a carried position would compound month after month.
    positions = panel.loc[mask].sort_values("date").index
    factors = np.power(1.012, np.arange(len(positions)))
    panel.loc[positions, "close"] = panel.loc[positions, "close"].to_numpy() * factors
    market = dict(market)
    market["universe"] = universe
    market["panel"] = panel
    market["newcomer"] = newcomer
    return market


def test_newcomer_holdings_respect_all_published_caps(tmp_path):
    market = _newcomer_market()
    newcomer = market["newcomer"]
    config_overrides = {
        "universe": {"min_listed_days": 250},
        "backtest": {
            "constraints": {
                "max_weight": 0.05,
                "max_active_weight": 0.025,
                "industry_active_limit": 0.03,
                "benchmark_weight_cap": 0.05,
                "turnover_limit": 0.60,
                "min_names": 3,
            }
        },
        "portfolio": {
            "defaults": {
                "max_weight": 0.05,
                "max_active_weight": 0.025,
                "te_target": 0.05,
                "turnover_limit": 0.60,
            },
            "constraints_bounds": {
                "max_weight": [0.005, 0.06],
                "max_active_weight": [0.001, 0.03],
                "te_target": [0.01, 0.12],
                "turnover_limit": [0.05, 1.0],
            },
            "benchmark_weight_cap": 0.05,
        },
    }
    hypothesis = sup.make_hypothesis(
        market,
        constraints={
            "max_weight": 0.05,
            "max_active_weight": 0.025,
            "te_target": 0.05,
            "turnover_limit": 0.60,
        },
    )
    ctx, market, paths = sup.make_run_context(
        tmp_path / "newcomer",
        market=market,
        config_overrides=config_overrides,
        hypothesis=hypothesis,
    )
    from aicwf.backtest.stage import run_backtest
    from aicwf.factors.stage import run_factor_build

    factor_output = run_factor_build(ctx)
    ctx.register("factor_build", factor_output)
    # Stage-level audit raises DataQualityError on a breach, so a successful call
    # already proves the invariants; assert the same numbers verification checks.
    backtest_output = run_backtest(ctx)
    holdings = pd.read_parquet(backtest_output.artifacts["holdings"])
    cfg = ctx.config["strategy"]
    max_weight = float(cfg["backtest"]["constraints"]["max_weight"])
    max_active = float(cfg["backtest"]["constraints"]["max_active_weight"])
    industry_limit = float(cfg["backtest"]["constraints"]["industry_active_limit"])
    assert len(holdings) > 0
    assert float(holdings["weight"].max()) <= max_weight + 1e-6
    assert float(holdings["active_weight"].abs().max()) <= max_active + 1e-6
    industry_active = holdings.groupby(["date", "industry"])["active_weight"].sum()
    assert float(industry_active.abs().max()) <= industry_limit + 1e-6
    # The newcomer is a real holding that is clamped to its benchmark proxy weight,
    # not carried at a compounded level.
    newcomer_rows = holdings[holdings["code"] == newcomer]
    assert len(newcomer_rows) > 0
    assert float(newcomer_rows["weight"].max()) <= max_weight + 1e-6
    assert float(newcomer_rows["weight"].max()) <= float(
        newcomer_rows["benchmark_weight"].max()
    ) + max_active + 1e-6
    # The first holdings date is the first execution date and is already compliant.
    first_date = holdings["date"].min()
    first_rows = holdings[holdings["date"] == first_date]
    assert float(first_rows["weight"].max()) <= max_weight + 1e-6


# ---------------------------------------------------------------------------
# Point-in-time invariance and determinism
# ---------------------------------------------------------------------------


def test_metrics_and_ledger_are_invariant_to_future_data(tmp_path):
    overrides = {"backtest": {"start_date": "2023-05-01", "end_date": "2024-06-28"}}
    ctx_a, market, paths_a, _, baseline = _run_synthetic(tmp_path / "a", config_overrides=overrides)
    panel = pd.read_parquet(paths_a["panel"])
    future_mask = panel["date"] > "2024-06-28"
    assert bool(future_mask.any())
    panel.loc[future_mask, "close"] = panel.loc[future_mask, "close"] * 2.0
    panel.loc[future_mask, "ret"] = 0.05
    write_parquet_atomic(paths_a["panel"], panel, index=False)

    from aicwf.backtest.stage import run_backtest
    from aicwf.factors.stage import run_factor_build

    factor_output = run_factor_build(ctx_a)
    ctx_a.register("factor_build", factor_output)
    rerun = run_backtest(ctx_a)
    baseline_metrics = _metrics_doc({"outputs": {"backtest": baseline}})
    rerun_metrics = json.loads(Path(rerun.artifacts["metrics"]).read_text(encoding="utf-8"))
    assert baseline_metrics["metrics"]["annualized_excess_return"] == rerun_metrics["metrics"]["annualized_excess_return"]
    assert baseline_metrics["metrics"]["tracking_error"] == rerun_metrics["metrics"]["tracking_error"]
    baseline_ledger = pd.read_parquet(baseline.artifacts["ledger"])
    rerun_ledger = pd.read_parquet(rerun.artifacts["ledger"])
    assert hash_dataframe(baseline_ledger) == hash_dataframe(rerun_ledger)


def test_backtest_is_deterministic(tmp_path):
    pipeline = sup.run_pipeline(tmp_path / "first", with_robustness=False)
    second = sup.run_pipeline(tmp_path / "second", with_robustness=False)
    for logical in ("ledger", "holdings", "metrics", "equity_curve"):
        first_path = sup.artifact_path(pipeline, "backtest", logical)
        second_path = sup.artifact_path(second, "backtest", logical)
        assert sup.content_hash(first_path) == sup.content_hash(second_path)


def test_equity_curve_identities():
    pipeline = sup.cached_fixture_pipeline()
    curve = pd.read_csv(sup.artifact_path(pipeline, "backtest", "equity_curve"))
    # The curve starts at the first execution date with the post-cost NAV (still
    # normalised by the initial capital, so it is very close to one).
    assert abs(float(curve["nav"].iloc[0]) - 1.0) < 0.05
    assert curve["date"].is_monotonic_increasing
    recomputed_return = curve["nav"].pct_change()
    np.testing.assert_allclose(
        curve["daily_return"].to_numpy(dtype=float)[1:],
        recomputed_return.to_numpy(dtype=float)[1:],
        rtol=1e-5,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        curve["excess_return"].dropna().to_numpy(dtype=float),
        (curve["daily_return"] - curve["benchmark_return"]).dropna().to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        curve["active_nav"].dropna().to_numpy(dtype=float),
        (curve["nav"] / curve["benchmark_nav"]).dropna().to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-12,
    )
