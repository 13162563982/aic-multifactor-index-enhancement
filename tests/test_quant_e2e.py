"""End-to-end mini-fixture pipeline and artifact-contract integration tests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_quant_support as sup  # noqa: E402

from aicwf.factors import library as factor_library  # noqa: E402

LEDGER_COLUMNS = [
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
]
HOLDINGS_COLUMNS = [
    "date",
    "code",
    "shares",
    "price",
    "weight",
    "benchmark_weight",
    "active_weight",
    "industry",
]
EQUITY_COLUMNS = [
    "date",
    "nav",
    "benchmark_nav",
    "active_nav",
    "daily_return",
    "benchmark_return",
    "excess_return",
    "drawdown",
]
TARGET_WEIGHT_COLUMNS = ["date", "code", "weight", "benchmark_weight", "active_weight", "industry"]
ORDER_COLUMNS = ["date", "code", "side", "shares", "weight_change", "notional", "participation_pct"]


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_mini_fixture_pipeline_end_to_end():
    assert sup.mini_fixture_available(), "mini snapshot fixture is missing"
    pipeline = sup.cached_fixture_pipeline(with_robustness=True)
    outputs = pipeline["outputs"]
    assert set(outputs) == {"factor_build", "backtest", "robustness", "portfolio"}
    for stage, output in outputs.items():
        assert output.artifacts, f"stage {stage} declared no artifacts"
        for logical, relative in output.artifacts.items():
            path = Path(relative)
            if not path.is_absolute():
                path = sup.ROOT / path
            assert path.exists(), f"declared artifact {stage}:{logical} missing on disk"

    factor_panel = pd.read_parquet(sup.artifact_path(pipeline, "factor_build", "factor_panel"))
    expected = ["date", "code"]
    for factor_id in factor_library.FACTOR_IDS:
        expected.extend([f"{factor_id}_raw", f"{factor_id}_z"])
    expected.extend(["composite_score", "eligible"])
    assert list(factor_panel.columns) == expected
    factor_report = json.loads(
        sup.artifact_path(pipeline, "factor_build", "factor_report").read_text(encoding="utf-8")
    )
    for key in (
        "coverage",
        "rank_ic",
        "ic_t_stat",
        "autocorrelation",
        "correlation_matrix",
        "composite_weights",
        "warnings",
    ):
        assert key in factor_report

    ledger = pd.read_parquet(sup.artifact_path(pipeline, "backtest", "ledger"))
    holdings = pd.read_parquet(sup.artifact_path(pipeline, "backtest", "holdings"))
    equity = pd.read_csv(sup.artifact_path(pipeline, "backtest", "equity_curve"))
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert list(holdings.columns) == HOLDINGS_COLUMNS
    assert list(equity.columns) == EQUITY_COLUMNS
    assert len(ledger) > 0 and len(holdings) > 0 and len(equity) > 1

    metrics_doc = json.loads(
        sup.artifact_path(pipeline, "backtest", "metrics").read_text(encoding="utf-8")
    )
    for key in ("metrics", "benchmark_metrics", "by_year", "constraints", "config_hash"):
        assert key in metrics_doc
    assert metrics_doc["initial_capital"] > 0
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
    ):
        assert key in metrics_doc["metrics"]
        assert key in metrics_doc["benchmark_metrics"] or key in {
            "annualized_excess_return",
            "tracking_error",
            "information_ratio",
        }

    robustness = json.loads(
        sup.artifact_path(pipeline, "robustness", "robustness").read_text(encoding="utf-8")
    )
    for key in (
        "subperiods",
        "cost_sensitivity",
        "leave_one_out",
        "parameter_grid",
        "placebo",
        "bias_stress",
        "deflated_sharpe",
        "summary",
    ):
        assert key in robustness
    assert "pass" in robustness["summary"] and "concerns" in robustness["summary"]
    checks = json.loads(
        sup.artifact_path(pipeline, "robustness", "robustness_checks").read_text(encoding="utf-8")
    )
    assert isinstance(checks, list) and checks
    for check in checks:
        assert {"name", "status", "value", "threshold", "detail"} <= set(check)
        assert check["status"] in {"pass", "warn", "fail"}

    target = pd.read_parquet(sup.artifact_path(pipeline, "portfolio", "target_weights"))
    orders = pd.read_csv(sup.artifact_path(pipeline, "portfolio", "orders"))
    risk_report = json.loads(
        sup.artifact_path(pipeline, "portfolio", "risk_report").read_text(encoding="utf-8")
    )
    assert list(target.columns) == TARGET_WEIGHT_COLUMNS
    assert list(orders.columns) == ORDER_COLUMNS
    for key in ("ex_ante_te", "industry_active", "factor_active", "turnover", "n_names", "optimizer", "constraints"):
        assert key in risk_report

    backtest_summary = outputs["backtest"].summary
    for key in (
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
        "max_drawdown",
        "rebalance_count",
    ):
        assert key in backtest_summary
    robustness_summary = outputs["robustness"].summary
    for key in ("passed_checks", "failed_checks", "concerns"):
        assert key in robustness_summary
    portfolio_summary = outputs["portfolio"].summary
    for key in ("date", "n_names", "ex_ante_te", "turnover", "fallback_used"):
        assert key in portfolio_summary
    assert portfolio_summary["date"] == risk_report["date"]
    assert portfolio_summary["fallback_used"] == risk_report["optimizer"]["fallback_used"]


# ---------------------------------------------------------------------------
# Inception / window alignment
# ---------------------------------------------------------------------------


def test_equity_curve_starts_at_first_rebalance():
    """The return series must start at the first execution date, not data start."""

    pipeline = sup.cached_fixture_pipeline()
    ledger = pd.read_parquet(sup.artifact_path(pipeline, "backtest", "ledger"))
    equity = pd.read_csv(sup.artifact_path(pipeline, "backtest", "equity_curve"))
    metrics_doc = json.loads(
        sup.artifact_path(pipeline, "backtest", "metrics").read_text(encoding="utf-8")
    )
    first_execution = str(ledger["date"].min())
    assert metrics_doc["metrics"]["start_date"] == first_execution
    assert equity["date"].iloc[0] == first_execution
    assert equity["date"].iloc[-1] == metrics_doc["metrics"]["end_date"]

    panel_dates = sorted(pipeline["market"]["panel"]["date"].astype(str).unique().tolist())
    expected_window = [day for day in panel_dates if first_execution <= day <= "2024-12-31"]
    assert len(equity) == len(expected_window)
    assert equity["date"].tolist() == expected_window
    benchmark = pd.read_parquet(pipeline["paths"]["benchmark"])
    benchmark["date"] = benchmark["date"].astype(str).str[:10]
    benchmark_window = benchmark[
        (benchmark["date"] >= first_execution) & (benchmark["date"] <= "2024-12-31")
    ]
    assert len(equity) == len(benchmark_window)
    assert int(metrics_doc["metrics"]["n_days"]) == len(expected_window)
    assert int(metrics_doc["metrics"]["n_returns"]) == len(expected_window) - 1


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------


def test_pipeline_is_deterministic_across_runs(tmp_path):
    first = sup.run_pipeline(tmp_path / "first", with_robustness=True)
    second = sup.run_pipeline(tmp_path / "second", with_robustness=True)
    for stage, logical in (
        ("factor_build", "factor_panel"),
        ("factor_build", "factor_report"),
        ("backtest", "ledger"),
        ("backtest", "holdings"),
        ("backtest", "metrics"),
        ("backtest", "equity_curve"),
        ("robustness", "robustness"),
        ("robustness", "robustness_checks"),
        ("portfolio", "target_weights"),
        ("portfolio", "orders"),
        ("portfolio", "risk_report"),
    ):
        first_hash = sup.content_hash(sup.artifact_path(first, stage, logical))
        second_hash = sup.content_hash(sup.artifact_path(second, stage, logical))
        assert first_hash == second_hash, f"{stage}:{logical} is not deterministic"


def test_robustness_is_deterministic_on_rerun():
    pipeline = sup.cached_fixture_pipeline()
    from aicwf.backtest.stage import run_robustness

    rerun = run_robustness(pipeline["ctx"])
    cached_hash = sup.content_hash(sup.artifact_path(pipeline, "robustness", "robustness"))
    rerun_hash = sup.content_hash(Path(rerun.artifacts["robustness"]))
    assert cached_hash == rerun_hash


# ---------------------------------------------------------------------------
# Independent reconstruction of the published numbers
# ---------------------------------------------------------------------------


def test_declared_metrics_match_independent_reconstruction():
    from aicwf.verify.independent_backtest import (
        benchmark_nav,
        compare_metrics,
        compute_benchmark_metrics,
        compute_performance_metrics,
        rebuild_from_ledger,
    )

    pipeline = sup.cached_fixture_pipeline()
    ledger = pd.read_parquet(sup.artifact_path(pipeline, "backtest", "ledger"))
    metrics_doc = json.loads(
        sup.artifact_path(pipeline, "backtest", "metrics").read_text(encoding="utf-8")
    )
    benchmark = pd.read_parquet(pipeline["paths"]["benchmark"])
    rebuild = rebuild_from_ledger(
        ledger,
        pipeline["market"]["panel"],
        benchmark=benchmark,
        initial_capital=float(metrics_doc["initial_capital"]),
        as_of="2024-12-31",
    )
    benchmark_series = benchmark_nav(benchmark)
    # The equity curve (and therefore the declared metrics) starts at the first
    # execution date; the verifier truncates its ledger rebuild the same way.
    first_execution = str(ledger["date"].min())
    recomputed = compute_performance_metrics(rebuild.nav.loc[first_execution:], benchmark_series)
    assert compare_metrics(metrics_doc["metrics"], recomputed, rel_tol=1e-9) == []
    recomputed_benchmark = compute_benchmark_metrics(benchmark_series)
    assert compare_metrics(
        metrics_doc["benchmark_metrics"],
        recomputed_benchmark,
        keys=tuple(metrics_doc["benchmark_metrics"]),
        rel_tol=1e-9,
        section="benchmark_metrics",
    ) == []


# ---------------------------------------------------------------------------
# Runtime budget on the mini fixture
# ---------------------------------------------------------------------------


def test_mini_fixture_runtime_is_bounded(tmp_path):
    from aicwf.backtest.stage import run_backtest, run_robustness
    from aicwf.factors.stage import run_factor_build
    from aicwf.portfolio.stage import run_portfolio

    ctx, market, paths = sup.fixture_run_context(tmp_path / "timed")
    started = time.perf_counter()
    factor_output = run_factor_build(ctx)
    ctx.register("factor_build", factor_output)
    backtest_output = run_backtest(ctx)
    ctx.register("backtest", backtest_output)
    portfolio_output = run_portfolio(ctx)
    ctx.register("portfolio", portfolio_output)
    core_seconds = time.perf_counter() - started
    assert core_seconds < 20.0, f"factor+backtest+portfolio took {core_seconds:.1f}s"
    started = time.perf_counter()
    robustness_output = run_robustness(ctx)
    robustness_seconds = time.perf_counter() - started
    assert robustness_output.summary["passed_checks"] >= 1
    assert robustness_seconds < 30.0, f"robustness took {robustness_seconds:.1f}s"
