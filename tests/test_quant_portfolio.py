"""Projection, optimiser, constraint-clamping, fallback and order tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_quant_support as sup  # noqa: E402

from aicwf.factors import common as factor_common  # noqa: E402
from aicwf.portfolio import optimizer as portfolio_optimizer  # noqa: E402
from aicwf.portfolio import orders as order_module  # noqa: E402
from aicwf.portfolio import projection as portfolio_projection  # noqa: E402
from aicwf.portfolio import stage as portfolio_stage  # noqa: E402


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_box_sum_projection_properties():
    rng = np.random.default_rng(11)
    n = 40
    vector = rng.normal(size=n)
    lower = np.zeros(n)
    upper = np.full(n, 0.2)
    projected = portfolio_projection.project_uniform_box(vector, lower, upper, total=1.0)
    assert projected.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(projected >= lower - 1e-12)
    assert np.all(projected <= upper + 1e-12)
    # Euclidean projection optimality: moving mass towards the target must not help.
    assert (vector - projected) @ (projected) >= -1e-9


def test_industry_neutral_projection_matches_reference_solution():
    from scipy.optimize import minimize

    rng = np.random.default_rng(5)
    groups = ["A", "A", "A", "B", "B", "B"]
    n = len(groups)
    vector = rng.uniform(0.0, 0.5, size=n)
    wb = rng.uniform(0.05, 0.3, size=n)
    wb = wb / wb.sum()
    targets = np.array([wb[[index for index, group in enumerate(groups) if group == name]].sum() for name in ("A", "B")])
    lower = np.zeros(n)
    upper = np.full(n, 0.35)
    result = portfolio_projection.project_group_box(vector, lower, upper, groups, targets)

    assert result.status == "exact"
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
    for group, target in zip(("A", "B"), targets):
        mask = np.array([name == group for name in groups])
        assert result.weights[mask].sum() == pytest.approx(float(target), abs=1e-10)

    def objective(weights):
        return 0.5 * float(np.sum((weights - vector) ** 2))

    def gradient(weights):
        return weights - vector

    constraints = [{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}]
    for group, target in zip(("A", "B"), targets):
        mask = np.array([name == group for name in groups], dtype=bool)
        constraints.append(
            {"type": "eq", "fun": (lambda weights, mask=mask, target=target: float(weights[mask].sum() - target))}
        )
    reference = minimize(
        objective,
        x0=result.weights,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 0.35)] * n,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert reference.success
    np.testing.assert_allclose(result.weights, reference.x, atol=1e-6)


def test_projection_relaxes_infeasible_group_targets():
    groups = ["A", "A", "B", "B"]
    vector = np.array([0.5, 0.5, 0.5, 0.5])
    lower = np.zeros(4)
    upper = np.full(4, 0.3)
    targets = np.array([0.9, 0.1])  # group A target 0.9 is infeasible (max 0.6)
    result = portfolio_projection.project_group_box(vector, lower, upper, groups, targets)
    assert result.status == "relaxed_group_targets"
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(result.weights <= upper + 1e-12)
    assert np.all(result.weights >= lower - 1e-12)
    # Group A is pushed to its maximum and the remaining mass is shifted to B.
    mask_a = np.array([group == "A" for group in groups])
    assert result.weights[mask_a].sum() == pytest.approx(0.6, abs=1e-10)


# ---------------------------------------------------------------------------
# Mean-variance optimiser
# ---------------------------------------------------------------------------


def _synthetic_optimizer_problem(seed: int = 21, n: int = 24):
    rng = np.random.default_rng(seed)
    codes = [f"C{index:02d}" for index in range(n)]
    industry = pd.Series([["Banks", "Tech", "Consumer", "Materials"][index % 4] for index in range(n)], index=codes)
    wb = pd.Series(rng.uniform(0.5, 1.5, size=n), index=codes)
    wb = wb / wb.sum()
    returns = pd.DataFrame(
        rng.normal(0.0, 0.015, size=(160, n)),
        columns=codes,
    )
    covariance = portfolio_optimizer.ledoit_wolf_covariance(returns, annualization=252).covariance
    alpha = pd.Series(rng.normal(size=n), index=codes)
    constraints = {
        "max_weight": 0.20,
        "max_active_weight": 0.15,
        "industry_active_limit": 0.10,
        "turnover_limit": 1.0,
        "te_target": 0.06,
        "cash_buffer": 0.0,
        "min_names": 0,
        "benchmark_weight_cap": 0.20,
        "weight_margin": 0.985,
    }
    return alpha, wb, covariance, industry, constraints


def test_te_bisection_hits_target_and_respects_constraints():
    alpha, wb, covariance, industry, constraints = _synthetic_optimizer_problem()
    result = portfolio_optimizer.optimize_weights(
        alpha=alpha,
        benchmark=wb,
        covariance=covariance,
        industry=industry,
        constraints=constraints,
        optimizer_cfg={"lambda_bisection_steps": 24, "max_iter": 200, "tol": 1e-9},
    )
    assert result.status == "ok"
    assert result.ex_ante_te == pytest.approx(constraints["te_target"], abs=0.01)
    weights = result.weights
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(weights.max()) <= constraints["max_weight"] * constraints["weight_margin"] + 1e-9
    active = weights - wb
    assert float(active.abs().max()) <= constraints["max_active_weight"] * constraints["weight_margin"] + 1e-9
    industry_active = weights.groupby(industry).sum() - wb.groupby(industry).sum()
    assert float(industry_active.abs().max()) < 1e-8
    # Ex-ante TE recomputed from the covariance matrix agrees with the optimiser.
    recomputed = portfolio_optimizer.ex_ante_te(active.to_numpy(dtype=float), covariance.to_numpy(dtype=float))
    assert recomputed == pytest.approx(result.ex_ante_te, rel=1e-9)


def test_optimizer_is_deterministic():
    alpha, wb, covariance, industry, constraints = _synthetic_optimizer_problem()
    first = portfolio_optimizer.optimize_weights(
        alpha=alpha, benchmark=wb, covariance=covariance, industry=industry, constraints=constraints
    )
    second = portfolio_optimizer.optimize_weights(
        alpha=alpha, benchmark=wb, covariance=covariance, industry=industry, constraints=constraints
    )
    np.testing.assert_array_equal(first.weights.to_numpy(), second.weights.to_numpy())


def test_rank_proportional_fallback_is_valid():
    alpha, wb, covariance, industry, constraints = _synthetic_optimizer_problem()
    result = portfolio_optimizer.rank_proportional_fallback(
        alpha=alpha, benchmark=wb, industry=industry, constraints=constraints
    )
    assert result.fallback_used is True
    weights = result.weights
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(weights.max()) <= constraints["max_weight"] * constraints["weight_margin"] + 1e-9
    active = weights - wb
    assert float(active.abs().max()) <= constraints["max_active_weight"] * constraints["weight_margin"] + 1e-9
    industry_active = weights.groupby(industry).sum() - wb.groupby(industry).sum()
    assert float(industry_active.abs().max()) < 1e-8


# ---------------------------------------------------------------------------
# Constraint clamping
# ---------------------------------------------------------------------------


def test_hypothesis_constraints_are_clamped_to_config_bounds():
    cfg = factor_common.read_strategy_file()
    hypothesis = {
        "constraints": {
            "max_weight": 0.50,
            "max_active_weight": 0.20,
            "te_target": 0.50,
            "turnover_limit": 0.90,
        }
    }
    values, info = factor_common.clamp_hypothesis_constraints(hypothesis, cfg)
    bounds = factor_common.DEFAULT_CONSTRAINT_BOUNDS
    assert values["max_weight"] == pytest.approx(0.03)
    assert values["max_active_weight"] == pytest.approx(0.015)
    assert values["te_target"] == pytest.approx(0.03)
    assert values["turnover_limit"] == pytest.approx(0.60)
    assert set(info["clamped"]) == {"max_weight", "max_active_weight", "te_target", "turnover_limit"}
    assert info["source"]["max_weight"] == "hypothesis"
    assert len(info["notes"]) == 4
    for key, (low, high) in bounds.items():
        assert values[key] >= low - 1e-15 and values[key] <= high + 1e-15


def test_constraints_inside_bounds_pass_through_unchanged():
    cfg = factor_common.read_strategy_file()
    hypothesis = {
        "constraints": {
            "max_weight": 0.03,
            "max_active_weight": 0.015,
            "te_target": 0.03,
            "turnover_limit": 0.30,
        }
    }
    values, info = factor_common.clamp_hypothesis_constraints(hypothesis, cfg)
    assert values["max_weight"] == pytest.approx(0.03)
    assert values["te_target"] == pytest.approx(0.03)
    assert info["clamped"] == {}
    assert all(source == "hypothesis" for source in info["source"].values())


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_orders_respect_board_lots_and_participation_cap():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    current_shares = {"A": 1000}
    prices = pd.Series({"A": 10.0, "B": 20.0, "C": 5.0})
    amount = pd.Series({"A": 1e5, "B": 1e6, "C": 1e6})
    orders = order_module.build_orders(
        date="2024-12-31",
        target_weights=weights,
        current_shares=current_shares,
        nav=1_000_000.0,
        prices=prices,
        amount=amount,
        lot_size=100,
        participation_cap=0.10,
    )
    assert list(orders.columns) == list(order_module.ORDER_COLUMNS)
    assert set(orders["side"]).issubset({"buy", "sell"})
    assert (orders["shares"] % 100 == 0).all()
    assert (orders["shares"] > 0).all()
    assert float(orders["participation_pct"].max()) <= 0.10 + 1e-12
    for row in orders.itertuples(index=False):
        assert float(row.notional) == pytest.approx(float(row.shares) * float(prices[row.code]))
        assert abs(float(row.weight_change)) == pytest.approx(
            float(row.shares) * float(prices[row.code]) / 1_000_000.0, rel=1e-9
        )
    # A is capped by participation (10% of 100k traded amount at price 10 -> 1000 shares).
    row_a = orders[orders["code"] == "A"].iloc[0]
    assert int(row_a["shares"]) == 1000
    # No same-day round trip of the same name.
    assert orders.groupby("code")["side"].nunique().max() == 1


# ---------------------------------------------------------------------------
# Portfolio stage on the frozen mini fixture
# ---------------------------------------------------------------------------


def test_portfolio_stage_target_weights_and_risk_report():
    pipeline = sup.cached_fixture_pipeline()
    target = pd.read_parquet(sup.artifact_path(pipeline, "portfolio", "target_weights"))
    risk = json.loads(
        sup.artifact_path(pipeline, "portfolio", "risk_report").read_text(encoding="utf-8")
    )
    metrics_doc = json.loads(
        sup.artifact_path(pipeline, "backtest", "metrics").read_text(encoding="utf-8")
    )
    constraints = metrics_doc["constraints"]
    assert list(target.columns) == [
        "date",
        "code",
        "weight",
        "benchmark_weight",
        "active_weight",
        "industry",
    ]
    assert len(target) > 0
    assert float(target["weight"].sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(target["weight"].max()) <= float(constraints["max_weight"]) + 1e-9
    assert float(target["active_weight"].abs().max()) <= float(constraints["max_active_weight"]) + 1e-9
    industry_active = target.groupby("industry")["active_weight"].sum()
    assert float(industry_active.abs().max()) <= float(constraints["industry_active_limit"]) + 1e-9
    for key in (
        "ex_ante_te",
        "industry_active",
        "factor_active",
        "turnover",
        "n_names",
        "optimizer",
        "constraints",
    ):
        assert key in risk
    for key in ("status", "iterations", "te_lambda", "fallback_used"):
        assert key in risk["optimizer"]
    assert risk["optimizer"]["fallback_used"] is False
    assert risk["ex_ante_te"] == pytest.approx(risk["te_target"], abs=0.02)
    assert risk["constraints"]["bounds"]
    summary = pipeline["outputs"]["portfolio"].summary
    assert summary["date"] == risk["date"]
    assert summary["n_names"] >= 1
    assert summary["fallback_used"] is False

    orders = pd.read_csv(sup.artifact_path(pipeline, "portfolio", "orders"))
    assert list(orders.columns) == list(order_module.ORDER_COLUMNS)
    assert orders["date"].astype(str).str[:10].eq(risk["date"]).all()


def test_portfolio_stage_uses_fallback_when_optimizer_fails(monkeypatch):
    pipeline = sup.cached_fixture_pipeline()
    inputs = portfolio_stage.load_portfolio_inputs(pipeline["ctx"])

    def _boom(**kwargs):
        raise RuntimeError("forced optimizer failure for the fallback test")

    monkeypatch.setattr(portfolio_stage.portfolio_optimizer, "optimize_weights", _boom)
    built = portfolio_stage.build_portfolio(inputs)
    assert built["risk_report"]["optimizer"]["fallback_used"] is True
    assert built["risk_report"]["optimizer"]["status"] == "fallback_rank_proportional"
    weights = built["weights"]
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(weights.max()) <= float(inputs["constraints"]["max_weight"]) + 1e-9
    assert any("fallback" in warning for warning in built["risk_report"]["warnings"])


def test_portfolio_build_is_deterministic():
    pipeline = sup.cached_fixture_pipeline()
    inputs = portfolio_stage.load_portfolio_inputs(pipeline["ctx"])
    first = portfolio_stage.build_portfolio(inputs)
    second = portfolio_stage.build_portfolio(inputs)
    np.testing.assert_array_equal(
        first["weights"].to_numpy(dtype=float), second["weights"].to_numpy(dtype=float)
    )
    pd.testing.assert_frame_equal(first["target_weights"], second["target_weights"])
    pd.testing.assert_frame_equal(first["orders"], second["orders"])
