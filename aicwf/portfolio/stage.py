"""Stage entrypoint: constrained final portfolio construction (``run_portfolio``).

The stage builds the *final* index-enhancement portfolio for the latest decision
date of the backtest window:

1. read the composite alpha from ``factor_build`` (formed at the previous month
   end, i.e. lagged one month) and the current holdings / NAV from ``backtest``;
2. clamp the hypothesis-proposed constraints (weight cap, active cap, TE target,
   turnover cap) to the bounds published in ``config/strategy.yaml``;
3. estimate the covariance of trailing daily returns with Ledoit-Wolf (diagonal
   fallback when there is not enough data);
4. maximise ``alpha'w - lambda/2 (w-wb)' Sigma (w-wb)`` with a projected-gradient
   ascent where every iterate is exactly projected onto the box + industry-neutral
   set, bisecting on ``lambda`` to hit the ex-ante tracking-error target;
5. apply the turnover cap by blending towards the current feasible portfolio;
6. fall back to deterministic rank-proportional clipped weights when the
   optimiser fails or the problem is degenerate (``fallback_used = true``);
7. write ``target_weights.parquet``, ``orders.csv`` and ``risk_report.json``.

Artifact conventions and the (documented) benchmark-proxy policy are shared with
the backtest stage; all values are deterministic and no wall clock is written.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import DataQualityError, RunContext, StageOutput, ValidationError
from aicwf.factors import common
from aicwf.factors import library as factor_library
from aicwf.portfolio import optimizer as portfolio_optimizer
from aicwf.portfolio import orders as order_module
from aicwf.portfolio.projection import industry_active_table, project_industry_neutral_box

DEFAULT_BOARD_LIMITS: dict[str, float] = {
    "main": 0.10,
    "chinext": 0.20,
    "star": 0.20,
    "bse": 0.30,
    "st": 0.05,
    "limit_tolerance": 0.005,
}
GROWTH_BOARDS = {"chinext", "gem", "star", "kcb", "sse_star", "szse_gem", "300", "688", "cyb"}
BSE_BOARDS = {"bse", "bj", "beijing", "北交所", "北证"}


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _read_upstream_parquet(
    ctx: RunContext, stage: str, candidates: tuple[str, ...], filename: str
) -> pd.DataFrame | None:
    try:
        path = common.find_upstream_artifact(ctx, stage, candidates, filename=filename)
    except ValidationError:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - absent/corrupt upstream artifact -> fallback
        return None


def _read_upstream_csv(
    ctx: RunContext, stage: str, candidates: tuple[str, ...], filename: str
) -> pd.DataFrame | None:
    try:
        path = common.find_upstream_artifact(ctx, stage, candidates, filename=filename)
    except ValidationError:
        return None
    try:
        return pd.read_csv(path, dtype={"date": str})
    except Exception:  # noqa: BLE001
        return None


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(index="date", columns="code", values=column, aggfunc="last")


def _board_limit(board: Any, is_st: bool) -> float:
    if is_st:
        return DEFAULT_BOARD_LIMITS["st"]
    text = str(board or "").strip().lower()
    if text in GROWTH_BOARDS or text in {"创业板", "科创板"}:
        return DEFAULT_BOARD_LIMITS["chinext"]
    if text in BSE_BOARDS:
        return DEFAULT_BOARD_LIMITS["bse"]
    return DEFAULT_BOARD_LIMITS["main"]


def load_portfolio_inputs(ctx: RunContext) -> dict[str, Any]:
    cfg = common.load_strategy_config(ctx)
    panel = common.load_panel(ctx)
    factor_path = common.find_upstream_artifact(
        ctx, "factor_build", ("factor_panel", "factor_panel.parquet"), filename="factor_panel.parquet"
    )
    factor_panel = pd.read_parquet(factor_path)
    if "composite_score" not in factor_panel.columns:
        raise ValidationError("factor_panel.parquet has no composite_score column")
    factor_panel = factor_panel.copy()
    factor_panel["date"] = factor_panel["date"].astype(str).str[:10]
    factor_panel["code"] = factor_panel["code"].astype(str)
    universe = common.load_snapshot_table(ctx, "universe", columns=("code",))
    hypothesis = common.load_hypothesis(ctx)
    constraints, constraint_info = common.clamp_hypothesis_constraints(hypothesis, cfg)
    holdings = _read_upstream_parquet(ctx, "backtest", ("holdings", "holdings.parquet"), "holdings.parquet")
    equity_curve = _read_upstream_csv(ctx, "backtest", ("equity_curve", "equity_curve.csv"), "equity_curve.csv")
    return {
        "cfg": cfg,
        "panel": panel,
        "factor_panel": factor_panel,
        "universe": universe,
        "hypothesis": hypothesis,
        "constraints": constraints,
        "constraint_info": constraint_info,
        "holdings": holdings,
        "equity_curve": equity_curve,
        "as_of": str(ctx.as_of)[:10],
    }


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def _universe_frame(inputs: Mapping[str, Any], codes: list[str]) -> pd.DataFrame:
    universe = inputs["universe"]
    if universe is None:
        frame = pd.DataFrame({"code": codes})
        frame["weight"] = 1.0 / max(len(codes), 1)
        frame["industry"] = "UNKNOWN"
        frame["is_st"] = False
        frame["board"] = "main"
        return frame
    frame = pd.DataFrame(universe).copy()
    frame["code"] = frame["code"].astype(str)
    frame = frame.drop_duplicates(subset=["code"], keep="last")
    return frame


def build_portfolio(
    inputs: Mapping[str, Any],
    *,
    decision_date: str | None = None,
) -> dict[str, Any]:
    cfg = dict(inputs["cfg"])
    constraints = dict(inputs["constraints"])
    panel = inputs["panel"].copy()
    panel["date"] = panel["date"].astype(str).str[:10]
    panel["code"] = panel["code"].astype(str)
    panel = panel[panel["date"] <= inputs["as_of"]]
    if panel.empty:
        raise DataQualityError("panel has no rows up to as_of")
    panel_dates = sorted(panel["date"].unique().tolist())
    decision = decision_date or panel_dates[-1]
    if decision not in panel_dates:
        raise DataQualityError(f"decision date {decision} is not a trading day in the panel")

    factor_panel = inputs["factor_panel"]
    factor_dates = sorted(factor_panel["date"].unique().tolist())
    cutoff = common.previous_month_end(decision)
    signal_date = common.latest_on_or_before(factor_dates, cutoff)
    if signal_date is None:
        raise DataQualityError(
            f"no factor score available at or before {cutoff}; cannot build a lagged portfolio"
        )
    eligible_rows = factor_panel[factor_panel["date"] == signal_date].copy()

    close = _pivot(panel, "close").sort_index()
    valuation_close = close.ffill()
    ret = _pivot(panel, "ret").sort_index() if "ret" in panel.columns else close.pct_change()
    amount = _pivot(panel, "amount").sort_index()
    in_universe_pivot = (
        panel.assign(_u=panel["in_universe"].astype(bool))
        .pivot_table(index="date", columns="code", values="_u", aggfunc="last")
        .reindex(index=panel_dates)
    )
    in_universe_values = in_universe_pivot.to_numpy()
    in_universe = pd.DataFrame(
        np.where(pd.isna(in_universe_values), False, in_universe_values),
        index=in_universe_pivot.index,
        columns=in_universe_pivot.columns,
    ).astype(bool)
    industries = (
        panel.pivot_table(index="date", columns="code", values="industry", aggfunc="last")
        .sort_index()
        .ffill()
    )
    listed_days = _pivot(panel, "listed_days").sort_index().ffill() if "listed_days" in panel.columns else None

    universe = _universe_frame(inputs, sorted(panel["code"].unique().tolist()))
    universe = universe[universe["code"].isin(set(panel["code"].unique().tolist()))].copy()
    if "weight" in universe.columns:
        universe["weight"] = pd.to_numeric(universe["weight"], errors="coerce").fillna(0.0)
    else:
        universe["weight"] = 1.0 / max(len(universe), 1)
    universe = universe[universe["weight"] > 0]
    if universe.empty:
        raise DataQualityError("universe has no investable members with positive benchmark weight")
    universe_codes = universe["code"].tolist()
    is_st_map = (
        dict(zip(universe["code"], universe["is_st"].fillna(False).astype(bool)))
        if "is_st" in universe.columns
        else {code: False for code in universe_codes}
    )
    board_map = (
        dict(zip(universe["code"], universe["board"].astype(str)))
        if "board" in universe.columns
        else {code: "main" for code in universe_codes}
    )
    suspended_pivot = (
        panel.assign(_s=panel["suspended"].astype(bool))
        .pivot_table(index="date", columns="code", values="_s", aggfunc="last")
        .reindex(index=panel_dates)
    )
    suspended_values = suspended_pivot.to_numpy()
    suspended = pd.DataFrame(
        np.where(pd.isna(suspended_values), True, suspended_values),
        index=suspended_pivot.index,
        columns=suspended_pivot.columns,
    ).astype(bool)

    # Current holdings / NAV from the backtest stage (loaded before construction so
    # untradable names can be carried at their current holding weight).
    capital = float(common.section(cfg, "backtest", "initial_capital", default=1.0e8) or 1.0e8)
    lot = int(common.section(cfg, "backtest", "lot_size", default=100) or 100)
    current_shares: dict[str, int] = {}
    holdings = inputs.get("holdings")
    held_weights = pd.Series(dtype=float)
    if isinstance(holdings, pd.DataFrame) and len(holdings):
        frame = holdings.copy()
        frame["date"] = frame["date"].astype(str).str[:10]
        frame["code"] = frame["code"].astype(str)
        last_date = frame["date"].max()
        latest = frame[frame["date"] == last_date]
        current_shares = {str(row.code): int(row.shares) for row in latest.itertuples(index=False)}
        held_weights = pd.Series(
            [float(row.weight) for row in latest.itertuples(index=False)],
            index=[str(row.code) for row in latest.itertuples(index=False)],
            dtype=float,
        )
        if float(held_weights.sum()) > 0:
            held_weights = held_weights / float(held_weights.sum())
    nav = capital
    equity_curve = inputs.get("equity_curve")
    if isinstance(equity_curve, pd.DataFrame) and len(equity_curve):
        frame = equity_curve.copy()
        frame["date"] = frame["date"].astype(str).str[:10]
        frame = frame.sort_values("date", kind="mergesort")
        selected = frame[frame["date"] == decision]
        nav_value = float(selected["nav"].iloc[0]) if len(selected) else float(frame["nav"].iloc[-1])
        if np.isfinite(nav_value) and nav_value > 0:
            nav = float(nav_value) * capital

    # Code set: every benchmark member with a frozen price, plus any held name.
    decision_prices = (
        valuation_close.loc[decision] if decision in valuation_close.index else pd.Series(dtype=float)
    )
    member_codes = [
        code
        for code in universe_codes
        if code in decision_prices.index
        and np.isfinite(float(decision_prices.get(code, np.nan)))
        and float(decision_prices.get(code, 0.0)) > 0
    ]
    codes = sorted(set(member_codes) | set(held_weights.index.tolist()))
    if len(codes) < 2:
        raise DataQualityError(f"only {len(codes)} investable names on {decision}; cannot optimise")

    # Benchmark proxy: capped free-float weights, renormalised over the code set.
    max_weight = float(constraints.get("max_weight", 0.03))
    benchmark_cap = float(constraints.get("benchmark_weight_cap", max_weight))
    weight_margin = float(constraints.get("weight_margin", 0.985))
    effective_cap = min(benchmark_cap, max_weight) * weight_margin
    raw_weights = universe.set_index("code")["weight"].reindex(codes).fillna(0.0)
    capped = common.project_capped_benchmark(raw_weights, effective_cap)
    benchmark = capped / float(capped.sum())
    industry = pd.Series(
        [
            str(industries.at[decision, code])
            if decision in industries.index and code in industries.columns
            else "UNKNOWN"
            for code in codes
        ],
        index=codes,
    )
    current_weights = held_weights.reindex(codes).fillna(0.0)
    if float(current_weights.sum()) > 0:
        current_weights = current_weights / float(current_weights.sum())

    # Alpha: composite score at the lagged signal date (0 for names the factor
    # stage did not cover, e.g. a name suspended on the signal date).
    score_rows = factor_panel[factor_panel["date"] == signal_date].set_index("code")
    alpha = pd.to_numeric(score_rows["composite_score"], errors="coerce").reindex(codes).astype(float).fillna(0.0)
    spread = float(alpha.std(ddof=0))
    if spread > 0:
        alpha = (alpha - float(alpha.mean())) / spread

    # "Tradable" means the market lets us trade the name (in the index, priced, not
    # suspended).  Factor eligibility only controls the alpha tilt: an index newcomer
    # that is not yet factor-eligible must still be tradable so the projection can
    # clamp it back inside the caps instead of carrying a compounded position.
    eligible = set(eligible_rows.loc[eligible_rows["eligible"].astype(bool), "code"].astype(str))
    min_listed = float(common.section(cfg, "universe", "min_listed_days", default=0) or 0)
    availability = []
    tilt_flags = []
    for code in codes:
        priced = True
        price = decision_prices.get(code, np.nan)
        if not np.isfinite(float(price)) or float(price) <= 0:
            priced = False
        if decision in in_universe.index and code in in_universe.columns and not bool(in_universe.at[decision, code]):
            priced = False
        if decision in suspended.index and code in suspended.columns and bool(suspended.at[decision, code]):
            priced = False
        availability.append(priced)
        tilt_flags.append(
            priced and code in eligible and not bool(is_st_map.get(code, False))
        )
    tradable = pd.Series(availability, index=codes, dtype=bool)
    tilt_mask = pd.Series(tilt_flags, index=codes, dtype=bool)
    if int(tradable.sum()) < 2:
        raise DataQualityError(f"only {int(tradable.sum())} tradable names on {decision}; cannot optimise")

    # Trailing volatility for the tracking-error scaling of the fallback path.
    window = panel_dates[max(0, len(panel_dates) - 60) :]
    sigma = (ret.reindex(index=window, columns=codes).std(ddof=1) * np.sqrt(252.0)).reindex(codes).fillna(0.0)

    # Covariance of trailing daily returns.
    covariance_cfg = dict(common.section(cfg, "portfolio", "covariance", default={}) or {})
    lookback = int(covariance_cfg.get("lookback_days", 120) or 120)
    annualization = int(covariance_cfg.get("annualization", 252) or 252)
    covariance_window = panel_dates[max(0, len(panel_dates) - lookback) :]
    returns_window = ret.reindex(index=covariance_window, columns=codes).astype(float)
    covariance_result = portfolio_optimizer.ledoit_wolf_covariance(
        returns_window,
        annualization=annualization,
        min_obs=int(covariance_cfg.get("min_obs", 20) or 20),
        method=str(covariance_cfg.get("method", "ledoit_wolf")),
        fallback=str(covariance_cfg.get("fallback", "diagonal")),
    )
    covariance = covariance_result.covariance.reindex(index=codes, columns=codes).fillna(0.0)

    # Minimum one-lot holding weight (half-capital safety margin, limit-move buffer).
    limits = pd.Series(
        [_board_limit(board_map.get(code), bool(is_st_map.get(code, False))) for code in codes],
        index=codes,
    )
    prices = decision_prices.reindex(codes).astype(float)
    min_holding = lot * prices * (1.0 + limits + 0.02) / max(0.5 * capital, 1.0)
    min_holding = min_holding.where(np.isfinite(min_holding), 0.0)

    optimizer_cfg = dict(common.section(cfg, "portfolio", "optimizer", default={}) or {})
    fallback_reason: str | None = None
    try:
        result = portfolio_optimizer.optimize_weights(
            alpha=alpha,
            benchmark=benchmark,
            covariance=covariance,
            industry=industry,
            constraints=constraints,
            optimizer_cfg=optimizer_cfg,
            previous_weights=current_weights,
            min_holding=min_holding,
            weight_margin=weight_margin,
            tradable=tradable,
            tilt_mask=tilt_mask,
            current_weights=current_weights,
        )
    except Exception as exc:  # noqa: BLE001 - deterministic fallback is required
        fallback_reason = f"{type(exc).__name__}: {exc}"
        try:
            result = portfolio_optimizer.rank_proportional_fallback(
                alpha=alpha,
                benchmark=benchmark,
                industry=industry,
                constraints=constraints,
                min_holding=min_holding,
                weight_margin=weight_margin,
                tradable=tradable,
                tilt_mask=tilt_mask,
                current_weights=current_weights,
                covariance=covariance,
            )
        except Exception as fallback_exc:  # noqa: BLE001 - surface as a data-quality error
            raise DataQualityError(
                "portfolio optimisation failed and the deterministic fallback is also infeasible:"
                f" {type(fallback_exc).__name__}: {fallback_exc}"
            ) from fallback_exc
    weights = pd.Series(result.weights, index=codes, dtype=float)
    ex_ante_te_before_blend = float(result.ex_ante_te) if np.isfinite(result.ex_ante_te) else float(
        portfolio_optimizer.ex_ante_te(
            (weights - benchmark).to_numpy(dtype=float), covariance.to_numpy(dtype=float)
        )
    )

    # Feasible current portfolio for the turnover blend: tradable names use the
    # published bounds, untradable names are pinned at their current weight.
    lower_full, upper_full = portfolio_optimizer.feasible_bounds(
        benchmark, constraints, weight_margin=weight_margin, min_holding=min_holding
    )
    fixed_lower = np.where(tradable.to_numpy(dtype=bool), lower_full, current_weights.to_numpy(dtype=float))
    fixed_upper = np.where(tradable.to_numpy(dtype=bool), upper_full, current_weights.to_numpy(dtype=float))
    previous_feasible_projection = project_industry_neutral_box(
        current_weights.to_numpy(dtype=float),
        fixed_lower,
        fixed_upper,
        industry.tolist(),
        benchmark.to_numpy(dtype=float),
        bisection_steps=60,
    )
    previous_feasible = pd.Series(previous_feasible_projection.weights, index=codes, dtype=float)

    # Turnover cap blend (both ends feasible for the same polytope).
    turnover_limit = float(constraints.get("turnover_limit", 0.30))
    turnover_raw = float(0.5 * np.abs(weights.to_numpy(dtype=float) - previous_feasible.to_numpy(dtype=float)).sum())
    if turnover_limit > 0 and turnover_raw > turnover_limit:
        theta = turnover_limit / turnover_raw
        blended = previous_feasible + theta * (weights - previous_feasible)
        blended = blended / float(blended.sum()) if float(blended.sum()) > 0 else blended
        weights = blended
    active = (weights - benchmark).to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    ex_ante_te = portfolio_optimizer.ex_ante_te(active, covariance_values)
    turnover = float(0.5 * np.abs(weights.to_numpy(dtype=float) - previous_feasible.to_numpy(dtype=float)).sum())

    industry_active = industry_active_table(weights, benchmark, industry)
    factor_active: dict[str, float] = {}
    z_columns = [column for column in factor_panel.columns if column.endswith("_z")]
    if z_columns:
        signal_frame = factor_panel[factor_panel["date"] == signal_date].set_index("code")
        for column in z_columns:
            values = pd.to_numeric(signal_frame[column], errors="coerce").reindex(codes).fillna(0.0)
            factor_active[column[: -len("_z")]] = float(np.dot(active, values.to_numpy(dtype=float)))
    composite_active = float(np.dot(active, alpha.to_numpy(dtype=float)))
    n_names = int((weights > 1e-10).sum())

    constraints_payload = {
        key: constraints.get(key)
        for key in (
            "max_weight",
            "max_active_weight",
            "industry_active_limit",
            "turnover_limit",
            "te_target",
            "cash_buffer",
            "min_names",
            "benchmark_weight_cap",
            "weight_margin",
        )
        if key in constraints
    }
    constraints_payload.update(
        {
            "bounds": inputs["constraint_info"].get("bounds") or {},
            "sources": inputs["constraint_info"].get("source") or {},
            "clamped": inputs["constraint_info"].get("clamped") or {},
            "clamping_notes": list(inputs["constraint_info"].get("notes") or []),
            "applied_from_hypothesis": {
                key: bool(inputs["constraint_info"].get("source", {}).get(key) == "hypothesis")
                for key in ("max_weight", "max_active_weight", "te_target", "turnover_limit")
            },
        }
    )

    target_weights_frame = pd.DataFrame(
        {
            "date": decision,
            "code": codes,
            "weight": weights.to_numpy(dtype=float),
            "benchmark_weight": benchmark.to_numpy(dtype=float),
            "active_weight": (weights - benchmark).to_numpy(dtype=float),
            "industry": industry.to_numpy(),
        }
    )
    target_weights_frame = target_weights_frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    participation_cap = float(common.section(cfg, "portfolio", "participation_cap", default=0.10) or 0.0)
    min_trade_notional = float(
        common.section(cfg, "backtest", "costs", "min_trade_notional", default=0.0) or 0.0
    )
    amounts = amount.reindex(index=[decision]).iloc[0] if len(amount) else pd.Series(dtype=float)
    orders = order_module.build_orders(
        date=decision,
        target_weights=weights,
        current_shares=current_shares,
        nav=nav,
        prices=prices,
        amount=amounts,
        lot_size=lot,
        participation_cap=participation_cap,
        min_trade_notional=min_trade_notional,
    )
    max_participation = float(orders["participation_pct"].max()) if len(orders) else 0.0
    warnings: list[str] = []
    if fallback_reason:
        warnings.append(f"optimizer failed, deterministic rank-proportional fallback used: {fallback_reason}")
    if result.projection_status not in (None, "exact"):
        warnings.append(f"projection status is {result.meta.get('projection_status')!r}")
    if covariance_result.meta.get("fallback_used"):
        warnings.append("covariance fell back to a diagonal estimate: " + str(covariance_result.meta.get("reason", "")))

    optimizer_payload = {
        "status": result.status,
        "iterations": int(result.iterations),
        "te_lambda": None if not np.isfinite(float(result.te_lambda or float("nan"))) else float(result.te_lambda),
        "fallback_used": bool(result.fallback_used),
        "method": "projected_gradient",
        "projection": result.projection_status,
        "ex_ante_te_before_turnover_blend": float(ex_ante_te_before_blend),
        "covariance": {
            "method": covariance_result.method,
            "fallback_used": bool(covariance_result.meta.get("fallback_used", False)),
            "n_observations": int(covariance_result.meta.get("n_observations", 0)),
            "shrinkage": covariance_result.meta.get("shrinkage"),
            "lookback_days": lookback,
            "annualization": annualization,
        },
        "constraints_satisfied": _constraints_satisfied(
            weights=weights,
            current_weights=current_weights,
            tradable=tradable,
            lower_full=lower_full,
            upper_full=upper_full,
            industry_active=industry_active,
            industry_limit=float(constraints.get("industry_active_limit", 1.0)),
            turnover=turnover,
            turnover_limit=turnover_limit,
        ),
    }
    risk_report = {
        "date": decision,
        "signal_date": signal_date,
        "ex_ante_te": float(ex_ante_te),
        "te_target": float(constraints.get("te_target", 0.0)),
        "industry_active": {str(key): float(value) for key, value in industry_active.items()},
        "factor_active": {str(key): float(value) for key, value in factor_active.items()},
        "composite_active_exposure": composite_active,
        "turnover": float(turnover),
        "n_names": n_names,
        "optimizer": optimizer_payload,
        "constraints": constraints_payload,
        "benchmark_weight_policy": {
            "source": "universe.weight (frozen free-float cap proxy)",
            "cap": float(effective_cap),
            "capped_and_renormalised": True,
            "note": (
                "raw CSI 300 weights above the name cap make a hard |active| <= max_active_weight "
                "constraint infeasible for megacaps, so the benchmark proxy is capped at the name "
                "cap and renormalised"
            ),
        },
        "participation": {
            "cap": participation_cap,
            "max_order_participation_pct": max_participation,
            "units": "fraction of decision-day traded amount (0.05 = 5% of ADV)",
        },
        "nav": float(nav),
        "warnings": warnings,
    }
    return {
        "decision_date": decision,
        "signal_date": signal_date,
        "target_weights": target_weights_frame,
        "orders": orders,
        "risk_report": risk_report,
        "weights": weights,
    }


def _constraints_satisfied(
    *,
    weights: pd.Series,
    current_weights: pd.Series,
    tradable: pd.Series,
    lower_full: "np.ndarray",
    upper_full: "np.ndarray",
    industry_active: Mapping[str, float],
    industry_limit: float,
    turnover: float,
    turnover_limit: float,
) -> dict[str, bool]:
    """Boolean audit of the published constraints (tradable bounds + fixed names)."""

    values = weights.to_numpy(dtype=float)
    current = current_weights.to_numpy(dtype=float)
    tradeable = tradable.to_numpy(dtype=bool)
    fixed_ok = bool(np.all(np.abs(values[~tradeable] - current[~tradeable]) <= 1e-9)) if (~tradeable).any() else True
    box_ok = (
        bool(np.all(values[tradeable] <= upper_full[tradeable] + 1e-9))
        and bool(np.all(values[tradeable] >= lower_full[tradeable] - 1e-9))
        if tradeable.any()
        else True
    )
    return {
        "box": bool(box_ok and fixed_ok),
        "fully_invested": bool(abs(float(weights.sum()) - 1.0) <= 1e-9),
        "industry_neutral": bool(
            (max(abs(value) for value in industry_active.values()) if industry_active else 0.0)
            <= float(industry_limit) + 1e-9
        ),
        "turnover_within_cap": bool(turnover <= turnover_limit + 1e-9),
        "untradable_names_pinned": bool(fixed_ok),
    }


# ---------------------------------------------------------------------------
# Stage handler
# ---------------------------------------------------------------------------


def run_portfolio(ctx: RunContext) -> StageOutput:
    inputs = load_portfolio_inputs(ctx)
    built = build_portfolio(inputs)
    target_path = ctx.artifact_path("portfolio", "target_weights.parquet")
    orders_path = ctx.artifact_path("portfolio", "orders.csv")
    risk_path = ctx.artifact_path("portfolio", "risk_report.json")
    common.write_frame_artifact(target_path, built["target_weights"])
    orders_frame = built["orders"]
    text = orders_frame.to_csv(index=False, float_format="%.10f", lineterminator="\n")
    from aicwf.util.jsonio import write_text

    write_text(orders_path, text)
    common.write_json_artifact(risk_path, built["risk_report"])

    risk_report = built["risk_report"]
    summary = {
        "date": built["decision_date"],
        "n_names": int(risk_report["n_names"]),
        "ex_ante_te": float(risk_report["ex_ante_te"]),
        "turnover": float(risk_report["turnover"]),
        "fallback_used": bool(risk_report["optimizer"]["fallback_used"]),
        "signal_date": built["signal_date"],
        "n_orders": int(len(orders_frame)),
    }
    artifacts = {
        "target_weights": common.rel_artifact(target_path),
        "orders": common.rel_artifact(orders_path),
        "risk_report": common.rel_artifact(risk_path),
        "target_weights.parquet": common.rel_artifact(target_path),
        "orders.csv": common.rel_artifact(orders_path),
        "risk_report.json": common.rel_artifact(risk_path),
    }
    ctx.event(
        "portfolio_completed",
        date=built["decision_date"],
        n_names=summary["n_names"],
        ex_ante_te=summary["ex_ante_te"],
        fallback_used=summary["fallback_used"],
    )
    return StageOutput(artifacts=artifacts, summary=summary)


#: Controller registry mapping (the Lead wires these in ``aicwf/pipeline.py``).
HANDLERS = {"portfolio": run_portfolio}


__all__ = ["HANDLERS", "build_portfolio", "load_portfolio_inputs", "run_portfolio"]
