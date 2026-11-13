"""Robustness suite for the index-enhancement backtest.

Every block re-runs the *same* engine with one dimension changed, so differences
are attributable to that dimension alone:

* ``subperiods`` — the strategy over disjoint date ranges;
* ``cost_sensitivity`` — 0x / 1x / 3x the configured transaction costs;
* ``leave_one_out`` — drop one composite factor at a time and rebuild the score;
* ``parameter_grid`` — tracking-error target x rebalance frequency;
* ``placebo`` — seeded random composite scores ("monkey with a dartboard");
* ``bias_stress`` — adversarial survivorship stress: remove the top-k index
  winners from the investable universe and re-run (if the edge only exists
  because the frozen universe contains the eventual winners, this collapses);
* ``deflated_sharpe`` — multiplicity-corrected Sharpe ratio following Bailey &
  López de Prado: ``DSR = Phi( (SR - SR0) sqrt(T-1) / sqrt(1 - g3 SR + ((g4-1)/4) SR^2) )``
  with ``SR0`` the expected maximum Sharpe of ``N`` independent trials.

All randomness is seeded (``ctx.seed``), every run is deterministic and the
artifact contains no wall-clock values.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm as _normal
from scipy.stats import skew as _skew

from aicwf.backtest import metrics as metric_model
from aicwf.backtest.engine import BacktestModel, BacktestResult
from aicwf.factors import preprocess as factor_preprocess

EULER_GAMMA = 0.5772156649015329


DEFAULT_THRESHOLDS: dict[str, float] = {
    "subperiod_pass_share": 0.66,
    "cost_3x_excess_min": 0.0,
    "loo_min_information_ratio": 0.0,
    "grid_positive_share": 0.60,
    "placebo_p_value_max": 0.10,
    "bias_stress_excess_min": 0.0,
    "dsr_min": 0.95,
}

METRIC_KEYS = (
    "annualized_return",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
    "sharpe",
    "max_drawdown",
)


def thresholds_from_config(cfg: Mapping[str, Any]) -> dict[str, float]:
    configured = dict(((cfg.get("robustness") or {}).get("thresholds") or {}))
    values = dict(DEFAULT_THRESHOLDS)
    for key, value in configured.items():
        if key in values and value is not None:
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                continue
    return values


def _metric_subset(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    metrics = metrics or {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        try:
            number = float(value) if value is not None else float("nan")
        except (TypeError, ValueError):
            number = float("nan")
        result[key] = None if not math.isfinite(number) else number
    result["n_returns"] = int(metrics.get("n_returns") or 0)
    return result


def _delta(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("annualized_excess_return", "information_ratio", "tracking_error", "sharpe"):
        left = metrics.get(key)
        right = baseline.get(key)
        try:
            out[key] = float(left) - float(right)
        except (TypeError, ValueError):
            out[key] = None
    return out


def window_metrics(result: BacktestResult, start: str, end: str) -> dict[str, Any]:
    """Metrics measured on the NAV path restricted to ``[start, end]``."""

    nav = result.nav.loc[start:end]
    benchmark = result.benchmark_nav.reindex(nav.index)
    if len(nav) < 3:
        return {}
    return metric_model.compute_performance_metrics(nav, benchmark)


def _safe_run(model: BacktestModel, label: str, warnings: list[str], **kwargs) -> BacktestResult | None:
    try:
        return model.run(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a failed variant becomes a concern, not a crash
        warnings.append(f"{label}: variant failed: {type(exc).__name__}: {exc}")
        return None


def _factor_scores(
    model: BacktestModel,
    weights: Mapping[str, float],
    *,
    skip: str | None = None,
) -> pd.DataFrame | None:
    """Weighted sum of neutralised factor z-scores on the model's factor grid."""

    total: pd.DataFrame | None = None
    for factor_id, weight in weights.items():
        if skip is not None and factor_id == skip:
            continue
        if abs(float(weight)) <= 1e-12:
            continue
        matrix = model.factor_zscores.get(factor_id)
        if matrix is None:
            continue
        masked = matrix.where(model.eligible)
        filled = factor_preprocess.fill_neutral(masked)
        contribution = (model.eligible.astype(float) * filled) * float(weight)
        total = contribution if total is None else total + contribution
    if total is None:
        return None
    return total.where(model.eligible)


def _deflated_sharpe(
    result: BacktestResult,
    trial_sharpes: Sequence[float],
    *,
    n_trials: int,
) -> dict[str, Any]:
    daily = result.nav.pct_change().dropna()
    n_observations = int(len(daily))
    if n_observations < 3 or float(daily.std(ddof=1)) <= 0:
        return {
            "sharpe_annual": result.metrics.get("sharpe"),
            "deflated_sharpe": None,
            "n_trials": int(n_trials),
            "n_observations": n_observations,
            "skew": None,
            "kurtosis": None,
        }
    sharpe_daily = float(daily.mean() / daily.std(ddof=1))
    skewness = float(_skew(daily, bias=False))
    kurt = float(_kurtosis(daily, fisher=False, bias=False))
    trials = [float(value) for value in trial_sharpes if value is not None and math.isfinite(float(value))]
    if len(trials) >= 2:
        variance = float(np.var(trials, ddof=1)) / float(metric_model.PERIODS_PER_YEAR)
    else:
        variance = max(sharpe_daily**2, 1e-12)
    trials_count = max(int(n_trials), len(trials) + 1, 2)
    expected_max = math.sqrt(max(variance, 1e-16)) * (
        (1.0 - EULER_GAMMA) * float(_normal.ppf(1.0 - 1.0 / trials_count))
        + EULER_GAMMA * float(_normal.ppf(1.0 - 1.0 / (trials_count * math.e)))
    )
    denominator = 1.0 - skewness * sharpe_daily + ((kurt - 1.0) / 4.0) * sharpe_daily**2
    denominator = math.sqrt(denominator) if denominator > 1e-12 else 1e-6
    statistic = (sharpe_daily - expected_max) * math.sqrt(n_observations - 1) / denominator
    deflated = float(_normal.cdf(statistic))
    return {
        "sharpe_annual": float(result.metrics.get("sharpe") or 0.0),
        "sharpe_daily": sharpe_daily,
        "expected_max_sharpe_daily": float(expected_max),
        "deflated_sharpe": deflated,
        "z_statistic": float(statistic),
        "n_trials": int(trials_count),
        "n_observations": n_observations,
        "skew": skewness,
        "kurtosis": kurt,
        "trial_sharpe_std_annual": float(math.sqrt(max(variance, 0.0)) * math.sqrt(metric_model.PERIODS_PER_YEAR)),
    }


def run_robustness_suite(
    *,
    model: BacktestModel,
    baseline: BacktestResult,
    composite_weights: Mapping[str, float],
    cfg: Mapping[str, Any],
    seed: int = 0,
    constraints: Mapping[str, Any] | None = None,
    hypothesis: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run every robustness block; returns (robustness document, checks list)."""

    robustness_cfg = dict(cfg.get("robustness") or {})
    thresholds = thresholds_from_config(cfg)
    seed = int(seed)
    warnings: list[str] = []
    baseline_metrics = _metric_subset(baseline.metrics)
    trial_sharpes: list[float] = []
    run_kwargs = {"constraints": constraints}

    # -- 1. subperiods ------------------------------------------------------
    subperiods: list[dict[str, Any]] = []
    subperiod_specs = list(robustness_cfg.get("subperiods") or [])
    if not subperiod_specs:
        subperiod_specs = [[baseline.equity_curve["date"].iloc[0], baseline.equity_curve["date"].iloc[-1]]]
    for spec in subperiod_specs:
        start, end = str(spec[0])[:10], str(spec[1])[:10]
        if start > end:
            warnings.append(f"subperiod {start}..{end} has start > end; skipped")
            continue
        result = _safe_run(model, f"subperiod {start}..{end}", warnings, start=start, end=end, **run_kwargs)
        if result is None:
            continue
        metrics = _metric_subset(window_metrics(result, start, end))
        trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
        subperiods.append(
            {
                "start": start,
                "end": end,
                "metrics": metrics,
                "delta_vs_baseline": _delta(metrics, baseline_metrics),
                "n_rebalances": len(result.rebalances),
            }
        )
    positive_subperiods = [
        item for item in subperiods if (item["metrics"].get("annualized_excess_return") or 0.0) > 0.0
    ]
    subperiod_share = (len(positive_subperiods) / len(subperiods)) if subperiods else 0.0

    # -- 2. cost sensitivity -------------------------------------------------
    cost_sensitivity: list[dict[str, Any]] = []
    multipliers = [float(value) for value in (robustness_cfg.get("cost_multipliers") or [0.0, 1.0, 3.0])]
    for multiplier in multipliers:
        if abs(multiplier - 1.0) < 1e-12:
            metrics = baseline_metrics
            result = baseline
        else:
            result = _safe_run(model, f"cost x{multiplier:g}", warnings, cost_multiplier=multiplier, **run_kwargs)
            if result is None:
                continue
            metrics = _metric_subset(result.metrics)
        if result is not baseline:
            trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
        cost_sensitivity.append(
            {
                "multiplier": float(multiplier),
                "metrics": metrics,
                "delta_vs_baseline": _delta(metrics, baseline_metrics),
            }
        )
    cost_3x = next(
        (item for item in cost_sensitivity if abs(float(item["multiplier"]) - 3.0) < 1e-12),
        cost_sensitivity[-1] if cost_sensitivity else None,
    )
    cost_3x_excess = None
    if cost_3x is not None:
        value = cost_3x["metrics"].get("annualized_excess_return")
        cost_3x_excess = None if value is None else float(value)

    # -- 3. leave-one-factor-out --------------------------------------------
    leave_one_out: dict[str, Any] = {}
    max_loo = int(robustness_cfg.get("max_loo_factors", 13) or 13)
    loo_factors = sorted(
        ((factor_id, float(weight)) for factor_id, weight in composite_weights.items() if abs(float(weight)) > 1e-12),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:max_loo]
    if bool(robustness_cfg.get("factor_leave_one_out", True)):
        for factor_id, weight in loo_factors:
            scores = _factor_scores(model, composite_weights, skip=factor_id)
            if scores is None:
                continue
            result = _safe_run(
                model, f"leave-one-out {factor_id}", warnings, scores=scores, **run_kwargs
            )
            if result is None:
                continue
            metrics = _metric_subset(result.metrics)
            trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
            leave_one_out[factor_id] = {
                "weight": float(weight),
                "metrics": metrics,
                "delta_vs_baseline": _delta(metrics, baseline_metrics),
            }
    loo_irs = [
        item["metrics"]["information_ratio"]
        for item in leave_one_out.values()
        if item["metrics"].get("information_ratio") is not None
    ]
    loo_min_ir = float(min(loo_irs)) if loo_irs else None

    # -- 4. parameter grid ---------------------------------------------------
    parameter_grid: list[dict[str, Any]] = []
    grid_cfg = dict(robustness_cfg.get("parameter_grid") or {})
    te_values = [float(value) for value in (grid_cfg.get("te_target") or [])] or [float((constraints or {}).get("te_target", 0.05))]
    rebalance_values = list(grid_cfg.get("rebalance") or []) or ["monthly", "quarterly"]
    for te_value in te_values:
        for rebalance in rebalance_values:
            result = _safe_run(
                model,
                f"grid te={te_value:g} {rebalance}",
                warnings,
                rebalance=str(rebalance),
                te_target=te_value,
                **run_kwargs,
            )
            if result is None:
                continue
            metrics = _metric_subset(result.metrics)
            trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
            parameter_grid.append(
                {
                    "te_target": float(te_value),
                    "rebalance": str(rebalance),
                    "metrics": metrics,
                    "delta_vs_baseline": _delta(metrics, baseline_metrics),
                }
            )
    positive_grid = [
        item for item in parameter_grid if (item["metrics"].get("annualized_excess_return") or 0.0) > 0.0
    ]
    grid_positive_share = (len(positive_grid) / len(parameter_grid)) if parameter_grid else 0.0

    # -- 5. seeded placebo ---------------------------------------------------
    placebo_cfg = int(robustness_cfg.get("placebo_repetitions", 20) or 0)
    placebo_runs: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    n_dates = len(model.factor_dates)
    n_codes = len(model.codes)
    baseline_ir = float(baseline_metrics.get("information_ratio") or 0.0)
    for repetition in range(placebo_cfg):
        random_scores = pd.DataFrame(
            rng.standard_normal((n_dates, n_codes)),
            index=model.factor_dates,
            columns=model.codes,
        )
        result = _safe_run(model, f"placebo #{repetition}", warnings, scores=random_scores, **run_kwargs)
        if result is None:
            continue
        metrics = _metric_subset(result.metrics)
        trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
        placebo_runs.append(metrics)
    placebo_irs = [float(item.get("information_ratio") or 0.0) for item in placebo_runs]
    if placebo_irs:
        p_value = float((1.0 + sum(1 for value in placebo_irs if value >= baseline_ir)) / (len(placebo_irs) + 1.0))
        placebo = {
            "repetitions": len(placebo_irs),
            "seed": seed,
            "baseline_information_ratio": baseline_ir,
            "placebo_information_ratio_mean": float(np.mean(placebo_irs)),
            "placebo_information_ratio_std": float(np.std(placebo_irs, ddof=1)) if len(placebo_irs) > 1 else 0.0,
            "placebo_information_ratio_max": float(np.max(placebo_irs)),
            "p_value": p_value,
            "placebo_samples": placebo_irs,
        }
    else:
        p_value = None
        placebo = {"repetitions": 0, "seed": seed, "p_value": None, "placebo_samples": []}

    # -- 6. adversarial survivorship stress ----------------------------------
    bias_stress: dict[str, Any] = {"levels": [], "drop_top_k": 0}
    drop_k = int(robustness_cfg.get("bias_stress_drop_top_k", 30) or 0)
    levels = [float(value) for value in (robustness_cfg.get("bias_stress_levels") or [0.5, 1.0])]
    if drop_k > 0 and len(model.codes) > 5:
        returns = pd.Series(dtype=float)
        prices = model.valuation_close
        for code in model.codes:
            series = prices[code].dropna()
            if len(series) >= 2 and float(series.iloc[0]) > 0:
                returns[code] = float(series.iloc[-1] / series.iloc[0] - 1.0)
        winners = [str(code) for code in returns.sort_values(ascending=False).index]
        bias_stress["winners"] = winners[: max(1, int(drop_k))]
        for level in levels:
            k = max(1, int(round(drop_k * float(level))))
            k = min(k, max(1, len(winners) - 3))
            dropped = winners[:k]
            result = _safe_run(
                model,
                f"bias stress drop {k}",
                warnings,
                exclude_codes=dropped,
                **run_kwargs,
            )
            if result is None:
                continue
            metrics = _metric_subset(result.metrics)
            trial_sharpes.append(float(metrics.get("sharpe") or 0.0))
            bias_stress["levels"].append(
                {
                    "k": int(k),
                    "dropped": dropped,
                    "metrics": metrics,
                    "delta_vs_baseline": _delta(metrics, baseline_metrics),
                }
            )
        bias_stress["drop_top_k"] = int(drop_k)

    # -- 7. deflated Sharpe ratio -------------------------------------------
    n_trials = int(robustness_cfg.get("dsr_trials", 24) or 24)
    deflated = _deflated_sharpe(baseline, trial_sharpes, n_trials=n_trials)

    # -- checks --------------------------------------------------------------
    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: str, value: Any, threshold: Any, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "value": value,
                "threshold": threshold,
                "detail": detail,
            }
        )

    if subperiods:
        status = "pass" if subperiod_share >= thresholds["subperiod_pass_share"] else "fail"
        add_check(
            "subperiod_consistency",
            status,
            subperiod_share,
            thresholds["subperiod_pass_share"],
            f"{len(positive_subperiods)}/{len(subperiods)} subperiods have positive annualized excess return",
        )
    else:
        add_check("subperiod_consistency", "warn", None, thresholds["subperiod_pass_share"], "no subperiod run completed")

    if cost_3x_excess is not None:
        status = "pass" if cost_3x_excess >= thresholds["cost_3x_excess_min"] else "fail"
        add_check(
            "cost_sensitivity_3x",
            status,
            cost_3x_excess,
            thresholds["cost_3x_excess_min"],
            f"annualized excess return with 3x transaction costs is {cost_3x_excess:.4f}",
        )
    else:
        add_check("cost_sensitivity_3x", "warn", None, thresholds["cost_3x_excess_min"], "cost sensitivity run missing")

    if loo_min_ir is not None:
        status = "pass" if loo_min_ir >= thresholds["loo_min_information_ratio"] else "fail"
        add_check(
            "leave_one_out",
            status,
            loo_min_ir,
            thresholds["loo_min_information_ratio"],
            f"worst leave-one-factor-out information ratio is {loo_min_ir:.4f} over {len(leave_one_out)} variants",
        )
    else:
        add_check("leave_one_out", "warn", None, thresholds["loo_min_information_ratio"], "no leave-one-out run completed")

    if parameter_grid:
        status = "pass" if grid_positive_share >= thresholds["grid_positive_share"] else "fail"
        add_check(
            "parameter_grid_stability",
            status,
            grid_positive_share,
            thresholds["grid_positive_share"],
            f"{len(positive_grid)}/{len(parameter_grid)} parameter combinations keep a positive annualized excess return",
        )
    else:
        add_check("parameter_grid_stability", "warn", None, thresholds["grid_positive_share"], "no grid run completed")

    if p_value is not None:
        status = "pass" if p_value <= thresholds["placebo_p_value_max"] else "fail"
        add_check(
            "placebo_significance",
            status,
            p_value,
            thresholds["placebo_p_value_max"],
            f"seeded random-score placebo p-value is {p_value:.4f} over {placebo['repetitions']} repetitions",
        )
    else:
        add_check("placebo_significance", "warn", None, thresholds["placebo_p_value_max"], "no placebo run completed")

    stress_metrics = bias_stress["levels"][-1]["metrics"] if bias_stress["levels"] else {}
    stress_excess = stress_metrics.get("annualized_excess_return") if stress_metrics else None
    if stress_excess is not None:
        status = "pass" if float(stress_excess) >= thresholds["bias_stress_excess_min"] else "fail"
        add_check(
            "survivorship_bias_stress",
            status,
            float(stress_excess),
            thresholds["bias_stress_excess_min"],
            "annualized excess return after removing the top full-period index winners"
            f" (k={bias_stress['levels'][-1]['k']}) is {float(stress_excess):.4f}",
        )
    else:
        add_check("survivorship_bias_stress", "warn", None, thresholds["bias_stress_excess_min"], "bias-stress run missing")

    dsr_value = deflated.get("deflated_sharpe")
    if dsr_value is not None:
        status = "pass" if float(dsr_value) >= thresholds["dsr_min"] else "fail"
        add_check(
            "deflated_sharpe",
            status,
            float(dsr_value),
            thresholds["dsr_min"],
            f"deflated Sharpe ratio {float(dsr_value):.4f} with {deflated['n_trials']} trials and {deflated['n_observations']} observations",
        )
    else:
        add_check("deflated_sharpe", "warn", None, thresholds["dsr_min"], "deflated Sharpe could not be computed")

    concerns = [f"{check['name']}: {check['detail']}" for check in checks if check["status"] != "pass"]
    passed = [str(check["name"]) for check in checks if check["status"] == "pass"]
    document = {
        "baseline": {
            "metrics": baseline_metrics,
            "n_rebalances": len(baseline.rebalances),
            "information_ratio": baseline_ir,
        },
        "subperiods": subperiods,
        "cost_sensitivity": cost_sensitivity,
        "leave_one_out": leave_one_out,
        "parameter_grid": parameter_grid,
        "placebo": placebo,
        "bias_stress": bias_stress,
        "deflated_sharpe": deflated,
        "checks": checks,
        "summary": {
            "pass": passed,
            "concerns": concerns,
            "n_pass": len(passed),
            "n_fail": len(concerns),
        },
        "seed": seed,
        "thresholds": thresholds,
        "warnings": warnings,
    }
    return document, checks


__all__ = [
    "DEFAULT_THRESHOLDS",
    "METRIC_KEYS",
    "run_robustness_suite",
    "thresholds_from_config",
    "window_metrics",
]
