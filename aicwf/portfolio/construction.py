"""Index-enhancement target-weight construction (shared by backtest and portfolio).

Given a cross-sectional alpha signal (neutralised composite score, lagged one
month), a benchmark proxy, industry labels and the *current* portfolio, this
module builds a fully invested target portfolio that:

* is long-only, fully invested (``sum(w) = 1``),
* respects the per-name weight cap, the per-name active-weight cap and the
  full-investment equality,
* is **industry neutral relative to the benchmark proxy**: for every industry
  ``g``, ``sum_{i in g} w_i = sum_{i in g} wb_i`` (exact projection, see
  :mod:`aicwf.portfolio.projection`),
* keeps at least one board lot of every benchmark constituent that the name-cap
  bounds allow (so the realised holdings are also industry neutral and the
  ``min_names`` constraint holds),
* handles untradable names (suspended / not in the index at the decision date /
  not covered by the factor signal) by **fixing** their weight at the current
  holding level inside the projection, so the remaining names absorb the
  industry-neutrality and capacity adjustments,
* respects the one-way turnover cap by blending towards the current feasible
  portfolio.

The active tilt is scaled from the diagonal tracking-error estimate

.. math::  a_i = clip(k \\alpha_i, -cap_a, cap_a), \\quad
           k = TE_{target} / \\sqrt{\\sum_i (\\alpha_i \\sigma_i)^2}

and then projected with
:func:`aicwf.portfolio.projection.project_industry_neutral_box`.  All inputs
are aligned by ``code`` and the function is deterministic.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.portfolio.projection import project_industry_neutral_box


def construct_index_enhancement_weights(
    *,
    alpha: pd.Series,
    benchmark: pd.Series,
    industry: pd.Series,
    prices: pd.Series,
    sigma: pd.Series,
    current_weights: pd.Series,
    tradable: pd.Series,
    constraints: Mapping[str, Any],
    te_target: float | None = None,
    min_holding: pd.Series | None = None,
    lot: int = 100,
    capital: float = 1.0e8,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build the constrained target weights; returns ``(weights, diagnostics)``."""

    codes = sorted(
        set(benchmark.index.tolist())
        | set(current_weights.index.tolist())
        | set(alpha.index.tolist())
    )
    if not codes:
        raise ValueError("no instruments to construct a portfolio from")

    def _series(values: pd.Series, fill: float = 0.0) -> pd.Series:
        return pd.Series(values, dtype=float).reindex(codes).fillna(fill).astype(float)

    wb = _series(benchmark)
    if float(wb.sum()) <= 0:
        raise ValueError("benchmark proxy weights sum to zero")
    wb = wb / float(wb.sum())
    alpha_values = _series(alpha)
    spread = float(alpha_values.std(ddof=0))
    if spread > 0:
        alpha_values = (alpha_values - float(alpha_values.mean())) / spread
    sigma_values = _series(sigma).clip(lower=0.05)
    current = _series(current_weights)
    current_total = float(current.sum())
    current = current / current_total if current_total > 0 else current
    tradeable = pd.Series(tradable, dtype=bool).reindex(codes).fillna(False).astype(bool)
    industry_values = (
        pd.Series(industry, dtype=object).reindex(codes).fillna("UNKNOWN").astype(str).tolist()
    )
    prices_values = _series(prices)

    max_weight = float(constraints.get("max_weight", 1.0))
    max_active = float(constraints.get("max_active_weight", 1.0))
    turnover_limit = float(constraints.get("turnover_limit", 1.0))
    min_names = int(constraints.get("min_names", 0) or 0)
    cash_buffer = float(constraints.get("cash_buffer", 0.0) or 0.0)
    weight_margin = float(constraints.get("weight_margin", 0.985))
    target_te = float(te_target if te_target is not None else constraints.get("te_target", 0.05))

    upper_bound = np.minimum(max_weight * weight_margin, wb.to_numpy() + max_active * weight_margin)
    lower_bound = np.maximum(wb.to_numpy() - max_active * weight_margin, 0.0)
    if min_holding is not None:
        floor = pd.Series(min_holding, dtype=float).reindex(codes).fillna(0.0).to_numpy()
        floor = np.where(np.isfinite(floor), floor, 0.0)
        floor_extra = np.where(floor <= upper_bound, np.maximum(floor - lower_bound, 0.0), 0.0)
    else:
        floor_extra = np.zeros(len(codes))
    floor_extra = _fit_industry_floors(
        lower_bound=lower_bound,
        floor_extra=floor_extra,
        wb_values=wb.to_numpy(dtype=float),
        industry=industry_values,
    )
    lower_bound = np.minimum(
        np.maximum(wb.to_numpy(dtype=float) - max_active * weight_margin, 0.0) + floor_extra,
        upper_bound,
    )

    # Untradable names are fixed at the current holding; names outside the
    # benchmark proxy but tradable are forced to zero (exit).
    tradeable_values = tradeable.to_numpy()
    current_values = current.to_numpy(dtype=float)
    non_member = wb.to_numpy(dtype=float) <= 0.0
    lower_final = np.where(tradeable_values, lower_bound, current_values)
    upper_final = np.where(tradeable_values, upper_bound, current_values)
    forced_exit = tradeable_values & non_member
    lower_final = np.where(forced_exit, 0.0, lower_final)
    upper_final = np.where(forced_exit, 0.0, upper_final)
    # Keep the fixed block + minimum lots feasible for full investment.
    lower_final = np.minimum(lower_final, upper_final)
    if float(upper_final.sum()) < 1.0 - 1e-9 or float(lower_final.sum()) > 1.0 + 1e-9:
        raise ValueError(
            "infeasible weight bounds: the name caps and fixed holdings cannot fund a fully invested portfolio"
        )

    def project(vector: np.ndarray):
        return project_industry_neutral_box(
            vector,
            lower_final,
            upper_final,
            industry_values,
            wb.to_numpy(dtype=float),
            bisection_steps=60,
        )

    weights = current.copy()
    info: dict[str, Any] = {}
    effective_te = target_te
    for attempt in range(14):
        tilt_k = _tilt_scale(alpha_values, sigma_values, effective_te)
        active = (tilt_k * alpha_values).clip(-max_active * weight_margin, max_active * weight_margin)
        raw = wb + active
        raw_values = raw.to_numpy(dtype=float)
        raw_values = np.where(tradeable_values, raw_values, current_values)
        projection = project(raw_values)
        weights = pd.Series(projection.weights, index=codes, dtype=float)
        held = int(((weights > 1e-8) & tradeable_values).sum() + ((current_values > 1e-8) & ~tradeable_values).sum())
        info = {
            "projection_status": projection.status,
            "industry_active_max": float(projection.max_abs_industry_active),
            "n_eligible": int(tradeable_values.sum()),
            "n_held": held,
            "tilt_scale": float(tilt_k),
            "constraints_feasible": True,
        }
        if held >= min_names or tilt_k <= 1e-12 or attempt == 13:
            break
        effective_te = effective_te * 0.5

    # Turnover cap: blend towards the current portfolio inside the same polytope.
    current_feasible = pd.Series(project(current_values).weights, index=codes, dtype=float)
    target_turnover = float(0.5 * np.abs(weights.to_numpy(dtype=float) - current_feasible.to_numpy(dtype=float)).sum())
    blended = weights
    if len(codes) and turnover_limit > 0 and target_turnover > float(turnover_limit):
        theta = float(turnover_limit) / target_turnover
        blended = current_feasible + theta * (weights - current_feasible)
        info["turnover_blended"] = True
        info["turnover_theta"] = float(theta)
    actual_turnover = float(
        0.5 * np.abs(blended.to_numpy(dtype=float) - current_feasible.to_numpy(dtype=float)).sum()
    )
    if cash_buffer > 0:
        blended = blended * (1.0 - cash_buffer)
    total = float(blended.sum())
    if total > 0:
        blended = blended / total * (1.0 - cash_buffer) if cash_buffer > 0 else blended / total

    active_final = blended.to_numpy(dtype=float) - wb.to_numpy(dtype=float)
    info.update(
        {
            "turnover": actual_turnover,
            "te_target": float(target_te),
            "effective_te_target": float(effective_te),
            "te_estimate_diagonal": float(
                np.sqrt(np.sum(active_final**2 * sigma_values.to_numpy(dtype=float) ** 2))
            ),
            "max_weight": float(blended.max()),
            "max_active": float(np.max(np.abs(active_final))) if len(active_final) else 0.0,
            "industry_active": {
                str(key): float(value)
                for key, value in pd.DataFrame({"active": active_final, "industry": industry_values})
                .groupby("industry", sort=True)["active"]
                .sum()
                .items()
            },
            "benchmark_weights": {str(code): float(value) for code, value in wb.items()},
            "upper_bounds": {str(code): float(value) for code, value in zip(codes, upper_final)},
            "codes": codes,
        }
    )
    return blended.astype(float), info


def _tilt_scale(alpha: pd.Series, sigma: pd.Series, target_te: float) -> float:
    denominator = float(np.sqrt(np.sum((alpha.to_numpy(dtype=float) * sigma.to_numpy(dtype=float)) ** 2)))
    if denominator <= 1e-12 or target_te <= 0:
        return 0.0
    return float(target_te) / denominator


def _fit_industry_floors(
    *,
    lower_bound: np.ndarray,
    floor_extra: np.ndarray,
    wb_values: np.ndarray,
    industry: Sequence[str],
) -> np.ndarray:
    """Shrink minimum-lot floors until every industry target stays feasible."""

    floor_extra = np.asarray(floor_extra, dtype=float).copy()
    base = np.maximum(lower_bound - floor_extra, 0.0)
    industry_array = np.asarray([str(value) for value in industry], dtype=object)
    for group in sorted(set(industry_array.tolist())):
        mask = industry_array == group
        target = float(wb_values[mask].sum())
        base_total = float(base[mask].sum())
        headroom = target - base_total
        if headroom <= 0:
            floor_extra[mask] = 0.0
            continue
        extra = float(floor_extra[mask].sum())
        if extra > headroom * 0.98:
            floor_extra[mask] = floor_extra[mask] * (headroom * 0.98 / extra)
    return floor_extra


__all__ = ["construct_index_enhancement_weights"]
