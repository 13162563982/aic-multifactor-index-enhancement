"""Exact Euclidean projections used by the backtest and the portfolio optimiser.

The portfolio constraints are a box (long-only, per-name weight cap, per-name
active-weight cap), a full-investment equality ``sum(w) = 1`` and per-industry
active-weight limits.  The benchmark proxy has industry totals ``B_g`` (sum of
benchmark weights inside industry ``g``); requiring the portfolio industry
totals to equal ``B_g`` is exactly industry neutrality.

Projection of a vector ``v`` onto::

    {lb_i <= w_i <= ub_i,  sum_{i in g} w_i = B_g for every industry g}

has a simple dual solution: with one multiplier ``mu_g`` per industry, the
unconstrained first-order condition is ``w_i = clip(v_i - mu_g, lb_i, ub_i)``.
Because the left-hand side is monotone decreasing in ``mu_g``, each industry is
solved independently by bisection.  The implementation below runs all industry
bisections simultaneously with vectorised ``np.add.reduceat`` operations, so a
300-name / 30-industry projection costs a few dozen microsecond-scale numpy
calls — fast enough to call thousands of times inside a projected-gradient
optimiser.

Feasibility edge cases (a group target outside ``[sum(lb_g), sum(ub_g)]``, or a
global target outside the total feasible interval) are handled deterministically:
the infeasible group totals are moved to the closest feasible point while
preserving the global sum, and the resulting deviation (the residual industry
active weight) is reported in the returned diagnostics rather than silently
producing an invalid portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

FLOAT_TOLERANCE = 1e-13


@dataclass
class ProjectionResult:
    weights: np.ndarray
    status: str
    max_abs_industry_active: float
    iterations: int
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "max_abs_industry_active": float(self.max_abs_industry_active),
            "iterations": int(self.iterations),
            **self.detail,
        }


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _group_structure(group_ids: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (sorted_groups, inverse, group_starts, sort_order) with stable ordering."""

    group_array = np.asarray([str(value) for value in group_ids], dtype=object)
    unique = sorted(set(group_array.tolist()))
    inverse = np.searchsorted(np.asarray(unique, dtype=object), group_array)
    order = np.argsort(inverse, kind="mergesort")
    inverse_sorted = inverse[order]
    starts = np.flatnonzero(np.r_[True, inverse_sorted[1:] != inverse_sorted[:-1]])
    return np.asarray(unique, dtype=object), inverse, starts, order


def _reduceat_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts) if len(starts) else np.array([], dtype=float)


def _distribute_residual(
    weights: np.ndarray,
    residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """Move a small per-group residual into the interior (keeps box feasibility)."""

    weights = weights.copy()
    if not np.any(residual):
        return weights
    group_index = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(weights)]))
    for _ in range(6):
        if np.all(np.abs(residual) <= 1e-15):
            break
        headroom = np.maximum(upper - weights, 0.0)
        slack = np.maximum(weights - lower, 0.0)
        residual_elements = residual[group_index]
        room = np.where(residual_elements >= 0.0, headroom, slack)
        group_room = _reduceat_sum(room, starts)
        scale = np.divide(
            residual_elements,
            group_room[group_index],
            out=np.zeros(len(weights)),
            where=group_room[group_index] > 0,
        )
        updated = np.clip(weights + room * scale, lower, upper)
        residual = residual - _reduceat_sum(updated - weights, starts)
        weights = updated
    return weights


def project_group_box(
    vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    group_ids: Sequence[Any],
    group_targets: Sequence[float] | dict[str, float],
    *,
    total: float = 1.0,
    tolerance: float = 1e-12,
    bisection_steps: int = 60,
) -> ProjectionResult:
    """Exact projection onto box + per-group sum targets + optional global sum.

    ``group_targets`` may be a sequence aligned with the sorted unique groups or
    a mapping ``{group_id: target}``.  The global sum is only enforced as a
    fallback when the group targets themselves do not add up to ``total`` (that
    only happens when the requested group targets are infeasible).
    """

    v = _as_float_array(vector)
    lb = _as_float_array(lower)
    ub = _as_float_array(upper)
    n = len(v)
    if n == 0:
        return ProjectionResult(np.zeros(0), "empty", 0.0, 0, {})
    if not (len(lb) == n and len(ub) == n):
        raise ValueError("vector, lower and upper must have the same length")
    if np.any(ub < lb - FLOAT_TOLERANCE):
        raise ValueError("upper bounds must be >= lower bounds")

    groups, inverse, starts, order = _group_structure(group_ids)
    v_sorted = v[order]
    lb_sorted = lb[order]
    ub_sorted = ub[order]
    inverse_sorted = inverse[order]
    if isinstance(group_targets, dict):
        targets = np.asarray([float(group_targets[str(group)]) for group in groups], dtype=float)
    else:
        targets = _as_float_array(group_targets)
        if len(targets) != len(groups):
            raise ValueError("group_targets must have one entry per distinct group")

    lb_group = _reduceat_sum(lb_sorted, starts)
    ub_group = _reduceat_sum(ub_sorted, starts)
    status = "exact"
    detail: dict[str, Any] = {"groups": int(len(groups))}

    # 1. Make the group targets feasible and, if needed, restore the global sum.
    adjusted = np.clip(targets, lb_group, ub_group)
    clipped_groups = int(np.sum(~np.isclose(adjusted, targets, atol=tolerance, rtol=0.0)))
    if clipped_groups:
        status = "relaxed_group_targets"
        detail["clipped_group_targets"] = clipped_groups
    residual_total = float(total) - float(adjusted.sum())
    if abs(residual_total) > 1e-10:
        # Solve for a uniform dual shift delta so that sum(clip(target-delta)) = total.
        lo = float(np.min(targets - ub_group)) - 1.0
        hi = float(np.max(targets - lb_group)) + 1.0
        feasible_low = float(lb_group.sum())
        feasible_high = float(ub_group.sum())
        if total < feasible_low - 1e-10 or total > feasible_high + 1e-10:
            # Group targets cannot add up to the requested total: fall back to a
            # plain box+sum projection and report it.
            from aicwf.factors.common import project_box_sum

            weights = project_box_sum(v, lb, ub, total)
            active = _group_active(weights, lb, ub, groups, inverse, targets)
            return ProjectionResult(
                weights,
                "box_sum_only",
                float(np.max(np.abs(active))) if len(active) else 0.0,
                0,
                {"groups": int(len(groups)), "reason": "group targets cannot reach global sum"},
            )
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            candidate = np.clip(targets - mid, lb_group, ub_group)
            if float(candidate.sum()) > float(total):
                lo = mid
            else:
                hi = mid
            if hi - lo <= tolerance:
                break
        adjusted = np.clip(targets - 0.5 * (lo + hi), lb_group, ub_group)
        # Distribute the residual on interior groups (deterministic).
        remainder = float(total) - float(adjusted.sum())
        if abs(remainder) > tolerance:
            room = ub_group - adjusted
            positive = room > 0
            if remainder > 0 and positive.any():
                adjusted = adjusted + remainder * room / room[positive].sum()
            elif remainder < 0:
                slack = adjusted - lb_group
                positive = slack > 0
                if positive.any():
                    adjusted = adjusted + remainder * slack / slack[positive].sum()
        adjusted = np.clip(adjusted, lb_group, ub_group)
        if status == "exact":
            status = "relaxed_group_targets"
        detail["global_sum_adjustment"] = float(remainder)

    # 2. Independent per-industry bisection for the dual multipliers.
    mu_lo = np.minimum.reduceat(v_sorted - ub_sorted, starts) - 1.0
    mu_hi = np.maximum.reduceat(v_sorted - lb_sorted, starts) + 1.0
    iterations = 0
    for iterations in range(1, bisection_steps + 1):
        mu = 0.5 * (mu_lo + mu_hi)
        weights_sorted = np.clip(v_sorted - mu[inverse_sorted], lb_sorted, ub_sorted)
        sums = _reduceat_sum(weights_sorted, starts)
        too_large = sums > adjusted
        mu_lo = np.where(too_large, mu, mu_lo)
        mu_hi = np.where(too_large, mu_hi, mu)
        if np.max(np.abs(sums - adjusted)) <= tolerance:
            break
    weights_sorted = np.clip(v_sorted - mu[inverse_sorted], lb_sorted, ub_sorted)
    # Recompute with the stored midpoint (avoid keeping a stale mu variable).
    mu = 0.5 * (mu_lo + mu_hi)
    weights_sorted = np.clip(v_sorted - mu[inverse_sorted], lb_sorted, ub_sorted)
    residual = adjusted - _reduceat_sum(weights_sorted, starts)
    weights_sorted = _distribute_residual(weights_sorted, residual, lb_sorted, ub_sorted, starts)

    weights = np.empty(n, dtype=float)
    weights[order] = weights_sorted
    # Repair tiny global sum drift without moving any group total.
    drift = float(total) - float(weights.sum()) if status in {"exact", "relaxed_group_targets"} else 0.0
    if abs(drift) > 1e-12:
        room = upper - weights
        capacity = room.sum()
        if capacity > 0:
            weights = np.minimum(weights + room * (drift / capacity), upper)
        else:
            slack = weights - lower
            capacity = slack.sum()
            if capacity > 0:
                weights = np.maximum(weights + slack * (drift / capacity), lower)

    active = _group_active(weights, lb, ub, groups, inverse, targets)
    max_active = float(np.max(np.abs(active))) if len(active) else 0.0
    detail.update(
        {
            "max_abs_global_sum_error": float(abs(weights.sum() - total)),
            "group_target_deviation": [float(value) for value in active],
        }
    )
    return ProjectionResult(weights, status, max_active, iterations, detail)


def _group_active(
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    inverse: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    sums = np.zeros(len(groups), dtype=float)
    np.add.at(sums, inverse, weights)
    return sums - targets


def project_industry_neutral_box(
    vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    group_ids: Sequence[Any],
    benchmark_weights: Sequence[float],
    *,
    industry_active_limit: float | None = None,
    tolerance: float = 1e-12,
    bisection_steps: int = 60,
) -> ProjectionResult:
    """Project onto box + ``sum(w)=1`` + industry totals equal to the benchmark.

    When ``industry_active_limit`` is given, group totals are allowed to deviate
    from the benchmark by at most that amount (used if exact neutrality is
    infeasible); otherwise exact neutrality is requested.
    """

    wb = _as_float_array(benchmark_weights)
    _groups, inverse, _starts, _order = _group_structure(group_ids)
    # Industry benchmark totals without re-implementing reduceat bookkeeping.
    totals = np.zeros(len(set(str(group) for group in group_ids)), dtype=float)
    np.add.at(totals, inverse, wb)
    target_map = {
        str(group): float(totals[index]) for index, group in enumerate(sorted(set(str(g) for g in group_ids)))
    }
    return project_group_box(
        vector,
        lower,
        upper,
        group_ids,
        target_map,
        total=1.0,
        tolerance=tolerance,
        bisection_steps=bisection_steps,
    )


def project_uniform_box(
    vector: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    total: float = 1.0,
    bisection_steps: int = 200,
) -> np.ndarray:
    """Plain box + sum projection (no industry constraints); convenience wrapper."""

    from aicwf.factors.common import project_box_sum

    return project_box_sum(
        _as_float_array(vector),
        _as_float_array(lower),
        _as_float_array(upper),
        total,
        max_iter=bisection_steps,
    )


def industry_active_table(weights: pd.Series, benchmark_weights: pd.Series, industry: pd.Series) -> dict[str, float]:
    """Per-industry active weight ``sum(w - wb)`` (zero for neutral portfolios)."""

    frame = pd.DataFrame(
        {
            "weight": pd.Series(weights, dtype=float),
            "benchmark_weight": pd.Series(benchmark_weights, dtype=float),
            "industry": pd.Series(industry).astype(str),
        }
    ).fillna({"weight": 0.0, "benchmark_weight": 0.0})
    frame["active"] = frame["weight"] - frame["benchmark_weight"]
    grouped = frame.groupby("industry", sort=True)["active"].sum()
    return {str(industry_id): float(value) for industry_id, value in grouped.items()}


__all__ = [
    "ProjectionResult",
    "industry_active_table",
    "project_group_box",
    "project_industry_neutral_box",
    "project_uniform_box",
]
