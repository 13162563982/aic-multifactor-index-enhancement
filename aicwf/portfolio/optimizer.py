"""Constrained mean-variance portfolio optimiser for index enhancement.

Objective
---------
At a decision date, with ``wb`` the benchmark proxy weights, ``alpha`` the
cross-sectional composite score and ``Sigma`` the Ledoit-Wolf covariance of daily
returns (annualised), solve

.. math::

    \\max_w \\; \\alpha^\\top w - \\frac{\\lambda}{2} (w - w_b)^\\top \\Sigma (w - w_b)

subject to

* ``0 <= w_i <= max_weight`` (long-only, per-name weight cap),
* ``|w_i - wb_i| <= max_active_weight`` (per-name active cap),
* ``sum_i w_i = 1`` (fully invested),
* ``|sum_{i in g} w_i - sum_{i in g} wb_i| ~ 0`` (industry neutrality).

The feasible set is a box with industry-total equalities; the **exact**
Euclidean projection onto it is :func:`aicwf.portfolio.projection.project_group_box`
(one dual bisection per industry, vectorised).  The optimiser is a projected
gradient ascent with Nesterov acceleration and adaptive restart; because the
gradient is ``alpha - lambda * Sigma (w - wb)`` its Lipschitz constant is
``lambda * lambda_max(Sigma)``, which gives a step size ``1/L`` with guaranteed
convergence.

The tracking-error target is hit by **bisection on lambda**: ex-ante tracking
error ``sqrt((w-wb)' Sigma (w-wb))`` is monotone decreasing in ``lambda`` (the
risk-aversion parameter), so a geometric bisection over
``optimizer.lambda_bounds`` finds the ``lambda`` whose portfolio TE is closest to
the (clamped) target.  Every iteration is deterministic: no random initialisation,
fixed bisection count, stable tie-breaking.

Covariance
----------
``sklearn.covariance.LedoitWolf`` shrinks the sample covariance of the trailing
daily returns (``portfolio.covariance.lookback_days``) and the result is
annualised by ``sqrt(annualization)`` on both axes.  When there are too few
observations or the estimator fails, a deterministic diagonal covariance
(per-name variance, median-imputed) is used and flagged in the diagnostics.

Fallback
--------
:func:`rank_proportional_fallback` builds deterministic weights proportional to
the composite-score rank inside the same feasible set; the portfolio stage uses
it when the optimiser fails or the problem is degenerate, setting
``fallback_used = true`` in ``risk_report.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.portfolio.projection import (
    ProjectionResult,
    industry_active_table,
    project_industry_neutral_box,
)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


@dataclass
class CovarianceResult:
    covariance: pd.DataFrame
    method: str
    meta: dict[str, Any] = field(default_factory=dict)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualization: int = 252,
    min_obs: int = 20,
    eigen_floor: float = 1.0e-10,
    method: str = "ledoit_wolf",
    fallback: str = "diagonal",
) -> CovarianceResult:
    """Annualised Ledoit-Wolf covariance with deterministic diagonal fallback."""

    frame = pd.DataFrame(returns).astype(float)
    codes = list(frame.columns)
    if len(codes) == 0:
        return CovarianceResult(pd.DataFrame(), "empty", {"fallback_used": True})
    values = frame.to_numpy(dtype=float)
    finite_rows = np.isfinite(values).all(axis=1)
    clean = values[finite_rows]
    meta: dict[str, Any] = {
        "n_codes": len(codes),
        "n_observations": int(len(clean)),
        "annualization": int(annualization),
    }
    if len(clean) < max(5, int(min_obs)) or len(codes) < 2 or str(method).lower() != "ledoit_wolf":
        covariance = _diagonal_covariance(clean, values, codes, annualization)
        meta.update({"fallback_used": True, "method": fallback, "reason": "insufficient observations"})
        return CovarianceResult(covariance, str(fallback), meta)
    try:
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf(store_precision=False, assume_centered=False)
        estimator.fit(clean)
        matrix = np.asarray(estimator.covariance_, dtype=float) * float(annualization)
        shrinkage = float(getattr(estimator, "shrinkage_", float("nan")))
    except Exception as exc:  # noqa: BLE001 - degenerate inputs fall back deterministically
        covariance = _diagonal_covariance(clean, values, codes, annualization)
        meta.update({"fallback_used": True, "method": fallback, "reason": f"{type(exc).__name__}: {exc}"})
        return CovarianceResult(covariance, str(fallback), meta)
    matrix = _symmetrise_with_floor(matrix, eigen_floor)
    meta.update({"fallback_used": False, "method": "ledoit_wolf", "shrinkage": shrinkage})
    return CovarianceResult(pd.DataFrame(matrix, index=codes, columns=codes), "ledoit_wolf", meta)


def _diagonal_covariance(
    clean: np.ndarray, raw: np.ndarray, codes: Sequence[str], annualization: int
) -> pd.DataFrame:
    if len(clean) >= 2:
        variances = clean.var(axis=0, ddof=1)
    else:
        variances = np.zeros(len(codes), dtype=float)
    variances = np.where(np.isfinite(variances) & (variances >= 0), variances, np.nan)
    if np.all(~np.isfinite(variances)):
        variances = np.full(len(codes), 1e-4 if len(clean) < 2 else 1e-4, dtype=float)
    median = float(np.nanmedian(variances))
    if not np.isfinite(median) or median <= 0:
        median = 1e-4
    variances = np.where(np.isfinite(variances), variances, median)
    matrix = np.diag(np.maximum(variances, 1e-12) * float(annualization))
    return pd.DataFrame(matrix, index=list(codes), columns=list(codes))


def _symmetrise_with_floor(matrix: np.ndarray, eigen_floor: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        floor = max(float(eigen_floor), float(np.max(np.abs(eigenvalues))) * float(eigen_floor))
        eigenvalues = np.maximum(eigenvalues, floor)
        matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
        matrix = 0.5 * (matrix + matrix.T)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical corner case
        matrix = matrix + np.eye(len(matrix)) * 1e-8
    return matrix


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


@dataclass
class OptimizerResult:
    weights: pd.Series
    status: str
    iterations: int
    te_lambda: float
    ex_ante_te: float
    fallback_used: bool
    projection_status: str = "exact"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "iterations": int(self.iterations),
            "te_lambda": None if self.te_lambda is None else float(self.te_lambda),
            "ex_ante_te": None if self.ex_ante_te is None else float(self.ex_ante_te),
            "fallback_used": bool(self.fallback_used),
            "projection_status": self.projection_status,
            **{key: value for key, value in self.meta.items() if key not in {"weights"}},
        }


def ex_ante_te(active_weights: np.ndarray, covariance: np.ndarray) -> float:
    value = float(active_weights @ covariance @ active_weights)
    return float(math.sqrt(max(value, 0.0)))


def _group_targets(benchmark: pd.Series, industry: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"benchmark": benchmark.astype(float), "industry": industry.astype(str)})
    grouped = frame.groupby("industry", sort=True)["benchmark"].sum()
    return {str(key): float(value) for key, value in grouped.items()}


def feasible_bounds(
    benchmark: pd.Series,
    constraints: Mapping[str, Any],
    *,
    weight_margin: float = 0.985,
    min_holding: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds implied by the per-name weight and active-weight caps.

    ``min_holding`` (optional) is the minimum weight needed to buy one board lot;
    it is only applied where it fits inside the active cap.
    """

    max_weight = float(constraints.get("max_weight", 1.0)) * float(weight_margin)
    max_active = float(constraints.get("max_active_weight", 1.0)) * float(weight_margin)
    wb = benchmark.to_numpy(dtype=float)
    upper = np.minimum(max_weight, wb + max_active)
    lower = np.maximum(wb - max_active, 0.0)
    if min_holding is not None:
        floor = min_holding.reindex(benchmark.index).fillna(0.0).to_numpy(dtype=float)
        valid = floor <= upper
        lower = np.where(valid, np.maximum(lower, np.minimum(floor, upper)), lower)
    lower = np.minimum(lower, upper)
    return lower, upper


def _fit_group_floors(
    lower: np.ndarray,
    upper: np.ndarray,
    wb: np.ndarray,
    industry: Sequence[str],
    *,
    max_active: float,
) -> np.ndarray:
    """Keep the box bounds feasible with respect to the industry totals."""

    industry_array = np.asarray([str(value) for value in industry], dtype=object)
    base = np.maximum(wb - float(max_active), 0.0)
    floors = np.maximum(lower - base, 0.0)
    for group in sorted(set(industry_array.tolist())):
        mask = industry_array == group
        target = float(wb[mask].sum())
        base_total = float(base[mask].sum())
        headroom = target - base_total
        if headroom <= 0:
            floors[mask] = 0.0
            continue
        extra = float(floors[mask].sum())
        if extra > headroom * 0.98:
            floors[mask] = floors[mask] * (headroom * 0.98 / extra)
    return np.minimum(base + floors, upper)


def optimize_weights(
    *,
    alpha: pd.Series,
    benchmark: pd.Series,
    covariance: pd.DataFrame,
    industry: pd.Series,
    constraints: Mapping[str, Any],
    optimizer_cfg: Mapping[str, Any] | None = None,
    previous_weights: pd.Series | None = None,
    min_holding: pd.Series | None = None,
    weight_margin: float = 0.985,
) -> OptimizerResult:
    """Projected-gradient mean-variance solution with TE targeting by lambda bisection."""

    codes = list(alpha.index)
    wb = benchmark.reindex(codes).fillna(0.0).astype(float)
    total = float(wb.sum())
    if total <= 0:
        raise ValueError("benchmark proxy weights sum to zero")
    wb = wb / total
    cov = covariance.reindex(index=codes, columns=codes).fillna(0.0).to_numpy(dtype=float)
    cov = 0.5 * (cov + cov.T)
    alpha_values = alpha.reindex(codes).astype(float).to_numpy(dtype=float)
    alpha_values = np.where(np.isfinite(alpha_values), alpha_values, 0.0)
    spread = float(alpha_values.std(ddof=0))
    if spread > 0:
        alpha_values = (alpha_values - alpha_values.mean()) / spread
    wb_values = wb.to_numpy(dtype=float)
    industry_values = industry.reindex(codes).fillna("UNKNOWN").astype(str).tolist()
    group_targets = _group_targets(wb, pd.Series(industry_values, index=codes))

    lower, upper = feasible_bounds(
        wb, constraints, weight_margin=weight_margin, min_holding=min_holding
    )
    if np.any(lower > upper + 1e-12):
        raise ValueError("infeasible weight bounds")
    if float(upper.sum()) < 1.0 - 1e-9 or float(lower.sum()) > 1.0 + 1e-9:
        raise ValueError(
            "infeasible weight bounds: the name caps cannot fund a fully invested portfolio"
        )
    max_active_bound = float(constraints.get("max_active_weight", 1.0)) * float(weight_margin)
    lower = _fit_group_floors(
        lower, upper, wb_values, industry_values, max_active=max_active_bound
    )

    def project(vector: np.ndarray) -> ProjectionResult:
        return project_industry_neutral_box(
            vector,
            lower,
            upper,
            industry_values,
            wb_values,
            bisection_steps=60,
        )

    base_projection = project(wb_values)
    feasible_benchmark = base_projection.weights
    start = feasible_benchmark
    if previous_weights is not None:
        previous = previous_weights.reindex(codes).fillna(0.0).to_numpy(dtype=float)
        start = project(previous).weights

    try:
        eigenvalue_max = float(np.max(np.linalg.eigvalsh(cov))) if len(codes) > 1 else float(cov[0, 0])
    except np.linalg.LinAlgError:  # pragma: no cover
        eigenvalue_max = float(np.max(np.diag(cov)))
    if not np.isfinite(eigenvalue_max) or eigenvalue_max <= 0:
        eigenvalue_max = float(np.max(np.diag(cov))) if len(codes) else 1e-8
    if not np.isfinite(eigenvalue_max) or eigenvalue_max <= 0:
        raise ValueError("covariance is degenerate")

    optimizer_cfg = dict(optimizer_cfg or {})
    bisection_steps = int(optimizer_cfg.get("lambda_bisection_steps", 24) or 24)
    max_iter = int(optimizer_cfg.get("max_iter", 200) or 200)
    tolerance = float(optimizer_cfg.get("tol", 1e-9) or 1e-9)
    lambda_bounds = [float(value) for value in (optimizer_cfg.get("lambda_bounds") or [1e-6, 1e6])]
    lambda_low = float(min(lambda_bounds))
    lambda_high = float(max(lambda_bounds))
    te_target = float(constraints.get("te_target", 0.05))

    solver = _ProjectedGradientSolver(
        alpha=alpha_values,
        benchmark=wb_values,
        covariance=cov,
        project=project,
        max_iter=max_iter,
        tolerance=tolerance,
        eigenvalue_max=eigenvalue_max,
    )

    def solve(lam: float, warm: _SolveResult | None = None) -> _SolveResult:
        return solver.solve(lam, start=start if warm is None else warm.weights)

    high_result = solve(lambda_high)
    low_result = solve(lambda_low)
    if high_result.ex_ante_te > te_target:
        best = high_result
        status = "te_target_unreachable_low"
    elif low_result.ex_ante_te < te_target:
        best = low_result
        status = "te_target_unreachable_high"
    else:
        best = low_result
        lo, hi = lambda_low, lambda_high
        prior = low_result
        for _ in range(max(1, bisection_steps)):
            mid = math.sqrt(lo * hi)
            candidate = solve(mid, warm=prior)
            if candidate.ex_ante_te > te_target:
                lo = mid
            else:
                hi = mid
            prior = candidate
            if abs(candidate.ex_ante_te - te_target) <= abs(best.ex_ante_te - te_target):
                best = candidate
            if abs(hi - lo) <= max(1e-12, 1e-9 * hi):
                break
        status = "ok"
    weights = pd.Series(best.weights, index=codes, dtype=float)
    projection = project(weights.to_numpy(dtype=float))
    industry_active = industry_active_table(weights, wb, industry.reindex(codes))
    max_industry_active = max(abs(value) for value in industry_active.values()) if industry_active else 0.0
    meta = {
        "covariance_method": "annualised",
        "max_industry_active": float(max_industry_active),
        "lambda_low": lambda_low,
        "lambda_high": lambda_high,
        "te_target": te_target,
        "bracket_low_te": float(low_result.ex_ante_te),
        "bracket_high_te": float(high_result.ex_ante_te),
        "n_names": int((weights > 0).sum()),
        "objective": float(
            float(alpha_values @ weights.to_numpy(dtype=float))
            - 0.5 * float(best.te_lambda) * float(best.ex_ante_te) ** 2
        ),
        "projection_status": projection.status,
    }
    return OptimizerResult(
        weights=weights,
        status=str(status),
        iterations=int(best.iterations),
        te_lambda=float(best.te_lambda),
        ex_ante_te=float(best.ex_ante_te),
        fallback_used=False,
        projection_status=projection.status,
        meta=meta,
    )


@dataclass
class _SolveResult:
    weights: np.ndarray
    iterations: int
    te_lambda: float
    ex_ante_te: float
    projection_status: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ProjectedGradientSolver:
    alpha: np.ndarray
    benchmark: np.ndarray
    covariance: np.ndarray
    project: Any
    max_iter: int
    tolerance: float
    eigenvalue_max: float

    def solve(self, lam: float, *, start: np.ndarray) -> _SolveResult:
        lam = float(lam)
        lipschitz = max(lam * float(self.eigenvalue_max), 1e-12)
        step = 1.0 / lipschitz
        w = np.asarray(start, dtype=float).copy()
        y = w.copy()
        t = 1.0
        active = w - self.benchmark
        objective = float(self.alpha @ w) - 0.5 * lam * float(active @ self.covariance @ active)
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            y_active = y - self.benchmark
            gradient = self.alpha - lam * (self.covariance @ y_active)
            candidate = self.project(y + step * gradient).weights
            candidate_active = candidate - self.benchmark
            candidate_objective = float(self.alpha @ candidate) - 0.5 * lam * float(
                candidate_active @ self.covariance @ candidate_active
            )
            change = float(np.max(np.abs(candidate - w))) if len(candidate) else 0.0
            if candidate_objective >= objective:  # FISTA momentum step
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                y = candidate + ((t - 1.0) / t_next) * (candidate - w)
                t = t_next
            else:  # adaptive restart keeps the accelerated scheme monotone
                y = candidate
                t = 1.0
            w = candidate
            objective = candidate_objective
            if change <= self.tolerance:
                break
        active = w - self.benchmark
        te = ex_ante_te(active, self.covariance)
        projection = self.project(w)
        return _SolveResult(
            weights=np.asarray(w, dtype=float),
            iterations=int(iterations),
            te_lambda=lam,
            ex_ante_te=float(te),
            projection_status=projection.status,
            meta={"projection_max_industry_active": float(projection.max_abs_industry_active)},
        )


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def rank_proportional_fallback(
    *,
    alpha: pd.Series,
    benchmark: pd.Series,
    industry: pd.Series,
    constraints: Mapping[str, Any],
    min_holding: pd.Series | None = None,
    weight_margin: float = 0.985,
) -> OptimizerResult:
    """Deterministic rank-proportional clipped weights (used when the QP fails).

    Scores are converted to cross-sectional ranks in ``[0, 1]``; the active tilt
    is proportional to ``rank - mean(rank)`` scaled to at most
    ``max_active_weight`` and projected onto the same feasible set.
    """

    codes = list(alpha.index)
    wb = benchmark.reindex(codes).fillna(0.0).astype(float)
    total = float(wb.sum())
    if total <= 0:
        raise ValueError("benchmark proxy weights sum to zero")
    wb = wb / total
    wb_values = wb.to_numpy(dtype=float)
    industry_values = industry.reindex(codes).fillna("UNKNOWN").astype(str).tolist()
    group_targets = _group_targets(wb, pd.Series(industry_values, index=codes))
    lower, upper = feasible_bounds(
        wb, constraints, weight_margin=weight_margin, min_holding=min_holding
    )
    max_active_bound = float(constraints.get("max_active_weight", 1.0)) * float(weight_margin)
    lower = _fit_group_floors(
        lower, upper, wb_values, industry_values, max_active=max_active_bound
    )
    ranks = pd.Series(alpha.reindex(codes).astype(float)).rank(method="average", pct=True).to_numpy(dtype=float)
    ranks = np.where(np.isfinite(ranks), ranks, 0.5)
    tilt = ranks - float(np.mean(ranks)) if len(ranks) else np.zeros(0)
    max_abs = float(np.max(np.abs(tilt))) if len(tilt) else 0.0
    max_active = float(constraints.get("max_active_weight", 0.1)) * float(weight_margin) * 0.9
    scale = (max_active / max_abs) if max_abs > 0 else 0.0
    raw = wb_values + scale * tilt
    projection = project_industry_neutral_box(
        raw,
        lower,
        upper,
        industry_values,
        wb_values,
        bisection_steps=60,
    )
    weights = pd.Series(projection.weights, index=codes, dtype=float)
    industry_active = industry_active_table(weights, wb, industry.reindex(codes))
    te = None
    return OptimizerResult(
        weights=weights,
        status="fallback_rank_proportional",
        iterations=0,
        te_lambda=float("nan"),
        ex_ante_te=float("nan") if te is None else float(te),
        fallback_used=True,
        projection_status=projection.status,
        meta={
            "max_industry_active": float(max(abs(value) for value in industry_active.values()))
            if industry_active
            else 0.0,
            "n_names": int((weights > 0).sum()),
        },
    )


__all__ = [
    "CovarianceResult",
    "OptimizerResult",
    "ex_ante_te",
    "ledoit_wolf_covariance",
    "optimize_weights",
    "rank_proportional_fallback",
]
