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

Untradable names (suspended / not covered by the signal) are **fixed** at their
current holding weights; the optimiser works on the remaining free names with
industry targets reduced by the fixed block's industry weights.  The feasible set
is a box with industry-total equalities; the exact Euclidean projection onto it is
:func:`aicwf.portfolio.projection.project_group_box` (one dual bisection per
industry, vectorised).  The optimiser is a projected gradient ascent with Nesterov
acceleration and adaptive restart; because the gradient is
``alpha - lambda * Sigma (w - wb)`` its Lipschitz constant is
``lambda * lambda_max(Sigma_ff)`` (the free-free block), which gives a step size
``1/L`` with guaranteed convergence.

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
    project_group_box,
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
        variances = np.full(len(codes), 1e-4, dtype=float)
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
# Constraints helpers
# ---------------------------------------------------------------------------


def feasible_bounds(
    benchmark: pd.Series,
    constraints: Mapping[str, Any],
    *,
    weight_margin: float = 0.985,
    min_holding: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds implied by the per-name weight and active-weight caps."""

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
    """Shrink minimum-lot floors until every industry target stays feasible."""

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


def _group_targets(benchmark: pd.Series, industry: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"benchmark": benchmark.astype(float), "industry": industry.astype(str)})
    grouped = frame.groupby("industry", sort=True)["benchmark"].sum()
    return {str(key): float(value) for key, value in grouped.items()}


def _industry_totals(wb: np.ndarray, industry: Sequence[str]) -> dict[str, float]:
    frame = pd.DataFrame({"benchmark": wb, "industry": [str(value) for value in industry]})
    return {str(key): float(value) for key, value in frame.groupby("industry", sort=True)["benchmark"].sum().items()}


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
            "te_lambda": None if self.te_lambda is None or not math.isfinite(float(self.te_lambda)) else float(self.te_lambda),
            "ex_ante_te": None if self.ex_ante_te is None else float(self.ex_ante_te),
            "fallback_used": bool(self.fallback_used),
            "projection_status": self.projection_status,
            **{key: value for key, value in self.meta.items()},
        }


def ex_ante_te(active_weights: np.ndarray, covariance: np.ndarray) -> float:
    value = float(active_weights @ covariance @ active_weights)
    return float(math.sqrt(max(value, 0.0)))


@dataclass
class _Problem:
    """Free-variable view of the constrained problem (fixed names held constant)."""

    codes: list[str]
    industry: list[str]
    wb: np.ndarray
    free: np.ndarray
    fixed_weights: np.ndarray
    covariance: np.ndarray
    alpha: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    free_codes: list[str]
    industry_free: list[str]
    wb_free: np.ndarray
    target_free: dict[str, float]
    total_free: float

    @property
    def n_free(self) -> int:
        return int(self.free.sum())

    def full_vector(self, free_values: np.ndarray) -> np.ndarray:
        full = self.fixed_weights.copy()
        full[self.free] = np.asarray(free_values, dtype=float)
        return full

    def project(self, free_values: np.ndarray) -> ProjectionResult:
        result = project_group_box(
            free_values,
            self.lower,
            self.upper,
            self.industry_free,
            self.target_free,
            total=self.total_free,
            bisection_steps=60,
        )
        full = self.full_vector(result.weights)
        return ProjectionResult(
            weights=full,
            status=result.status,
            max_abs_industry_active=result.max_abs_industry_active,
            iterations=result.iterations,
            detail=result.detail,
        )


def _build_problem(
    *,
    alpha: pd.Series,
    benchmark: pd.Series,
    covariance: pd.DataFrame,
    industry: pd.Series,
    constraints: Mapping[str, Any],
    weight_margin: float,
    min_holding: pd.Series | None,
    tradable: pd.Series | None,
    tilt_mask: pd.Series | None,
    current_weights: pd.Series | None,
) -> _Problem:
    codes = list(alpha.index)
    wb = pd.Series(benchmark, dtype=float).reindex(codes).fillna(0.0).astype(float)
    total = float(wb.sum())
    if total <= 0:
        raise ValueError("benchmark proxy weights sum to zero")
    wb = wb / total
    industry_values = pd.Series(industry, dtype=object).reindex(codes).fillna("UNKNOWN").astype(str).tolist()
    if tradable is None:
        tradeable = np.ones(len(codes), dtype=bool)
    else:
        tradeable = pd.Series(tradable, dtype=bool).reindex(codes).fillna(False).to_numpy(dtype=bool)
    current = pd.Series(current_weights, dtype=float).reindex(codes).fillna(0.0).astype(float) if current_weights is not None else pd.Series(0.0, index=codes)
    current_values = current.to_numpy(dtype=float)
    wb_values_raw = wb.to_numpy(dtype=float)
    fixed_values = np.where(tradeable, 0.0, np.where(current_values > 1e-12, current_values, wb_values_raw))
    fixed_total = float(fixed_values.sum())
    free_mask = tradeable.copy()
    if fixed_total > 1.0 + 1e-9:
        raise ValueError("fixed holdings already exceed a fully invested portfolio")
    total_free = 1.0 - fixed_total

    wb_values = wb.to_numpy(dtype=float)
    lower_full, upper_full = feasible_bounds(
        wb, constraints, weight_margin=weight_margin, min_holding=min_holding
    )
    max_active_bound = float(constraints.get("max_active_weight", 1.0)) * float(weight_margin)
    lower_full = _fit_group_floors(lower_full, upper_full, wb_values, industry_values, max_active=max_active_bound)

    free_lower = lower_full[free_mask]
    free_upper = upper_full[free_mask]
    wb_free = wb_values[free_mask]
    industry_free = [industry_values[index] for index in np.flatnonzero(free_mask)]
    free_codes = [codes[index] for index in np.flatnonzero(free_mask)]

    full_targets = _industry_totals(wb_values, industry_values)
    fixed_targets = _industry_totals(fixed_values, industry_values)
    target_free = {key: float(full_targets.get(key, 0.0) - fixed_targets.get(key, 0.0)) for key in full_targets}
    if sum(target_free.values()) > total_free + 1e-9:
        # The fixed block may "over-hold" an industry relative to its benchmark
        # weight; the projection clips the free targets deterministically.
        pass
    covariance_values = covariance.reindex(index=codes, columns=codes).fillna(0.0).to_numpy(dtype=float)
    covariance_values = 0.5 * (covariance_values + covariance_values.T)
    alpha_raw = pd.Series(alpha, dtype=float).reindex(codes).fillna(0.0)
    alpha_values = np.zeros(len(codes), dtype=float)
    if tilt_mask is None:
        tilt_values = tradeable.copy()
    else:
        tilt_values = pd.Series(tilt_mask, dtype=bool).reindex(codes).fillna(False).to_numpy(dtype=bool) & tradeable
    if tilt_values.any():
        spread = float(alpha_raw[tilt_values].std(ddof=0))
        if spread > 0:
            alpha_values[tilt_values] = (
                alpha_raw[tilt_values].to_numpy(dtype=float) - float(alpha_raw[tilt_values].mean())
            ) / spread
    return _Problem(
        codes=codes,
        industry=industry_values,
        wb=wb_values,
        free=free_mask,
        fixed_weights=fixed_values,
        covariance=covariance_values,
        alpha=alpha_values,
        lower=free_lower,
        upper=free_upper,
        free_codes=free_codes,
        industry_free=industry_free,
        wb_free=wb_free,
        target_free=target_free,
        total_free=total_free,
    )


@dataclass
class _SolveResult:
    free_values: np.ndarray
    iterations: int
    te_lambda: float
    ex_ante_te: float
    projection_status: str


def _run_solver(
    problem: _Problem,
    *,
    lam: float,
    start_free: np.ndarray,
    max_iter: int,
    tolerance: float,
    eigenvalue_max: float,
    alpha_full_dot_fixed: float,
    linear_free: np.ndarray,
    quadratic_free: np.ndarray,
    full_benchmark: np.ndarray,
) -> _SolveResult:
    """Accelerated projected gradient on the free variables for one lambda."""

    cov = problem.covariance
    free = problem.free
    fixed = problem.fixed_weights
    cov_ff = cov[np.ix_(free, free)]
    active_fixed = fixed - full_benchmark
    offset = cov[np.ix_(free, np.flatnonzero(~free))] @ active_fixed[~free] if (~free).any() else np.zeros(problem.n_free)
    lipschitz = max(float(lam) * float(eigenvalue_max), 1e-12)
    step = 1.0 / lipschitz
    wb_free = problem.wb_free
    alpha_free = problem.alpha[free]

    def full_of(free_values: np.ndarray) -> np.ndarray:
        return problem.full_vector(free_values)

    def objective(free_values: np.ndarray) -> float:
        full = full_of(free_values)
        active = full - full_benchmark
        return float(problem.alpha @ full) - 0.5 * float(lam) * float(active @ cov @ active)

    def gradient(free_values: np.ndarray) -> np.ndarray:
        return alpha_free - float(lam) * (cov_ff @ (free_values - wb_free) + offset)

    x = np.asarray(start_free, dtype=float).copy()
    y = x.copy()
    t = 1.0
    value = objective(x)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        candidate = problem.project(y + step * gradient(y)).weights[free]
        candidate_value = objective(candidate)
        change = float(np.max(np.abs(candidate - x))) if len(candidate) else 0.0
        if candidate_value >= value:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = candidate + ((t - 1.0) / t_next) * (candidate - x)
            t = t_next
        else:
            y = candidate
            t = 1.0
        x = candidate
        value = candidate_value
        if change <= tolerance:
            break
    full = full_of(x)
    active = full - full_benchmark
    te = ex_ante_te(active, problem.covariance)
    projection = problem.project(x)
    return _SolveResult(
        free_values=x,
        iterations=int(iterations),
        te_lambda=float(lam),
        ex_ante_te=float(te),
        projection_status=projection.status,
    )


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
    tradable: pd.Series | None = None,
    tilt_mask: pd.Series | None = None,
    current_weights: pd.Series | None = None,
) -> OptimizerResult:
    """Projected-gradient mean-variance solution with TE targeting by lambda bisection."""

    problem = _build_problem(
        alpha=alpha,
        benchmark=benchmark,
        covariance=covariance,
        industry=industry,
        constraints=constraints,
        weight_margin=weight_margin,
        min_holding=min_holding,
        tradable=tradable,
        tilt_mask=tilt_mask,
        current_weights=current_weights,
    )
    free_mask = problem.free
    if problem.n_free < 1:
        weights = pd.Series(problem.fixed_weights, index=problem.codes, dtype=float)
        return OptimizerResult(
            weights=weights,
            status="no_tradable_names",
            iterations=0,
            te_lambda=float("nan"),
            ex_ante_te=ex_ante_te(problem.fixed_weights - problem.wb, problem.covariance),
            fallback_used=False,
            projection_status="exact",
            meta={"n_names": int((weights > 0).sum()), "max_industry_active": 0.0},
        )
    if float(problem.upper.sum()) < problem.total_free - 1e-9 or float(problem.lower.sum()) > problem.total_free + 1e-9:
        raise ValueError("infeasible free-variable bounds for a fully invested portfolio")

    cov_ff = problem.covariance[np.ix_(free_mask, free_mask)]
    try:
        eigenvalue_max = float(np.max(np.linalg.eigvalsh(cov_ff))) if problem.n_free > 0 else 1.0
    except np.linalg.LinAlgError:  # pragma: no cover
        eigenvalue_max = float(np.max(np.diag(cov_ff)))
    if not np.isfinite(eigenvalue_max) or eigenvalue_max <= 0:
        eigenvalue_max = float(np.max(np.diag(cov_ff))) if problem.n_free else 1e-8
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

    if previous_weights is not None:
        previous = pd.Series(previous_weights, dtype=float).reindex(problem.codes).fillna(0.0).to_numpy()
        previous_free = previous[free_mask]
        start_free = problem.project(previous_free).weights[free_mask]
    else:
        start_free = problem.project(problem.wb_free).weights[free_mask]

    def solve(lam: float, warm: _SolveResult | None = None) -> _SolveResult:
        return _run_solver(
            problem,
            lam=lam,
            start_free=start_free if warm is None else warm.free_values,
            max_iter=max_iter,
            tolerance=tolerance,
            eigenvalue_max=eigenvalue_max,
            alpha_full_dot_fixed=0.0,
            linear_free=np.zeros(problem.n_free),
            quadratic_free=np.zeros(problem.n_free),
            full_benchmark=problem.wb,
        )

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

    full_weights = problem.full_vector(best.free_values)
    weights = pd.Series(full_weights, index=problem.codes, dtype=float)
    projection = problem.project(best.free_values)
    industry_active = industry_active_table(weights, pd.Series(problem.wb, index=problem.codes), pd.Series(problem.industry, index=problem.codes))
    max_industry_active = max(abs(value) for value in industry_active.values()) if industry_active else 0.0
    meta = {
        "max_industry_active": float(max_industry_active),
        "lambda_low": lambda_low,
        "lambda_high": lambda_high,
        "te_target": te_target,
        "bracket_low_te": float(low_result.ex_ante_te),
        "bracket_high_te": float(high_result.ex_ante_te),
        "n_names": int((weights > 0).sum()),
        "n_free": int(problem.n_free),
        "fixed_weight": float(float(problem.fixed_weights.sum())),
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
    tradable: pd.Series | None = None,
    tilt_mask: pd.Series | None = None,
    current_weights: pd.Series | None = None,
    covariance: pd.DataFrame | None = None,
) -> OptimizerResult:
    """Deterministic rank-proportional clipped weights (used when the QP fails).

    Scores are converted to cross-sectional ranks in ``[0, 1]``; the active tilt
    is proportional to ``rank - mean(rank)`` scaled to at most
    ``max_active_weight`` and projected onto the same feasible set.  Untradable
    names stay fixed at their current holding weights, exactly as in the optimiser.
    """

    if covariance is None:
        codes = list(alpha.index)
        covariance = pd.DataFrame(np.eye(len(codes)), index=codes, columns=codes)
    problem = _build_problem(
        alpha=alpha,
        benchmark=benchmark,
        covariance=covariance,
        industry=industry,
        constraints=constraints,
        weight_margin=weight_margin,
        min_holding=min_holding,
        tradable=tradable,
        tilt_mask=tilt_mask,
        current_weights=current_weights,
    )
    free_mask = problem.free
    if problem.n_free < 1:
        weights = pd.Series(problem.fixed_weights, index=problem.codes, dtype=float)
        return OptimizerResult(
            weights=weights,
            status="fallback_rank_proportional",
            iterations=0,
            te_lambda=float("nan"),
            ex_ante_te=float("nan"),
            fallback_used=True,
            projection_status="exact",
            meta={"n_names": int((weights > 0).sum()), "max_industry_active": 0.0},
        )
    alpha_free = problem.alpha[free_mask]
    ranks = pd.Series(alpha_free).rank(method="average", pct=True).to_numpy(dtype=float)
    ranks = np.where(np.isfinite(ranks), ranks, 0.5)
    tilt = ranks - float(np.mean(ranks))
    max_abs = float(np.max(np.abs(tilt))) if len(tilt) else 0.0
    max_active = float(constraints.get("max_active_weight", 0.1)) * float(weight_margin) * 0.9
    scale = (max_active / max_abs) if max_abs > 0 else 0.0
    raw_free = problem.wb_free + scale * tilt
    projection = problem.project(raw_free)
    weights = pd.Series(projection.weights, index=problem.codes, dtype=float)
    industry_active = industry_active_table(
        weights,
        pd.Series(problem.wb, index=problem.codes),
        pd.Series(problem.industry, index=problem.codes),
    )
    active = problem.full_vector(projection.weights[free_mask]) - problem.wb
    return OptimizerResult(
        weights=weights,
        status="fallback_rank_proportional",
        iterations=0,
        te_lambda=float("nan"),
        ex_ante_te=ex_ante_te(active, problem.covariance),
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
    "feasible_bounds",
    "ledoit_wolf_covariance",
    "optimize_weights",
    "rank_proportional_fallback",
]
