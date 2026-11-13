"""Industry + size neutralisation of cross-sectional factor scores.

For every formation date we run one cross-sectional ordinary least squares
regression of the rank z-score on industry dummies and log market size, then
keep the residual (which is orthogonal to both exposures by construction) and
re-standardise it:

.. math::

    z_{i,t} = X_{i,t} \\beta_t + \\varepsilon_{i,t}, \\qquad
    X_{i,t} = [1,\\; D^{ind}_{i,t},\\; \\log(\\text{size})_{i,t}]

    \\tilde z_{i,t} = \\frac{\\varepsilon_{i,t} - \\bar\\varepsilon_t}{sd(\\varepsilon_t)}

where :math:`D^{ind}` is the full set of industry dummies (the first category is
dropped because the intercept spans it) and size is centred and scaled for
numerical conditioning.  The residual z-score :math:`\\tilde z` therefore has a
zero cross-sectional mean, unit standard deviation, zero industry-demeaned mean
in every industry, and zero sample correlation with log size up to numerical
precision.  Inactive or missing names (``z`` is NaN) are excluded from the
regression for that date and stay NaN afterwards.

A date with fewer than ``min_obs`` valid names cannot support the regression;
for such dates we fall back to a plain cross-sectional de-mean and record the
event in the returned diagnostics so it shows up in ``factor_report.json``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def neutralise_cross_section(
    z: pd.DataFrame,
    industry: pd.DataFrame,
    log_size: pd.DataFrame,
    *,
    min_obs: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Neutralise one date x code matrix against industry and log size."""

    z = pd.DataFrame(z).astype(float)
    industry = pd.DataFrame(industry).reindex_like(z).fillna("UNKNOWN")
    log_size = pd.DataFrame(log_size).reindex_like(z).astype(float)

    codes = list(z.columns)
    out = np.full(z.shape, np.nan, dtype=float)
    z_values = z.to_numpy(dtype=float)
    size_values = log_size.to_numpy(dtype=float)
    industry_values = industry.to_numpy(dtype=object)

    fallback_dates: list[str] = []
    used_dates = 0
    for row in range(z_values.shape[0]):
        scores = z_values[row]
        valid = np.isfinite(scores)
        count = int(valid.sum())
        if count == 0:
            continue
        if count < max(2, int(min_obs)):
            fallback_dates.append(str(z.index[row]))
            residual = _demean_and_standardise(scores[valid])
            out[row, valid] = residual
            used_dates += 1
            continue
        design = _design_matrix(industry_values[row][valid], size_values[row][valid])
        if design.shape[1] >= count:
            fallback_dates.append(str(z.index[row]))
            residual = _demean_and_standardise(scores[valid])
            out[row, valid] = residual
            used_dates += 1
            continue
        beta, *_ = np.linalg.lstsq(design, scores[valid], rcond=None)
        fitted = design @ beta
        raw_residual = scores[valid] - fitted
        residual = _demean_and_standardise(raw_residual)
        out[row, valid] = residual
        used_dates += 1

    result = pd.DataFrame(out, index=z.index, columns=codes)
    diagnostics = {
        "dates_processed": used_dates,
        "fallback_dates": len(fallback_dates),
        "fallback_date_examples": fallback_dates[:5],
        "min_obs": int(min_obs),
    }
    return result, diagnostics


def _design_matrix(industries: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    n = len(industries)
    categories, inverse = np.unique(industries.astype(str), return_inverse=True)
    columns = [np.ones(n, dtype=float)]
    if len(categories) >= 2:
        dummies = np.eye(len(categories), dtype=float)[inverse][:, 1:]
        columns.append(dummies)
    size = np.asarray(sizes, dtype=float)
    finite = np.isfinite(size)
    if finite.sum() >= 2:
        fill = float(np.nanmean(size))
        size = np.where(finite, size, fill)
        spread = float(np.nanstd(size))
        if not np.isfinite(spread) or spread <= 1e-12:
            spread = 1.0
        columns.append(((size - float(np.nanmean(size))) / spread).reshape(n, 1))
    elif finite.any():
        columns.append(np.zeros((n, 1)))
    return np.column_stack(columns)


def _demean_and_standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    demeaned = values - float(np.mean(values))
    spread = float(np.std(demeaned, ddof=1)) if len(demeaned) > 1 else 0.0
    if not np.isfinite(spread) or spread <= 1e-12:
        return np.zeros_like(demeaned)
    return demeaned / spread


def neutralise_factors(
    z_scores: Mapping[str, pd.DataFrame],
    industry: pd.DataFrame,
    log_size: pd.DataFrame,
    *,
    min_obs: int = 5,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Neutralise every factor matrix; returns the neutralised matrices + diagnostics."""

    neutralised: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    for factor_id, matrix in z_scores.items():
        neutral, detail = neutralise_cross_section(matrix, industry, log_size, min_obs=min_obs)
        neutralised[factor_id] = neutral
        diagnostics[factor_id] = detail
    return neutralised, diagnostics


__all__ = ["neutralise_cross_section", "neutralise_factors"]
