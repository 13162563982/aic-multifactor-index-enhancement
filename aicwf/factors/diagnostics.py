"""Factor diagnostics: coverage, RankIC, persistence and cross-factor correlation.

All diagnostics are computed on the investable cross-section (``eligible``) and
on the monthly formation grid.  RankIC for formation date ``t`` is the Spearman
correlation between the neutralised factor score at ``t`` and the *next*
monthly formation-to-formation return ``r(t -> t+1)``; because the forward
return only uses grid points that already exist inside the factor panel, the
diagnostic itself introduces no additional look-ahead beyond what an in-sample
IC report inherently is (the backtest never reads it).

Definitions
-----------
* ``rank_ic_mean = mean_t( corr_spearman(z_t, r_{t->t+1}) )``
* ``ic_t_stat = rank_ic_mean / (sd(IC_t) / sqrt(n_dates))``
* ``autocorrelation`` = mean over consecutive months of the cross-sectional
  Spearman correlation of factor scores (rank stability / factor turnover).
* ``correlation_matrix`` = mean over months of the cross-sectional Pearson
  correlation between two neutralised factors (pooling months would mix time
  variation into the estimate).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.factors.preprocess import _average_ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    rank_left = _average_ranks(left)
    rank_right = _average_ranks(right)
    if np.std(rank_left) == 0 or np.std(rank_right) == 0:
        return float("nan")
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _as_bool_frame(frame, like) -> pd.DataFrame:
    """Boolean frame reindexed to ``like`` with missing values treated as False."""

    out = pd.DataFrame(frame).reindex_like(like)
    values = out.to_numpy()
    if values.dtype == bool:
        return out.astype(bool)
    filled = np.where(pd.isna(values), False, values)
    return pd.DataFrame(filled.astype(bool), index=out.index, columns=out.columns)


def rank_ic_series(
    score: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    min_names: int = 5,
) -> pd.Series:
    """Per-date cross-sectional RankIC of one factor."""

    scores = pd.DataFrame(score).reindex_like(forward_returns)
    elig = _as_bool_frame(eligible, forward_returns)
    values: dict[str, float] = {}
    for date in forward_returns.index:
        row_score = scores.loc[date].to_numpy(dtype=float)
        row_forward = forward_returns.loc[date].to_numpy(dtype=float)
        mask = elig.loc[date].to_numpy(dtype=bool) & np.isfinite(row_score) & np.isfinite(row_forward)
        if int(mask.sum()) < max(3, min_names):
            continue
        value = _spearman(row_score[mask], row_forward[mask])
        if np.isfinite(value):
            values[str(date)] = round(float(value), 12)
    return pd.Series(values, dtype=float)


def summarise_ic(ic: pd.Series) -> dict[str, Any]:
    if len(ic) == 0:
        return {"mean": None, "std": None, "t_stat": None, "n_dates": 0, "positive_share": None}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    if len(ic) > 1 and std > 1e-12:
        t_stat = mean / (std / np.sqrt(len(ic)))
    else:
        t_stat = None
    return {
        "mean": mean,
        "std": std,
        "t_stat": None if t_stat is None else float(t_stat),
        "n_dates": int(len(ic)),
        "positive_share": float((ic > 0).mean()),
    }


def rank_ic_table(
    scores: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    min_names: int = 5,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    """Summary statistics and per-date series for every factor."""

    summary: dict[str, dict[str, Any]] = {}
    series: dict[str, dict[str, float]] = {}
    for factor_id, matrix in scores.items():
        ic = rank_ic_series(matrix, forward_returns, eligible, min_names=min_names)
        summary[factor_id] = summarise_ic(ic)
        series[factor_id] = {str(date): float(value) for date, value in ic.items()}
    return summary, series


def autocorrelation_table(
    scores: Mapping[str, pd.DataFrame],
    eligible: pd.DataFrame,
    *,
    min_names: int = 5,
) -> dict[str, float]:
    """Mean month-to-month cross-sectional rank correlation per factor."""

    ages = sorted(pd.DataFrame(eligible).index)
    result: dict[str, float] = {}
    for factor_id, matrix in scores.items():
        values: list[float] = []
        data = pd.DataFrame(matrix)
        elig = _as_bool_frame(eligible, eligible)
        for previous, current in zip(ages[:-1], ages[1:]):
            left = data.loc[previous].to_numpy(dtype=float)
            right = data.loc[current].to_numpy(dtype=float)
            mask = (
                elig.loc[previous].to_numpy(dtype=bool)
                & elig.loc[current].to_numpy(dtype=bool)
                & np.isfinite(left)
                & np.isfinite(right)
            )
            if int(mask.sum()) < max(3, min_names):
                continue
            value = _spearman(left[mask], right[mask])
            if np.isfinite(value):
                values.append(float(value))
        result[factor_id] = float(np.mean(values)) if values else float("nan")
    return result


def correlation_matrix(
    scores: Mapping[str, pd.DataFrame],
    eligible: pd.DataFrame,
    *,
    min_names: int = 5,
) -> dict[str, dict[str, float | None]]:
    """Mean monthly cross-sectional Pearson correlation between factors."""

    factor_ids = list(scores.keys())
    matrix: dict[str, dict[str, float | None]] = {fid: {} for fid in factor_ids}
    elig = _as_bool_frame(eligible, eligible)
    dates = sorted(elig.index)
    values = {fid: pd.DataFrame(scores[fid]) for fid in factor_ids}
    for i, left_id in enumerate(factor_ids):
        matrix[left_id][left_id] = 1.0
        for right_id in factor_ids[i + 1 :]:
            monthly: list[float] = []
            for date in dates:
                left = values[left_id].loc[date].to_numpy(dtype=float)
                right = values[right_id].loc[date].to_numpy(dtype=float)
                mask = (
                    elig.loc[date].to_numpy(dtype=bool)
                    & np.isfinite(left)
                    & np.isfinite(right)
                )
                if int(mask.sum()) < max(3, min_names):
                    continue
                value = _pearson(left[mask], right[mask])
                if np.isfinite(value):
                    monthly.append(float(value))
            value = float(np.mean(monthly)) if monthly else None
            matrix[left_id][right_id] = None if value is None else round(value, 8)
            matrix[right_id][left_id] = matrix[left_id][right_id]
    return matrix


def coverage_report(
    raw: Mapping[str, pd.DataFrame],
    scores: Mapping[str, pd.DataFrame],
    eligible: pd.DataFrame,
) -> dict[str, Any]:
    """Data coverage for each raw factor and its neutralised z-score."""

    elig = _as_bool_frame(eligible, eligible)
    eligible_rows = int(elig.to_numpy().sum())
    by_factor: dict[str, dict[str, Any]] = {}
    for factor_id in raw:
        raw_matrix = pd.DataFrame(raw[factor_id]).reindex_like(elig)
        z_matrix = pd.DataFrame(scores[factor_id]).reindex_like(elig)
        raw_finite = np.isfinite(raw_matrix.to_numpy(dtype=float)) & elig.to_numpy(dtype=bool)
        z_finite = np.isfinite(z_matrix.to_numpy(dtype=float)) & elig.to_numpy(dtype=bool)
        by_factor[factor_id] = {
            "raw_rows": int(raw_finite.sum()),
            "z_rows": int(z_finite.sum()),
            "raw_coverage": float(raw_finite.sum() / eligible_rows) if eligible_rows else None,
            "z_coverage": float(z_finite.sum() / eligible_rows) if eligible_rows else None,
        }
    return {
        "eligible_rows": eligible_rows,
        "dates": int(elig.shape[0]),
        "codes": int(elig.shape[1]),
        "eligible_share": float(elig.to_numpy().mean()) if elig.size else None,
        "by_factor": by_factor,
    }


__all__ = [
    "autocorrelation_table",
    "correlation_matrix",
    "coverage_report",
    "rank_ic_series",
    "rank_ic_table",
    "summarise_ic",
]
