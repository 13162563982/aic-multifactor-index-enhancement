"""Cross-sectional preprocessing: winsorisation and rank z-scores.

Both transforms are applied *date by date* across the investable cross-section,
which is what makes them robust to level/scale differences between months and
to differing units in the raw inputs (e.g. ``revenue_yoy`` in percent vs
fraction does not matter after a within-month rank transform).

Definitions
-----------
For a cross-section of ``n`` valid values ``x_(1) <= ... <= x_(n)``:

* **winsorise** (quantile method): clip ``x`` to
  ``[Q(lower), Q(upper)]`` where ``Q`` is the empirical quantile
  (``lower = 0.01``, ``upper = 0.99`` by default).  Cross-sections with fewer
  than three valid values are left unchanged (a quantile is meaningless then).
* **rank z-score**: let ``r_i`` be the average-tie rank of ``x_i`` in
  ``1..n``.  Then

  .. math::  z_i = \\frac{r_i - (n+1)/2}{\\sqrt{(n^2 - 1) / 12}}

  which is the Van der Waerden normal score normalised to mean zero and unit
  variance over the cross-section.  Missing values stay missing (they are *not*
  imputed here); the composite step treats a missing neutralised score as a
  neutral zero exposure.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _as_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame).astype(float) if not isinstance(frame, pd.DataFrame) else frame.astype(float)


def _per_date(matrix: pd.DataFrame, function) -> pd.DataFrame:
    """Apply ``function(values_1d) -> values_1d`` to every date row."""

    matrix = pd.DataFrame(matrix).astype(float)
    out = np.full(matrix.shape, np.nan, dtype=float)
    values = matrix.to_numpy(dtype=float)
    for row in range(values.shape[0]):
        out[row, :] = function(values[row, :])
    return pd.DataFrame(out, index=matrix.index, columns=matrix.columns)


def winsorize(
    frame: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip extreme cross-sectional values at empirical quantiles (per date)."""

    def _clip(row: np.ndarray) -> np.ndarray:
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count < 3:
            return row.copy()
        low = float(np.quantile(row[valid], lower, method="linear"))
        high = float(np.quantile(row[valid], upper, method="linear"))
        return np.clip(row, low, high)

    return _per_date(frame, _clip)


def rank_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank-based z-score of each cross-section (Van der Waerden normal scores)."""

    def _zscore(row: np.ndarray) -> np.ndarray:
        result = np.full(row.shape, np.nan, dtype=float)
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count == 0:
            return result
        ranks = _average_ranks(row[valid])
        if count == 1:
            result[valid] = 0.0
            return result
        scale = np.sqrt((count**2 - 1) / 12.0)
        result[valid] = (ranks - (count + 1) / 2.0) / scale
        return result

    return _per_date(frame, _zscore)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1..n) without scipy, deterministic."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Resolve ties by replacing runs of equal values with their mean rank.
    sorted_values = values[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                average = float(np.mean(np.arange(start + 1, index + 1)))
                ranks[order[start:index]] = average
            start = index
    return ranks


def fill_neutral(frame: pd.DataFrame, *, value: float = 0.0) -> pd.DataFrame:
    """Fill missing cross-sectional scores with a neutral value (zero)."""

    return pd.DataFrame(frame).astype(float).fillna(value)


def preprocess_factor(
    raw: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorise then rank-zscore one raw factor matrix (date x code)."""

    clipped = winsorize(raw, lower=lower, upper=upper)
    return rank_zscore(clipped)


__all__ = ["fill_neutral", "preprocess_factor", "rank_zscore", "winsorize"]
