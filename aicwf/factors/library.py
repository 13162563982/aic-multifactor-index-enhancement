"""Raw factor library (monthly, point-in-time).

Every factor is evaluated on a *formation grid* of the last trading day of each
calendar month.  For a formation date ``t`` a factor value may only use data
whose date is ``<= t``:

* price / liquidity / risk factors use trailing daily windows that end at ``t``;
* fundamental factors use the most recent report whose ``notice_date``
  (announcement date) is ``<= t`` — never the report period end and never a
  later-announced number.  When ``notice_date`` is missing the snapshot marks
  the row as estimated and we fall back to ``report_date + 90 days``;
* momentum is the classic 12-1 construction ``close(t-1m)/close(t-12m) - 1``:
  the most recent month is skipped, so a signal formed at month end ``t`` is
  only traded in the following month ("signals lagged one month"); the backtest
  additionally executes on the T+1 close after the decision date.

Factor equations (``c`` = formation close, ``eps`` = EPS, ``bps`` = book value
per share, ``revenue``/``net_profit`` in the same currency unit):

======================  ===========================================================
id                      raw value
======================  ===========================================================
``value_ep``            ``eps / c``
``value_bp``            ``bps / c``
``value_spp``           ``(revenue / shares) / c`` with ``shares = net_profit/eps``
``quality_roe``         ``roe`` (fallback ``eps / bps``)
``quality_gross_margin````gross_margin``
``quality_cashflow``    ``cfo_per_share / eps`` (defined only when ``eps > 0``)
``growth_revenue``      ``revenue_yoy``
``growth_profit``       ``net_profit_yoy``
``momentum_12_1``       ``c(t-1m) / c(t-12m) - 1``
``reversal_1m``         ``c(t) / c(t-1m) - 1``
``low_vol_60d``         ``std(ret_d, ddof=1)`` over the last 60 trading days
``turnover_20d``        ``mean(turnover_d)`` over the last 20 trading days
``lottery_max_20d``     ``max(ret_d)`` over the last 20 trading days
======================  ===========================================================

Signs are *not* applied here: the raw value is the economic quantity.  The
catalog ``direction`` (``+1``/``-1``) is applied when the composite score is
built, so that a factor such as 60-day volatility can be used with a negative
portfolio weight.

The output of :func:`build_monthly_factor_data` is a
:class:`MonthlyFactorData` bundle of wide matrices (formation date x code),
which the preprocessing/neutralisation stages consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.contracts import ValidationError

FACTOR_IDS: tuple[str, ...] = (
    "value_ep",
    "value_bp",
    "value_spp",
    "quality_roe",
    "quality_gross_margin",
    "quality_cashflow",
    "growth_revenue",
    "growth_profit",
    "momentum_12_1",
    "reversal_1m",
    "low_vol_60d",
    "turnover_20d",
    "lottery_max_20d",
)

#: Fallback catalog used when ``config/strategy.yaml`` has no catalog section
#: (tests pass a minimal config).  Values mirror the shipped configuration.
DEFAULT_CATALOG: dict[str, dict[str, Any]] = {
    "value_ep": {"direction": 1.0, "category": "value", "description": "trailing EPS / price"},
    "value_bp": {"direction": 1.0, "category": "value", "description": "book value per share / price"},
    "value_spp": {"direction": 1.0, "category": "value", "description": "revenue per share / price"},
    "quality_roe": {"direction": 1.0, "category": "quality", "description": "EPS / BPS"},
    "quality_gross_margin": {"direction": 1.0, "category": "quality", "description": "gross profit / revenue"},
    "quality_cashflow": {"direction": 1.0, "category": "quality", "description": "operating cash flow per share / EPS"},
    "growth_revenue": {"direction": 1.0, "category": "growth", "description": "revenue year over year"},
    "growth_profit": {"direction": 1.0, "category": "growth", "description": "net profit year over year"},
    "momentum_12_1": {"direction": 1.0, "category": "momentum", "description": "12-1 month return"},
    "reversal_1m": {"direction": -1.0, "category": "momentum", "description": "one month reversal"},
    "low_vol_60d": {"direction": -1.0, "category": "risk", "description": "60 day realized volatility"},
    "turnover_20d": {"direction": -1.0, "category": "liquidity", "description": "20 day turnover"},
    "lottery_max_20d": {"direction": -1.0, "category": "risk", "description": "max daily return over 20 days"},
}

FUNDAMENTAL_COLUMNS = (
    "code",
    "report_date",
    "notice_date",
    "eps",
    "bps",
    "roe",
    "gross_margin",
    "revenue",
    "revenue_yoy",
    "net_profit",
    "net_profit_yoy",
    "cfo_per_share",
    "notice_date_estimated",
)

#: Trading days in a month used to size the daily trailing windows.
DAILY_MOMENTUM_MONTHS = 12
TRADING_DAYS_PER_MONTH = 21


def factor_catalog(cfg: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return the validated factor catalog (id -> direction/category/...).

    The ids must match :data:`FACTOR_IDS` exactly because the hypothesis agent
    emits these ids; anything else is a configuration error, never silently
    dropped.
    """

    raw: Any = None
    if isinstance(cfg, Mapping):
        raw = (cfg.get("factors") or {}).get("catalog")
    if not raw:
        return {fid: dict(meta) for fid, meta in DEFAULT_CATALOG.items()}
    if not isinstance(raw, Mapping):
        raise ValidationError("factors.catalog must be a mapping")
    unknown = sorted(set(raw) - set(FACTOR_IDS))
    missing = sorted(set(FACTOR_IDS) - set(raw))
    if unknown:
        raise ValidationError(
            f"factors.catalog contains unknown factor ids {unknown}; supported ids are {list(FACTOR_IDS)}"
        )
    if missing:
        raise ValidationError(
            f"factors.catalog is missing factor ids {missing}; all catalog ids are required"
        )
    catalog: dict[str, dict[str, Any]] = {}
    for fid in FACTOR_IDS:
        meta = dict(raw[fid] or {})
        direction = float(meta.get("direction", DEFAULT_CATALOG[fid]["direction"]))
        if direction == 0.0:
            raise ValidationError(f"factor {fid!r} has direction 0; use +1 or -1")
        meta["direction"] = direction
        meta.setdefault("category", DEFAULT_CATALOG[fid]["category"])
        catalog[fid] = meta
    return catalog


def factor_directions(cfg: Mapping[str, Any] | None) -> dict[str, float]:
    return {fid: float(meta["direction"]) for fid, meta in factor_catalog(cfg).items()}


@dataclass
class MonthlyFactorData:
    """Wide (formation date x code) matrices for one point-in-time factor build."""

    raw: dict[str, pd.DataFrame]
    close: pd.DataFrame
    industry: pd.DataFrame
    log_size: pd.DataFrame
    eligible: pd.DataFrame
    forward_returns: pd.DataFrame
    formation_dates: list[str]
    codes: list[str]
    warnings: list[str]

    def long_frame(self) -> pd.DataFrame:
        """Long ``(date, code)`` frame with raw factors plus eligibility context."""

        dates = self.formation_dates
        codes = self.codes
        index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
        data: dict[str, np.ndarray] = {}
        for fid in FACTOR_IDS:
            matrix = self.raw[fid].reindex(index=dates, columns=codes)
            data[fid] = matrix.to_numpy(dtype=float).ravel()
        data["close"] = self.close.reindex(index=dates, columns=codes).to_numpy(dtype=float).ravel()
        data["log_size"] = (
            self.log_size.reindex(index=dates, columns=codes).to_numpy(dtype=float).ravel()
        )
        data["industry"] = (
            self.industry.reindex(index=dates, columns=codes)
            .fillna("UNKNOWN")
            .to_numpy(dtype=object)
            .ravel()
        )
        data["eligible"] = (
            self.eligible.reindex(index=dates, columns=codes).fillna(False).to_numpy(dtype=bool).ravel()
        )
        frame = pd.DataFrame(data, index=index).reset_index()
        return frame


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _pivot(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="code", values=column, aggfunc="last")


def _bool_pivot(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    values = panel[column].astype(bool)
    pivot = panel.assign(_flag=values).pivot_table(
        index="date", columns="code", values="_flag", aggfunc="last"
    )
    raw = pivot.to_numpy()
    filled = np.where(pd.isna(raw), False, raw)
    return pd.DataFrame(filled.astype(bool), index=pivot.index, columns=pivot.columns)


def _reindex_bool_matrix(
    frame: pd.DataFrame, dates: Sequence[str], codes: Sequence[str], *, default: bool
) -> pd.DataFrame:
    """Reindex a boolean matrix, filling missing (date, code) pairs with ``default``."""

    out = frame.reindex(index=list(dates), columns=list(codes))
    raw = out.to_numpy()
    if raw.dtype == bool:
        return out.astype(bool)
    filled = np.where(pd.isna(raw), bool(default), raw)
    return pd.DataFrame(filled.astype(bool), index=out.index, columns=out.columns)


def _format_return(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator.to_numpy(dtype=float) / denominator.to_numpy(dtype=float) - 1.0
    values[~np.isfinite(values)] = np.nan
    return pd.DataFrame(values, index=numerator.index, columns=numerator.columns)


def _prepare_fundamentals(
    fundamentals: pd.DataFrame | None, warnings: list[str]
) -> pd.DataFrame | None:
    if fundamentals is None or len(fundamentals) == 0:
        warnings.append("fundamentals snapshot unavailable: fundamental factors are NaN")
        return None
    frame = pd.DataFrame(fundamentals).copy()
    missing = [column for column in FUNDAMENTAL_COLUMNS if column not in frame.columns]
    if missing:
        warnings.append(f"fundamentals snapshot missing columns {missing}: affected factors are NaN")
    frame["code"] = frame["code"].astype(str)
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["notice_date"] = pd.to_datetime(frame["notice_date"], errors="coerce")
    estimated = frame["notice_date"].isna()
    if estimated.any():
        warnings.append(
            f"{int(estimated.sum())} fundamental rows have no notice_date; using report_date + 90 days"
        )
        fallback = frame.loc[estimated, "report_date"] + pd.Timedelta(days=90)
        frame.loc[estimated, "notice_date"] = fallback
    frame = frame[frame["report_date"].notna() & frame["notice_date"].notna()]
    numeric = [
        "eps",
        "bps",
        "roe",
        "gross_margin",
        "revenue",
        "revenue_yoy",
        "net_profit",
        "net_profit_yoy",
        "cfo_per_share",
    ]
    for column in numeric:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame = frame.sort_values(["code", "notice_date", "report_date"], kind="mergesort")
    frame = frame.drop_duplicates(subset=["code", "notice_date"], keep="last").reset_index(drop=True)
    return frame.loc[:, ["code", "report_date", "notice_date", *numeric]]


def _asof_fundamentals(
    fundamentals: pd.DataFrame, formation_dates: Sequence[str], codes: Sequence[str]
) -> pd.DataFrame:
    """Attach the most recent announced report (notice_date <= formation date)."""

    grid = pd.DataFrame(
        {
            "date": np.repeat(pd.to_datetime(list(formation_dates)).to_numpy(), len(codes)),
            "code": np.tile(np.asarray(list(codes), dtype=object), len(formation_dates)),
        }
    )
    grid = grid.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    right = fundamentals.sort_values("notice_date", kind="mergesort").reset_index(drop=True)
    merged = pd.merge_asof(
        grid,
        right,
        left_on="date",
        right_on="notice_date",
        by="code",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def _fundamental_matrices(
    merged: pd.DataFrame, formation_dates: Sequence[str], codes: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Compute the fundamental raw factors from the as-of joined report table."""

    merged = merged.copy()
    close = merged["_close"].to_numpy(dtype=float)

    def matrix(values: np.ndarray) -> pd.DataFrame:
        values = np.asarray(values, dtype=float)
        values[~np.isfinite(values)] = np.nan
        frame = pd.DataFrame(
            {"date": merged["date"].to_numpy(), "code": merged["code"].to_numpy(), "value": values}
        )
        pivot = frame.pivot_table(index="date", columns="code", values="value", aggfunc="last")
        return pivot.reindex(index=list(formation_dates), columns=list(codes)).astype(float)

    eps = merged["eps"].to_numpy(dtype=float)
    bps = merged["bps"].to_numpy(dtype=float)
    revenue = merged["revenue"].to_numpy(dtype=float)
    net_profit = merged["net_profit"].to_numpy(dtype=float)
    cfo = merged["cfo_per_share"].to_numpy(dtype=float)
    roe = merged["roe"].to_numpy(dtype=float)
    eps_positive = eps > 0
    profit_positive = net_profit > 0
    shares = np.where(eps_positive & profit_positive, net_profit / np.where(eps != 0, eps, np.nan), np.nan)
    revenue_per_share = np.where(shares != 0, revenue / shares, np.nan)

    roe_fallback = np.where(bps > 0, eps / np.where(bps != 0, bps, np.nan), np.nan)
    roe_final = np.where(np.isfinite(roe), roe, roe_fallback)

    return {
        "value_ep": matrix(eps / close),
        "value_bp": matrix(bps / close),
        "value_spp": matrix(revenue_per_share / close),
        "quality_roe": matrix(roe_final),
        "quality_gross_margin": matrix(merged["gross_margin"].to_numpy(dtype=float)),
        "quality_cashflow": matrix(np.where(eps_positive, cfo / np.where(eps != 0, eps, np.nan), np.nan)),
        "growth_revenue": matrix(merged["revenue_yoy"].to_numpy(dtype=float)),
        "growth_profit": matrix(merged["net_profit_yoy"].to_numpy(dtype=float)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_monthly_factor_data(
    panel: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    *,
    end_date: Any = None,
    min_listed_days: int = 250,
    vol_window: int = 60,
    short_window: int = 20,
) -> MonthlyFactorData:
    """Build the wide point-in-time factor matrices on the monthly grid."""

    warnings: list[str] = []
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["date", "code"], kind="mergesort")
    panel_dates = sorted(frame["date"].unique().tolist())
    if end_date is not None:
        cutoff = pd.Timestamp(end_date).date().isoformat()
        panels_in_range = [day for day in panel_dates if day <= cutoff]
    else:
        panels_in_range = panel_dates
    if len(panels_in_range) == 0:
        raise ValidationError("panel has no trading days inside the requested date range")

    from aicwf.factors.common import month_end_dates

    formation_dates = month_end_dates(panels_in_range)
    codes = sorted(frame["code"].astype(str).unique().tolist())
    if not formation_dates or not codes:
        raise ValidationError("cannot build a monthly factor grid from the panel")

    # Monthly close matrix (formation dates x codes).  Only closes up to the
    # formation date are used, so this grid is invariant to any future rows.
    monthly_close = _pivot(frame, "close").reindex(index=formation_dates, columns=codes)
    monthly_close = monthly_close.astype(float)

    daily_close = _pivot(frame, "close").sort_index()
    daily_ret = _pivot(frame, "ret").sort_index() if "ret" in frame.columns else daily_close.pct_change()
    daily_turnover = _pivot(frame, "turnover").sort_index()
    if "suspended" in frame.columns:
        daily_suspended = _bool_pivot(frame, "suspended")
    else:
        daily_suspended = pd.DataFrame(False, index=daily_close.index, columns=daily_close.columns)
    if "in_universe" in frame.columns:
        daily_in_universe = _bool_pivot(frame, "in_universe")
    else:
        daily_in_universe = pd.DataFrame(True, index=daily_close.index, columns=daily_close.columns)
    daily_log_size = _pivot(frame, "log_size").sort_index()
    industry_matrix = (
        frame.pivot_table(index="date", columns="code", values="industry", aggfunc="last")
        .sort_index()
        .ffill()
    )
    industry = industry_matrix.reindex(index=formation_dates, columns=codes).fillna("UNKNOWN")

    raw: dict[str, pd.DataFrame] = {}
    raw["momentum_12_1"] = _format_return(monthly_close.shift(1), monthly_close.shift(12))
    raw["reversal_1m"] = _format_return(monthly_close, monthly_close.shift(1))
    raw["low_vol_60d"] = daily_ret.rolling(vol_window, min_periods=max(10, vol_window // 2)).std(ddof=1).reindex(
        index=formation_dates
    )
    raw["turnover_20d"] = daily_turnover.rolling(
        short_window, min_periods=max(5, short_window // 2)
    ).mean().reindex(index=formation_dates)
    raw["lottery_max_20d"] = daily_ret.rolling(
        short_window, min_periods=max(5, short_window // 2)
    ).max().reindex(index=formation_dates)
    for fid in ("low_vol_60d", "turnover_20d", "lottery_max_20d"):
        raw[fid] = raw[fid].reindex(columns=codes).astype(float)
        raw[fid][~np.isfinite(raw[fid].to_numpy())] = np.nan

    prepared = _prepare_fundamentals(fundamentals, warnings)
    if prepared is not None:
        merged = _asof_fundamentals(prepared, formation_dates, codes)
        close_flat = monthly_close.to_numpy(dtype=float).ravel()
        merged["_close"] = close_flat
        fundamental = _fundamental_matrices(merged, formation_dates, codes)
        raw.update(fundamental)
    else:
        for fid in (
            "value_ep",
            "value_bp",
            "value_spp",
            "quality_roe",
            "quality_gross_margin",
            "quality_cashflow",
            "growth_revenue",
            "growth_profit",
        ):
            raw[fid] = pd.DataFrame(np.nan, index=formation_dates, columns=codes)

    ordered_raw: dict[str, pd.DataFrame] = {}
    for fid in FACTOR_IDS:
        matrix = raw[fid].reindex(index=formation_dates, columns=codes).astype(float)
        ordered_raw[fid] = matrix

    # Point-in-time eligibility: in the index, trading, seasoned, priced, non-ST.
    suspended = _reindex_bool_matrix(daily_suspended, formation_dates, codes, default=True)
    in_universe = _reindex_bool_matrix(daily_in_universe, formation_dates, codes, default=False)
    if "listed_days" in frame.columns:
        listed_days = _pivot(frame, "listed_days").reindex(index=formation_dates, columns=codes)
        listed_days = listed_days.astype(float).fillna(0.0)
    else:
        listed_days = pd.DataFrame(1_000_000.0, index=formation_dates, columns=codes)
    eligible = (
        monthly_close.notna()
        & (monthly_close > 0)
        & in_universe
        & (~suspended)
        & (listed_days >= float(min_listed_days))
    )
    if universe is not None and "is_st" in getattr(universe, "columns", []):
        st_codes = set(
            universe.loc[universe["is_st"].astype(str).str.lower().isin({"true", "1"}), "code"].astype(str)
        )
        if st_codes:
            for code in st_codes:
                if code in eligible.columns:
                    eligible[code] = False
    eligible = eligible.reindex(index=formation_dates, columns=codes)
    eligible = pd.DataFrame(
        np.where(pd.isna(eligible.to_numpy()), False, eligible.to_numpy()),
        index=formation_dates,
        columns=codes,
    ).astype(bool)

    log_size = daily_log_size.reindex(index=formation_dates, columns=codes).astype(float)
    forward = monthly_close.pct_change().shift(-1)
    forward = forward.where(np.isfinite(forward))

    return MonthlyFactorData(
        raw=ordered_raw,
        close=monthly_close,
        industry=industry,
        log_size=log_size,
        eligible=eligible,
        forward_returns=forward,
        formation_dates=list(formation_dates),
        codes=list(codes),
        warnings=warnings,
    )
