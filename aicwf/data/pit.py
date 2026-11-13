"""Point-in-time (PIT) fundamentals availability helpers.

The snapshot stores one fundamentals row per report period. A report period is
only *knowable* after its announcement date (``notice_date``); using the report
period end date would leak future information. Two helpers are provided and are
reused by the data-prep stage and by independent verification:

``available_as_of``
    Filter fundamentals to the rows whose effective notice date is on or before
    the decision date. Rows with a missing ``notice_date`` fall back to
    ``report_date + 90 calendar days`` and are flagged
    ``notice_date_estimated=True``.

``assert_no_lookahead``
    Raise :class:`aicwf.contracts.ValidationError` when any row of a frame has a
    date after the decision date (used to reject future data in panel, ledger,
    holdings, factor, equity or target-weight artifacts).
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Sequence

import pandas as pd

from aicwf.contracts import ValidationError, parse_date

#: Fallback lag used when a report period has no announcement date.
FUNDAMENTAL_NOTICE_FALLBACK_DAYS = 90


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _iso_series(series: pd.Series) -> pd.Series:
    parsed = _to_datetime(series)
    formatted = parsed.dt.strftime("%Y-%m-%d")
    out = formatted.astype(object)
    return out.where(parsed.notna(), None)


def fill_estimated_notice_dates(
    fundamentals: pd.DataFrame,
    *,
    fallback_days: int = FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
) -> pd.DataFrame:
    """Fill missing ``notice_date`` with ``report_date + fallback_days``.

    Returns a copy where ``notice_date`` is always populated (when the report
    date is known) and ``notice_date_estimated`` marks the estimated rows. Rows
    that already carry an estimate flag are left untouched.
    """

    if fundamentals is None or fundamentals.empty:
        return fundamentals.copy() if fundamentals is not None else pd.DataFrame()
    frame = fundamentals.copy()
    if "notice_date_estimated" not in frame.columns:
        frame["notice_date_estimated"] = False
    else:
        frame["notice_date_estimated"] = frame["notice_date_estimated"].map(
            lambda value: bool(value) if value is not None and not (isinstance(value, float) and pd.isna(value)) else False
        )
    report_dates = _to_datetime(frame["report_date"])
    notice_dates = _to_datetime(frame["notice_date"])
    estimated = notice_dates.isna() & report_dates.notna()
    if estimated.any():
        fallback = report_dates.loc[estimated] + pd.Timedelta(days=int(fallback_days))
        notice_dates = notice_dates.copy()
        notice_dates.loc[estimated] = fallback
        frame["notice_date_estimated"] = frame["notice_date_estimated"] | estimated
    frame["notice_date"] = _iso_series(notice_dates)
    return frame


def effective_notice_dates(
    fundamentals: pd.DataFrame,
    *,
    fallback_days: int = FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
) -> pd.Series:
    """Effective availability date per row (``notice_date`` or fallback)."""

    if fundamentals is None or fundamentals.empty:
        return pd.Series(dtype=object)
    notice = _to_datetime(fundamentals["notice_date"])
    report = _to_datetime(fundamentals["report_date"])
    fallback = report + pd.Timedelta(days=int(fallback_days))
    resolved = notice.where(notice.notna(), fallback)
    return _iso_series(resolved)


def available_as_of(
    fundamentals: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
    *,
    fallback_days: int = FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
) -> pd.DataFrame:
    """Return the fundamental rows available on ``as_of`` (PIT filter).

    The result keeps the contract columns, replaces missing notice dates with the
    ``report_date + fallback_days`` estimate, appends ``effective_notice_date``
    and sets ``notice_date_estimated`` accordingly. Only periods with
    ``report_date <= as_of`` and ``effective_notice_date <= as_of`` survive.
    """

    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=list(fundamentals.columns if fundamentals is not None else ()) + ["effective_notice_date"])
    cutoff = parse_date(str(as_of))
    frame = fill_estimated_notice_dates(fundamentals, fallback_days=fallback_days)
    report_dates = _to_datetime(frame["report_date"])
    notice_dates = _to_datetime(frame["notice_date"])
    mask = (report_dates.notna()) & (report_dates.dt.date <= cutoff) & (notice_dates.notna()) & (notice_dates.dt.date <= cutoff)
    available = frame.loc[mask].copy()
    available["effective_notice_date"] = _iso_series(notice_dates.loc[mask])
    keys = [column for column in ("code", "report_date") if column in available.columns]
    if keys:
        available = available.sort_values(keys, kind="mergesort")
    return available.reset_index(drop=True)


def date_violations(
    frame: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
    *,
    date_col: str = "date",
    name: str = "frame",
    limit: int = 20,
) -> list[dict[str, object]]:
    """Return up to ``limit`` future-dated rows as ``{name, code, date, as_of}``."""

    if frame is None or frame.empty or date_col not in frame.columns:
        return []
    cutoff = parse_date(str(as_of))
    dates = _to_datetime(frame[date_col])
    mask = dates.notna() & (dates.dt.date > cutoff)
    if not mask.any():
        return []
    bad = frame.loc[mask]
    violations: list[dict[str, object]] = []
    for index, row in bad.head(limit).iterrows():
        violations.append(
            {
                "artifact": name,
                "column": date_col,
                "code": str(row["code"]) if "code" in bad.columns else None,
                "date": str(dates.loc[index].date().isoformat()),  # type: ignore[union-attr]
                "as_of": cutoff.isoformat(),
            }
        )
    return violations


def assert_no_lookahead(
    frame: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
    *,
    date_col: str = "date",
    name: str = "frame",
) -> None:
    """Raise :class:`ValidationError` if any row is dated after ``as_of``."""

    violations = date_violations(frame, as_of, date_col=date_col, name=name, limit=5)
    if violations:
        first = violations[0]
        raise ValidationError(
            f"{name} contains {len(violations)}+ row(s) after as_of={first['as_of']} "
            f"(e.g. code={first['code']} date={first['date']}); future data is forbidden"
        )


def assert_notice_dates_valid(
    fundamentals: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
    *,
    fallback_days: int = FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
) -> None:
    """Raise when a fundamentals row available *after* ``as_of`` leaks in.

    Callers that already applied :func:`available_as_of` can use this as a
    cheap assertion in tests and in the verification stage.
    """

    if fundamentals is None or fundamentals.empty:
        return
    cutoff = parse_date(str(as_of))
    notice = _to_datetime(effective_notice_dates(fundamentals, fallback_days=fallback_days))
    mask = notice.notna() & (notice.dt.date > cutoff)
    if mask.any():
        row = fundamentals.loc[mask].iloc[0]
        raise ValidationError(
            f"fundamentals contain a row with notice_date after as_of={cutoff.isoformat()} "
            f"(code={row.get('code')}, report_date={row.get('report_date')})"
        )


def max_available_report_date(
    fundamentals: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
    *,
    fallback_days: int = FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
) -> str | None:
    """Latest report period end available on ``as_of`` (ISO string) or ``None``."""

    available = available_as_of(fundamentals, as_of, fallback_days=fallback_days)
    if available.empty:
        return None
    values = _to_datetime(available["report_date"]).dropna()
    if values.empty:
        return None
    return values.max().date().isoformat()
