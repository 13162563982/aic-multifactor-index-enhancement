"""Trading calendar helpers derived from the frozen snapshot benchmark.

The snapshot's ``calendar.parquet`` is authoritative: it is the set of index
trading days covered by the frozen data. No exchange calendar library is used,
so behaviour is fully determined by the snapshot (deterministic and offline).
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Sequence

import pandas as pd

from aicwf.contracts import parse_date
from aicwf.data.snapshot import CALENDAR_FILE, Snapshot, load_snapshot


def build_calendar(dates: Iterable[str | dt.date | dt.datetime]) -> pd.DataFrame:
    """Canonical calendar frame (``date`` column, ISO strings, sorted, unique)."""

    parsed = [parse_date(str(value)).isoformat() for value in dates]
    frame = pd.DataFrame({"date": sorted(set(parsed))})
    return frame


def trading_days_from_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Calendar derived from benchmark dates (the index trades every trading day)."""

    if benchmark is None or benchmark.empty or "date" not in benchmark.columns:
        return pd.DataFrame({"date": []})
    return build_calendar(benchmark["date"].tolist())


def trading_days(calendar: pd.DataFrame | Sequence[str]) -> list[str]:
    """Return the calendar as a sorted list of ISO date strings."""

    if isinstance(calendar, pd.DataFrame):
        values = calendar["date"].tolist()
    else:
        values = list(calendar)
    return sorted({parse_date(str(value)).isoformat() for value in values})


def is_trading_day(calendar: pd.DataFrame | Sequence[str], date: str | dt.date) -> bool:
    return parse_date(str(date)).isoformat() in set(trading_days(calendar))


def next_trading_day(calendar: pd.DataFrame | Sequence[str], date: str | dt.date, *, inclusive: bool = False) -> str | None:
    """First trading day strictly after ``date`` (or on it when ``inclusive``)."""

    target = parse_date(str(date))
    for day in trading_days(calendar):
        candidate = parse_date(day)
        if candidate > target or (inclusive and candidate == target):
            return day
    return None


def previous_trading_day(calendar: pd.DataFrame | Sequence[str], date: str | dt.date, *, inclusive: bool = False) -> str | None:
    """Last trading day strictly before ``date`` (or on it when ``inclusive``)."""

    target = parse_date(str(date))
    result: str | None = None
    for day in trading_days(calendar):
        candidate = parse_date(day)
        if candidate < target or (inclusive and candidate == target):
            result = day
        else:
            break
    return result


def month_end_trading_days(
    calendar: pd.DataFrame | Sequence[str],
    *,
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
) -> list[str]:
    """Last trading day of each month present in the calendar (optionally clipped)."""

    lower = parse_date(str(start)) if start is not None else None
    upper = parse_date(str(end)) if end is not None else None
    last_by_month: dict[tuple[int, int], str] = {}
    for day in trading_days(calendar):
        current = parse_date(day)
        if lower is not None and current < lower:
            continue
        if upper is not None and current > upper:
            continue
        last_by_month[(current.year, current.month)] = day
    return [last_by_month[key] for key in sorted(last_by_month)]


def calendar_from_snapshot(snapshot_dir: str) -> Snapshot:
    """Convenience loader used by stages that only need the calendar."""

    return load_snapshot(snapshot_dir)


__all__ = [
    "CALENDAR_FILE",
    "build_calendar",
    "calendar_from_snapshot",
    "is_trading_day",
    "month_end_trading_days",
    "next_trading_day",
    "previous_trading_day",
    "trading_days",
    "trading_days_from_benchmark",
]
