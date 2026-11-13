"""Time helpers. All persisted timestamps are UTC ISO-8601; business dates are Asia/Shanghai."""

from __future__ import annotations

import datetime as dt

MARKET_TZ = dt.timezone(dt.timedelta(hours=8))


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def market_now() -> dt.datetime:
    return dt.datetime.now(MARKET_TZ)


def month_end_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    """Calendar month ends clipped to ``end`` (informational only).

    The trading calendar decides the actual rebalance days; this helper is used by
    scheduling and tests.
    """

    dates: list[dt.date] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        if month == 12:
            nxt = dt.date(year + 1, 1, 1)
        else:
            nxt = dt.date(year, month + 1, 1)
        candidate = nxt - dt.timedelta(days=1)
        if candidate < start:
            candidate = start
        if candidate > end:
            candidate = end
        dates.append(candidate)
        year, month = nxt.year, nxt.month
    return dates
