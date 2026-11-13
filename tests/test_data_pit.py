"""Point-in-time helper tests: availability filtering and future-data guards."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from aicwf.contracts import ValidationError
from aicwf.data.pit import (
    FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
    assert_no_lookahead,
    assert_notice_dates_valid,
    available_as_of,
    date_violations,
    effective_notice_dates,
    fill_estimated_notice_dates,
    max_available_report_date,
)


def fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["600001.SH", "600001.SH", "600001.SH", "600002.SH"],
            "report_date": ["2023-12-31", "2024-03-31", "2024-06-30", "2023-09-30"],
            "notice_date": ["2024-03-20", None, "2024-08-20", "2023-10-25"],
            "eps": [1.0, 0.3, 0.4, 0.5],
            "bps": [10.0, 10.2, 10.4, 5.0],
            "roe": [10.0, 3.0, 3.5, 9.0],
            "gross_margin": [40.0, 40.0, 41.0, 30.0],
            "revenue": [1.0e9, 2.5e8, 2.6e8, 1.0e8],
            "revenue_yoy": [5.0, 4.0, 6.0, 3.0],
            "net_profit": [1.0e8, 2.5e7, 2.6e7, 1.0e7],
            "net_profit_yoy": [3.0, 2.0, 4.0, 1.0],
            "cfo_per_share": [1.2, 0.3, 0.4, 0.5],
            "notice_date_estimated": [False, False, False, False],
        }
    )


def test_available_as_of_filters_on_notice_date():
    frame = fundamentals_frame()
    available = available_as_of(frame, "2024-02-01")
    # 2023 annual report announced 2024-03-20 is not yet available
    assert set(available["report_date"]) == {"2023-09-30"}

    available = available_as_of(frame, "2024-03-20")
    assert set(available["report_date"]) == {"2023-09-30", "2023-12-31"}

    available = available_as_of(frame, "2024-12-31")
    # 2024 Q1 missing notice -> estimated 2024-03-31 + 90d = 2024-06-29, so available
    assert set(available["report_date"]) == {"2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"}
    assert "effective_notice_date" in available.columns


def test_missing_notice_date_uses_90_day_fallback_and_flags():
    frame = fundamentals_frame()
    filled = fill_estimated_notice_dates(frame.copy())
    row = filled.loc[(filled["code"] == "600001.SH") & (filled["report_date"] == "2024-03-31")].iloc[0]
    assert row["notice_date"] == "2024-06-29"
    assert bool(row["notice_date_estimated"]) is True

    before = available_as_of(frame, "2024-06-28")
    assert "2024-03-31" not in set(before["report_date"])
    after = available_as_of(frame, "2024-06-29")
    assert "2024-03-31" in set(after["report_date"])
    estimated_rows = after.loc[after["report_date"] == "2024-03-31"]
    assert bool(estimated_rows.iloc[0]["notice_date_estimated"]) is True


def test_fallback_days_is_configurable():
    frame = fundamentals_frame()
    available = available_as_of(frame, "2024-06-01", fallback_days=30)
    # 2024-03-31 + 30d = 2024-04-30 <= 2024-06-01 -> included
    assert "2024-03-31" in set(available["report_date"])
    assert FUNDAMENTAL_NOTICE_FALLBACK_DAYS == 90


def test_future_report_periods_are_excluded():
    frame = fundamentals_frame()
    # report_date after the cutoff can never be available
    available = available_as_of(frame, "2023-10-01")
    assert available.empty or (pd.to_datetime(available["report_date"]).dt.date <= dt.date(2023, 10, 1)).all()


def test_effective_notice_dates_series():
    frame = fundamentals_frame()
    effective = effective_notice_dates(frame)
    assert effective.iloc[0] == "2024-03-20"
    assert effective.iloc[1] == "2024-06-29"


def test_assert_no_lookahead_accepts_rows_on_cutoff():
    panel = pd.DataFrame({"date": ["2024-01-02", "2024-01-05"], "code": ["A", "A"]})
    assert_no_lookahead(panel, "2024-01-05")  # no raise
    assert date_violations(panel, "2024-01-05") == []


def test_assert_no_lookahead_rejects_future_rows():
    panel = pd.DataFrame({"date": ["2024-01-02", "2024-01-08"], "code": ["A", "A"]})
    with pytest.raises(ValidationError, match="after as_of"):
        assert_no_lookahead(panel, "2024-01-05", name="panel")
    violations = date_violations(panel, "2024-01-05", name="panel")
    assert len(violations) == 1
    assert violations[0]["date"] == "2024-01-08"
    assert violations[0]["code"] == "A"


def test_assert_no_lookahead_ignores_missing_date_column():
    frame = pd.DataFrame({"x": [1, 2]})
    assert_no_lookahead(frame, "2024-01-05")


def test_assert_notice_dates_valid_rejects_leakage():
    available = available_as_of(fundamentals_frame(), "2024-01-31")
    assert_notice_dates_valid(available, "2024-01-31")  # filtered frame is fine
    with pytest.raises(ValidationError):
        assert_notice_dates_valid(fundamentals_frame(), "2024-01-31")


def test_max_available_report_date():
    frame = fundamentals_frame()
    assert max_available_report_date(frame, "2024-02-01") == "2023-09-30"
    assert max_available_report_date(frame, "2024-12-31") == "2024-06-30"
    assert max_available_report_date(frame.iloc[0:0], "2024-12-31") is None
