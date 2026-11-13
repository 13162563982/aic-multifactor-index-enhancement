"""Market-data subsystem: frozen snapshots, Eastmoney client, PIT helpers.

Public entry points used by the rest of the workflow:

* :func:`aicwf.data.stage.run_data_prep` -- data-prep stage entry point.
* :func:`aicwf.data.snapshot.load_snapshot` / :func:`write_snapshot` -- frozen
  snapshot IO and manifest hashing.
* :func:`aicwf.data.pit.available_as_of` / :func:`assert_no_lookahead` -- PIT
  fundamentals and future-data guards reused by verification.
* :class:`aicwf.data.eastmoney.EastmoneyClient` -- token-less public data client
  with retries/host rotation (used by ``scripts/build_snapshot.py``).
"""

from __future__ import annotations

from aicwf.data.calendar import (
    build_calendar,
    is_trading_day,
    month_end_trading_days,
    next_trading_day,
    previous_trading_day,
    trading_days,
    trading_days_from_benchmark,
)
from aicwf.data.eastmoney import (
    EastmoneyClient,
    board_for_code,
    load_data_config,
    normalize_code,
    parse_constituents,
    parse_fundamentals,
    parse_kline,
    secid_for_code,
)
from aicwf.data.http import HttpClient, HttpTransport, RequestsTransport, TransportError
from aicwf.data.pit import (
    FUNDAMENTAL_NOTICE_FALLBACK_DAYS,
    assert_no_lookahead,
    assert_notice_dates_valid,
    available_as_of,
    effective_notice_dates,
    fill_estimated_notice_dates,
    max_available_report_date,
)
from aicwf.data.snapshot import (
    FRAME_SPECS,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    SNAPSHOT_FILES,
    Snapshot,
    compute_snapshot_hash,
    content_hashes,
    file_hashes,
    load_snapshot,
    snapshot_dir_for,
    verify_snapshot,
    write_snapshot,
)
from aicwf.data.validate import (
    QUALITY_DEFAULT,
    compute_quality_checks,
    load_quality_config,
    quality_failures,
)
from aicwf.data.stage import run_data_prep

__all__ = [
    "FUNDAMENTAL_NOTICE_FALLBACK_DAYS",
    "FRAME_SPECS",
    "MANIFEST_FILE",
    "EastmoneyClient",
    "HttpClient",
    "HttpTransport",
    "QUALITY_DEFAULT",
    "RequestsTransport",
    "SCHEMA_VERSION",
    "SNAPSHOT_FILES",
    "Snapshot",
    "TransportError",
    "assert_no_lookahead",
    "assert_notice_dates_valid",
    "available_as_of",
    "board_for_code",
    "build_calendar",
    "compute_quality_checks",
    "compute_snapshot_hash",
    "content_hashes",
    "effective_notice_dates",
    "file_hashes",
    "fill_estimated_notice_dates",
    "is_trading_day",
    "load_data_config",
    "load_quality_config",
    "load_snapshot",
    "max_available_report_date",
    "month_end_trading_days",
    "next_trading_day",
    "normalize_code",
    "parse_constituents",
    "parse_fundamentals",
    "parse_kline",
    "previous_trading_day",
    "quality_failures",
    "run_data_prep",
    "secid_for_code",
    "snapshot_dir_for",
    "trading_days",
    "trading_days_from_benchmark",
    "verify_snapshot",
    "write_snapshot",
]
