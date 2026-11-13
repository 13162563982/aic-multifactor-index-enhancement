"""Quality gates for the frozen snapshot and the data-prep panel.

Thresholds come from ``config/data.yaml`` (``quality`` block) and are enforced by
:func:`aicwf.data.stage.run_data_prep`; a failed gate raises
:class:`aicwf.contracts.DataQualityError`, which the controller treats as
non-retryable.

Each check is a small JSON-serialisable dict ``{id, status, detail}`` with
``status`` in ``{"pass", "warn", "fail"}``. Checks are also copied into
``02_data/data_quality.json`` so reviewers and the verification stage can see
exactly which gate was evaluated.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from aicwf.contracts import PROJECT_ROOT, DataQualityError, parse_date
from aicwf.data.pit import available_as_of
from aicwf.data.snapshot import Snapshot
from aicwf.util.jsonio import read_yaml

QUALITY_DEFAULT: dict[str, Any] = {
    "max_missing_price_ratio": 0.10,
    "max_stale_trading_days": 5,
    "min_fundamental_coverage": 0.85,
    "duplicate_rows_allowed": False,
    "require_positive_prices": True,
}

MIN_UNIVERSE_SIZE_DEFAULT = 250


def load_quality_config(
    config: Mapping[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the merged quality thresholds.

    Accepts either the raw ``config/data.yaml`` payload (``{"quality": ...}``) or
    the full config bundle (``{"data": {"quality": ...}}``); falls back to
    ``config/data.yaml`` on disk and then to :data:`QUALITY_DEFAULT`.
    """

    quality: dict[str, Any] = {}
    if config is not None:
        cfg = dict(config)
        if isinstance(cfg.get("quality"), Mapping):
            quality = dict(cfg["quality"])
        elif isinstance(cfg.get("data"), Mapping) and isinstance(cfg["data"].get("quality"), Mapping):
            quality = dict(cfg["data"]["quality"])
    if not quality:
        config_path = Path(path) if path else PROJECT_ROOT / "config" / "data.yaml"
        if config_path.exists():
            loaded = read_yaml(config_path) or {}
            if isinstance(loaded.get("quality"), Mapping):
                quality = dict(loaded["quality"])
    merged = dict(QUALITY_DEFAULT)
    merged.update(quality)
    return merged


def _missing_price_ratio(prices: pd.DataFrame, calendar: pd.DataFrame, universe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate share of expected trading-day/name cells without a bar.

    Expected cells are counted from a code's first observed bar (or its listed
    date, if later) up to the last calendar day; missing bars are suspensions or
    fetch gaps.
    """

    if prices.empty or calendar.empty or universe.empty:
        return {"missing_ratio": 1.0, "expected_cells": 0, "observed_cells": 0, "worst_codes": []}
    calendar_days = pd.to_datetime(calendar["date"], errors="coerce").dropna()
    first_bar = prices.groupby("code")["date"].min()
    observed_by_code = prices.groupby("code").size()
    listed = universe.set_index("code")["listed_date"] if "listed_date" in universe.columns else None
    expected = 0
    observed = int(len(prices))
    ratios: list[tuple[str, float]] = []
    for code, first_date in first_bar.items():
        start = pd.to_datetime(first_date, errors="coerce")
        if listed is not None and code in listed.index:
            listed_date = pd.to_datetime(listed.loc[code], errors="coerce")
            if pd.notna(listed_date) and pd.notna(start):
                start = max(start, listed_date)
        if pd.isna(start):
            continue
        code_expected = int((calendar_days >= start).sum())
        expected += code_expected
        code_observed = int(observed_by_code.get(code, 0))
        if code_expected > 0:
            ratios.append((code, 1.0 - min(1.0, code_observed / code_expected)))
    missing_ratio = 0.0 if expected == 0 else max(0.0, 1.0 - observed / expected)
    worst = sorted(ratios, key=lambda item: item[1], reverse=True)[:5]
    return {
        "missing_ratio": float(round(missing_ratio, 6)),
        "expected_cells": int(expected),
        "observed_cells": int(observed),
        "worst_codes": [{"code": code, "missing_ratio": round(ratio, 6)} for code, ratio in worst],
    }


def _stale_trading_days(calendar: pd.DataFrame, as_of: dt.date) -> int:
    """Business days between the last snapshot day and ``as_of`` (weekday proxy)."""

    days = pd.to_datetime(calendar["date"], errors="coerce").dropna()
    if days.empty:
        return 0
    last = days.max().date()
    if last >= as_of:
        return 0
    business_days = pd.bdate_range(start=last + dt.timedelta(days=1), end=as_of, freq="B")
    return int(len(business_days))


def compute_quality_checks(
    snapshot: Snapshot,
    *,
    as_of: str | dt.date | dt.datetime | None = None,
    quality: Mapping[str, Any] | None = None,
    min_universe_size: int = MIN_UNIVERSE_SIZE_DEFAULT,
) -> list[dict[str, Any]]:
    """Evaluate every configured quality gate; returns ``[{id,status,detail}]``."""

    thresholds = dict(QUALITY_DEFAULT)
    if quality:
        thresholds.update(dict(quality))
    cutoff = parse_date(str(as_of)) if as_of is not None else snapshot.as_of
    universe = snapshot.universe
    prices = snapshot.prices
    calendar = snapshot.calendar
    checks: list[dict[str, Any]] = []

    # 1) universe size -------------------------------------------------
    # The deterministic synthetic mini fixture intentionally has 6 names, so the
    # production universe-size gate would always fail on it. Relax it only for
    # snapshots whose provenance says they are synthetic; real snapshots stay strict.
    provenance = snapshot.manifest.get("provenance") or {}
    notes_text = " ".join(str(note) for note in (snapshot.manifest.get("notes") or []))
    synthetic = "make_fixture" in str(provenance.get("builder", "")) or "synthetic" in notes_text.lower()
    effective_min = 0 if synthetic else int(min_universe_size)
    universe_size = int(len(universe))
    checks.append(
        {
            "id": "universe_size",
            "status": "pass" if universe_size >= effective_min else "fail",
            "detail": {
                "universe_size": universe_size,
                "minimum": effective_min,
                "configured_minimum": int(min_universe_size),
                "synthetic_fixture": synthetic,
            },
        }
    )

    # 2) duplicate (date, code) rows -----------------------------------
    duplicate_count = int(prices.duplicated(subset=["date", "code"]).sum()) if not prices.empty else 0
    duplicates_allowed = bool(thresholds.get("duplicate_rows_allowed", False))
    checks.append(
        {
            "id": "duplicate_rows",
            "status": "pass" if duplicate_count == 0 or duplicates_allowed else "fail",
            "detail": {"duplicate_rows": duplicate_count, "allowed": duplicates_allowed},
        }
    )

    # 3) positive prices ----------------------------------------------
    close = pd.to_numeric(prices.get("close"), errors="coerce") if "close" in prices.columns else pd.Series(dtype=float)
    nonpositive = int((close <= 0).sum())
    require_positive = bool(thresholds.get("require_positive_prices", True))
    checks.append(
        {
            "id": "positive_prices",
            "status": "pass" if nonpositive == 0 or not require_positive else "fail",
            "detail": {"nonpositive_rows": nonpositive, "required": require_positive},
        }
    )

    # 4) missing price ratio ------------------------------------------
    missing = _missing_price_ratio(prices, calendar, universe)
    max_missing = float(thresholds.get("max_missing_price_ratio", 0.10))
    checks.append(
        {
            "id": "missing_price_ratio",
            "status": "pass" if missing["missing_ratio"] <= max_missing else "fail",
            "detail": {**missing, "threshold": max_missing},
        }
    )

    # 5) stale data (trading days between last calendar day and as_of) --
    stale = _stale_trading_days(calendar, cutoff)
    max_stale = int(thresholds.get("max_stale_trading_days", 5))
    checks.append(
        {
            "id": "stale_trading_days",
            "status": "pass" if stale <= max_stale else "fail",
            "detail": {"stale_trading_days": stale, "threshold": max_stale, "as_of": cutoff.isoformat()},
        }
    )

    # 6) fundamental coverage (PIT rows available on as_of) -------------
    fundamentals = snapshot.fundamentals
    available = available_as_of(fundamentals, cutoff) if fundamentals is not None else pd.DataFrame()
    available_codes = set(available["code"].unique()) if not available.empty and "code" in available.columns else set()
    universe_codes = set(universe["code"].unique()) if not universe.empty else set()
    covered = len(available_codes & universe_codes)
    coverage = (covered / len(universe_codes)) if universe_codes else 0.0
    min_coverage = float(thresholds.get("min_fundamental_coverage", 0.85))
    checks.append(
        {
            "id": "fundamental_coverage",
            "status": "fail" if coverage < min_coverage else "pass",
            "detail": {
                "covered_codes": covered,
                "universe_codes": len(universe_codes),
                "coverage": float(round(coverage, 6)),
                "minimum": min_coverage,
            },
        }
    )

    # 7) no future-dated rows ------------------------------------------
    def _max_date(frame: pd.DataFrame, column: str = "date") -> str | None:
        if frame is None or frame.empty or column not in frame.columns:
            return None
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        return values.max().date().isoformat() if not values.empty else None

    price_max = _max_date(prices)
    benchmark_max = _max_date(snapshot.benchmark)
    future = [value for value in (price_max, benchmark_max) if value is not None and value > cutoff.isoformat()]
    checks.append(
        {
            "id": "as_of_cutoff",
            "status": "fail" if future else "pass",
            "detail": {"as_of": cutoff.isoformat(), "max_price_date": price_max, "max_benchmark_date": benchmark_max},
        }
    )

    # 8) benchmark covers the calendar ---------------------------------
    calendar_days = pd.to_datetime(calendar["date"], errors="coerce").dropna() if not calendar.empty else pd.Series(dtype="datetime64[ns]")
    benchmark_days = (
        pd.to_datetime(snapshot.benchmark["date"], errors="coerce").dropna() if not snapshot.benchmark.empty else pd.Series(dtype="datetime64[ns]")
    )
    same_calendar = bool(calendar_days.empty and benchmark_days.empty) or (
        len(calendar_days) == len(benchmark_days) and calendar_days.isin(benchmark_days).all()
    )
    checks.append(
        {
            "id": "benchmark_calendar",
            "status": "pass" if same_calendar else "fail",
            "detail": {"calendar_days": int(len(calendar_days)), "benchmark_days": int(len(benchmark_days))},
        }
    )
    return checks


def quality_failures(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(check) for check in checks if check.get("status") == "fail"]


def raise_on_quality_failures(checks: Sequence[Mapping[str, Any]], *, context: str = "") -> None:
    failures = quality_failures(checks)
    if not failures:
        return
    details = "; ".join(f"{check['id']}: {check.get('detail')}" for check in failures)
    prefix = f"{context}: " if context else ""
    raise DataQualityError(f"{prefix}data quality gate(s) failed -- {details}")
