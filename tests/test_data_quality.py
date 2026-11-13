"""Quality-gate tests: each gate must fail loudly on a broken snapshot."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from aicwf.contracts import DataQualityError
from aicwf.data.snapshot import load_snapshot, write_snapshot
from aicwf.data.validate import (
    MIN_UNIVERSE_SIZE_DEFAULT,
    QUALITY_DEFAULT,
    compute_quality_checks,
    load_quality_config,
    quality_failures,
    raise_on_quality_failures,
)

AS_OF = "2024-12-31"


@pytest.fixture(scope="module")
def mini_frames(mini_snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    snapshot = load_snapshot(mini_snapshot_dir)
    return {name: frame.copy() for name, frame in snapshot.frames.items()}


def write_variant(tmp_path: Path, frames: dict[str, pd.DataFrame], *, notes: list[str] | None = None) -> Path:
    target = tmp_path / "variant"
    write_snapshot(
        target,
        frames,
        as_of=AS_OF,
        created_at_utc="2025-01-01T00:00:00+00:00",
        source_urls={"test": "variant"},
        notes=notes if notes is not None else ["synthetic test fixture"],
    )
    return target


def checks_by_id(checks: list[dict]) -> dict[str, dict]:
    return {check["id"]: check for check in checks}


def test_clean_mini_snapshot_passes_all_gates(mini_snapshot_dir: Path):
    snapshot = load_snapshot(mini_snapshot_dir)
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert quality_failures(list(checks.values())) == []
    assert checks["universe_size"]["status"] == "pass"
    assert checks["missing_price_ratio"]["status"] == "pass"
    assert checks["fundamental_coverage"]["status"] == "pass"
    assert checks["as_of_cutoff"]["status"] == "pass"


def test_duplicate_rows_gate(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    frames["prices"] = pd.concat([frames["prices"], frames["prices"].head(3)], ignore_index=True)
    snapshot = load_snapshot(write_variant(tmp_path, frames))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["duplicate_rows"]["status"] == "fail"
    assert checks["duplicate_rows"]["detail"]["duplicate_rows"] == 3
    failures = quality_failures(list(checks.values()))
    assert any(check["id"] == "duplicate_rows" for check in failures)
    with pytest.raises(DataQualityError):
        raise_on_quality_failures(list(checks.values()), context="test")


def test_nonpositive_prices_gate(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    frames["prices"].loc[5, "close"] = -1.0
    snapshot = load_snapshot(write_variant(tmp_path, frames))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["positive_prices"]["status"] == "fail"
    assert checks["positive_prices"]["detail"]["nonpositive_rows"] == 1


def test_stale_snapshot_gate(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    calendar = frames["calendar"].sort_values("date").reset_index(drop=True)
    keep = calendar["date"].max()
    trimmed_days = calendar.loc[calendar["date"] >= "2024-12-16", "date"]
    frames["calendar"] = calendar.loc[~calendar["date"].isin(trimmed_days)].reset_index(drop=True)
    frames["benchmark"] = frames["benchmark"].loc[~frames["benchmark"]["date"].isin(trimmed_days)].reset_index(drop=True)
    frames["prices"] = frames["prices"].loc[~frames["prices"]["date"].isin(trimmed_days)].reset_index(drop=True)
    # as_of stays 2024-12-31 while the data stops ~2 trading weeks earlier
    snapshot = load_snapshot(write_variant(tmp_path, frames))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["stale_trading_days"]["status"] == "fail"
    assert checks["stale_trading_days"]["detail"]["stale_trading_days"] > 5
    assert keep  # keep the variable referenced (documents original max date)


def test_fundamental_coverage_gate(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    dropped_codes = sorted(frames["universe"]["code"].unique())[:3]
    frames["fundamentals"] = frames["fundamentals"].loc[~frames["fundamentals"]["code"].isin(dropped_codes)].reset_index(drop=True)
    snapshot = load_snapshot(write_variant(tmp_path, frames))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["fundamental_coverage"]["status"] == "fail"
    detail = checks["fundamental_coverage"]["detail"]
    assert detail["covered_codes"] == 3
    assert detail["coverage"] == pytest.approx(0.5)


def test_universe_size_gate_for_non_synthetic_snapshot(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    snapshot = load_snapshot(write_variant(tmp_path, frames, notes=["production build"]))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["universe_size"]["status"] == "fail"
    assert checks["universe_size"]["detail"]["universe_size"] == 6
    assert checks["universe_size"]["detail"]["minimum"] == MIN_UNIVERSE_SIZE_DEFAULT


def test_universe_size_gate_relaxed_for_synthetic(mini_snapshot_dir: Path):
    snapshot = load_snapshot(mini_snapshot_dir)
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF, min_universe_size=250))
    assert checks["universe_size"]["status"] == "pass"
    assert checks["universe_size"]["detail"]["synthetic_fixture"] is True


def test_future_dated_rows_gate(tmp_path: Path, mini_frames):
    frames = copy.deepcopy(mini_frames)
    extra = frames["prices"].tail(1).copy()
    extra["date"] = "2025-01-02"
    frames["prices"] = pd.concat([frames["prices"], extra], ignore_index=True)
    snapshot = load_snapshot(write_variant(tmp_path, frames))
    checks = checks_by_id(compute_quality_checks(snapshot, as_of=AS_OF))
    assert checks["as_of_cutoff"]["status"] == "fail"


def test_load_quality_config_merges_defaults(tmp_path: Path):
    defaults = load_quality_config(None)
    for key, value in QUALITY_DEFAULT.items():
        assert defaults[key] == value
    overridden = load_quality_config({"data": {"quality": {"max_missing_price_ratio": 0.02}}})
    assert overridden["max_missing_price_ratio"] == 0.02
    assert overridden["max_stale_trading_days"] == QUALITY_DEFAULT["max_stale_trading_days"]
    direct = load_quality_config({"quality": {"min_fundamental_coverage": 0.5}})
    assert direct["min_fundamental_coverage"] == 0.5


def test_raise_on_quality_failures_noop_when_clean():
    checks = [{"id": "x", "status": "pass", "detail": {}}, {"id": "y", "status": "warn", "detail": {}}]
    raise_on_quality_failures(checks)  # no raise
