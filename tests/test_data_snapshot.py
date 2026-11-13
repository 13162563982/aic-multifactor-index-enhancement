"""Snapshot layout, hashing, manifest and integrity tests (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aicwf.data.snapshot import (
    FRAME_SPECS,
    SNAPSHOT_FILES,
    compute_snapshot_hash,
    content_hashes,
    file_hashes,
    load_snapshot,
    verify_snapshot,
    write_snapshot,
)
from aicwf.util.hashing import hash_json


def make_frames(as_of: str = "2024-12-04") -> dict[str, pd.DataFrame]:
    dates = ["2024-12-02", "2024-12-03", "2024-12-04"]
    codes = ["600001.SH", "000001.SZ"]
    price_rows = []
    for code, base in zip(codes, (10.0, 20.0)):
        for offset, day in enumerate(dates):
            close = base + offset
            price_rows.append(
                {
                    "date": day,
                    "code": code,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + offset,
                    "amount": close * 1000,
                    "turnover": 0.5,
                    "pct_chg": 0.1 if offset else 0.0,
                    "board": "main",
                }
            )
    prices = pd.DataFrame(price_rows)
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": [4000.0, 4010.0, 4005.0],
            "ret": [float("nan"), 0.0025, -0.00125],
        }
    )
    calendar = pd.DataFrame({"date": dates})
    universe = pd.DataFrame(
        {
            "code": codes,
            "name": ["Alpha", "Beta"],
            "industry": ["银行", "白酒Ⅱ"],
            "weight": [0.6, 0.4],
            "free_cap": [6.0e10, 4.0e10],
            "total_cap": [9.0e10, 6.0e10],
            "listed_date": ["2010-01-04", "2012-06-08"],
            "is_st": [False, False],
            "board": ["main", "main"],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "code": codes * 2,
            "report_date": ["2024-09-30"] * 2 + ["2024-06-30"] * 2,
            "notice_date": ["2024-10-25"] * 2 + ["2024-07-20"] * 2,
            "eps": [1.0, 2.0, 0.9, 1.9],
            "bps": [10.0, 20.0, 9.9, 19.9],
            "roe": [10.0, 10.0, 9.0, 9.5],
            "gross_margin": [40.0, 50.0, 41.0, 49.0],
            "revenue": [1.0e9, 2.0e9, 0.9e9, 1.9e9],
            "revenue_yoy": [5.0, 6.0, 4.0, 5.5],
            "net_profit": [1.0e8, 2.0e8, 0.9e8, 1.9e8],
            "net_profit_yoy": [3.0, 4.0, 2.0, 3.5],
            "cfo_per_share": [1.2, 2.2, 1.1, 2.1],
            "notice_date_estimated": [False, False, False, False],
        }
    )
    return {
        "prices": prices,
        "fundamentals": fundamentals,
        "universe": universe,
        "benchmark": benchmark,
        "calendar": calendar,
    }


@pytest.fixture
def written_snapshot(tmp_path: Path) -> Path:
    snapshot_dir = tmp_path / "snap"
    write_snapshot(
        snapshot_dir,
        make_frames(),
        as_of="2024-12-04",
        created_at_utc="2025-01-01T00:00:00+00:00",
        source_urls={"test": "https://example.invalid"},
        notes=["test snapshot"],
    )
    return snapshot_dir


def test_write_and_load_round_trip(written_snapshot: Path):
    snapshot = load_snapshot(written_snapshot)
    assert snapshot.as_of.isoformat() == "2024-12-04"
    assert set(snapshot.frames) == set(FRAME_SPECS)
    assert len(snapshot.prices) == 6
    assert len(snapshot.universe) == 2
    assert snapshot.manifest["row_counts"]["prices"] == 6
    assert snapshot.manifest["date_ranges"]["prices"] == ["2024-12-02", "2024-12-04"]
    # dates stay ISO strings, not datetimes
    assert snapshot.prices["date"].map(type).eq(str).all()
    assert snapshot.universe["is_st"].dtype == bool
    assert snapshot.fundamentals["notice_date_estimated"].dtype == bool
    # sorted by (date, code)
    ordered = snapshot.prices.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    assert snapshot.prices[["date", "code"]].equals(ordered[["date", "code"]])


def test_manifest_hashes_are_consistent(written_snapshot: Path):
    manifest = json.loads((written_snapshot / "manifest.json").read_text(encoding="utf-8"))
    for required in ("schema_version", "as_of", "created_at_utc", "source_urls", "row_counts", "date_ranges", "files", "snapshot_hash", "notes"):
        assert required in manifest
    assert manifest["files"], "files mapping must not be empty"
    for name, digest in manifest["files"].items():
        assert isinstance(name, str) and name.endswith((".parquet", ".json"))
        assert isinstance(digest, str) and len(digest) == 64
    # snapshot_hash is reproducible from the flat files mapping
    assert hash_json(manifest["files"]) == manifest["snapshot_hash"]
    # content hashes cover the same files and are deterministic artifact hashes
    assert set(manifest["content_hashes"]) == set(manifest["files"])
    for name in SNAPSHOT_FILES:
        assert name in manifest["files"]
    assert verify_snapshot(written_snapshot, deep=True)["ok"] is True
    assert verify_snapshot(written_snapshot, deep=False)["ok"] is True


def test_rebuild_is_content_deterministic(tmp_path: Path):
    first = write_snapshot(
        tmp_path / "one",
        make_frames(),
        as_of="2024-12-04",
        created_at_utc="2025-01-01T00:00:00+00:00",
    )
    second = write_snapshot(
        tmp_path / "two",
        make_frames(),
        as_of="2024-12-04",
        created_at_utc="2025-01-01T00:00:00+00:00",
    )
    assert first["files"] == second["files"]
    assert first["content_hashes"] == second["content_hashes"]
    assert first["snapshot_hash"] == second["snapshot_hash"]


def test_tampering_is_detected(written_snapshot: Path):
    prices_path = written_snapshot / "prices.parquet"
    frame = pd.read_parquet(prices_path)
    frame.loc[0, "close"] = frame.loc[0, "close"] + 1.0
    frame.to_parquet(prices_path, index=False)
    result = verify_snapshot(written_snapshot, deep=True)
    assert result["ok"] is False
    assert "prices.parquet" in result["file_mismatches"]
    assert "prices.parquet" in result["content_mismatches"]


def test_missing_file_is_reported(written_snapshot: Path):
    (written_snapshot / "calendar.parquet").unlink()
    result = verify_snapshot(written_snapshot, deep=True)
    assert result["ok"] is False
    assert result["missing_files"]


def test_mini_fixture_round_trip(mini_snapshot_dir: Path):
    snapshot = load_snapshot(mini_snapshot_dir)
    assert snapshot.as_of.isoformat() == "2024-12-31"
    assert len(snapshot.universe) == 6
    assert snapshot.universe["code"].nunique() == 6
    assert set(snapshot.universe["board"]) == {"main", "chinext", "star"}
    # 36 months of business days ending 2024-12-31
    days = pd.to_datetime(snapshot.calendar["date"])
    assert days.min() >= pd.Timestamp("2022-01-01")
    assert days.max() == pd.Timestamp("2024-12-31")
    assert len(days) >= 750
    assert verify_snapshot(mini_snapshot_dir, deep=True)["ok"] is True
    assert snapshot.manifest_hash_consistent() is True
    # estimated notice dates are stored and flagged for the fixture
    assert snapshot.fundamentals["notice_date"].isna().sum() == 0
    assert int(snapshot.fundamentals["notice_date_estimated"].sum()) >= 1
    # no prices before a listed date
    for row in snapshot.universe.itertuples():
        code_prices = snapshot.prices.loc[snapshot.prices["code"] == row.code]
        assert code_prices["date"].min() >= row.listed_date


def test_committed_fixture_hashes_are_self_consistent(mini_snapshot_dir: Path):
    manifest = json.loads((mini_snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_hash"] == compute_snapshot_hash(manifest["files"])
    assert file_hashes(mini_snapshot_dir, files=list(manifest["files"])) == manifest["files"]
    assert content_hashes(mini_snapshot_dir, files=list(manifest["files"])) == manifest["content_hashes"]
    assert manifest["as_of"] == "2024-12-31"


def test_real_snapshot_manifest_self_consistency(project_root: Path):
    snapshot_dir = project_root / "data" / "snapshots" / "2026-11-12"
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("real snapshot has not been built in this workspace")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["as_of"] == "2026-11-12"
    assert hash_json(manifest["files"]) == manifest["snapshot_hash"]
    for name in ("prices.parquet", "fundamentals.parquet", "universe.parquet", "benchmark.parquet", "calendar.parquet"):
        assert (snapshot_dir / name).exists()
    assert manifest["row_counts"]["universe"] >= 250
    assert manifest["date_ranges"]["prices"][0] == "2016-01-04"
