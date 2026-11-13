"""Offline parser tests using the small real payloads in tests/fixtures/raw/.

The raw JSON files were captured from the live public endpoints on 2026-11-13
and trimmed to a few rows; no network is used by these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aicwf.data.eastmoney import (
    FUNDAMENTAL_COLUMNS,
    KLINE_FIELDS,
    board_for_code,
    code_from_secid,
    normalize_code,
    parse_constituents,
    parse_fundamentals,
    parse_kline,
    secid_for_code,
)


@pytest.fixture(scope="module")
def raw_dir(project_root: Path) -> Path:
    path = project_root / "tests" / "fixtures" / "raw"
    if not path.exists():
        pytest.skip("raw payload fixtures are missing")
    return path


def load_payload(raw_dir: Path, name: str) -> dict:
    return json.loads((raw_dir / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Constituents
# ---------------------------------------------------------------------------


def test_parse_constituents_columns_and_values(raw_dir: Path):
    payload = load_payload(raw_dir, "constituents_small.json")
    frame = parse_constituents(payload)
    assert list(frame.columns) == [
        "code",
        "name",
        "industry",
        "price",
        "pct_chg",
        "turnover",
        "pe",
        "pb",
        "total_cap",
        "free_cap",
        "listed_date",
        "board",
    ]
    assert len(frame) == 5
    assert frame["code"].is_unique
    raw_row = payload["data"]["diff"][0]
    first = frame.iloc[0]
    # parser output must equal the raw payload fields (values move intraday, so
    # compare against the captured payload instead of hard-coded numbers)
    assert first["code"] == "000596.SZ"
    assert first["name"] == "古井贡酒"
    assert first["industry"] == "白酒Ⅱ"
    assert first["listed_date"] == "1996-09-27"
    assert first["board"] == "main"
    assert first["free_cap"] == pytest.approx(float(raw_row["f21"]))
    assert first["total_cap"] == pytest.approx(float(raw_row["f20"]))
    assert first["price"] == pytest.approx(float(raw_row["f2"]))
    assert first["pct_chg"] == pytest.approx(float(raw_row["f3"]))
    assert first["free_cap"] < first["total_cap"]
    assert "ST" not in first["name"]


def test_parse_constituents_accepts_dict_diff():
    payload = {
        "data": {
            "total": 1,
            "diff": {
                "0": {
                    "f12": "688981",
                    "f13": 1,
                    "f14": "中芯国际",
                    "f2": 55.0,
                    "f3": 1.25,
                    "f8": 0.8,
                    "f9": 60.0,
                    "f20": 1.0e11,
                    "f21": 8.0e10,
                    "f23": 3.3,
                    "f26": 20200716,
                    "f100": "半导体",
                }
            },
        }
    }
    frame = parse_constituents(payload)
    assert len(frame) == 1
    assert frame.iloc[0]["code"] == "688981.SH"
    assert frame.iloc[0]["board"] == "star"
    assert frame.iloc[0]["listed_date"] == "2020-07-16"


def test_parse_constituents_handles_placeholders():
    payload = {"data": {"diff": [{"f12": "600000", "f13": 1, "f14": "X", "f2": "-", "f21": None}]}}
    frame = parse_constituents(payload)
    assert pd.isna(frame.iloc[0]["price"])
    assert pd.isna(frame.iloc[0]["free_cap"])


# ---------------------------------------------------------------------------
# K-line
# ---------------------------------------------------------------------------


def test_parse_kline_stock_fields(raw_dir: Path):
    frame = parse_kline(load_payload(raw_dir, "kline_600519_small.json"))
    assert list(frame.columns) == list(KLINE_FIELDS)
    assert len(frame) == 22
    assert frame["date"].is_monotonic_increasing
    assert frame["date"].iloc[0] == "2024-01-02"
    assert frame["date"].iloc[-1] == "2024-01-31"
    first = frame.iloc[0]
    assert first["open"] == pytest.approx(1580.66)
    assert first["close"] == pytest.approx(1550.67)
    assert first["high"] == pytest.approx(1583.85)
    assert first["low"] == pytest.approx(1543.76)
    assert first["pct_chg"] == pytest.approx(-2.58)
    assert first["turnover"] == pytest.approx(0.26)
    # OHLC consistency holds for every bar
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (frame["volume"] > 0).all()


def test_parse_kline_index_fields(raw_dir: Path):
    frame = parse_kline(load_payload(raw_dir, "kline_index_small.json"))
    assert len(frame) == 22
    assert frame["date"].iloc[0] == "2024-01-02"
    assert frame["close"].iloc[0] == pytest.approx(3386.35, rel=1e-4)
    assert frame["volume"].iloc[0] > 0
    # index rows carry an amplitude/pct_chg; turnover is reported as a small number
    assert frame["pct_chg"].notna().all()
    assert frame["pct_chg"].iloc[0] == pytest.approx(-1.30)


def test_parse_kline_empty_payload_returns_typed_frame():
    frame = parse_kline({"rc": 0, "data": None})
    assert frame.empty
    assert list(frame.columns) == list(KLINE_FIELDS)


def test_parse_kline_skips_malformed_lines():
    payload = {"data": {"klines": ["2024-01-02,1,2,3,4,5,6,7,8,9,10", "garbage", ""]}}
    frame = parse_kline(payload)
    assert len(frame) == 1
    assert frame.iloc[0]["date"] == "2024-01-02"


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


def test_parse_fundamentals_maps_contract_fields(raw_dir: Path):
    payload = load_payload(raw_dir, "fundamentals_small.json")
    frame = parse_fundamentals(payload)
    assert list(frame.columns) == list(FUNDAMENTAL_COLUMNS)
    assert len(frame) == 2
    # parser output is sorted ascending by report_date; compare against the raw row
    raw_row = payload["result"]["data"][-1]
    first = frame.iloc[0]
    assert first["code"] == "600519.SH"
    assert first["report_date"] == "2026-06-30"
    assert first["notice_date"] == "2026-08-15"
    assert first["eps"] == pytest.approx(raw_row["EPSJB"])
    assert first["bps"] == pytest.approx(raw_row["BPS"])
    assert first["roe"] == pytest.approx(raw_row["ROEJQ"])
    assert first["gross_margin"] == pytest.approx(raw_row["XSMLL"])
    assert first["revenue"] == pytest.approx(raw_row["TOTALOPERATEREVE"])
    assert first["net_profit"] == pytest.approx(raw_row["PARENTNETPROFIT"])
    assert first["revenue_yoy"] == pytest.approx(raw_row["TOTALOPERATEREVETZ"])
    assert first["net_profit_yoy"] == pytest.approx(raw_row["PARENTNETPROFITTZ"])
    assert first["cfo_per_share"] == pytest.approx(raw_row["MGJYXJJE"])
    assert bool(first["notice_date_estimated"]) is False
    assert frame["report_date"].is_monotonic_increasing  # normalised to ascending


def test_parse_fundamentals_empty_result():
    frame = parse_fundamentals({"success": True, "result": None})
    assert frame.empty
    assert list(frame.columns) == list(FUNDAMENTAL_COLUMNS)


def test_parse_fundamentals_missing_notice_date_is_null():
    payload = {
        "result": {
            "data": [
                {
                    "SECUCODE": "000001.SZ",
                    "REPORT_DATE": "2023-06-30 00:00:00",
                    "NOTICE_DATE": None,
                    "EPSJB": 1.0,
                }
            ]
        }
    }
    frame = parse_fundamentals(payload)
    assert len(frame) == 1
    assert frame.iloc[0]["notice_date"] is None
    assert bool(frame.iloc[0]["notice_date_estimated"]) is False


# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------


def test_normalize_code_and_secid():
    assert normalize_code("600519") == "600519.SH"
    assert normalize_code("000001") == "000001.SZ"
    assert normalize_code("688981") == "688981.SH"
    assert normalize_code("830799") == "830799.BJ"
    assert normalize_code("600519", market=1) == "600519.SH"
    assert normalize_code("000001", market=0) == "000001.SZ"
    assert normalize_code("600519.sh") == "600519.SH"
    assert secid_for_code("600519.SH") == "1.600519"
    assert secid_for_code("000001.SZ") == "0.000001"
    assert code_from_secid("1.000300") == "000300.SH"


def test_board_for_code():
    assert board_for_code("600519.SH") == "main"
    assert board_for_code("000001.SZ") == "main"
    assert board_for_code("300750.SZ") == "chinext"
    assert board_for_code("301269.SZ") == "chinext"
    assert board_for_code("688981.SH") == "star"
    assert board_for_code("830799.BJ") == "bse"
