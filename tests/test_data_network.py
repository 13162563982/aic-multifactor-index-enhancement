"""Live-endpoint tests. Skipped by default; run with AICWF_NETWORK_TESTS=1.

These are the only tests in the market-data suite that touch the network. They
verify the public Eastmoney endpoints still answer with the documented shape;
normal CI/test runs skip them so the suite stays offline and fast.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from aicwf.data.eastmoney import (
    EastmoneyClient,
    load_data_config,
    parse_constituents,
    parse_fundamentals,
    parse_kline,
)

NETWORK_ENABLED = os.environ.get("AICWF_NETWORK_TESTS") == "1"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not NETWORK_ENABLED, reason="network tests disabled (set AICWF_NETWORK_TESTS=1)"),
]


@pytest.fixture(scope="module")
def client() -> EastmoneyClient:
    return EastmoneyClient.from_config(load_data_config())


def test_live_constituents(client: EastmoneyClient):
    frame = client.fetch_constituents()
    assert len(frame) >= 250
    assert frame["code"].is_unique
    for column in ("code", "name", "industry", "free_cap", "listed_date", "board"):
        assert column in frame.columns
    assert (frame["free_cap"] > 0).sum() >= 250


def test_live_stock_kline(client: EastmoneyClient):
    frame = client.fetch_kline("1.600519", "2024-01-02", "2024-01-31")
    assert len(frame) == 22
    assert frame["date"].iloc[0] == "2024-01-02"
    assert (frame["close"] > 0).all()
    assert list(frame.columns)[:8] == [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
    ]


def test_live_index_kline(client: EastmoneyClient):
    frame = client.fetch_index_kline("2024-01-02", "2024-01-31")
    assert len(frame) == 22
    assert frame["close"].iloc[0] > 1000


def test_live_fundamentals(client: EastmoneyClient):
    frame = client.fetch_fundamentals("600519.SH")
    assert len(frame) >= 10
    assert frame["code"].eq("600519.SH").all()
    assert frame["report_date"].is_monotonic_decreasing
    assert frame["notice_date"].notna().sum() >= 1
