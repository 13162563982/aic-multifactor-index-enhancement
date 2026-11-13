"""Build the deterministic, fully offline mini snapshot used by the test suite.

Priority-one deliverable: other subsystems (quant, verification, platform, agents)
run their offline tests against ``tests/fixtures/mini_snapshot/`` instead of the
real 300-name snapshot. The fixture mirrors the real snapshot layout exactly
(see ``docs/artifact_contract.md`` section 1)::

    tests/fixtures/mini_snapshot/
        prices.parquet
        fundamentals.parquet
        universe.parquet
        benchmark.parquet
        calendar.parquet
        coverage_report.json
        manifest.json

Specs: 6 synthetic stocks, 36 months of business-day bars ending 2024-12-31,
fixed seed (``--seed``), no network, deterministic content hashes. One stock is
listed late and one has a handful of missing (suspended) bars; one stock has
fundamental rows with a missing announcement date so the point-in-time fallback
(``report_date + 90 days``) and its ``notice_date_estimated`` flag are exercised.

Usage::

    python scripts/make_fixture.py                 # writes tests/fixtures/mini_snapshot
    python scripts/make_fixture.py --out /tmp/mini --seed 7
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicwf.data.pit import fill_estimated_notice_dates  # noqa: E402
from aicwf.data.snapshot import (  # noqa: E402
    FRAME_ORDER,
    SCHEMA_VERSION,
    write_snapshot,
)

DEFAULT_OUT = PROJECT_ROOT / "tests" / "fixtures" / "mini_snapshot"
DEFAULT_SEED = 20240101
DEFAULT_AS_OF = "2024-12-31"
DEFAULT_CREATED_AT = "2025-01-01T00:00:00+00:00"

# code, name, industry, board, free_cap (CNY), total_cap, listed_date, start_price
STOCKS = (
    ("600001.SH", "Synth Alpha", "白酒Ⅱ", "main", 120.0e9, 150.0e9, "2010-04-19", 42.0),
    ("600002.SH", "Synth Beta", "银行", "main", 85.0e9, 120.0e9, "2008-09-25", 18.5),
    ("000001.SZ", "Synth Gamma", "软件开发", "main", 60.0e9, 80.0e9, "2012-06-08", 27.3),
    ("000002.SZ", "Synth Delta", "银行", "main", 45.0e9, 60.0e9, "2015-01-06", 12.8),
    ("300001.SZ", "Synth Epsilon", "通信设备", "chinext", 30.0e9, 40.0e9, "2019-11-21", 35.6),
    ("688001.SH", "Synth Zeta", "半导体", "star", 20.0e9, 26.0e9, "2022-06-01", 55.0),
)

BENCHMARK_START = 4000.0


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Synthetic trading calendar: every weekday (no exchange holidays)."""

    days = pd.bdate_range(start, end, freq="B")
    return pd.DatetimeIndex([day for day in days if day.weekday() < 5])


def build_prices(
    stock: tuple,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    missing_dates: tuple[str, ...] = (),
) -> pd.DataFrame:
    code, name, industry, board, free_cap, total_cap, listed_date, start_price = stock
    n = len(dates)
    daily_ret = rng.normal(0.00035, 0.0155, size=n)
    # occasional earnings-day jumps keep the series interesting but realistic
    jump_mask = rng.random(n) < 0.02
    daily_ret = daily_ret + jump_mask * rng.normal(0.0, 0.04, size=n)
    close = start_price * np.exp(np.cumsum(daily_ret))
    overnight = rng.normal(0.0, 0.004, size=n)
    open_ = close * np.exp(overnight)
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0.0, 0.006, size=n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0.0, 0.006, size=n)))
    volume = np.round(rng.lognormal(mean=13.2, sigma=0.45, size=n)).astype("int64")
    amount = np.round(volume * close, 2)
    float_shares = free_cap / start_price  # constant synthetic free-float shares
    turnover = np.round(volume / float_shares * 100.0, 4)
    pct_chg = (pd.Series(close).pct_change() * 100.0).fillna(0.0).to_numpy()

    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "code": code,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
            "amount": amount,
            "turnover": turnover,
            "pct_chg": np.round(pct_chg, 4),
            "board": board,
        }
    )
    if missing_dates:
        frame = frame.loc[~frame["date"].isin(missing_dates)]
    return frame.reset_index(drop=True)


def build_benchmark(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    ret = rng.normal(0.00012, 0.0115, size=n)
    close = BENCHMARK_START * np.exp(np.cumsum(ret))
    ret_series = pd.Series(close).pct_change()
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": np.round(close, 4),
            "ret": ret_series.to_numpy(),
        }
    )


def quarter_ends(start: str, end: str) -> list[str]:
    dates: list[str] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            candidate = dt.date(year, month, day)
            if dt.date.fromisoformat(start) <= candidate <= dt.date.fromisoformat(end):
                dates.append(candidate.isoformat())
    return dates


def build_fundamentals(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    report_dates = quarter_ends("2022-01-01", "2024-12-31")
    base_notice_lag_days = {"600001.SH": 25, "600002.SH": 28, "000001.SZ": 30, "000002.SZ": 27, "300001.SZ": 26, "688001.SH": 24}
    for idx, stock in enumerate(STOCKS):
        code = stock[0]
        eps = 0.6 + idx * 0.25 + rng.normal(0, 0.05)
        bps = 6.0 + idx * 1.5
        revenue = 5.0e9 + idx * 1.2e9
        net_profit = 6.0e8 + idx * 1.4e8
        for period in report_dates:
            report_date = dt.date.fromisoformat(period)
            eps += rng.normal(0.03, 0.04)
            bps += rng.normal(0.05, 0.03)
            revenue *= 1.0 + rng.normal(0.012, 0.02)
            net_profit *= 1.0 + rng.normal(0.010, 0.025)
            roe = float(np.round(eps / max(bps, 0.1) * 4.0 * 100.0 + rng.normal(0, 0.4), 4))
            gross_margin = float(np.round(35.0 + idx * 3.0 + rng.normal(0, 1.0), 4))
            cfo_per_share = float(np.round(eps * (0.9 + rng.normal(0, 0.2)), 4))
            notice = report_date + dt.timedelta(days=base_notice_lag_days[code])
            # a few genuinely missing announcement dates -> 90-day fallback flag
            missing_notice = (code == "600002.SH" and period in ("2022-06-30", "2023-03-31")) or (
                code == "688001.SH" and period == "2024-09-30"
            )
            rows.append(
                {
                    "code": code,
                    "report_date": period,
                    "notice_date": None if missing_notice else notice.isoformat(),
                    "eps": float(np.round(eps, 4)),
                    "bps": float(np.round(bps, 4)),
                    "roe": roe,
                    "gross_margin": gross_margin,
                    "revenue": float(np.round(revenue, 2)),
                    "revenue_yoy": float(np.round(rng.normal(8.0, 3.0), 4)),
                    "net_profit": float(np.round(net_profit, 2)),
                    "net_profit_yoy": float(np.round(rng.normal(9.0, 4.0), 4)),
                    "cfo_per_share": cfo_per_share,
                    "notice_date_estimated": False,
                }
            )
    return pd.DataFrame(rows)


def build_universe() -> pd.DataFrame:
    free_caps = np.array([stock[4] for stock in STOCKS], dtype=float)
    weights = free_caps / free_caps.sum()
    rows = []
    for stock, weight in zip(STOCKS, weights):
        code, name, industry, board, free_cap, total_cap, listed_date, start_price = stock
        rows.append(
            {
                "code": code,
                "name": name,
                "industry": industry,
                "weight": float(round(weight, 10)),
                "free_cap": float(free_cap),
                "total_cap": float(total_cap),
                "listed_date": listed_date,
                "is_st": False,
                "board": board,
            }
        )
    return pd.DataFrame(rows)


def build_coverage_report(manifest: dict) -> dict:
    return {
        "as_of": manifest["as_of"],
        "requested_names": len(STOCKS),
        "covered_names": len(STOCKS),
        "failed_names": [],
        "coverage_ratio": 1.0,
        "synthetic": True,
        "notes": [
            "synthetic fixture produced by scripts/make_fixture.py; no network access",
            "stock 688001.SH is listed late (2022-06-01) and has 4 missing bars",
            "stock 600002.SH has two estimated notice dates (report_date + 90 days)",
        ],
    }


def build_fixture(
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    as_of: str = DEFAULT_AS_OF,
    start: str = "2022-01-04",
    created_at_utc: str = DEFAULT_CREATED_AT,
) -> dict:
    """Build the mini snapshot at ``out_dir`` and return its manifest."""

    rng = np.random.default_rng(seed)
    dates = _business_days(start, as_of)

    price_frames = []
    for index, stock in enumerate(STOCKS):
        stock_rng = np.random.default_rng(seed + 101 * (index + 1))
        missing: tuple[str, ...] = ()
        if stock[0] == "688001.SH":
            # suspended bars: pick four valid trading days after listing
            candidates = [date.strftime("%Y-%m-%d") for date in dates if date.strftime("%Y-%m-%d") >= "2022-06-01"]
            missing = tuple(candidates[5:9])
        frame = build_prices(stock, dates, stock_rng, missing_dates=missing)
        # no bars before the synthetic listing date (late-listed stock 688001.SH)
        listed_date = stock[6]
        frame = frame.loc[frame["date"] >= listed_date].reset_index(drop=True)
        price_frames.append(frame)
    prices = pd.concat(price_frames, ignore_index=True)

    benchmark = build_benchmark(dates, np.random.default_rng(seed + 7))
    fundamentals = fill_estimated_notice_dates(build_fundamentals(np.random.default_rng(seed + 3)))
    universe = build_universe()
    calendar = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})

    frames = {
        "prices": prices,
        "fundamentals": fundamentals,
        "universe": universe,
        "benchmark": benchmark,
        "calendar": calendar,
    }
    coverage = build_coverage_report({"as_of": as_of})
    manifest = write_snapshot(
        out_dir,
        frames,
        as_of=as_of,
        schema_version=SCHEMA_VERSION,
        source_urls={"fixture": "synthetic://scripts/make_fixture.py"},
        notes=[
            "deterministic synthetic mini snapshot for offline tests (no network)",
            f"fixed seed {seed}; 6 synthetic stocks; 36 months ending {as_of}",
            "synthetic trading calendar = weekdays (exchange holidays are not modelled)",
            "industry/name/cap values are synthetic and not real market data",
        ],
        created_at_utc=created_at_utc,
        provenance={
            "builder": "scripts/make_fixture.py",
            "seed": seed,
            "generator": "aicwf.data.snapshot.write_snapshot",
            "stocks": len(STOCKS),
        },
        extra_json_files={"coverage_report.json": coverage},
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output snapshot directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--start", default="2022-01-04")
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT, help="fixed manifest timestamp")
    args = parser.parse_args(argv)

    manifest = build_fixture(
        Path(args.out),
        seed=args.seed,
        as_of=args.as_of,
        start=args.start,
        created_at_utc=args.created_at,
    )
    summary = {
        "out": str(Path(args.out)),
        "as_of": manifest["as_of"],
        "row_counts": manifest["row_counts"],
        "date_ranges": manifest["date_ranges"],
        "snapshot_hash": manifest["snapshot_hash"],
        "files": list(manifest["files"]),
    }
    print(json.dumps(summary, indent=2))
    missing = [name for name in FRAME_ORDER if name not in manifest["row_counts"]]
    if missing:
        print(f"ERROR: missing frames {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
