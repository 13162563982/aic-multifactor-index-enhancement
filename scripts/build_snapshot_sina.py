"""Fallback frozen-snapshot builder using Sina daily bars and Eastmoney fundamentals.

The primary builder (``scripts/build_snapshot.py``) uses Eastmoney quote hosts, which
are sometimes rate-limited for hours. This script produces the same snapshot contract
using Sina's public daily-bar endpoint (through ``akshare``) for prices and
constituents, plus Eastmoney's data centre for financial statements and industries
(the data centre host is separate from the rate-limited quote hosts).

Usage:
    python scripts/build_snapshot_sina.py --as-of 2026-11-12 --verify
    python scripts/build_snapshot_sina.py --as-of 2026-11-12 --limit 10   # smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicwf.data.eastmoney import parse_fundamentals  # noqa: E402
from aicwf.data.snapshot import verify_snapshot, write_snapshot  # noqa: E402

DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
FUNDAMENTAL_REPORT = "RPT_F10_FINANCE_MAINFINADATA"
COMPONENT_REPORT = "RPT_INDEX_TS_COMPONENT"


def sina_symbol(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def full_code(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    return f"{code}.BJ"


def board_for(code: str) -> str:
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301")):
        return "CHINEXT"
    if code.startswith(("8", "4")):
        return "BJ"
    return "MAIN"


def request_json(url: str, *, attempts: int = 4, timeout: float = 20.0) -> dict:
    import urllib.request

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AICWF academic research)"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network layer
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_exc}")


def fetch_constituents() -> pd.DataFrame:
    import akshare as ak

    frame = ak.index_stock_cons(symbol="000300")
    frame = frame.rename(columns={"品种代码": "code", "品种名称": "name", "纳入日期": "inclusion_date"})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame[["code", "name", "inclusion_date"]].drop_duplicates(subset=["code"]).reset_index(drop=True)


def fetch_industry_map() -> dict[str, str]:
    """Best-effort code -> industry map from the Eastmoney data centre."""

    mapping: dict[str, str] = {}
    for page in range(1, 13):
        url = (
            f"{DATACENTER}?reportName={COMPONENT_REPORT}&columns=ALL"
            f"&pageNumber={page}&pageSize=500"
        )
        try:
            payload = request_json(url)
        except Exception:
            break
        rows = ((payload or {}).get("result") or {}).get("data") or []
        if not rows:
            break
        for row in rows:
            code = str(row.get("SECURITY_CODE") or "").zfill(6)
            industry = row.get("INDUSTRY")
            if code and industry:
                mapping.setdefault(code, str(industry))
        if len(rows) < 500:
            break
        time.sleep(0.2)
    return mapping


def fetch_prices(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_daily(
                symbol=sina_symbol(code),
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            if raw is None or raw.empty:
                raise RuntimeError("empty frame")
            frame = raw.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frame = frame[frame["date"] <= end].sort_values("date")
            frame["code"] = full_code(code)
            frame["board"] = board_for(code)
            frame["pct_chg"] = frame["close"].pct_change() * 100.0
            for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
                if column not in frame.columns:
                    frame[column] = 0.0
            keep = [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turnover",
                "pct_chg",
                "board",
            ]
            if "outstanding_share" in frame.columns:
                keep.append("outstanding_share")
            return frame[keep].reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - third-party endpoint
            last_exc = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"price fetch failed for {code}: {last_exc}")


def fetch_fundamentals(code: str) -> pd.DataFrame:
    secucode = full_code(code)
    url = (
        f"{DATACENTER}?reportName={FUNDAMENTAL_REPORT}&columns=ALL"
        f"&filter=(SECUCODE%3D%22{secucode}%22)&pageNumber=1&pageSize=60"
        f"&sortColumns=REPORT_DATE&sortTypes=-1"
    )
    payload = request_json(url)
    return parse_fundamentals(payload)


def fetch_benchmark(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_index_daily(symbol="sh000300")
    frame = raw[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].sort_values("date").reset_index(drop=True)
    frame["ret"] = frame["close"].pct_change()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen CSI 300 snapshot from Sina prices + Eastmoney fundamentals")
    parser.add_argument("--as-of", default="2026-11-12")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N names")
    parser.add_argument("--min-names", type=int, default=250)
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    as_of = args.as_of
    out_dir = Path(args.output) if args.output else ROOT / "data" / "snapshots" / as_of
    if out_dir.exists() and not args.overwrite:
        print(f"snapshot already exists: {out_dir} (use --overwrite)")
        return 2

    print(f"[sina-builder] constituents as_of {as_of}")
    cons = fetch_constituents()
    if args.limit:
        cons = cons.head(args.limit)
    print(f"[sina-builder] constituents: {len(cons)}")

    print("[sina-builder] industry map from Eastmoney data centre")
    industries = fetch_industry_map()
    print(f"[sina-builder] industries mapped: {len(industries)}")

    print("[sina-builder] benchmark")
    benchmark = fetch_benchmark(args.start, as_of)
    calendar = benchmark[["date"]].copy()
    print(f"[sina-builder] benchmark bars: {len(benchmark)}")

    price_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    last_close: dict[str, float] = {}
    first_date: dict[str, str] = {}
    for index, row in cons.iterrows():
        code = str(row["code"])
        try:
            frame = fetch_prices(code, args.start, as_of)
            price_frames.append(frame)
            last_close[code] = float(frame["close"].iloc[-1])
            first_date[code] = str(frame["date"].iloc[0])
        except Exception as exc:  # noqa: BLE001 - documented coverage failures
            failures.append({"code": code, "stage": "prices", "error": str(exc)[:200]})
        if (index + 1) % 25 == 0:
            print(f"[sina-builder] prices {index + 1}/{len(cons)} ok={len(price_frames)} failed={len(failures)}")
        time.sleep(args.delay)

    if len(price_frames) < args.min_names:
        print(f"[sina-builder] ABORT: only {len(price_frames)} names (< min {args.min_names})")
        return 3

    prices = pd.concat(price_frames, ignore_index=True)

    fundamentals_frames: list[pd.DataFrame] = []
    if not args.skip_fundamentals:
        for index, code in enumerate(sorted(last_close)):
            try:
                fundamentals_frames.append(fetch_fundamentals(code))
            except Exception as exc:  # noqa: BLE001
                failures.append({"code": code, "stage": "fundamentals", "error": str(exc)[:200]})
            if (index + 1) % 25 == 0:
                print(f"[sina-builder] fundamentals {index + 1}/{len(last_close)} ok={len(fundamentals_frames)}")
            time.sleep(min(args.delay, 0.15))
    fundamentals = (
        pd.concat(fundamentals_frames, ignore_index=True)
        if fundamentals_frames
        else pd.DataFrame(
            columns=[
                "code",
                "report_date",
                "notice_date",
                "eps",
                "bps",
                "roe",
                "gross_margin",
                "revenue",
                "revenue_yoy",
                "net_profit",
                "net_profit_yoy",
                "cfo_per_share",
                "notice_date_estimated",
            ]
        )
    )

    codes_with_fundamentals = set(fundamentals["code"].unique()) if len(fundamentals) else set()
    covered = [code for code in last_close if (args.skip_fundamentals or full_code(code) in codes_with_fundamentals)]
    if len(covered) < args.min_names:
        print(f"[sina-builder] ABORT: only {len(covered)} names with prices+fundamentals (< min {args.min_names})")
        return 3

    universe_rows = []
    for code in covered:
        name = str(cons.loc[cons["code"] == code, "name"].iloc[0])
        free_cap = float(last_close[code]) * 1.0  # replaced below by shares when available
        universe_rows.append(
            {
                "code": full_code(code),
                "name": name,
                "industry": industries.get(code, "Unknown"),
                "weight": 0.0,
                "free_cap": free_cap,
                "total_cap": free_cap,
                "listed_date": first_date.get(code, args.start),
                "is_st": "ST" in name.upper(),
                "board": board_for(code),
            }
        )
    universe = pd.DataFrame(universe_rows)

    # Free-float proxy: last close x outstanding shares from Sina's daily frame.
    shares: dict[str, float] = {}
    for frame in price_frames:
        code = str(frame["code"].iloc[0]).split(".")[0]
        if "outstanding_share" in frame.columns and len(frame):
            try:
                shares[code] = float(frame["outstanding_share"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
    universe["free_cap"] = [
        float(last_close.get(str(code).split(".")[0], 0.0)) * shares.get(str(code).split(".")[0], 1.0)
        for code in universe["code"]
    ]
    universe["total_cap"] = universe["free_cap"]
    total = float(universe["free_cap"].sum())
    universe["weight"] = universe["free_cap"] / total if total > 0 else 1.0 / len(universe)
    universe = universe.sort_values("code").reset_index(drop=True)

    coverage = {
        "as_of": as_of,
        "requested_names": int(len(cons)),
        "priced_names": int(len(price_frames)),
        "names_with_fundamentals": int(len(codes_with_fundamentals)),
        "covered_names": int(len(covered)),
        "coverage": round(len(covered) / max(len(cons), 1), 4),
        "failures": failures[:500],
        "provider": "sina_prices_akshare+eastmoney_datacenter_fundamentals",
    }

    frames = {
        "prices": prices,
        "fundamentals": fundamentals,
        "universe": universe,
        "benchmark": benchmark,
        "calendar": calendar,
    }
    manifest = write_snapshot(
        out_dir,
        frames,
        as_of=as_of,
        schema_version="1.1.0",
        source_urls={
            "constituents": "https://finance.sina.com.cn (akshare index_stock_cons 000300)",
            "prices": "https://finance.sina.com.cn (akshare stock_zh_a_daily qfq)",
            "fundamentals": f"{DATACENTER}?reportName={FUNDAMENTAL_REPORT}",
            "industry": f"{DATACENTER}?reportName={COMPONENT_REPORT}",
            "benchmark": "https://finance.sina.com.cn (akshare stock_zh_index_daily sh000300)",
        },
        notes=[
            "Fallback builder: prices and constituents from Sina via akshare; fundamentals and industry from Eastmoney data centre.",
            "Benchmark weights are a free-float (outstanding-share) proxy, not the official CSI 300 weights.",
            "Current-constituent universe: survivorship and reconstitution bias documented in data/README.md.",
        ],
        extra_json_files={"coverage_report.json": coverage},
    )
    print(f"[sina-builder] wrote {out_dir}")
    print(f"[sina-builder] snapshot_hash {manifest.get('snapshot_hash')}")
    print(f"[sina-builder] covered {len(covered)} names; coverage {coverage['coverage']:.2%}")

    if args.verify:
        verified = verify_snapshot(out_dir, deep=True)
        print(f"[sina-builder] verify: {json.dumps(verified.get('verification', verified), default=str)[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
