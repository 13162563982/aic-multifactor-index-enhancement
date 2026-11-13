"""Documented recovery: add missing CSI 300 constituents to a frozen snapshot.

Background
----------
The Sina fallback builder (``scripts/build_snapshot_sina.py``) takes constituents from
``akshare.index_stock_cons('000300')``. That Sina-derived table returns 300 rows but only
**289 unique codes** (11 duplicate/stale rows), so the frozen snapshot would contain a
289-name universe (96.3% of the index). The official CSIndex table
``akshare.index_stock_cons_csindex('000300')`` has exactly 300 unique codes.

This script performs a single, documented augmentation of an existing frozen snapshot:

1. loads the snapshot,
2. diffs its universe against the official CSIndex constituent list,
3. fetches qfq daily prices from Sina (akshare ``stock_zh_a_daily``),
   fundamentals + industry from the working Eastmoney data centre for the missing codes,
4. appends the rows, normalises ``board`` values to the documented lowercase set
   ``{main, chinext, star, bse}``, refreshes free-float cap/weights and fills estimated
   notice dates,
5. rewrites the snapshot atomically through :func:`aicwf.data.snapshot.write_snapshot`,
   producing a new manifest/sha256/``snapshot_hash`` and an updated
   ``coverage_report.json`` that records what was augmented and why.

It is idempotent: when the snapshot already contains every official constituent it is a
no-op. It never runs concurrently with the snapshot builder.

Usage::

    python scripts/augment_snapshot.py --as-of 2026-11-12
    python scripts/augment_snapshot.py --as-of 2026-11-12 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicwf.data.eastmoney import parse_fundamentals  # noqa: E402
from aicwf.data.pit import fill_estimated_notice_dates  # noqa: E402
from aicwf.data.snapshot import load_snapshot, verify_snapshot, write_snapshot  # noqa: E402
from aicwf.util.jsonio import read_json, write_json  # noqa: E402
from aicwf.util.timeutil import utc_now_iso  # noqa: E402

DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
FUNDAMENTAL_REPORT = "RPT_F10_FINANCE_MAINFINADATA"
COMPONENT_REPORT = "RPT_INDEX_TS_COMPONENT"
BOARD_BY_CODE = {"MAIN": "main", "CHINEXT": "chinext", "STAR": "star", "BJ": "bse"}


def log(message: str) -> None:
    print(f"[augment] {message}", flush=True)


def full_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    return f"{code}.BJ"


def sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def board_for(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("8", "4")):
        return "bse"
    return "main"


def request_json(url: str, *, attempts: int = 5, timeout: float = 20.0) -> dict:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AICWF academic research)"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network layer
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_exc}")


def official_constituents() -> pd.DataFrame:
    import akshare as ak

    frame = ak.index_stock_cons_csindex(symbol="000300")
    frame = frame.rename(columns={"成分券代码": "code", "成分券名称": "name", "日期": "date"})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return frame[["code", "name"]]


def fetch_industry_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for page in range(1, 15):
        url = f"{DATACENTER}?reportName={COMPONENT_REPORT}&columns=ALL&pageNumber={page}&pageSize=500"
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


def fetch_prices(code: str, start: str, end: str, *, delay: float = 0.2) -> tuple[pd.DataFrame, float]:
    import akshare as ak

    last_exc: Exception | None = None
    for attempt in range(4):
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
            frame = frame.loc[frame["date"] <= end].sort_values("date").reset_index(drop=True)
            if frame.empty:
                raise RuntimeError("no rows in range")
            shares = float(frame["outstanding_share"].iloc[-1]) if "outstanding_share" in frame.columns else 0.0
            frame["code"] = full_code(code)
            frame["board"] = board_for(code)
            frame["pct_chg"] = pd.to_numeric(frame["close"], errors="coerce").pct_change() * 100.0
            for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
                if column not in frame.columns:
                    frame[column] = 0.0
            keep = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg", "board"]
            return frame[keep].reset_index(drop=True), shares
        except Exception as exc:  # noqa: BLE001 - third-party endpoint
            last_exc = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"price fetch failed for {code}: {last_exc}")


def fetch_fundamentals(code: str) -> pd.DataFrame:
    secucode = full_code(code)
    url = (
        f"{DATACENTER}?reportName={FUNDAMENTAL_REPORT}&columns=ALL"
        f"&filter={urllib.parse.quote(f'(SECUCODE=\"{secucode}\")')}"
        f"&pageNumber=1&pageSize=60&sortColumns=REPORT_DATE&sortTypes=-1"
    )
    payload = request_json(url)
    return parse_fundamentals(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default="2026-11-12")
    parser.add_argument("--snapshot", default=None, help="snapshot directory (default data/snapshots/<as-of>)")
    parser.add_argument("--delay", type=float, default=0.2, help="pause between Sina calls")
    parser.add_argument("--dry-run", action="store_true", help="only print the augmentation plan")
    args = parser.parse_args(argv)

    snapshot_dir = Path(args.snapshot) if args.snapshot else ROOT / "data" / "snapshots" / args.as_of
    if not (snapshot_dir / "manifest.json").exists():
        log(f"snapshot not found: {snapshot_dir}")
        return 2
    snapshot = load_snapshot(snapshot_dir)
    original_manifest = snapshot.manifest
    as_of = str(original_manifest["as_of"])
    if args.dry_run:
        log(f"as_of={as_of} existing universe={len(snapshot.universe)}")

    official = official_constituents()
    official_codes = [str(code).zfill(6) for code in official["code"].tolist()]
    existing_full = set(snapshot.universe["code"].astype(str))
    missing = [code for code in official_codes if full_code(code) not in existing_full]
    log(f"official constituents={len(official_codes)} existing={len(existing_full)} missing={len(missing)}")

    existing_board_values = set(snapshot.prices["board"].astype(str)) | set(snapshot.universe["board"].astype(str))
    needs_board_normalisation = any(value != value.lower() for value in existing_board_values)
    log(
        f"board values={sorted(existing_board_values)} "
        f"needs_normalisation={needs_board_normalisation}"
    )
    if not missing and not needs_board_normalisation:
        log("nothing to augment; snapshot already covers all official constituents with normalised boards")
        return 0
    if args.dry_run:
        return 0

    industries: dict[str, str] = {}
    if missing:
        log("fetching industry map from Eastmoney data centre")
        industries = fetch_industry_map()
        log(f"industry map entries={len(industries)}")

    new_prices: list[pd.DataFrame] = []
    new_fundamentals: list[pd.DataFrame] = []
    new_universe_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for code in missing:
        name = str(official.loc[official["code"] == code, "name"].iloc[0])
        try:
            price_frame, shares = fetch_prices(code, "2016-01-04", as_of, delay=args.delay)
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "stage": "prices", "error": str(exc)[:200]})
            continue
        try:
            fundamental_frame = fetch_fundamentals(code)
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "stage": "fundamentals", "error": str(exc)[:200]})
            fundamental_frame = pd.DataFrame(columns=snapshot.fundamentals.columns)
        last_close = float(pd.to_numeric(price_frame["close"], errors="coerce").dropna().iloc[-1])
        free_cap = last_close * shares if shares > 0 else 0.0
        new_prices.append(price_frame)
        if len(fundamental_frame):
            new_fundamentals.append(fundamental_frame)
        new_universe_rows.append(
            {
                "code": full_code(code),
                "name": name,
                "industry": industries.get(code, "Unknown"),
                "weight": 0.0,
                "free_cap": free_cap,
                "total_cap": free_cap,
                "listed_date": str(price_frame["date"].iloc[0]),
                "is_st": "ST" in name.upper(),
                "board": board_for(code),
            }
        )
        log(f"fetched {code} {name}: prices={len(price_frame)} fundamentals={len(fundamental_frame)} free_cap={free_cap:.3e}")
        time.sleep(args.delay)

    if failures:
        log(f"ABORT: {len(failures)} augmentation fetch failure(s): {failures[:10]}")
        return 3

    # ---- merge frames -------------------------------------------------
    prices = pd.concat([snapshot.prices, *new_prices], ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices = prices.dropna(subset=["date"]).drop_duplicates(subset=["date", "code"], keep="last")
    prices["board"] = prices["board"].map(lambda value: BOARD_BY_CODE.get(str(value).upper(), str(value).lower()))
    prices = prices.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    fundamentals = pd.concat([snapshot.fundamentals, *new_fundamentals], ignore_index=True) if new_fundamentals else snapshot.fundamentals.copy()
    fundamentals = fundamentals.drop_duplicates(subset=["code", "report_date"], keep="last")
    fundamentals = fundamentals.loc[fundamentals["report_date"].notna()]
    fundamentals = fundamentals.loc[fundamentals["report_date"] <= as_of]
    fundamentals = fill_estimated_notice_dates(fundamentals)
    fundamentals = fundamentals.sort_values(["code", "report_date"], kind="mergesort").reset_index(drop=True)

    universe = snapshot.universe.copy()
    universe["board"] = universe["board"].map(lambda value: BOARD_BY_CODE.get(str(value).upper(), str(value).lower()))
    if new_universe_rows:
        new_universe = pd.DataFrame(new_universe_rows)
        universe = pd.concat([universe, new_universe], ignore_index=True)
    universe = universe.drop_duplicates(subset=["code"], keep="last")
    caps = pd.to_numeric(universe["free_cap"], errors="coerce").clip(lower=0)
    total_cap = float(caps.sum())
    universe["weight"] = caps / total_cap if total_cap > 0 else 1.0 / len(universe)
    universe = universe.sort_values("code", kind="mergesort").reset_index(drop=True)

    benchmark = snapshot.benchmark.copy()
    calendar = snapshot.calendar.copy()

    # ---- coverage report + manifest provenance ------------------------
    coverage_path = snapshot_dir / "coverage_report.json"
    coverage = read_json(coverage_path) if coverage_path.exists() else {}
    coverage.update(
        {
            "as_of": as_of,
            "requested_names": len(official_codes),
            "covered_names": int(len(universe)),
            "coverage": round(float(len(universe)) / max(len(official_codes), 1), 6),
            "priced_names": int(prices["code"].nunique()),
            "fundamentals_ok": int(fundamentals["code"].nunique()),
            "augmented_codes": missing,
            "augmented_count": len(missing),
            "board_values_normalised": needs_board_normalisation,
            "board_values": "lowercase main/chinext/star/bse",
            "volume_units": "shares (Sina stock_zh_a_daily)",
            "turnover_units": "ratio volume/outstanding_share (multiply by 100 for percent)",
        }
    )
    if missing:
        coverage["augmentation"] = (
            "fallback builder's Sina constituent table returned 300 rows but only 289 unique codes; "
            "the missing official CSI 300 names were fetched from the CSIndex official constituent list "
            "plus Sina qfq prices and Eastmoney data-centre fundamentals/industry"
        )
    write_json(coverage_path, coverage, indent=2, sort_keys=True)

    provenance = dict(original_manifest.get("provenance") or {})
    provenance.update(
        {
            "augmented_at_utc": utc_now_iso(),
            "augmented_codes": missing,
            "augmentation_source": "CSIndex official constituent list + Sina qfq prices + Eastmoney datacenter fundamentals/industry",
            "board_normalisation": "lowercase main/chinext/star/bse",
        }
    )
    source_urls = dict(original_manifest.get("source_urls") or {})
    source_urls["constituents_official"] = "https://www.csindex.com.cn (akshare index_stock_cons_csindex 000300)"
    notes = list(original_manifest.get("notes") or [])
    if missing:
        notes.append(
            f"Augmented with {len(missing)} official CSI 300 constituents missing from the fallback Sina constituent list; "
            "instrument/sector data from the working Eastmoney data centre."
        )
    if needs_board_normalisation:
        notes.append("Board values normalised to the documented lowercase set {main, chinext, star, bse}.")
    manifest = write_snapshot(
        snapshot_dir,
        {
            "prices": prices,
            "fundamentals": fundamentals,
            "universe": universe,
            "benchmark": benchmark,
            "calendar": calendar,
        },
        as_of=as_of,
        schema_version=str(original_manifest.get("schema_version") or "1.1.0"),
        source_urls=source_urls,
        notes=notes,
        created_at_utc=str(original_manifest.get("created_at_utc") or utc_now_iso()),
        provenance=provenance,
        extra_json_files={"coverage_report.json": coverage},
    )
    log(f"wrote augmented snapshot: universe={len(universe)} prices={len(prices)} fundamentals={len(fundamentals)}")
    log(f"snapshot_hash={manifest['snapshot_hash']}")
    result = verify_snapshot(snapshot_dir, deep=True)
    log(f"deep verify ok={result.get('ok')} files={len(result.get('checked_files', []))}")
    print(json.dumps({
        "snapshot_dir": str(snapshot_dir),
        "as_of": as_of,
        "augmented_codes": missing,
        "row_counts": manifest["row_counts"],
        "date_ranges": manifest["date_ranges"],
        "universe_size": len(universe),
        "snapshot_hash": manifest["snapshot_hash"],
        "verify_ok": result.get("ok"),
    }, indent=2))
    return 0 if result.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
