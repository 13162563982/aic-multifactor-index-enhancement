"""Build the frozen CSI 300 data snapshot from public Eastmoney endpoints.

Produces ``data/snapshots/<as_of>/`` exactly per ``docs/artifact_contract.md``
section 1::

    prices.parquet       qfq-adjusted daily bars, all constituents
    fundamentals.parquet PIT fundamentals history (notice dates filled)
    universe.parquet     current CSI 300 members + normalised free-float weight
    benchmark.parquet    CSI 300 index (000300.SH) close and daily return
    calendar.parquet     trading days covered by the snapshot
    coverage_report.json fetch coverage / failures (documented data caveat)
    manifest.json        provenance, row counts, date ranges, sha256, snapshot_hash

Usage::

    python scripts/build_snapshot.py --as-of 2026-11-12
    python scripts/build_snapshot.py --as-of 2026-11-12 --limit 10   # smoke test
    python scripts/build_snapshot.py --check data/snapshots/2026-11-12

Notes
-----
* The public hosts reset connections frequently; the client retries with
  backoff/jitter and rotates hosts (see ``aicwf/data/http.py``).
* If fewer than ``--min-names`` names can be fetched the build aborts without
  writing a snapshot; between ``--min-names`` and the full 300 it proceeds and
  records the missing names in ``coverage_report.json``.
* ``--limit`` / ``--names`` are debugging aids; limited snapshots are flagged in
  the manifest notes and must never be used for the real research run.
* All dates are stored as ISO ``YYYY-MM-DD`` strings; prices are qfq-adjusted
  (``fqt=1``); non-positive adjusted prices (possible for very high-dividend
  names early in the qfq history) are dropped and recorded in the coverage
  report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicwf.contracts import DataQualityError, RetryableError, parse_date  # noqa: E402
from aicwf.data.eastmoney import (  # noqa: E402
    EastmoneyClient,
    board_for_code,
    load_data_config,
    normalize_code,
    secid_for_code,
)
from aicwf.data.pit import fill_estimated_notice_dates  # noqa: E402
from aicwf.data.snapshot import (  # noqa: E402
    FRAME_SPECS,
    SCHEMA_VERSION,
    snapshot_dir_for,
    verify_snapshot,
    write_snapshot,
)
from aicwf.util.jsonio import read_parquet, write_json, write_parquet_atomic  # noqa: E402
from aicwf.util.timeutil import utc_now_iso  # noqa: E402

PRICE_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg", "board"]
FUNDAMENTAL_COLUMNS = list(FRAME_SPECS["fundamentals"].columns)


def log(message: str) -> None:
    print(f"[build_snapshot] {message}", flush=True)


# ---------------------------------------------------------------------------
# Resumable raw-response cache
#
# The public hosts temporarily rate-limit this IP after long request bursts
# (all endpoints close connections immediately). Successful per-name responses
# are therefore cached as parquet under data/cache/<as_of>/ so a interrupted or
# rate-limited build can be restarted without re-downloading finished names.
# The cache never enters the snapshot directory and never affects snapshot_hash.
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, kind: str, code: str | None = None) -> Path:
    if code is None:
        return cache_dir / f"{kind}.parquet"
    return cache_dir / kind / f"{code.replace('.', '_')}.parquet"


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = read_parquet(path)
    except Exception:  # noqa: BLE001 - corrupt cache entry: refetch
        return None
    if frame is None or len(frame) == 0:
        return None
    return frame


def _store_cached(path: Path, frame: pd.DataFrame | None) -> None:
    if frame is None or len(frame) == 0:
        return
    try:
        write_parquet_atomic(path, frame)
    except Exception as exc:  # noqa: BLE001 - cache is an optimisation only
        log(f"cache write failed for {path.name}: {type(exc).__name__}")


def _fetch_cached(
    fetch: Any,
    *,
    cache_dir: Path | None,
    kind: str,
    code: str | None,
    label: str,
    rounds: int,
    pause_seconds: float,
) -> pd.DataFrame:
    if cache_dir is not None:
        cached = _load_cached(_cache_path(cache_dir, kind, code))
        if cached is not None:
            return cached
    frame = _fetch_with_rounds(fetch, label=label, rounds=rounds, pause_seconds=pause_seconds)
    if cache_dir is not None:
        _store_cached(_cache_path(cache_dir, kind, code), frame)
    return frame


def _fetch_with_rounds(
    fetch: Any,
    *,
    label: str,
    rounds: int = 2,
    pause_seconds: float = 2.0,
    pause_growth: float = 2.0,
) -> Any:
    """Run ``fetch`` with outer retry rounds (each round already retries internally).

    The pause between rounds grows geometrically (``pause_seconds * growth**i``)
    because a rate-limit ban generally needs a quiet period, not more requests.
    """

    last_exc: Exception | None = None
    pause = float(pause_seconds)
    for round_index in range(max(1, int(rounds))):
        try:
            return fetch()
        except RetryableError as exc:
            last_exc = exc
        if round_index < rounds - 1:
            log(f"retrying {label} (round {round_index + 2}/{rounds} after {type(last_exc).__name__}); quiet pause {pause:.0f}s")
            time.sleep(pause)
            pause *= max(1.0, float(pause_growth))
    if last_exc is not None:
        raise last_exc
    raise RetryableError(f"{label}: fetch produced no result")


def _wait_for_connectivity(client: EastmoneyClient, *, max_wait_seconds: float) -> None:
    """Block until the list endpoint answers (rate-limit bans are temporary)."""

    if max_wait_seconds <= 0:
        return
    import urllib.parse  # noqa: F401 - params handled by the transport

    started = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            payload = client.http.get_json(
                client.list_urls,
                params={
                    "pn": 1,
                    "pz": 1,
                    "po": 1,
                    "np": 1,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f3",
                    "fs": client.csi300_board,
                    "fields": "f12,f14",
                },
                validator=lambda body: bool(((body or {}).get("data") or {}).get("diff")),
                describe="connectivity preflight",
                attempts=2,
                backoff_seconds=(5.0, 10.0),
            )
            log(f"connectivity OK after {attempt} preflight probe(s), {time.time() - started:.0f}s")
            return
        except RetryableError:
            waited = time.time() - started
            if waited >= max_wait_seconds:
                raise DataQualityError(
                    f"public endpoints still rate-limited after {waited:.0f}s of preflight; "
                    "wait a few minutes and rerun (the resumable cache keeps finished work)"
                )
            pause = min(60.0, 10.0 + 10.0 * attempt)
            log(f"connectivity preflight failed (attempt {attempt}); waiting {pause:.0f}s before retrying")
            time.sleep(pause)
    result = verify_snapshot(snapshot_dir, deep=True)
    print(json.dumps(result, indent=2))
    if result.get("ok"):
        manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
        log(f"VERIFY OK snapshot_hash={manifest.get('snapshot_hash')} as_of={manifest.get('as_of')}")
        return 0
    log("VERIFY FAILED: snapshot does not match its manifest")
    return 1


def _check_only(snapshot_dir: Path) -> int:
    return _verify_only(snapshot_dir)


def _prepare_universe(
    constituents: pd.DataFrame,
    *,
    as_of: str,
) -> pd.DataFrame:
    universe = constituents.copy()
    universe["board"] = universe["code"].map(board_for_code)
    universe["is_st"] = universe["name"].fillna("").str.contains("ST", case=False)
    universe["listed_date"] = universe["listed_date"].where(universe["listed_date"].notna(), None)
    caps = pd.to_numeric(universe["free_cap"], errors="coerce")
    caps = caps.where(caps > 0)
    total = float(caps.sum())
    universe["weight"] = (caps / total).fillna(0.0) if total > 0 else 0.0
    return universe


def build_snapshot(
    *,
    as_of: str,
    start: str = "2016-01-04",
    end: str | None = None,
    config: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    limit: int | None = None,
    names: list[str] | None = None,
    min_names: int | None = None,
    skip_fundamentals: bool = False,
    overwrite: bool = False,
    log_every: int = 20,
    client: EastmoneyClient | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    preflight_wait_seconds: float = 300.0,
) -> dict[str, Any]:
    """Fetch and freeze the snapshot; returns the manifest.

    Successful raw responses are cached under ``data/cache/<as_of>/`` (unless
    ``use_cache`` is false) so a rate-limited or interrupted build can be
    restarted without re-downloading completed names.
    """

    config = dict(config or load_data_config())
    snapshot_cfg = dict(config.get("snapshot") or {})
    as_of_date = parse_date(str(as_of))
    end = str(end or as_of_date.isoformat())
    if parse_date(end) > as_of_date:
        raise ValueError(f"end {end} is after as_of {as_of_date.isoformat()}")
    required = int(min_names if min_names is not None else snapshot_cfg.get("min_universe_size", 250))
    schema_version = str(snapshot_cfg.get("schema_version") or SCHEMA_VERSION)
    root = Path(output_root) if output_root else Path(snapshot_cfg.get("root") or "data/snapshots")
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    final_dir = snapshot_dir_for(as_of_date, root=root)
    build_dir = root / f".{as_of_date.isoformat()}.building"
    cache_root: Path | None = None
    if use_cache:
        cache_root = Path(cache_dir) if cache_dir else (PROJECT_ROOT / "data" / "cache" / as_of_date.isoformat())
        if not cache_root.is_absolute():
            cache_root = PROJECT_ROOT / cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        cached_names = len(list((cache_root / "prices").glob("*.parquet"))) if (cache_root / "prices").exists() else 0
        log(f"resumable cache: {cache_root} ({cached_names} cached price frames)")
    if final_dir.exists() and not overwrite:
        raise DataQualityError(f"snapshot already exists: {final_dir} (use --overwrite to rebuild)")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    client = client or EastmoneyClient.from_config(config)
    _wait_for_connectivity(client, max_wait_seconds=float(preflight_wait_seconds))
    log(f"fetching CSI 300 constituents (as_of={as_of_date.isoformat()}, hosts={list(client.hosts)})")
    constituents = _fetch_cached(
        lambda: client.fetch_constituents(),
        cache_dir=cache_root,
        kind="constituents",
        code=None,
        label="constituents",
        rounds=6,
        pause_seconds=30.0,
        pause_growth=2.0,
    )
    log(f"constituents: {len(constituents)} names")
    if names:
        wanted = {normalize_code(name) for name in names}
        constituents = constituents.loc[constituents["code"].isin(wanted)].reset_index(drop=True)
        log(f"restricted to --names: {len(constituents)} names")
    if limit is not None and limit > 0:
        constituents = constituents.head(int(limit)).reset_index(drop=True)
        log(f"restricted to --limit: {len(constituents)} names")
    limited = bool(names) or (limit is not None and limit > 0)

    universe_raw = _prepare_universe(constituents, as_of=as_of_date.isoformat())
    requested = len(universe_raw)
    if requested < 3:
        raise DataQualityError(f"too few constituents to build a snapshot: {requested}")
    log(f"fetching benchmark index {client.index_secid}")
    benchmark_raw = _fetch_cached(
        lambda: client.fetch_index_kline(start, end),
        cache_dir=cache_root,
        kind="benchmark",
        code=None,
        label="benchmark",
        rounds=6,
        pause_seconds=30.0,
        pause_growth=2.0,
    )
    if benchmark_raw.empty:
        raise DataQualityError("benchmark index returned no bars")
    log(f"benchmark bars: {len(benchmark_raw)}")

    price_frames: dict[str, pd.DataFrame] = {}
    fundamental_frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, Any]] = []
    cache_hits = 0
    started = dt.datetime.now()
    for position, (_, row) in enumerate(universe_raw.iterrows(), start=1):
        code = str(row["code"])
        if cache_root is not None and _cache_path(cache_root, "prices", code).exists():
            cache_hits += 1
        try:
            price_frames[code] = _fetch_cached(
                lambda code=code: client.fetch_kline(secid_for_code(code), start, end),
                cache_dir=cache_root,
                kind="prices",
                code=code,
                label=f"kline {code}",
                rounds=2,
                pause_seconds=3.0,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue with other names
            failures.append({"code": code, "kind": "prices", "error": f"{type(exc).__name__}: {exc}"[:400]})
        if not skip_fundamentals:
            try:
                fundamental_frames[code] = _fetch_cached(
                    lambda code=code: client.fetch_fundamentals(code),
                    cache_dir=cache_root,
                    kind="fundamentals",
                    code=code,
                    label=f"fundamentals {code}",
                    rounds=2,
                    pause_seconds=3.0,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"code": code, "kind": "fundamentals", "error": f"{type(exc).__name__}: {exc}"[:400]})
        if position % max(1, int(log_every)) == 0 or position == requested:
            elapsed = (dt.datetime.now() - started).total_seconds()
            rate = position / elapsed if elapsed > 0 else 0.0
            eta = (requested - position) / rate if rate > 0 else 0.0
            log(
                f"progress {position}/{requested} names | prices_ok={len(price_frames)} "
                f"fundamentals_ok={len(fundamental_frames)} failures={len(failures)} "
                f"cache_hits={cache_hits} elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )

    # ---- prices frame -------------------------------------------------
    log("assembling prices")
    price_rows: list[pd.DataFrame] = []
    dropped_nonpositive: dict[str, int] = {}
    for code, frame in price_frames.items():
        if frame is None or frame.empty:
            failures.append({"code": code, "kind": "prices", "error": "empty bar frame"})
            continue
        frame = frame.copy()
        frame["code"] = code
        frame["board"] = board_for_code(code)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.loc[frame["date"].notna()]
        frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)]
        nonpositive = int((pd.to_numeric(frame["close"], errors="coerce") <= 0).sum())
        if nonpositive:
            dropped_nonpositive[code] = nonpositive
            frame = frame.loc[pd.to_numeric(frame["close"], errors="coerce") > 0]
        if frame.empty:
            failures.append({"code": code, "kind": "prices", "error": "no rows in requested range after cleaning"})
            continue
        price_rows.append(frame.loc[:, PRICE_COLUMNS])
    if not price_rows:
        raise DataQualityError("no price data fetched for any constituent")
    prices = pd.concat(price_rows, ignore_index=True)
    prices = prices.drop_duplicates(subset=["date", "code"], keep="last")
    prices = prices.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    # ---- fundamentals frame ------------------------------------------
    log("assembling fundamentals")
    fundamentals = pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    if fundamental_frames:
        fundamentals = pd.concat(fundamental_frames.values(), ignore_index=True)
        fundamentals = fundamentals.drop_duplicates(subset=["code", "report_date"], keep="last")
        fundamentals = fundamentals.loc[fundamentals["report_date"].notna()]
        fundamentals = fundamentals.loc[fundamentals["report_date"] <= as_of_date.isoformat()]
        fundamentals = fill_estimated_notice_dates(fundamentals)
        fundamentals = fundamentals.sort_values(["code", "report_date"], kind="mergesort").reset_index(drop=True)

    # ---- benchmark + calendar ----------------------------------------
    benchmark = benchmark_raw.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    benchmark = benchmark.loc[benchmark["date"].notna()]
    benchmark = benchmark.loc[(benchmark["date"] >= start) & (benchmark["date"] <= end)]
    benchmark = benchmark.sort_values("date", kind="mergesort").drop_duplicates(subset=["date"], keep="last")
    benchmark["ret"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change()
    benchmark = benchmark.loc[:, ["date", "close", "ret"]].reset_index(drop=True)
    calendar = pd.DataFrame({"date": benchmark["date"].tolist()})

    # ---- decide covered universe -------------------------------------
    price_codes = set(prices["code"].unique())
    fundamentals_codes = set(fundamentals["code"].unique()) if not fundamentals.empty else set()
    if skip_fundamentals:
        covered_codes = sorted(price_codes)
    else:
        covered_codes = sorted(price_codes & fundamentals_codes)
    coverage_ratio = len(covered_codes) / requested if requested else 0.0
    log(
        f"coverage: requested={requested} prices={len(price_codes)} "
        f"fundamentals={len(fundamentals_codes)} covered={len(covered_codes)} "
        f"ratio={coverage_ratio:.3f}"
    )
    if len(covered_codes) < required and not limited:
        raise DataQualityError(
            f"only {len(covered_codes)} of {requested} names covered (minimum {required}); "
            f"refusing to write snapshot. failures: {[f['code'] + ':' + f['kind'] for f in failures][:20]}"
        )
    prices = prices.loc[prices["code"].isin(covered_codes)].reset_index(drop=True)
    if not fundamentals.empty:
        fundamentals = fundamentals.loc[fundamentals["code"].isin(covered_codes)].reset_index(drop=True)

    universe = universe_raw.loc[universe_raw["code"].isin(covered_codes)].copy()
    # listed_date fallback: first bar date (lower bound) for names without f26
    first_dates = prices.groupby("code")["date"].min()
    universe["listed_date"] = universe.apply(
        lambda row: row["listed_date"]
        if isinstance(row["listed_date"], str) and row["listed_date"]
        else first_dates.get(row["code"]),
        axis=1,
    )
    caps = pd.to_numeric(universe["free_cap"], errors="coerce")
    caps = caps.where(caps > 0)
    total_cap = float(caps.sum())
    universe["weight"] = (caps / total_cap).fillna(0.0) if total_cap > 0 else 0.0
    universe = universe.sort_values("code", kind="mergesort").reset_index(drop=True)

    # ---- coverage report ---------------------------------------------
    failed_codes = sorted({failure["code"] for failure in failures} - set(covered_codes))
    coverage = {
        "as_of": as_of_date.isoformat(),
        "requested_names": requested,
        "covered_names": len(covered_codes),
        "coverage_ratio": round(coverage_ratio, 6),
        "prices_ok": len(price_codes),
        "fundamentals_ok": len(fundamentals_codes),
        "failed_names": [
            {"code": code, "errors": [f["error"] for f in failures if f["code"] == code][:5]}
            for code in failed_codes
        ],
        "dropped_nonpositive_qfq_bars": dropped_nonpositive,
        "price_date_range": [
            str(prices["date"].min()) if not prices.empty else None,
            str(prices["date"].max()) if not prices.empty else None,
        ],
        "fundamental_date_range": [
            str(fundamentals["report_date"].min()) if not fundamentals.empty else None,
            str(fundamentals["report_date"].max()) if not fundamentals.empty else None,
        ],
        "notes": [
            "constituents are the CURRENT CSI 300 members (survivorship bias; documented in data/README.md)",
            "qfq-adjusted prices from Eastmoney fqt=1; non-positive adjusted bars dropped",
            "estimated notice dates = report_date + 90 days, flagged in fundamentals.notice_date_estimated",
        ],
    }
    write_json(build_dir / "coverage_report.json", coverage, indent=2, sort_keys=True)

    notes = [
        "Frozen Eastmoney public-data snapshot for academic research use.",
        f"prices: qfq-adjusted (fqt=1) daily bars from {start} to {end}.",
        "universe: current CSI 300 members; weight = normalised free-float market cap proxy.",
        "fundamentals: report_date/notice_date PIT history; missing notice dates estimated at report_date+90d and flagged.",
        "benchmark: CSI 300 index 000300.SH close and daily return.",
        "Known limitation: current-constituent survivorship bias (see data/README.md).",
    ]
    if limited:
        notes.append(f"DEBUG build: restricted to {requested} names (--limit/--names); not for research runs.")
    if failures:
        notes.append(f"{len(failures)} fetch failure(s) recorded in coverage_report.json; covered names only.")

    log("writing snapshot files + manifest")
    manifest = write_snapshot(
        build_dir,
        {
            "prices": prices,
            "fundamentals": fundamentals,
            "universe": universe,
            "benchmark": benchmark,
            "calendar": calendar,
        },
        as_of=as_of_date,
        schema_version=schema_version,
        source_urls={
            "constituents": "https://{host}/api/qt/clist/get?fs=b:BK0500&pz=100&fields=f12,f13,f14,f2,f3,f8,f9,f20,f21,f23,f26,f100",
            "kline": "https://{host}/api/qt/stock/kline/get?secid={secid}&klt=101&fqt=1&fields2=f51..f61&ut=<public token>",
            "fundamentals": "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL&filter=(SECUCODE=\"{code}\")",
        },
        notes=notes,
        created_at_utc=utc_now_iso(),
        provenance={
            "builder": "scripts/build_snapshot.py",
            "source": "eastmoney-public",
            "fetched_at_utc": utc_now_iso(),
            "start_date": start,
            "end_date": end,
            "price_adjustment": "qfq",
            "klt": 101,
            "requested_names": requested,
            "covered_names": len(covered_codes),
            "failures": len(failures),
            "limited": limited,
        },
        extra_json_files={"coverage_report.json": coverage},
    )

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(build_dir, final_dir)
    log(f"snapshot written to {final_dir}")
    log(
        f"row counts: prices={manifest['row_counts'].get('prices')} "
        f"fundamentals={manifest['row_counts'].get('fundamentals')} "
        f"universe={manifest['row_counts'].get('universe')} "
        f"benchmark={manifest['row_counts'].get('benchmark')} "
        f"calendar={manifest['row_counts'].get('calendar')}"
    )
    log(f"snapshot_hash={manifest.get('snapshot_hash')}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default="2026-11-12", help="decision / freeze date (ISO)")
    parser.add_argument("--start", default="2016-01-04", help="first price date (ISO)")
    parser.add_argument("--end", default=None, help="last price date (default: as_of)")
    parser.add_argument("--output", default=None, help="snapshot root directory (default from config)")
    parser.add_argument("--config", default=None, help="path to config/data.yaml")
    parser.add_argument("--limit", type=int, default=None, help="debug: build only the first N names")
    parser.add_argument("--names", default=None, help="debug: comma-separated codes to build")
    parser.add_argument("--min-names", type=int, default=None, help="minimum covered names (default 250)")
    parser.add_argument("--skip-fundamentals", action="store_true", help="debug: skip fundamentals fetch")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing snapshot directory")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cache-dir", default=None, help="resumable raw-response cache dir (default data/cache/<as_of>)")
    parser.add_argument("--no-cache", action="store_true", help="disable the resumable raw-response cache")
    parser.add_argument(
        "--preflight-wait-seconds",
        type=float,
        default=300.0,
        help="wait up to this long for the rate-limited public endpoints to recover before fetching (default 300)",
    )
    parser.add_argument("--check", default=None, metavar="SNAPSHOT_DIR", help="verify an existing snapshot and exit")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the snapshot after building (or verify the existing one without rebuilding)",
    )
    args = parser.parse_args(argv)

    try:
        if args.check:
            return _verify_only(Path(args.check))
        config = load_data_config(args.config)
        snapshot_root = args.output or ((config.get("snapshot") or {}).get("root") or "data/snapshots")
        final_dir = snapshot_dir_for(args.as_of, root=snapshot_root)
        if args.verify and not args.overwrite and final_dir.exists():
            log(f"snapshot already exists, verifying {final_dir}")
            return _verify_only(final_dir)
        names = [name for name in (args.names or "").split(",") if name.strip()] or None
        manifest = build_snapshot(
            as_of=args.as_of,
            start=args.start,
            end=args.end,
            config=config,
            output_root=args.output,
            limit=args.limit,
            names=names,
            min_names=args.min_names,
            skip_fundamentals=args.skip_fundamentals,
            overwrite=args.overwrite,
            log_every=args.log_every,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            preflight_wait_seconds=args.preflight_wait_seconds,
        )
    except DataQualityError as exc:
        log(f"DATA QUALITY ERROR: {exc}")
        return 2
    except RetryableError as exc:
        log(f"NETWORK ERROR: {exc}")
        return 3
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        log("interrupted")
        return 130
    except Exception:  # noqa: BLE001 - CLI top-level diagnostic
        traceback.print_exc()
        return 1
    print(json.dumps({
        "snapshot_dir": str(snapshot_dir_for(manifest["as_of"], root=snapshot_root)),
        "as_of": manifest["as_of"],
        "snapshot_hash": manifest["snapshot_hash"],
        "row_counts": manifest["row_counts"],
        "date_ranges": manifest["date_ranges"],
    }, indent=2))
    if args.verify:
        return _verify_only(snapshot_dir_for(manifest["as_of"], root=snapshot_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
