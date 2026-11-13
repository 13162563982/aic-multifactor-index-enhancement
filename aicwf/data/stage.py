"""Data-prep stage: frozen snapshot -> panel / benchmark / quality / PIT audit.

Entry point: :func:`run_data_prep(ctx) -> StageOutput`.

Reads the frozen snapshot from ``ctx.snapshot_dir`` or the configured path
(``config/strategy.yaml`` ``data.snapshot``), enforces the run's ``as_of``
cutoff and point-in-time fundamentals availability, and writes the contract
artifacts into ``ctx.stage_dir("data_prep")`` (``02_data/``):

* ``panel.parquet``      ``date, code, close, ret, volume, amount, turnover,
  suspended, in_universe, industry, log_size, listed_days``
* ``benchmark.parquet``  ``date, close, ret``
* ``data_quality.json``  ``snapshot_hash, rows, universe_size, coverage,
  checks, warnings``
* ``pit_audit.json``     ``as_of, violations, max_panel_date,
  fundamentals_available_as_of, checks``

Design decisions (documented in ``data/README.md``):

* Panel dates are ISO ``YYYY-MM-DD`` strings; returns ``ret`` are decimal
  fractions. Suspended days (no bar) keep the previous close forward-filled,
  carry ``suspended=True``, zero volume/amount/turnover and ``ret=0`` (they are
  not tradable).
* ``log_size`` is ``ln(free_float_cap)`` from the snapshot universe. Market-cap
  history is not available from the public endpoint, so the freeze-date cap is
  used for every date; this is a *static* attribute known at ``as_of`` (not
  future information) but it is not point-in-time cap history.
* ``in_universe`` marks current members from their listing date onward (the
  snapshot universe is the current index membership: survivorship bias).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from aicwf.contracts import (
    PROJECT_ROOT,
    DataQualityError,
    RunContext,
    StageOutput,
    parse_date,
    relpath,
)
from aicwf.data.pit import available_as_of, date_violations, max_available_report_date
from aicwf.data.snapshot import Snapshot, load_snapshot, snapshot_dir_for
from aicwf.data.validate import (
    MIN_UNIVERSE_SIZE_DEFAULT,
    compute_quality_checks,
    load_quality_config,
    quality_failures,
)
from aicwf.util.jsonio import write_json, write_parquet_atomic

PANEL_COLUMNS: tuple[str, ...] = (
    "date",
    "code",
    "close",
    "ret",
    "volume",
    "amount",
    "turnover",
    "suspended",
    "in_universe",
    "industry",
    "log_size",
    "listed_days",
)
BENCHMARK_COLUMNS: tuple[str, ...] = ("date", "close", "ret")


# ---------------------------------------------------------------------------
# Snapshot resolution
# ---------------------------------------------------------------------------


def _config_blocks(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blocks: list[Mapping[str, Any]] = []
    if not isinstance(config, Mapping):
        return blocks
    blocks.append(config)
    strategy = config.get("strategy")
    if isinstance(strategy, Mapping):
        blocks.append(strategy)
    data_cfg = config.get("data")
    if isinstance(data_cfg, Mapping):
        blocks.append(data_cfg)
    return blocks


def resolve_snapshot_dir(ctx: RunContext) -> Path:
    """Locate the frozen snapshot directory for a run.

    Order: ``ctx.snapshot_dir`` -> config ``strategy.data.snapshot`` (string) ->
    a ``snapshot``/``snapshot_dir`` path found anywhere in the config bundle ->
    ``data/snapshots/<as_of>``.
    """

    if ctx.snapshot_dir is not None:
        return Path(ctx.snapshot_dir)
    candidates: list[str] = []
    for block in _config_blocks(ctx.config or {}):
        direct = block.get("snapshot_dir")
        if isinstance(direct, str) and direct:
            candidates.append(direct)
        data_section = block.get("data")
        if isinstance(data_section, Mapping):
            snapshot = data_section.get("snapshot")
            if isinstance(snapshot, str) and snapshot:
                candidates.append(snapshot)
        snapshot = block.get("snapshot")
        if isinstance(snapshot, str) and snapshot:
            candidates.append(snapshot)
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
    if candidates:
        path = Path(candidates[0])
        return path if path.is_absolute() else PROJECT_ROOT / path
    return snapshot_dir_for(ctx.as_of)


def _min_universe_size(config: Mapping[str, Any], quality: Mapping[str, Any] | None = None) -> int:
    for block in _config_blocks(config):
        snapshot = block.get("snapshot")
        if isinstance(snapshot, Mapping) and snapshot.get("min_universe_size") is not None:
            return int(snapshot["min_universe_size"])
        data_section = block.get("data")
        if isinstance(data_section, Mapping):
            snapshot = data_section.get("snapshot")
            if isinstance(snapshot, Mapping) and snapshot.get("min_universe_size") is not None:
                return int(snapshot["min_universe_size"])
    return MIN_UNIVERSE_SIZE_DEFAULT


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def build_panel(snapshot: Snapshot, as_of: str | dt.date | dt.datetime) -> pd.DataFrame:
    """Build the point-in-time daily panel from a frozen snapshot."""

    cutoff = parse_date(str(as_of))
    prices = snapshot.prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices = prices.loc[prices["date"].notna()]
    prices = prices.loc[prices["date"] <= cutoff.isoformat()]
    if prices.empty:
        raise DataQualityError(f"snapshot has no price rows on or before as_of={cutoff.isoformat()}")

    calendar = snapshot.calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    calendar = calendar.loc[calendar["date"].notna()]
    calendar = calendar.loc[calendar["date"] <= cutoff.isoformat()]
    if calendar.empty:
        raise DataQualityError(f"snapshot has no calendar days on or before as_of={cutoff.isoformat()}")

    universe = snapshot.universe.copy()
    universe["code"] = universe["code"].astype(str)
    universe["listed_date"] = pd.to_datetime(universe["listed_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    free_cap = pd.to_numeric(universe["free_cap"], errors="coerce")
    free_cap = free_cap.where(free_cap > 0)
    universe["log_size"] = free_cap.map(lambda cap: float(math.log(cap)) if pd.notna(cap) else None)
    universe = universe.set_index("code", drop=False)

    codes = [code for code in universe.index.unique() if (prices["code"] == code).any()]
    if not codes:
        raise DataQualityError("no snapshot universe code has price data")
    first_bar = prices.groupby("code")["date"].min()
    trading_days = calendar["date"].tolist()

    rows: list[pd.DataFrame] = []
    for code in codes:
        meta = universe.loc[code]
        start = first_bar.get(code)
        if start is None:
            continue
        listed = _as_str(meta.get("listed_date"))
        start = max(start, listed) if listed else start
        code_days = [day for day in trading_days if day >= start]
        if not code_days:
            continue
        base = pd.DataFrame({"date": code_days})
        base["code"] = code
        base["industry"] = _as_str(meta.get("industry"))
        base["log_size"] = meta.get("log_size")
        if listed:
            listed_date = parse_date(listed)
            base["listed_days"] = base["date"].map(lambda day: (parse_date(day) - listed_date).days)
        else:
            base["listed_days"] = base["date"].map(lambda day: (parse_date(day) - parse_date(start)).days)
        base["in_universe"] = True
        rows.append(base)
    grid = pd.concat(rows, ignore_index=True)

    price_slice = prices.loc[:, ["date", "code", "close", "volume", "amount", "turnover"]].copy()
    for column in ("close", "volume", "amount", "turnover"):
        price_slice[column] = pd.to_numeric(price_slice[column], errors="coerce")
    panel = grid.merge(price_slice, on=["date", "code"], how="left", indicator=True)
    panel["suspended"] = (panel["_merge"] == "left_only") | (panel["volume"].fillna(0) <= 0)
    panel = panel.drop(columns=["_merge"])
    panel = panel.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    panel["close"] = panel.groupby("code", sort=False)["close"].transform(lambda values: values.ffill())
    panel["ret"] = panel.groupby("code", sort=False)["close"].transform(lambda values: values.pct_change())
    # A suspended day carries no tradeable volume and zero return.
    panel.loc[panel["suspended"], "volume"] = 0.0
    panel.loc[panel["suspended"], "amount"] = 0.0
    panel.loc[panel["suspended"], "turnover"] = 0.0
    panel.loc[panel["suspended"], "ret"] = 0.0
    panel = panel.dropna(subset=["close"]).reset_index(drop=True)
    panel = panel.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    panel["suspended"] = panel["suspended"].astype(bool)
    panel["in_universe"] = panel["in_universe"].astype(bool)
    panel["industry"] = panel["industry"].map(_as_str)
    panel = panel.loc[:, list(PANEL_COLUMNS)].reset_index(drop=True)
    return panel


def build_benchmark(snapshot: Snapshot, as_of: str | dt.date | dt.datetime) -> pd.DataFrame:
    cutoff = parse_date(str(as_of))
    benchmark = snapshot.benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    benchmark = benchmark.loc[benchmark["date"].notna()]
    benchmark = benchmark.loc[benchmark["date"] <= cutoff.isoformat()]
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark["ret"] = pd.to_numeric(benchmark["ret"], errors="coerce")
    benchmark = benchmark.loc[:, list(BENCHMARK_COLUMNS)].sort_values("date", kind="mergesort").reset_index(drop=True)
    return benchmark


# ---------------------------------------------------------------------------
# Auditing helpers
# ---------------------------------------------------------------------------


def build_pit_audit(
    snapshot: Snapshot,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    as_of: str | dt.date | dt.datetime,
) -> dict[str, Any]:
    """Compute the PIT audit document (all checks must pass with no violations)."""

    cutoff = parse_date(str(as_of))
    available = available_as_of(snapshot.fundamentals, cutoff)
    violations: list[dict[str, Any]] = []
    violations.extend(date_violations(panel, cutoff, name="panel", limit=20))
    violations.extend(date_violations(benchmark, cutoff, name="benchmark", limit=20))
    if not available.empty:
        violations.extend(
            date_violations(available, cutoff, date_col="effective_notice_date", name="fundamentals", limit=20)
        )
    estimated_rows = int(available["notice_date_estimated"].sum()) if not available.empty and "notice_date_estimated" in available else 0

    max_panel_date = str(pd.to_datetime(panel["date"], errors="coerce").max().date()) if not panel.empty else None
    max_benchmark_date = str(pd.to_datetime(benchmark["date"], errors="coerce").max().date()) if not benchmark.empty else None
    latest_report = max_available_report_date(snapshot.fundamentals, cutoff)

    checks = [
        {
            "id": "panel_date_cutoff",
            "status": "pass" if max_panel_date is not None and max_panel_date <= cutoff.isoformat() else "fail",
            "detail": {"max_panel_date": max_panel_date, "as_of": cutoff.isoformat()},
        },
        {
            "id": "benchmark_date_cutoff",
            "status": "pass" if max_benchmark_date is not None and max_benchmark_date <= cutoff.isoformat() else "fail",
            "detail": {"max_benchmark_date": max_benchmark_date, "as_of": cutoff.isoformat()},
        },
        {
            "id": "fundamentals_notice_dates",
            "status": "pass"
            if not [v for v in violations if v.get("artifact") == "fundamentals"]
            else "fail",
            "detail": {
                "available_rows": int(len(available)),
                "estimated_notice_rows": estimated_rows,
                "latest_available_report_date": latest_report,
            },
        },
        {
            "id": "snapshot_as_of",
            "status": "pass" if snapshot.as_of >= cutoff else "fail",
            "detail": {"snapshot_as_of": snapshot.as_of.isoformat(), "run_as_of": cutoff.isoformat()},
        },
    ]
    if estimated_rows:
        checks.append(
            {
                "id": "estimated_notice_dates",
                "status": "warn",
                "detail": {"estimated_rows": estimated_rows, "rule": "report_date + 90 days"},
            }
        )
    audit = {
        "as_of": cutoff.isoformat(),
        "violations": violations,
        "max_panel_date": max_panel_date,
        "max_benchmark_date": max_benchmark_date,
        "fundamentals_available_as_of": int(len(available)),
        "fundamentals_latest_report_date": latest_report,
        "fundamentals_estimated_rows": estimated_rows,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_as_of": snapshot.as_of.isoformat(),
        "checks": checks,
    }
    return audit


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_data_prep(ctx: RunContext) -> StageOutput:
    """Data-prep stage: snapshot -> panel + benchmark + quality + PIT audit."""

    cutoff = parse_date(str(ctx.as_of))
    snapshot_path = resolve_snapshot_dir(ctx)
    ctx.event("data_prep_snapshot", path=str(snapshot_path), as_of=cutoff.isoformat())
    try:
        snapshot = load_snapshot(snapshot_path)
    except FileNotFoundError as exc:
        raise DataQualityError(f"frozen snapshot not found: {snapshot_path} ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 - structural snapshot problem
        raise DataQualityError(f"cannot load frozen snapshot {snapshot_path}: {type(exc).__name__}: {exc}") from exc

    if snapshot.as_of > cutoff:
        raise DataQualityError(
            f"snapshot as_of={snapshot.as_of.isoformat()} is later than the run decision date "
            f"{cutoff.isoformat()}; future data must not be used"
        )

    quality_cfg = load_quality_config(ctx.config)
    min_universe = _min_universe_size(ctx.config, quality_cfg)
    checks = compute_quality_checks(
        snapshot,
        as_of=cutoff,
        quality=quality_cfg,
        min_universe_size=min_universe,
    )
    failures = quality_failures(checks)
    if failures:
        # Fail before building artifacts: the snapshot itself is unusable.
        details = "; ".join(f"{check['id']}: {check.get('detail')}" for check in failures)
        raise DataQualityError(f"snapshot quality gate(s) failed: {details}")

    panel = build_panel(snapshot, cutoff)
    benchmark = build_benchmark(snapshot, cutoff)
    audit = build_pit_audit(snapshot, panel, benchmark, cutoff)
    failed_audit = [check for check in audit["checks"] if check.get("status") == "fail"]
    audit_violations = audit.get("violations", [])

    stage_dir = ctx.stage_dir("data_prep")
    panel_path = write_parquet_atomic(stage_dir / "panel.parquet", panel)
    benchmark_path = write_parquet_atomic(stage_dir / "benchmark.parquet", benchmark)

    coverage = 0.0
    for check in checks:
        if check.get("id") == "fundamental_coverage":
            coverage = float(check.get("detail", {}).get("coverage", 0.0))
    rows = int(len(panel))
    universe_size = int(panel["code"].nunique()) if not panel.empty else 0
    max_date = str(pd.to_datetime(panel["date"], errors="coerce").max().date()) if not panel.empty else None

    warnings: list[str] = []
    for check in checks:
        if check.get("status") == "warn":
            warnings.append(f"{check['id']}: {check.get('detail')}")
    for check in audit["checks"]:
        if check.get("status") == "warn":
            warnings.append(f"pit:{check['id']}: {check.get('detail')}")
    effective_min = None
    for check in checks:
        if check.get("id") == "universe_size":
            effective_min = int(check.get("detail", {}).get("minimum", 0) or 0)
    if effective_min and universe_size < effective_min:
        warnings.append(f"universe_size {universe_size} is below configured minimum {effective_min}")

    quality_report: dict[str, Any] = {
        "as_of": cutoff.isoformat(),
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_as_of": snapshot.as_of.isoformat(),
        "rows": rows,
        "universe_size": universe_size,
        "coverage": float(round(coverage, 6)),
        "max_date": max_date,
        "checks": checks,
        "warnings": warnings,
    }
    quality_path = write_json(stage_dir / "data_quality.json", quality_report, indent=2, sort_keys=True)
    audit_path = write_json(stage_dir / "pit_audit.json", audit, indent=2, sort_keys=True)

    summary = {
        "rows": rows,
        "universe_size": universe_size,
        "max_date": max_date,
        "coverage": float(round(coverage, 6)),
        "warnings": warnings,
    }
    output = StageOutput(
        artifacts={
            "panel": relpath(panel_path),
            "benchmark": relpath(benchmark_path),
            "data_quality": relpath(quality_path),
            "pit_audit": relpath(audit_path),
        },
        summary=summary,
    )
    ctx.register("data_prep", output)
    ctx.event(
        "data_prep_completed",
        rows=rows,
        universe_size=universe_size,
        max_date=max_date,
        snapshot_hash=snapshot.snapshot_hash,
    )

    if audit_violations or failed_audit:
        details = {
            "violations": audit_violations[:5],
            "failed_checks": failed_audit,
        }
        raise DataQualityError(f"PIT audit failed for run as_of={cutoff.isoformat()}: {details}")
    return output
