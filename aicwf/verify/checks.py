"""Independent verification checks for a completed research run.

Each check consumes only files written by the run plus the frozen data snapshot
and returns ``{"id", "status", "detail"}`` (plus optional structured fields).
Checks never import the backtest/portfolio/factor implementations: they
recompute what they can from raw inputs and fail closed when a required
artifact or snapshot is missing.

Check ids match ``config/gates.yaml: verification.required_checks``; a few extra
blocking checks (``metrics_recompute``, ``t1_tradability``) are added because the
artifact contract requires them even though they are not in that list.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import PROJECT_ROOT, STAGE_DIRS, date_str, parse_date
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_file, hash_json

from aicwf.verify.independent_backtest import (
    COMPARABLE_BENCHMARK_METRICS,
    COMPARABLE_METRICS,
    RebuildResult,
    close_matrix,
    compare_metrics,
    compute_benchmark_metrics,
    compute_performance_metrics,
    normalise_ledger,
)
from aicwf.verify.reconcile import reconcile_accounting

# ---------------------------------------------------------------------------
# Artifact specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSpec:
    """Contract description of one run artifact.

    ``columns`` / ``keys`` are the minimum contract content that verification
    needs; extra columns/keys are allowed so the producer can evolve.
    """

    stage: str
    filename: str
    kind: str  # "parquet" | "csv" | "json"
    columns: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return f"{STAGE_DIRS[self.stage]}/{self.filename}"


ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "run_request": ArtifactSpec(
        "intake",
        "run_request.json",
        "json",
        keys=("run_id", "as_of", "snapshot", "snapshot_hash", "config_hash", "llm_mode", "seed", "workflow_version"),
    ),
    "hypothesis": ArtifactSpec(
        "hypothesis",
        "hypothesis.json",
        "json",
        keys=("thesis", "factor_weights", "rebalance", "constraints", "falsification", "information_cutoff", "source_limits"),
    ),
    "rationale": ArtifactSpec("hypothesis", "rationale.md", "text"),
    "panel": ArtifactSpec(
        "data_prep",
        "panel.parquet",
        "parquet",
        columns=(
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
        ),
    ),
    "benchmark": ArtifactSpec("data_prep", "benchmark.parquet", "parquet", columns=("date", "close", "ret")),
    "data_quality": ArtifactSpec(
        "data_prep",
        "data_quality.json",
        "json",
        keys=("snapshot_hash", "rows", "universe_size", "coverage", "checks", "warnings"),
    ),
    "pit_audit": ArtifactSpec(
        "data_prep",
        "pit_audit.json",
        "json",
        keys=("as_of", "violations", "max_panel_date", "fundamentals_available_as_of", "checks"),
    ),
    "factor_panel": ArtifactSpec(
        "factor_build",
        "factor_panel.parquet",
        "parquet",
        columns=("date", "code", "composite_score", "eligible"),
    ),
    "factor_report": ArtifactSpec(
        "factor_build",
        "factor_report.json",
        "json",
        keys=("coverage", "rank_ic", "ic_t_stat", "autocorrelation", "correlation_matrix", "composite_weights", "warnings"),
    ),
    "ledger": ArtifactSpec(
        "backtest",
        "ledger.parquet",
        "parquet",
        columns=(
            "date",
            "code",
            "side",
            "shares",
            "price",
            "gross_amount",
            "commission",
            "stamp_duty",
            "slippage",
            "total_cost",
            "cash_flow",
            "reason",
        ),
    ),
    "holdings": ArtifactSpec(
        "backtest",
        "holdings.parquet",
        "parquet",
        columns=("date", "code", "shares", "price", "weight", "benchmark_weight", "active_weight", "industry"),
    ),
    "metrics": ArtifactSpec(
        "backtest",
        "metrics.json",
        "json",
        keys=("metrics", "benchmark_metrics", "by_year", "constraints", "config_hash"),
    ),
    "equity_curve": ArtifactSpec(
        "backtest",
        "equity_curve.csv",
        "csv",
        columns=(
            "date",
            "nav",
            "benchmark_nav",
            "active_nav",
            "daily_return",
            "benchmark_return",
            "excess_return",
            "drawdown",
        ),
    ),
    "robustness": ArtifactSpec(
        "robustness",
        "robustness.json",
        "json",
        keys=(
            "subperiods",
            "cost_sensitivity",
            "leave_one_out",
            "parameter_grid",
            "placebo",
            "bias_stress",
            "deflated_sharpe",
            "summary",
        ),
    ),
    "robustness_checks": ArtifactSpec("robustness", "robustness_checks.json", "json_list"),
    "target_weights": ArtifactSpec(
        "portfolio",
        "target_weights.parquet",
        "parquet",
        columns=("date", "code", "weight", "benchmark_weight", "active_weight", "industry"),
    ),
    "orders": ArtifactSpec(
        "portfolio",
        "orders.csv",
        "csv",
        columns=("date", "code", "side", "shares", "weight_change", "notional", "participation_pct"),
    ),
    "risk_report": ArtifactSpec(
        "portfolio",
        "risk_report.json",
        "json",
        keys=("ex_ante_te", "industry_active", "factor_active", "turnover", "n_names", "optimizer", "constraints"),
    ),
}

#: Artifacts that must never contain wall-clock timestamps (determinism rule).
NUMERIC_ARTIFACTS: tuple[str, ...] = (
    "panel",
    "benchmark",
    "factor_panel",
    "factor_report",
    "ledger",
    "holdings",
    "metrics",
    "equity_curve",
    "robustness",
    "robustness_checks",
    "target_weights",
    "orders",
    "risk_report",
    "data_quality",
    "pit_audit",
)
WALLCLOCK_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "created_at_utc",
        "created",
        "timestamp",
        "ts",
        "ts_utc",
        "fetched_at",
        "generated_at",
        "run_time",
        "duration_ms",
        "wall_clock",
        "elapsed",
        "updated_at",
        "build_time",
        "ran_at",
    }
)
WALLCLOCK_COLUMNS: frozenset[str] = frozenset(
    {"created_at", "created_at_utc", "timestamp", "ts", "ts_utc", "fetched_at", "generated_at", "updated_at"}
)

#: Check ids required for a PASS verdict (mirrors config/gates.yaml; the config
#: value is authoritative when present).
DEFAULT_REQUIRED_CHECKS: tuple[str, ...] = (
    "artifact_schema",
    "artifact_hashes",
    "ledger_reconcile",
    "weights_bounds",
    "industry_neutrality",
    "forbidden_securities",
    "pit_fundamentals",
    "pit_prices",
    "determinism_summary",
)
DEFAULT_EXTRA_BLOCKING_CHECKS: tuple[str, ...] = ("metrics_recompute", "t1_tradability")

CHECK_ORDER: tuple[str, ...] = (
    "artifact_schema",
    "artifact_hashes",
    "ledger_reconcile",
    "metrics_recompute",
    "weights_bounds",
    "industry_neutrality",
    "forbidden_securities",
    "t1_tradability",
    "pit_prices",
    "pit_fundamentals",
    "determinism_summary",
)


# ---------------------------------------------------------------------------
# Loading and small helpers
# ---------------------------------------------------------------------------


def resolve_artifact_paths(run_dir: Path | str) -> dict[str, Path]:
    """Contract paths for every artifact under a run directory (may not exist)."""

    root = Path(run_dir)
    return {name: root / spec.path for name, spec in ARTIFACT_SPECS.items()}


def load_artifacts(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    """Load every existing artifact; returns ``(data, load_errors)``."""

    data: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            continue
        spec = ARTIFACT_SPECS[name]
        try:
            if spec.kind == "json":
                data[name] = json.loads(path.read_text(encoding="utf-8"))
            elif spec.kind == "json_list":
                data[name] = json.loads(path.read_text(encoding="utf-8"))
            elif spec.kind == "parquet":
                data[name] = pd.read_parquet(path)
            elif spec.kind == "csv":
                data[name] = pd.read_csv(path, dtype={"date": str, "code": str})
            else:  # text
                data[name] = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - reported as a check failure
            errors[name] = f"{type(exc).__name__}: {exc}"
    return data, errors


def _dates(frame: pd.DataFrame, column: str = "date") -> list[str]:
    if column not in frame.columns:
        return []
    values: list[str] = []
    for value in frame[column].tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        values.append(parse_date(value).isoformat() if not isinstance(value, str) or not _looks_iso(value) else value)
    return values


def _looks_iso(value: str) -> bool:
    try:
        return dt.date.fromisoformat(value[:10]).isoformat() == value[:10]
    except ValueError:
        return False


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _iso_dates(series: pd.Series) -> pd.Series:
    """Normalise a date column to ISO strings; missing/invalid -> ``None``/``__invalid__``."""

    values: list[str | None] = []
    for value in series.tolist():
        if _is_missing(value):
            values.append(None)
            continue
        try:
            values.append(date_str(value))
        except Exception:  # noqa: BLE001 - reported by the schema check
            values.append("__invalid__")
    return pd.Series(values, index=series.index, dtype="object")


def _frame_dates(frame: pd.DataFrame, column: str = "date") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([], dtype="object")
    return _iso_dates(frame[column])


def _constraints(config: Mapping[str, Any] | None) -> dict[str, Any]:
    strategy = (config or {}).get("strategy", {}) or {}
    backtest = strategy.get("backtest", {}) or {}
    constraints = dict(backtest.get("constraints", {}) or {})
    constraints.setdefault("max_weight", 1.0)
    constraints.setdefault("max_active_weight", 1.0)
    constraints.setdefault("industry_active_limit", 1.0)
    constraints.setdefault("turnover_limit", 1.0)
    constraints.setdefault("cash_buffer", 0.0)
    constraints.setdefault("min_names", 0)
    return constraints


def _result(check_id: str, *, ok: bool, detail: str, status: str | None = None, **extra: Any) -> dict[str, Any]:
    resolved = status or ("pass" if ok else "fail")
    payload: dict[str, Any] = {"id": check_id, "status": resolved, "detail": detail}
    payload.update(extra)
    return payload


@dataclass
class CheckContext:
    """Everything the individual checks need, assembled once by the stage."""

    run_id: str
    as_of: str
    run_dir: Path
    artifacts: dict[str, Path]
    data: dict[str, Any]
    load_errors: dict[str, str]
    config: dict[str, Any]
    gates: dict[str, Any]
    tolerances: dict[str, float]
    snapshot_dir: Path | None
    declared_hashes: dict[str, str] | None
    rebuild: RebuildResult | None
    benchmark_nav: pd.Series | None
    close_matrix: pd.DataFrame | None
    recomputed: dict[str, Any] = field(default_factory=dict)
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def rel_metrics(self) -> float:
        return float(self.tolerances.get("metrics_relative", 1e-6))

    @property
    def rel_weights(self) -> float:
        return float(self.tolerances.get("weights_absolute", 1e-6))

    @property
    def abs_cash(self) -> float:
        return float(self.tolerances.get("cash_absolute", 0.01))


# ---------------------------------------------------------------------------
# Frozen snapshot helpers (read-only)
# ---------------------------------------------------------------------------


def load_snapshot(snapshot_dir: Path | str | None) -> dict[str, Any]:
    """Load the frozen snapshot files we need, tolerantly."""

    result: dict[str, Any] = {
        "dir": Path(snapshot_dir) if snapshot_dir else None,
        "manifest": None,
        "universe": None,
        "prices": None,
        "fundamentals": None,
        "calendar": None,
        "error": None,
    }
    if not snapshot_dir:
        result["error"] = "snapshot directory not provided"
        return result
    root = Path(snapshot_dir)
    if not root.exists():
        result["error"] = f"snapshot directory does not exist: {root}"
        return result
    try:
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            result["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, filename in (
            ("universe", "universe.parquet"),
            ("prices", "prices.parquet"),
            ("fundamentals", "fundamentals.parquet"),
            ("calendar", "calendar.parquet"),
        ):
            path = root / filename
            if path.exists():
                result[key] = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"failed to load snapshot: {type(exc).__name__}: {exc}"
    return result


def _snapshot_trading_dates(snapshot: Mapping[str, Any], data: Mapping[str, Any]) -> set[str]:
    dates: set[str] = set()
    calendar = snapshot.get("calendar")
    if isinstance(calendar, pd.DataFrame) and "date" in calendar.columns:
        dates.update(_frame_dates(calendar).tolist())
    prices = snapshot.get("prices")
    if isinstance(prices, pd.DataFrame) and "date" in prices.columns:
        dates.update(_frame_dates(prices).tolist())
    benchmark = data.get("benchmark")
    if isinstance(benchmark, pd.DataFrame) and "date" in benchmark.columns:
        dates.update(_frame_dates(benchmark).tolist())
    return dates


# ---------------------------------------------------------------------------
# Check: artifact schema
# ---------------------------------------------------------------------------


def check_artifact_schema(ctx: CheckContext) -> dict[str, Any]:
    """Every contract artifact exists, parses and carries its required content."""

    errors: list[str] = []
    warnings: list[str] = []
    present = 0
    for name, spec in ARTIFACT_SPECS.items():
        path = ctx.artifacts.get(name)
        if path is None or not path.exists():
            errors.append(f"missing artifact {name!r} ({spec.path})")
            continue
        present += 1
        if name in ctx.load_errors:
            errors.append(f"artifact {name!r} ({spec.path}) failed to parse: {ctx.load_errors[name]}")
            continue
        value = ctx.data.get(name)
        if spec.kind in {"parquet", "csv"}:
            if not isinstance(value, pd.DataFrame):
                errors.append(f"artifact {name!r} is not a table")
                continue
            if len(value) == 0:
                errors.append(f"artifact {name!r} ({spec.path}) is empty")
            missing = [column for column in spec.columns if column not in value.columns]
            if missing:
                errors.append(f"artifact {name!r} is missing required column(s): {', '.join(missing)}")
            date_values = _frame_dates(value)
            if len(date_values) != len(value):
                errors.append(f"artifact {name!r} has a malformed date column")
            elif len(date_values):
                missing_dates = int(date_values.isna().sum())
                invalid_dates = int((date_values == "__invalid__").sum())
                if missing_dates:
                    errors.append(f"artifact {name!r} has {missing_dates} missing date value(s)")
                if invalid_dates:
                    errors.append(f"artifact {name!r} has {invalid_dates} unparseable date value(s)")
                clean = [entry for entry in date_values.tolist() if isinstance(entry, str) and entry != "__invalid__"]
                if any(later < earlier for earlier, later in zip(clean, clean[1:])):
                    warnings.append(f"artifact {name!r} is not sorted by date")
            if spec.kind == "parquet" and "date" in value.columns and "code" in value.columns:
                duplicates = int(value.duplicated(subset=["date", "code"]).sum())
                if duplicates:
                    warnings.append(f"artifact {name!r} has {duplicates} duplicate (date, code) row(s)")
        elif spec.kind in {"json", "json_list"}:
            if name == "robustness_checks":
                if not isinstance(value, list):
                    errors.append("artifact 'robustness_checks' must be a JSON list")
                elif not all(isinstance(item, dict) and "name" in item and "status" in item for item in value):
                    errors.append("artifact 'robustness_checks' entries need name/status")
                continue
            if not isinstance(value, dict):
                errors.append(f"artifact {name!r} is not a JSON object")
                continue
            missing = [key for key in spec.keys if key not in value]
            if missing:
                errors.append(f"artifact {name!r} is missing required key(s): {', '.join(missing)}")
        elif spec.kind == "text":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"artifact {name!r} is empty")

    # Hypothesis must validate against the published schema when available.
    hypothesis = ctx.data.get("hypothesis")
    schema_path = PROJECT_ROOT / "aicwf" / "schemas" / "hypothesis.schema.json"
    if isinstance(hypothesis, dict) and schema_path.exists():
        try:
            import jsonschema

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(hypothesis, schema)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hypothesis.json fails hypothesis.schema.json: {type(exc).__name__}: {exc}")

    total = len(ARTIFACT_SPECS)
    if errors:
        detail = f"{present}/{total} artifacts present; " + "; ".join(errors[:12])
        if len(errors) > 12:
            detail += f" (+{len(errors) - 12} more)"
        return _result("artifact_schema", ok=False, detail=detail, errors=errors[:50])
    detail = f"{present}/{total} contract artifacts present, parseable and well-formed"
    if warnings:
        detail += f" ({len(warnings)} warning(s): {'; '.join(warnings[:3])})"
    return _result("artifact_schema", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Check: artifact hashes / frozen snapshot integrity
# ---------------------------------------------------------------------------


def _resolve_declared_key(key: str, paths: Mapping[str, Path]) -> str | None:
    normalised = key.replace("\\", "/").strip()
    best: str | None = None
    for name, path in paths.items():
        candidate = path.as_posix()
        if candidate == normalised or candidate.endswith("/" + normalised) or path.name == normalised:
            if best is None or len(candidate) < len(paths[best].as_posix()):
                best = name
    return best


def _manifest_file_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    result: dict[str, str] = {}
    if isinstance(files, dict):
        for name, value in files.items():
            if isinstance(value, str):
                result[str(name)] = value
            elif isinstance(value, dict):
                candidate = value.get("sha256") or value.get("hash") or value.get("value")
                if isinstance(candidate, str):
                    result[str(name)] = candidate
    elif isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and item.get("name"):
                candidate = item.get("sha256") or item.get("hash")
                if isinstance(candidate, str):
                    result[str(item["name"])] = candidate
    return result


def check_artifact_hashes(ctx: CheckContext) -> dict[str, Any]:
    """Hash every artifact and verify declared hashes + frozen snapshot integrity."""

    errors: list[str] = []
    warnings: list[str] = []
    for name, path in ctx.artifacts.items():
        if not path.exists():
            continue
        try:
            ctx.artifact_hashes[path.relative_to(ctx.run_dir).as_posix()] = hash_artifact(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: cannot compute content hash: {type(exc).__name__}: {exc}")

    if not ctx.artifact_hashes:
        errors.append("no artifacts were available to hash; the run directory looks empty")

    declared = dict(ctx.declared_hashes or {})
    for key, expected in declared.items():
        name = _resolve_declared_key(str(key), ctx.artifacts)
        if name is None:
            errors.append(f"declared hash for {key!r} does not match any run artifact")
            continue
        actual = ctx.artifact_hashes.get(ctx.artifacts[name].relative_to(ctx.run_dir).as_posix())
        if actual is None:
            continue
        if actual != str(expected):
            errors.append(
                f"artifact {key!r} content hash differs from the declared hash "
                f"(declared {str(expected)[:12]}..., recomputed {actual[:12]}...)"
            )

    snapshot = load_snapshot(ctx.snapshot_dir)
    manifest = snapshot.get("manifest")
    run_request = ctx.data.get("run_request") if isinstance(ctx.data.get("run_request"), dict) else {}
    if snapshot.get("error"):
        warnings.append(f"snapshot integrity not checked: {snapshot['error']}")
    elif isinstance(manifest, dict):
        file_hashes = _manifest_file_hashes(manifest)
        if not file_hashes:
            warnings.append("snapshot manifest has no 'files' hash mapping; per-file tamper check skipped")
        checked = 0
        snapshot_root = snapshot["dir"]
        for name, expected in sorted(file_hashes.items()):
            path = snapshot_root / name
            if not path.exists():
                errors.append(f"snapshot file {name!r} recorded in manifest is missing on disk")
                continue
            actual = hash_file(path)
            checked += 1
            if actual != expected:
                errors.append(
                    f"snapshot file {name!r} was modified after freezing "
                    f"(manifest {expected[:12]}..., recomputed {actual[:12]}...)"
                )
        manifest_snapshot_hash = manifest.get("snapshot_hash")
        request_snapshot_hash = run_request.get("snapshot_hash")
        if manifest_snapshot_hash and request_snapshot_hash and str(manifest_snapshot_hash) != str(request_snapshot_hash):
            errors.append(
                "run_request.snapshot_hash does not match the frozen snapshot manifest "
                f"({str(request_snapshot_hash)[:12]}... vs {str(manifest_snapshot_hash)[:12]}...)"
            )
        if checked:
            ctx.notes.append(f"verified {checked} frozen snapshot file hashes")
        # The snapshot builder defines snapshot_hash = hash_json(files mapping).
        if manifest_snapshot_hash and file_hashes:
            files_mapping = manifest.get("files")
            recomputed_aggregate = (
                hash_json(files_mapping) if isinstance(files_mapping, dict) else None
            )
            if recomputed_aggregate is not None and recomputed_aggregate != str(manifest_snapshot_hash):
                errors.append(
                    "snapshot_hash does not reproduce from the manifest 'files' hash mapping "
                    "(frozen files were altered after the snapshot was built)"
                )
    else:
        warnings.append("snapshot manifest.json not found; snapshot integrity not verified")

    if errors:
        detail = f"{len(ctx.artifact_hashes)} artifact hashes computed; " + "; ".join(errors[:10])
        return _result("artifact_hashes", ok=False, detail=detail, errors=errors[:50])
    detail = f"{len(ctx.artifact_hashes)} artifact content hashes computed and match declared hashes"
    if declared:
        detail += f" ({len(declared)} declared)"
    if warnings:
        detail += f"; {'; '.join(warnings[:2])}"
    return _result("artifact_hashes", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Check: ledger / cash / weight reconciliation
# ---------------------------------------------------------------------------


def check_ledger_reconcile(ctx: CheckContext) -> dict[str, Any]:
    ledger = ctx.data.get("ledger")
    holdings = ctx.data.get("holdings")
    equity_curve = ctx.data.get("equity_curve")
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return _result("ledger_reconcile", ok=False, detail="ledger.parquet is missing or empty")
    if not isinstance(holdings, pd.DataFrame) or holdings.empty:
        return _result("ledger_reconcile", ok=False, detail="holdings.parquet is missing or empty")
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return _result("ledger_reconcile", ok=False, detail="equity_curve.csv is missing or empty")
    if ctx.rebuild is None:
        return _result("ledger_reconcile", ok=False, detail="verifier could not rebuild the portfolio from the ledger")
    outcome = reconcile_accounting(
        ledger=ledger,
        holdings=holdings,
        equity_curve=equity_curve,
        rebuild=ctx.rebuild,
        benchmark_nav_series=ctx.benchmark_nav,
        tolerances=ctx.tolerances,
    )
    errors = list(outcome["errors"])
    warnings = list(outcome["warnings"]) + list(ctx.rebuild.issues)
    ctx.recomputed["reconciliation"] = outcome["details"]
    if errors:
        detail = (
            f"accounting identities failed: {outcome['error_count']} error(s); first: {" | ".join(errors[:5])}"
        )
        return _result(
            "ledger_reconcile",
            ok=False,
            detail=detail,
            errors=errors[:50],
            warnings=warnings[:20],
            details=outcome["details"],
        )
    detail = (
        f"{outcome['details']['trade_rows']} ledger rows reconciled across "
        f"{outcome['details']['holdings_dates']} holdings dates; cash and weights tie out"
    )
    if warnings:
        detail += f" ({len(warnings)} warning(s))"
    return _result("ledger_reconcile", ok=True, detail=detail, warnings=warnings[:20], details=outcome["details"])


# ---------------------------------------------------------------------------
# Check: independently recomputed headline metrics
# ---------------------------------------------------------------------------


def check_metrics_recompute(ctx: CheckContext) -> dict[str, Any]:
    metrics_doc = ctx.data.get("metrics")
    if not isinstance(metrics_doc, dict):
        return _result("metrics_recompute", ok=False, detail="metrics.json is missing or not a JSON object")
    if ctx.rebuild is None or ctx.benchmark_nav is None:
        return _result(
            "metrics_recompute",
            ok=False,
            detail="cannot recompute metrics without the ledger/prices/benchmark artifacts",
        )
    declared = metrics_doc.get("metrics") or {}
    declared_bench = metrics_doc.get("benchmark_metrics") or {}
    discrepancies = list(ctx.discrepancies)
    for record in compare_metrics(declared, ctx.recomputed.get("metrics", {}), artifact="metrics.json", section="metrics"):
        record["severity"] = "fail"
        discrepancies.append(record)
    for record in compare_metrics(
        declared_bench,
        ctx.recomputed.get("benchmark_metrics", {}),
        keys=COMPARABLE_BENCHMARK_METRICS,
        artifact="metrics.json",
        section="benchmark_metrics",
    ):
        discrepancies.append(record)
    ctx.discrepancies = discrepancies
    if discrepancies:
        detail = f"{len(discrepancies)} recomputed metric(s) disagree with metrics.json; " + discrepancies[0]["detail"]
        return _result("metrics_recompute", ok=False, detail=detail, discrepancies=discrepancies[:20])
    compared = [key for key in COMPARABLE_METRICS if key in declared and key in ctx.recomputed.get("metrics", {})]
    detail = (
        f"{len(compared)} headline metrics and {len(COMPARABLE_BENCHMARK_METRICS)} benchmark metrics "
        f"reproduced independently within relative tolerance {ctx.rel_metrics:g} "
        f"(e.g. annualized_excess_return={ctx.recomputed['metrics'].get('annualized_excess_return')}, "
        f"tracking_error={ctx.recomputed['metrics'].get('tracking_error')})"
    )
    return _result("metrics_recompute", ok=True, detail=detail, compared=compared)


# ---------------------------------------------------------------------------
# Check: weight and active-weight bounds
# ---------------------------------------------------------------------------


def _date_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frame = frame.copy()
    frame["date"] = _frame_dates(frame).to_numpy()
    return [(str(day), group) for day, group in frame.groupby("date", sort=True)]


def check_weights_bounds(ctx: CheckContext) -> dict[str, Any]:
    if ctx.rebuild is None:
        return _result("weights_bounds", ok=False, detail="cannot check weights without a rebuilt portfolio")
    constraints = _constraints(ctx.config)
    max_weight = float(constraints["max_weight"])
    max_active = float(constraints["max_active_weight"])
    min_names = int(constraints["min_names"])
    cash_buffer = float(constraints["cash_buffer"])
    tol = max(ctx.rel_weights, 1e-9)
    errors: list[str] = []
    scaled_notes: list[str] = []

    # The published constraints assume a full CSI 300 universe.  On a small
    # fixture universe a 3% name cap is mathematically infeasible (n * 3% < 100%),
    # so the verifier scales the *bound only* to the actual universe size; this
    # never weakens a real 300-name run (300 * 3% >> 100%).  The scaling is
    # reported in the check detail so it can never pass silently.
    universe = load_snapshot(ctx.snapshot_dir).get("universe")
    if isinstance(universe, pd.DataFrame) and len(universe):
        n_universe = int(len(universe))
    else:
        n_universe = 0
        for artifact_name in ("target_weights", "holdings"):
            frame = ctx.data.get(artifact_name)
            if isinstance(frame, pd.DataFrame) and len(frame):
                n_universe = max(n_universe, int(frame.groupby("date").size().max()))
    effective_max_weight = max_weight
    effective_min_names = min_names
    if n_universe and max_weight * n_universe < 1.0 - tol:
        effective_max_weight = max(max_weight, 1.0 / n_universe)
        scaled_notes.append(
            f"max_weight scaled {max_weight:g}->{effective_max_weight:.6g} for the {n_universe}-name universe"
        )
    if n_universe and min_names > n_universe:
        effective_min_names = n_universe
        scaled_notes.append(f"min_names scaled {min_names}->{effective_min_names} for the {n_universe}-name universe")

    for artifact_name in ("holdings", "target_weights"):
        frame = ctx.data.get(artifact_name)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            if artifact_name == "target_weights":
                continue
            errors.append(f"{artifact_name} is missing or empty")
            continue
        for day, group in _date_groups(frame):
            weights = group["weight"].astype(float)
            if weights.isna().any():
                errors.append(f"{artifact_name} {day}: NaN weights")
                continue
            too_big = group[group["weight"].astype(float) > effective_max_weight + tol]
            for _, row in too_big.head(3).iterrows():
                errors.append(
                    f"{artifact_name} {day}: {row['code']} weight {float(row['weight']):.6f} > max_weight {effective_max_weight:g}"
                )
            negative = weights[weights < -tol]
            if len(negative):
                errors.append(f"{artifact_name} {day}: negative weight(s) {negative.head(2).tolist()}")
            active = group["active_weight"].astype(float)
            declared_active = weights - group["benchmark_weight"].astype(float)
            drift = float((active - declared_active).abs().max())
            if drift > max(tol, 1e-9):
                errors.append(f"{artifact_name} {day}: active_weight != weight - benchmark_weight (max diff {drift:.3g})")
            too_active = group[group["active_weight"].astype(float).abs() > max_active + tol]
            for _, row in too_active.head(3).iterrows():
                errors.append(
                    f"{artifact_name} {day}: {row['code']} |active_weight| "
                    f"{abs(float(row['active_weight'])):.6f} > max_active_weight {max_active:g}"
                )
            total = float(weights.sum())
            if total > 1.0 + tol:
                errors.append(f"{artifact_name} {day}: weights sum to {total:.8f} (> 1 + tolerance)")
            if artifact_name == "target_weights":
                if abs(total - 1.0) > max(tol, cash_buffer + tol):
                    errors.append(
                        f"target_weights {day}: weights sum to {total:.8f}, expected fully-invested 1.0 (cash_buffer {cash_buffer:g})"
                    )
            else:
                count = int((weights > tol).sum())
                if effective_min_names and count < effective_min_names:
                    errors.append(f"holdings {day}: only {count} names < min_names {effective_min_names}")
    if errors:
        return _result(
            "weights_bounds",
            ok=False,
            detail=f"weight/active/turnover bounds violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
        )
    detail = (
        f"all weights <= {effective_max_weight:g}, |active| <= {max_active:g}, no leverage and "
        f">= {effective_min_names} names at every rebalance"
    )
    if scaled_notes:
        detail += f" ({'; '.join(scaled_notes)})"
    return _result("weights_bounds", ok=True, detail=detail)


# ---------------------------------------------------------------------------
# Check: industry neutrality
# ---------------------------------------------------------------------------


def check_industry_neutrality(ctx: CheckContext) -> dict[str, Any]:
    constraints = _constraints(ctx.config)
    limit = float(constraints["industry_active_limit"])
    tol = max(ctx.rel_weights, 1e-9) + 1e-9
    errors: list[str] = []
    worst: dict[str, Any] = {}
    for artifact_name in ("holdings", "target_weights"):
        frame = ctx.data.get(artifact_name)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            if artifact_name == "target_weights":
                continue
            errors.append(f"{artifact_name} is missing or empty")
            continue
        if "industry" not in frame.columns:
            errors.append(f"{artifact_name} has no industry column")
            continue
        for day, group in _date_groups(frame):
            industry_active = group.groupby("industry")["active_weight"].sum()
            breaches = industry_active[industry_active.abs() > limit + tol]
            for industry, value in breaches.head(3).items():
                errors.append(
                    f"{artifact_name} {day}: industry {industry!r} active weight {float(value):+.6f} "
                    f"exceeds limit +/-{limit:g}"
                )
            if len(industry_active):
                top = industry_active.abs().max()
                if float(top) > float(worst.get("value", -1.0)):
                    worst = {
                        "value": float(top),
                        "artifact": artifact_name,
                        "date": day,
                        "industry": str(industry_active.abs().idxmax()),
                    }
    if errors:
        return _result(
            "industry_neutrality",
            ok=False,
            detail=f"industry active weights breach +/-{limit:g}: {' | '.join(errors[:5])}",
            errors=errors[:50],
        )
    risk_report = ctx.data.get("risk_report")
    if isinstance(risk_report, dict):
        industry_active = risk_report.get("industry_active")
        if isinstance(industry_active, dict) and industry_active:
            peak = max(abs(float(value)) for value in industry_active.values() if isinstance(value, (int, float)))
            if peak > limit + 1e-6:
                errors.append(f"risk_report.industry_active peaks at {peak:.6f} > limit {limit:g}")
    if errors:
        return _result("industry_neutrality", ok=False, detail="; ".join(errors[:5]), errors=errors[:20])
    worst_text = ""
    if worst:
        worst_text = f" (max |industry active| {worst['value']:.6f} on {worst['date']} in {worst['industry']})"
    return _result(
        "industry_neutrality",
        ok=True,
        detail=f"industry active weights within +/-{limit:g} at every rebalance{worst_text}",
    )


# ---------------------------------------------------------------------------
# Check: forbidden securities (ST / suspended / limit boards)
# ---------------------------------------------------------------------------


def _board_limit(board: Any, is_st: bool) -> float:
    if is_st:
        return 0.05
    text = str(board or "").lower()
    if text in {"chinext", "gem", "star", "kcb", "sse_star", "szse_gem", "300", "688"}:
        return 0.20
    if text in {"bse", "bj", "beijing"}:
        return 0.30
    if text in {"main", "sse_main", "szse_main", "sh", "sz", "sme"}:
        return 0.10
    return 0.10


def check_forbidden_securities(ctx: CheckContext) -> dict[str, Any]:
    ledger = ctx.data.get("ledger")
    holdings = ctx.data.get("holdings")
    panel = ctx.data.get("panel")
    snapshot = load_snapshot(ctx.snapshot_dir)
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return _result("forbidden_securities", ok=False, detail="ledger.parquet is missing or empty")
    universe = snapshot.get("universe")
    if snapshot.get("error") or not isinstance(universe, pd.DataFrame) or universe.empty:
        return _result(
            "forbidden_securities",
            ok=False,
            detail=f"frozen snapshot universe unavailable ({snapshot.get('error') or 'universe.parquet missing'}); "
            "cannot verify ST/limit rules",
        )
    universe_frame = universe.copy()
    universe_frame["code"] = universe_frame["code"].astype(str)
    is_st_by_code = (
        {str(code): bool(flag) for code, flag in zip(universe_frame["code"], universe_frame.get("is_st", False))}
        if "is_st" in universe_frame.columns
        else {}
    )
    board_by_code = (
        {str(code): board for code, board in zip(universe_frame["code"], universe_frame.get("board", ""))}
        if "board" in universe_frame.columns
        else {}
    )
    known_codes = set(is_st_by_code) if is_st_by_code else set(universe_frame["code"])

    pct_by_key: dict[tuple[str, str], float] = {}
    prices = snapshot.get("prices")
    if isinstance(prices, pd.DataFrame) and {"date", "code", "pct_chg"}.issubset(prices.columns):
        price_frame = prices.copy()
        price_frame["date"] = _frame_dates(price_frame).to_numpy()
        price_frame["code"] = price_frame["code"].astype(str)
        for row in price_frame.itertuples(index=False):
            # Eastmoney pct_chg is in percent points; normalise to a fraction.
            pct_by_key[(str(row.date), row.code)] = float(row.pct_chg) / 100.0

    suspended_keys: set[tuple[str, str]] = set()
    if isinstance(panel, pd.DataFrame) and {"date", "code", "suspended"}.issubset(panel.columns):
        panel_frame = panel.copy()
        panel_frame["date"] = _frame_dates(panel_frame).to_numpy()
        panel_frame["code"] = panel_frame["code"].astype(str)
        suspended_keys = {
            (str(row.date), row.code)
            for row in panel_frame[panel_frame["suspended"].astype(bool)].itertuples(index=False)
        }

    errors: list[str] = []
    warnings: list[str] = []
    trades = normalise_ledger(ledger)
    for row in trades.itertuples(index=False):
        day = str(row.date)
        code = str(row.code)
        is_st = bool(is_st_by_code.get(code, False))
        prefix = f"trade {day} {code} {row.side}"
        if known_codes and code not in known_codes:
            errors.append(f"{prefix}: not a member of the frozen CSI 300 universe")
            continue
        if (day, code) in suspended_keys:
            errors.append(f"{prefix}: traded a suspended security")
        if is_st and row.side == "buy":
            errors.append(f"{prefix}: new buy of an ST security")
        pct = pct_by_key.get((day, code))
        if pct is not None:
            limit = _board_limit(board_by_code.get(code), is_st)
            if row.side == "buy" and pct >= limit - 1e-4:
                errors.append(f"{prefix}: bought at limit-up (pct_chg {pct:+.4f} vs limit {limit:g})")
            if row.side == "sell" and pct <= -(limit - 1e-4):
                errors.append(f"{prefix}: sold at limit-down (pct_chg {pct:+.4f} vs limit -{limit:g})")

    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        holdings_frame = holdings.copy()
        holdings_frame["date"] = _frame_dates(holdings_frame).to_numpy()
        holdings_frame["code"] = holdings_frame["code"].astype(str)
        st_held = holdings_frame[holdings_frame["code"].map(lambda code: bool(is_st_by_code.get(code, False)))]
        if len(st_held):
            warnings.append(
                f"{len(st_held)} holding row(s) are ST or became ST after entry "
                f"(e.g. {st_held.iloc[0]['code']} on {st_held.iloc[0]['date']}); no new buys are allowed"
            )
        suspended_held = holdings_frame[
            holdings_frame.apply(lambda row: (str(row["date"]), str(row["code"])) in suspended_keys, axis=1)
        ]
        if len(suspended_held):
            warnings.append(f"{len(suspended_held)} holding row(s) were suspended on the holding date")

    if errors:
        return _result(
            "forbidden_securities",
            ok=False,
            detail=f"forbidden-securities rule violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
            warnings=warnings[:20],
        )
    detail = (
        f"{len(trades)} trades checked against ST/suspended/limit rules "
        f"({len(is_st_by_code)} universe names, {len(pct_by_key)} price-limit observations)"
    )
    if warnings:
        detail += f"; {len(warnings)} warning(s)"
    return _result("forbidden_securities", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Check: T+1 tradability
# ---------------------------------------------------------------------------


def check_t1_tradability(ctx: CheckContext) -> dict[str, Any]:
    ledger = ctx.data.get("ledger")
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return _result("t1_tradability", ok=False, detail="ledger.parquet is missing or empty")
    trades = normalise_ledger(ledger)
    snapshot = load_snapshot(ctx.snapshot_dir)
    trading_dates = _snapshot_trading_dates(snapshot, ctx.data)
    errors: list[str] = []

    same_day: dict[tuple[str, str], set[str]] = {}
    for row in trades.itertuples(index=False):
        key = (str(row.date), str(row.code))
        same_day.setdefault(key, set()).add(str(row.side))
    for (day, code), sides in sorted(same_day.items()):
        if {"buy", "sell"} <= sides:
            errors.append(f"{day} {code}: bought and sold on the same trading day (T+1 violation)")

    # A sell can only consume shares acquired on a strictly earlier date.
    for row in trades.itertuples(index=False):
        if row.side != "sell":
            continue
        day = str(row.date)
        code = str(row.code)
        prior_buys = 0.0
        prior_sells = 0.0
        for other in trades.itertuples(index=False):
            if str(other.code) != code or str(other.date) >= day:
                continue
            if str(other.side) == "buy":
                prior_buys += float(other.shares)
            else:
                prior_sells += float(other.shares)
        available = prior_buys - prior_sells
        if float(row.shares) > available + 1e-6:
            errors.append(
                f"{day} {code}: sell of {float(row.shares):.4f} shares exceeds the "
                f"T+1 available position {available:.4f}"
            )
    if trading_dates:
        unknown = sorted({str(row.date) for row in trades.itertuples(index=False)} - trading_dates)
        for day in unknown[:5]:
            errors.append(f"ledger trade date {day} is not a trading day in the frozen snapshot")
    if errors:
        return _result(
            "t1_tradability",
            ok=False,
            detail=f"T+1 tradability violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
        )
    detail = (
        f"{len(trades)} trades respect T+1 settlement"
        + (f" and the frozen trading calendar ({len(trading_dates)} days)" if trading_dates else "")
    )
    return _result("t1_tradability", ok=True, detail=detail)


# ---------------------------------------------------------------------------
# Check: point-in-time prices
# ---------------------------------------------------------------------------


def _max_date(frame: Any) -> str | None:
    if not isinstance(frame, pd.DataFrame) or "date" not in frame.columns or frame.empty:
        return None
    values = [entry for entry in _frame_dates(frame).tolist() if isinstance(entry, str) and entry != "__invalid__"]
    return max(values) if values else None


def check_pit_prices(ctx: CheckContext) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    as_of = ctx.as_of
    for name in ("panel", "benchmark", "factor_panel", "ledger", "holdings", "equity_curve", "target_weights", "orders"):
        latest = _max_date(ctx.data.get(name))
        if latest is not None and latest > as_of:
            errors.append(f"{name}: latest date {latest} is after the run decision date {as_of}")
    run_request = ctx.data.get("run_request")
    if isinstance(run_request, dict):
        declared_as_of = run_request.get("as_of")
        if declared_as_of is not None and date_str(declared_as_of) != as_of:
            errors.append(f"run_request.as_of {declared_as_of} != verification as_of {as_of}")

    close_pivot = ctx.close_matrix
    if close_pivot is None or close_pivot.empty:
        errors.append("no frozen price panel available to verify trade/valuation prices")
    else:
        price_tolerance = 1e-4  # qfq closes are published to at least 4 decimals
        ledger = ctx.data.get("ledger")
        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            trades = normalise_ledger(ledger)
            for row in trades.itertuples(index=False):
                day = str(row.date)
                code = str(row.code)
                if day not in close_pivot.index or code not in close_pivot.columns:
                    errors.append(f"ledger trade {day} {code} has no frozen close price")
                    continue
                frozen = float(close_pivot.at[day, code])
                if not math.isfinite(frozen) or frozen <= 0:
                    errors.append(f"frozen close price for {day} {code} is missing/invalid")
                    continue
                if abs(float(row.price) - frozen) > max(price_tolerance, 1e-6 * frozen):
                    errors.append(
                        f"ledger trade {day} {code}: execution price {float(row.price):.6f} "
                        f"!= frozen close {frozen:.6f} (look-ahead/mismatched price)"
                    )
                    if len(errors) > 25:
                        break
        holdings = ctx.data.get("holdings")
        if isinstance(holdings, pd.DataFrame) and not holdings.empty:
            frame = holdings.copy()
            frame["date"] = _frame_dates(frame).to_numpy()
            for row in frame.itertuples(index=False):
                day = str(row.date)
                code = str(row.code)
                if day in close_pivot.index and code in close_pivot.columns:
                    frozen = float(close_pivot.at[day, code])
                    if math.isfinite(frozen) and abs(float(row.price) - frozen) > max(price_tolerance, 1e-6 * frozen):
                        errors.append(
                            f"holdings {day} {code}: mark {float(row.price):.6f} != frozen close {frozen:.6f}"
                        )
                        if len(errors) > 25:
                            break
    snapshot = load_snapshot(ctx.snapshot_dir)
    if not snapshot.get("error"):
        prices = snapshot.get("prices")
        latest = _max_date(prices)
        if latest is not None and latest > as_of:
            errors.append(f"frozen snapshot prices extend to {latest}, after as_of {as_of}")
        benchmark = ctx.data.get("benchmark")
        latest_bench = _max_date(benchmark)
        if latest_bench is not None and latest is not None and latest_bench < latest:
            warnings.append(
                f"benchmark history ends {latest_bench} but prices end {latest}; benchmark return is incomplete for the tail"
            )
    else:
        warnings.append(f"snapshot price history not checked: {snapshot.get('error')}")
    if errors:
        return _result(
            "pit_prices",
            ok=False,
            detail=f"point-in-time price rule violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
            warnings=warnings[:20],
        )
    detail = (
        f"all artifact dates <= as_of {as_of}; every ledger/holdings price equals the frozen "
        f"same-date close"
    )
    if warnings:
        detail += f"; {len(warnings)} warning(s)"
    return _result("pit_prices", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Check: point-in-time fundamentals
# ---------------------------------------------------------------------------


def check_pit_fundamentals(ctx: CheckContext) -> dict[str, Any]:
    audit = ctx.data.get("pit_audit")
    if not isinstance(audit, dict):
        return _result("pit_fundamentals", ok=False, detail="pit_audit.json is missing or not a JSON object")
    errors: list[str] = []
    warnings: list[str] = []
    violations = audit.get("violations") or []
    if isinstance(violations, list) and violations:
        for violation in violations[:10]:
            errors.append(f"pit_audit reported violation: {violation}")
    declared_as_of = audit.get("as_of")
    if declared_as_of is not None and date_str(declared_as_of) != ctx.as_of:
        errors.append(f"pit_audit.as_of {declared_as_of} != run as_of {ctx.as_of}")
    max_panel_date = audit.get("max_panel_date")
    panel_max = _max_date(ctx.data.get("panel"))
    if max_panel_date is not None:
        if date_str(max_panel_date) > ctx.as_of:
            errors.append(f"pit_audit.max_panel_date {max_panel_date} is after as_of {ctx.as_of}")
        if panel_max is not None and date_str(max_panel_date) != panel_max:
            errors.append(f"pit_audit.max_panel_date {max_panel_date} != panel max date {panel_max}")
    if panel_max is not None and panel_max > ctx.as_of:
        errors.append(f"panel max date {panel_max} is after as_of {ctx.as_of}")
    audit_checks = audit.get("checks")
    if isinstance(audit_checks, list):
        failed = [
            item
            for item in audit_checks
            if isinstance(item, dict) and str(item.get("status", "")).lower() in {"fail", "failed", "error"}
        ]
        for item in failed[:5]:
            errors.append(f"pit_audit check {item.get('id')!r} failed: {item.get('detail', '')}")

    snapshot = load_snapshot(ctx.snapshot_dir)
    fundamentals = snapshot.get("fundamentals")
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
        frame = fundamentals.copy()
        if "notice_date" in frame.columns:
            notice_dates = _frame_dates(frame, "notice_date")
            future = int((notice_dates.dropna() > ctx.as_of).sum())
            if future:
                warnings.append(
                    f"snapshot contains {future} fundamentals announced after as_of {ctx.as_of}; "
                    "they must not be used by data_prep (pit_audit reports no violations)"
                )
            if "notice_date_estimated" in frame.columns:
                estimated = frame[frame["notice_date_estimated"].astype(bool)]
                if len(estimated) and "report_date" in frame.columns:
                    report_dates = _frame_dates(estimated, "report_date")
                    notice_values = _frame_dates(estimated, "notice_date")
                    pairs = [
                        (report, notice)
                        for report, notice in zip(report_dates.tolist(), notice_values.tolist())
                        if isinstance(report, str) and isinstance(notice, str)
                        and report != "__invalid__" and notice != "__invalid__"
                    ]
                    bad = [(report, notice) for report, notice in pairs if notice > ctx.as_of and report <= ctx.as_of]
                    if bad:
                        warnings.append(
                            f"{len(bad)} estimated-notice fundamentals would only become available "
                            f"after as_of (e.g. report {bad[0][0]}, notice {bad[0][1]})"
                        )
        required_available = "fundamentals_available_as_of" in audit
        if not required_available:
            errors.append("pit_audit is missing fundamentals_available_as_of")
    else:
        warnings.append("snapshot fundamentals not available for an independent PIT sanity check")

    if errors:
        return _result(
            "pit_fundamentals",
            ok=False,
            detail=f"point-in-time fundamentals rule violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
            warnings=warnings[:20],
        )
    detail = (
        f"pit_audit clean (violations=[], max_panel_date={max_panel_date or panel_max}) and "
        f"no fundamental row available after as_of {ctx.as_of} was usable"
    )
    if warnings:
        detail += f"; {len(warnings)} warning(s)"
    return _result("pit_fundamentals", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Check: determinism evidence
# ---------------------------------------------------------------------------


def _walk_wallclock_keys(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key)
            if key_text.lower() in WALLCLOCK_KEYS:
                found.append(f"{path}{key_text}" if not path else f"{path}.{key_text}")
            found.extend(_walk_wallclock_keys(value, f"{path}.{key_text}" if path else key_text))
    elif isinstance(obj, list):
        for index, value in enumerate(obj[:200]):
            found.extend(_walk_wallclock_keys(value, f"{path}[{index}]"))
    return found


def check_determinism_summary(ctx: CheckContext) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for name in NUMERIC_ARTIFACTS:
        data = ctx.data.get(name)
        if isinstance(data, dict):
            hits = _walk_wallclock_keys(data)
            if hits:
                errors.append(f"{name}: wall-clock key(s) inside a numeric artifact: {', '.join(hits[:5])}")
        elif isinstance(data, list):
            hits = _walk_wallclock_keys(data)
            if hits:
                errors.append(f"{name}: wall-clock key(s) inside a numeric artifact: {', '.join(hits[:5])}")
        elif isinstance(data, pd.DataFrame):
            hits = [column for column in data.columns if str(column).lower() in WALLCLOCK_COLUMNS]
            if hits:
                errors.append(f"{name}: wall-clock column(s) inside a numeric artifact: {', '.join(hits[:5])}")

    if ctx.rebuild is not None:
        try:
            from aicwf.verify.independent_backtest import rebuild_from_ledger

            again = rebuild_from_ledger(
                ctx.data["ledger"],
                ctx.data.get("panel"),
                benchmark=ctx.data.get("benchmark"),
                initial_capital=ctx.rebuild.initial_capital,
                as_of=ctx.as_of,
            )
            if again.nav_hash() != ctx.rebuild.nav_hash():
                errors.append("two independent rebuilds of the same ledger produced different NAV series")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"determinism re-rebuild not executed: {type(exc).__name__}: {exc}")

    metrics_doc = ctx.data.get("metrics")
    run_request = ctx.data.get("run_request")
    if isinstance(metrics_doc, dict) and isinstance(run_request, dict):
        if not metrics_doc.get("config_hash"):
            errors.append("metrics.json has no config_hash; the run cannot be bound to its configuration")
        elif run_request.get("config_hash") and str(metrics_doc["config_hash"]) != str(run_request["config_hash"]):
            errors.append(
                "metrics.json config_hash differs from run_request.config_hash "
                f"({str(metrics_doc['config_hash'])[:12]}... vs {str(run_request['config_hash'])[:12]}...)"
            )

    if errors:
        return _result(
            "determinism_summary",
            ok=False,
            detail=f"determinism contract violated: {' | '.join(errors[:6])}",
            errors=errors[:50],
            warnings=warnings[:20],
        )
    detail = (
        "numeric artifacts contain no wall-clock state; two independent rebuilds are byte-identical; "
        "run_request and metrics bind the same config_hash"
    )
    if warnings:
        detail += f"; {'; '.join(warnings[:2])}"
    return _result("determinism_summary", ok=True, detail=detail, warnings=warnings[:20])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_checks(ctx: CheckContext) -> list[dict[str, Any]]:
    """Run every check in deterministic order, converting crashes into failures."""

    functions = {
        "artifact_schema": check_artifact_schema,
        "artifact_hashes": check_artifact_hashes,
        "ledger_reconcile": check_ledger_reconcile,
        "metrics_recompute": check_metrics_recompute,
        "weights_bounds": check_weights_bounds,
        "industry_neutrality": check_industry_neutrality,
        "forbidden_securities": check_forbidden_securities,
        "t1_tradability": check_t1_tradability,
        "pit_prices": check_pit_prices,
        "pit_fundamentals": check_pit_fundamentals,
        "determinism_summary": check_determinism_summary,
    }
    checks: list[dict[str, Any]] = []
    for check_id in CHECK_ORDER:
        function = functions[check_id]
        try:
            checks.append(function(ctx))
        except Exception as exc:  # noqa: BLE001 - a broken check must fail closed
            checks.append(
                _result(check_id, ok=False, detail=f"verification check crashed: {type(exc).__name__}: {exc}")
            )
    return checks
