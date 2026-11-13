"""Verification stage entry point (``09_verification``).

Public API used by the pipeline/controller:

``run_verification(ctx) -> StageOutput``
    Build the verification, write ``verification.json`` + ``verifier_note.md``
    into the stage directory and return the declared artifacts/summary.

``build_verification(run_dir=..., as_of=..., ...) -> dict``
    The same logic without a :class:`RunContext`; usable from CLI/tests and by
    the eval harness when it needs to re-verify a candidate run.

``is_blocking(verification) -> bool``
    ``True`` iff the verdict is ``FAIL``; the pipeline uses this to enter the
    bounded revision loop (``gates.verification.max_revision_rounds``).

``render_verifier_note(verification) -> str``
    Deterministic Markdown fallback; the read-only verifier agent
    (``aicwf.agents.stages.generate_verifier_note``) is preferred when present.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from aicwf.contracts import (
    PROJECT_ROOT,
    STAGE_DIRS,
    RunContext,
    StageOutput,
    date_str,
    parse_date,
    relpath,
)
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import read_json, read_yaml, write_json, write_text

from aicwf.verify.checks import (
    ARTIFACT_SPECS,
    CHECK_ORDER,
    DEFAULT_EXTRA_BLOCKING_CHECKS,
    DEFAULT_REQUIRED_CHECKS,
    CheckContext,
    load_artifacts,
    load_snapshot,
    resolve_artifact_paths,
    run_checks,
)
from aicwf.verify.independent_backtest import (
    benchmark_nav,
    close_matrix,
    compare_metrics,
    compute_benchmark_metrics,
    compute_performance_metrics,
    compute_turnover,
    json_safe,
    normalise_ledger,
    rebuild_from_ledger,
)

VERIFIER_VERSION = "1.0.0"
VERIFIER_NAME = "independent-verifier"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_gates(gates: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the gates config from an explicit mapping, a config bundle or disk."""

    if gates is not None:
        return dict(gates)
    bundle = dict(config or {})
    if isinstance(bundle.get("gates"), Mapping):
        return dict(bundle["gates"])
    path = PROJECT_ROOT / "config" / "gates.yaml"
    if path.exists():
        loaded = read_yaml(path)
        return dict(loaded or {})
    return {}


def verification_tolerances(gates: Mapping[str, Any] | None) -> dict[str, float]:
    values = ((gates or {}).get("verification", {}) or {}).get("tolerances", {}) or {}
    tolerances = {
        "metrics_relative": 1.0e-6,
        "weights_absolute": 1.0e-6,
        "cash_absolute": 0.01,
    }
    for key in list(tolerances):
        if key in values and values[key] is not None:
            try:
                tolerances[key] = float(values[key])
            except (TypeError, ValueError):
                continue
    return tolerances


def required_checks(gates: Mapping[str, Any] | None) -> list[str]:
    configured = ((gates or {}).get("verification", {}) or {}).get("required_checks")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return list(DEFAULT_REQUIRED_CHECKS)


def blocking_checks(gates: Mapping[str, Any] | None) -> list[str]:
    required = required_checks(gates)
    extras = [check_id for check_id in DEFAULT_EXTRA_BLOCKING_CHECKS if check_id not in required]
    return required + extras


# ---------------------------------------------------------------------------
# Artifact / snapshot resolution
# ---------------------------------------------------------------------------


def _artifact_paths_for_run(run_dir: Path) -> dict[str, Path]:
    paths = resolve_artifact_paths(run_dir)
    return paths


def _overlay_stage_outputs(paths: dict[str, Path], ctx: RunContext) -> dict[str, Path]:
    """Prefer the paths the controller recorded over the contract defaults."""

    by_stage: dict[str, dict[str, str]] = {}
    for stage, output in (ctx.stage_outputs or {}).items():
        if output is None:
            continue
        by_stage[stage] = {str(logical): str(value) for logical, value in output.artifacts.items()}
    if not by_stage:
        return paths
    for name, spec in ARTIFACT_SPECS.items():
        declared = by_stage.get(spec.stage, {})
        candidate = None
        for logical, value in declared.items():
            if Path(value).name == spec.filename:
                candidate = value
                break
        if candidate is None:
            candidate = declared.get(name)
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            paths[name] = path
    return paths


def resolve_snapshot_dir(
    ctx: RunContext | None = None,
    *,
    snapshot_dir: Path | str | None = None,
    run_request: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    as_of: str | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if snapshot_dir:
        candidates.append(Path(snapshot_dir))
    if ctx is not None and ctx.snapshot_dir is not None:
        candidates.append(Path(ctx.snapshot_dir))
    if run_request and run_request.get("snapshot"):
        candidates.append(PROJECT_ROOT / str(run_request["snapshot"]))
    strategy = (config or {}).get("strategy", {}) or {}
    declared = (strategy.get("data", {}) or {}).get("snapshot")
    if declared:
        candidates.append(PROJECT_ROOT / str(declared))
    if as_of:
        candidates.append(PROJECT_ROOT / "data" / "snapshots" / as_of)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _declared_hashes(run_dir: Path, ctx: RunContext | None = None, explicit: Mapping[str, str] | None = None) -> dict[str, str] | None:
    if explicit:
        return {str(key): str(value) for key, value in explicit.items()}
    for name in ("artifact_hashes.json", "state_snapshot.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:  # noqa: BLE001
            continue
        mapping = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(mapping, dict):
            result = {
                str(key): str(value.get("sha256") if isinstance(value, dict) else value)
                for key, value in mapping.items()
            }
            if result:
                return result
        if isinstance(mapping, list):
            result = {}
            for item in mapping:
                if isinstance(item, dict) and item.get("path") and (item.get("hash") or item.get("sha256")):
                    result[str(item["path"])] = str(item.get("hash") or item.get("sha256"))
            if result:
                return result
    if ctx is not None:
        store = getattr(ctx, "state_store", None) or getattr(ctx, "store", None)
        if store is not None and hasattr(store, "artifact_hashes"):
            try:
                mapping = store.artifact_hashes(ctx.run_id)
                if mapping:
                    return {str(key): str(value) for key, value in mapping.items()}
            except Exception:  # noqa: BLE001
                pass
    return None


def _initial_capital(metrics_doc: Any) -> float:
    if isinstance(metrics_doc, dict):
        for candidate in (metrics_doc.get("initial_capital"), (metrics_doc.get("metrics") or {}).get("initial_capital")):
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    # The equity curve is normally normalised to NAV 1.0; fall back to unit capital.
    return 1.0


# ---------------------------------------------------------------------------
# Core verification builder
# ---------------------------------------------------------------------------


def build_verification(
    *,
    run_dir: Path | str,
    as_of: dt.date | str | None = None,
    config: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    snapshot_dir: Path | str | None = None,
    declared_hashes: Mapping[str, str] | None = None,
    llm_mode: str = "replay",
    initial_capital: float | None = None,
    run_context: RunContext | None = None,
) -> dict[str, Any]:
    """Independently verify one run; returns the verification document."""

    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    config_bundle = dict(config or {})
    gates_config = load_gates(gates, config_bundle)
    tolerances = verification_tolerances(gates_config)

    paths = _artifact_paths_for_run(root)
    if run_context is not None:
        paths = _overlay_stage_outputs(paths, run_context)
    data, load_errors = load_artifacts(paths)
    run_request = data.get("run_request") if isinstance(data.get("run_request"), dict) else {}
    if as_of is None:
        if run_request.get("as_of") is not None:
            as_of = parse_date(run_request["as_of"])
        else:
            raise ValueError("as_of is required when run_request.json does not declare it")
    as_of_iso = date_str(as_of)
    resolved_run_id = run_id or str(run_request.get("run_id") or root.name)
    resolved_snapshot = resolve_snapshot_dir(
        run_context,
        snapshot_dir=snapshot_dir,
        run_request=run_request,
        config=config_bundle,
        as_of=as_of_iso,
    )

    # -- rebuild the portfolio from raw ledger + prices ------------------
    rebuild = None
    benchmark_series = None
    close_pivot = None
    rebuild_error = None
    capital = float(initial_capital) if initial_capital else _initial_capital(data.get("metrics"))
    try:
        prices = data.get("panel")
        if not isinstance(prices, pd.DataFrame) or prices.empty or "close" not in prices.columns:
            snapshot = load_snapshot(resolved_snapshot)
            prices = snapshot.get("prices")
        benchmark_frame = data.get("benchmark")
        if prices is not None and not isinstance(prices, pd.DataFrame):
            prices = None
        if prices is not None and "close" in prices.columns:
            close_pivot = close_matrix(prices)
        rebuild = rebuild_from_ledger(
            data.get("ledger"),
            prices,
            benchmark=benchmark_frame,
            initial_capital=capital,
            as_of=as_of_iso,
        )
        if isinstance(benchmark_frame, pd.DataFrame) and not benchmark_frame.empty:
            benchmark_series = benchmark_nav(benchmark_frame)
    except Exception as exc:  # noqa: BLE001 - reported as a check failure
        rebuild_error = f"{type(exc).__name__}: {exc}"

    recomputed: dict[str, Any] = {}
    discrepancies: list[dict[str, Any]] = []
    if rebuild is not None and benchmark_series is not None:
        # The backtest equity curve starts at the first T+1 execution date, while the
        # ledger rebuild covers the full frozen panel (cash before inception).  Metrics
        # must be recomputed on the same inception window; reconciliation and
        # determinism keep using the full rebuild.
        nav_for_metrics = rebuild.nav
        metrics_window_start = str(rebuild.nav.index[0])
        ledger_frame = data.get("ledger")
        if isinstance(ledger_frame, pd.DataFrame) and not ledger_frame.empty:
            try:
                first_execution = str(normalise_ledger(ledger_frame)["date"].min())
            except Exception:  # noqa: BLE001 - malformed ledger is reported by other checks
                first_execution = ""
            if first_execution and first_execution in rebuild.nav.index:
                nav_for_metrics = rebuild.nav.loc[first_execution:]
                metrics_window_start = first_execution
        metrics = compute_performance_metrics(nav_for_metrics, benchmark_series)
        benchmark_metrics = compute_benchmark_metrics(benchmark_series)
        turnover = compute_turnover(data.get("ledger"), rebuild.path)
        declared_metrics = (data.get("metrics") or {}).get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        discrepancies.extend(
            compare_metrics(
                declared_metrics,
                metrics,
                keys=tuple(key for key in declared_metrics if key in metrics),
                rel_tol=tolerances["metrics_relative"],
                artifact="metrics.json",
                section="metrics",
            )
        )
        nav_values = rebuild.nav.astype(float)
        recomputed = {
            "initial_capital": capital,
            "nav_hash": rebuild.nav_hash(),
            "nav_inception": float(nav_values.iloc[0]),
            "nav_start": float(nav_for_metrics.iloc[0]),
            "nav_end": float(nav_values.iloc[-1]),
            "metrics_window_start": metrics_window_start,
            "n_valuation_days": int(len(nav_values)),
            "n_metric_days": int(len(nav_for_metrics)),
            "metrics": metrics,
            "benchmark_metrics": benchmark_metrics,
            "turnover": turnover,
            "rebuild_issues": list(rebuild.issues),
        }
    else:
        recomputed = {"initial_capital": capital, "error": rebuild_error or "benchmark or ledger unavailable"}

    ctx = CheckContext(
        run_id=resolved_run_id,
        as_of=as_of_iso,
        run_dir=root,
        artifacts=paths,
        data=data,
        load_errors=load_errors,
        config=config_bundle,
        gates=gates_config,
        tolerances=tolerances,
        snapshot_dir=resolved_snapshot,
        declared_hashes=_declared_hashes(root, run_context, declared_hashes),
        rebuild=rebuild,
        benchmark_nav=benchmark_series,
        close_matrix=close_pivot,
        recomputed=recomputed,
        discrepancies=discrepancies,
    )

    if rebuild_error:
        ctx.notes.append(f"portfolio rebuild failed: {rebuild_error}")

    checks = run_checks(ctx)
    # The metrics check owns the final discrepancy list (it augments it).
    discrepancies = ctx.discrepancies
    recomputed = ctx.recomputed
    required = required_checks(gates_config)
    blocking = blocking_checks(gates_config)
    statuses = {check["id"]: str(check.get("status", "fail")) for check in checks}
    failed_required = [check_id for check_id in required if statuses.get(check_id) != "pass"]
    failed_blocking = [check_id for check_id in blocking if statuses.get(check_id) != "pass"]
    verdict = "FAIL" if failed_blocking else "PASS"

    snapshot_manifest = load_snapshot(resolved_snapshot).get("manifest") or {}
    snapshot_field: dict[str, Any] = {
        "snapshot_hash": snapshot_manifest.get("snapshot_hash") or run_request.get("snapshot_hash"),
        "as_of": snapshot_manifest.get("as_of"),
    }
    if resolved_snapshot is not None:
        try:
            snapshot_field["path"] = relpath(resolved_snapshot)
        except Exception:  # noqa: BLE001
            snapshot_field["path"] = Path(resolved_snapshot).name

    verification: dict[str, Any] = {
        "verdict": verdict,
        "run_id": resolved_run_id,
        "as_of": as_of_iso,
        "checks": checks,
        "recomputed": json_safe(recomputed),
        "discrepancies": json_safe(discrepancies),
        "artifact_hashes": {key: ctx.artifact_hashes[key] for key in sorted(ctx.artifact_hashes)},
        "tolerances": {key: tolerances[key] for key in sorted(tolerances)},
        "snapshot": snapshot_field,
        "verifier": {
            "name": VERIFIER_NAME,
            "version": VERIFIER_VERSION,
            "mode": "deterministic_recompute",
            "llm_mode": llm_mode,
            "implementation": "aicwf.verify (independent of aicwf.backtest/aicwf.portfolio)",
            "checks_total": len(checks),
            "checks_failed": sum(1 for check in checks if check.get("status") != "pass"),
            "required_checks": required,
            "blocking_checks": blocking,
            "failed_required_checks": failed_required,
            "failed_blocking_checks": failed_blocking,
            "notes": list(ctx.notes),
        },
    }
    return verification


def is_blocking(verification: Mapping[str, Any] | None) -> bool:
    """True when verification should stop the pipeline / trigger a revision."""

    if not verification:
        return True
    return str(verification.get("verdict", "FAIL")).upper() != "PASS"


# ---------------------------------------------------------------------------
# Verifier note (deterministic fallback; agent-generated when available)
# ---------------------------------------------------------------------------


def _escape_table(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_verifier_note(verification: Mapping[str, Any]) -> str:
    """Deterministic Markdown verifier note rendered from the verification dict."""

    verdict = str(verification.get("verdict", "FAIL")).upper()
    checks = list(verification.get("checks") or [])
    failed = [check for check in checks if check.get("status") != "pass"]
    metrics = (verification.get("recomputed") or {}).get("metrics") or {}
    bench_metrics = (verification.get("recomputed") or {}).get("benchmark_metrics") or {}
    discrepancies = list(verification.get("discrepancies") or [])
    verifier = verification.get("verifier") or {}
    lines = [
        f"# Independent verification note - {verdict}",
        "",
        f"- **Run:** `{verification.get('run_id')}` as of **{verification.get('as_of')}**",
        f"- **Verifier:** {verifier.get('name', VERIFIER_NAME)} v{verifier.get('version', VERIFIER_VERSION)} "
        f"({verifier.get('mode', 'deterministic_recompute')}, llm_mode={verifier.get('llm_mode', 'replay')})",
        f"- **Independence:** {verifier.get('implementation', 'aicwf.verify')}",
        f"- **Checks:** {len(checks)} total, {len(failed)} failed",
        "",
        "## Checks",
        "",
        "| id | status | detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {_escape_table(check.get('id'))} | {_escape_table(check.get('status'))} | {_escape_table(check.get('detail'))} |")
    lines.append("")
    lines.append("## Independently recomputed headline metrics")
    lines.append("")
    if metrics:
        lines.append("| metric | recomputed |")
        lines.append("|---|---|")
        for key in (
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "annualized_excess_return",
            "tracking_error",
            "information_ratio",
        ):
            if key in metrics:
                lines.append(f"| {key} | {metrics[key]} |")
    else:
        lines.append("_No metrics could be recomputed (see the failing checks)._")
    lines.append("")
    if bench_metrics:
        lines.append("## Benchmark")
        lines.append("")
        for key in ("annualized_return", "annualized_volatility", "sharpe", "max_drawdown"):
            if key in bench_metrics:
                lines.append(f"- `{key}` = {bench_metrics[key]}")
        lines.append("")
    lines.append("## Discrepancies")
    lines.append("")
    if discrepancies:
        for record in discrepancies[:10]:
            lines.append(f"- {record.get('detail', record)}")
        if len(discrepancies) > 10:
            lines.append(f"- ... and {len(discrepancies) - 10} more")
    else:
        lines.append("_None: every reported headline number was reproduced independently within tolerance._")
    lines.append("")
    lines.append("## Declaration")
    lines.append("")
    lines.append(
        "The verifier recomputed positions, cash, the daily NAV path, benchmark returns and all "
        "headline metrics from `ledger.parquet`, the frozen prices and `benchmark.parquet` without "
        "importing `aicwf.backtest`, `aicwf.portfolio` or `aicwf.factors`. It did not modify any "
        "upstream artifact. The frozen snapshot file hashes, the declared artifact content hashes, "
        "weight/active bounds, industry neutrality, forbidden-security rules, T+1 settlement, and "
        "point-in-time availability were checked independently."
    )
    return "\n".join(lines).rstrip() + "\n"


def _note_generator() -> Callable[[Any, Mapping[str, Any]], Any] | None:
    try:  # local import keeps the module importable before the agent runtime exists
        from aicwf.agents.stages import generate_verifier_note  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return generate_verifier_note


def generate_verifier_note_text(
    verification: Mapping[str, Any],
    ctx: RunContext | Any | None = None,
) -> str:
    """Prefer the read-only verifier agent; fall back to the deterministic renderer."""

    generator = _note_generator()
    if generator is not None and ctx is not None:
        try:
            text = generator(ctx, verification)
            if isinstance(text, str) and text.strip():
                return text.rstrip() + "\n"
        except Exception:  # noqa: BLE001 - the fallback is deterministic and always available
            pass
    return render_verifier_note(verification)


# ---------------------------------------------------------------------------
# RunContext stage entry point
# ---------------------------------------------------------------------------


def run_verification(ctx: RunContext) -> StageOutput:
    """Stage handler for ``verification`` (``09_verification``)."""

    stage_dir = ctx.stage_dir("verification")
    stage_dir.mkdir(parents=True, exist_ok=True)
    config_bundle = dict(ctx.config or {})
    verification = build_verification(
        run_dir=ctx.run_dir,
        as_of=ctx.as_of,
        config=config_bundle,
        run_id=ctx.run_id,
        snapshot_dir=ctx.snapshot_dir,
        llm_mode=getattr(ctx, "llm_mode", "replay"),
        initial_capital=None,
        run_context=ctx,
    )
    verification_path = stage_dir / "verification.json"
    write_json(verification_path, verification, indent=2, sort_keys=True)
    note_path = stage_dir / "verifier_note.md"
    write_text(note_path, generate_verifier_note_text(verification, ctx))

    summary = {
        "verdict": verification["verdict"],
        "checks_total": len(verification["checks"]),
        "checks_failed": verification["verifier"]["checks_failed"],
    }
    try:
        ctx.event("verification_completed", verdict=verification["verdict"], checks_failed=summary["checks_failed"])
    except Exception:  # noqa: BLE001 - events are optional
        pass
    return StageOutput(
        artifacts={
            "verification": relpath(verification_path),
            "verifier_note": relpath(note_path),
        },
        summary=summary,
    )


def verify_error_summary(verification: Mapping[str, Any]) -> str:
    """One-line summary for logs: ``FAIL (2/11 checks)``."""

    verifier = verification.get("verifier") or {}
    return f"{verification.get('verdict')} ({verifier.get('checks_failed', '?')}/{len(verification.get('checks') or [])} checks)"
