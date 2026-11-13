"""Monthly monitoring run (``python -m aicwf monitor``).

Produces ``outputs/monitor/<monitor_id>/`` with

* ``monitor_request.json``   -- what triggered this monitoring run
* ``performance_update.json`` -- realized return / TE / tracking difference since the last rebalance
* ``drift_report.json``      -- factor RankIC drift, exposure drift, data freshness, coverage
* ``monitoring.json``        -- alerts + thresholds + verdict
* ``monitoring_report.md``   -- human-readable note

Everything is deterministic (no wall clock, sorted JSON) and works offline: the
model is only optionally used for prose, never for numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import PROJECT_ROOT, date_str, parse_date, relpath
from aicwf.monitor import drift
from aicwf.util.jsonio import read_json, read_parquet, read_yaml, write_json, write_text

MONITOR_SCHEMA_VERSION = "1.0.0"
DEFAULT_MONITOR_ROOT = PROJECT_ROOT / "outputs" / "monitor"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"

DEFAULT_THRESHOLDS: dict[str, float] = {
    "te_alert": 0.08,
    "te_stop_multiple": 2.0,
    "excess_drawdown_alert": 0.10,
    "data_staleness_days": 3,
    "min_coverage": 0.90,
    "ic_decay_alert": -0.05,
    "industry_active_limit": 0.02,
}


# ---------------------------------------------------------------------------
# Configuration and path resolution
# ---------------------------------------------------------------------------


def load_config_bundle(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load the project config bundle (or normalise an explicit one)."""

    if config is not None:
        return dict(config)
    from aicwf.util.paths import load_config_bundle as _load_bundle

    return _load_bundle()


def monitoring_thresholds(config: Mapping[str, Any], gates: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Thresholds from ``strategy.monitoring`` + the strategy constraints."""

    strategy = (config or {}).get("strategy", {}) or {}
    monitoring = dict(strategy.get("monitoring", {}) or {})
    constraints = ((strategy.get("backtest", {}) or {}).get("constraints", {}) or {})
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in list(thresholds):
        if key in monitoring and monitoring[key] is not None:
            try:
                thresholds[key] = float(monitoring[key])
            except (TypeError, ValueError):
                continue
    if "industry_active_limit" in constraints:
        try:
            thresholds["industry_active_limit"] = float(constraints["industry_active_limit"])
        except (TypeError, ValueError):
            pass
    thresholds["te_stop_level"] = thresholds["te_alert"] * thresholds["te_stop_multiple"]
    thresholds["ic_decay_absolute_alert"] = abs(thresholds["ic_decay_alert"])
    return thresholds


def resolve_snapshot_dir(
    *,
    as_of: str,
    snapshot_dir: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    snapshot_root: Path | str | None = None,
) -> Path | None:
    if snapshot_dir:
        return Path(snapshot_dir)
    declared = (((config or {}).get("strategy", {}) or {}).get("data", {}) or {}).get("snapshot")
    if declared:
        candidate = PROJECT_ROOT / str(declared)
        if candidate.exists():
            return candidate
    root = Path(snapshot_root) if snapshot_root else DEFAULT_SNAPSHOT_ROOT
    candidates: list[tuple[str, Path]] = []
    if root.exists():
        for path in sorted(root.iterdir()):
            manifest = path / "manifest.json"
            if not path.is_dir() or not manifest.exists():
                continue
            try:
                payload = read_json(manifest)
            except Exception:  # noqa: BLE001
                continue
            candidate_as_of = date_str(payload.get("as_of")) if payload.get("as_of") else path.name
            if candidate_as_of <= as_of:
                candidates.append((candidate_as_of, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def _registered_run_ids(history_path: Path | None) -> set[str]:
    """Run ids recorded by promoted releases (the deployed research runs)."""

    if history_path is None or not Path(history_path).exists():
        return set()
    try:
        from aicwf.registry.store import load_history

        history = load_history(history_path, validate=False)
    except Exception:  # noqa: BLE001
        return set()
    run_ids: set[str] = set()
    for entry in history.get("entries") or []:
        if str(entry.get("kind")) == "release" and entry.get("run_id"):
            run_ids.add(str(entry["run_id"]))
    return run_ids


def resolve_run_dir(
    *,
    as_of: str,
    run_dir: Path | str | None = None,
    runs_root: Path | str | None = None,
    registered_run_ids: set[str] | None = None,
) -> Path | None:
    if run_dir:
        return Path(run_dir)
    root = Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT
    if not root.exists():
        return None
    # Only consider runs that actually produced the artifacts monitoring consumes
    # (avoids transient / replay-check run records that may exist without output).
    required_artifacts = (
        ("05_backtest", "metrics.json"),
        ("05_backtest", "equity_curve.csv"),
        ("05_backtest", "holdings.parquet"),
        ("03_factors", "factor_panel.parquet"),
    )
    candidates: list[tuple[str, str, Path]] = []
    for path in root.iterdir():
        request_path = path / "00_intake" / "run_request.json"
        if not path.is_dir() or not request_path.exists():
            continue
        if not all((path / stage / name).exists() for stage, name in required_artifacts):
            continue
        try:
            request = read_json(request_path)
        except Exception:  # noqa: BLE001
            continue
        if not request.get("as_of"):
            continue
        candidate_as_of = date_str(request["as_of"])
        if candidate_as_of <= as_of:
            candidates.append((candidate_as_of, path.name, path))
    if not candidates:
        return None
    # Prefer a run that was actually promoted to the registry (the deployed
    # release), so ad-hoc replay/verification runs cannot hijack monitoring.
    registered = set(registered_run_ids or ())
    if registered:
        promoted = [candidate for candidate in candidates if candidate[1] in registered]
        if promoted:
            candidates = promoted
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def resolve_previous_monitor(
    *,
    out_root: Path,
    as_of: str,
    previous_monitor_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    if previous_monitor_dir:
        request = Path(previous_monitor_dir) / "monitor_request.json"
        if request.exists():
            try:
                return dict(read_json(request))
            except Exception:  # noqa: BLE001
                return None
    if not out_root.exists():
        return None
    candidates: list[tuple[str, str, str, dict[str, Any]]] = []
    for path in sorted(out_root.iterdir()):
        request_path = path / "monitor_request.json"
        if not path.is_dir() or not request_path.exists():
            continue
        try:
            request = read_json(request_path)
        except Exception:  # noqa: BLE001
            continue
        candidate_as_of = request.get("as_of")
        if not candidate_as_of:
            continue
        try:
            candidate_date = date_str(candidate_as_of)
        except Exception:  # noqa: BLE001
            continue
        if candidate_date < as_of:
            candidates.append((candidate_date, str(request.get("monitor_id") or path.name), path.name, request))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[-1][3]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _read_optional(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            return read_parquet(path)
        if path.suffix == ".csv":
            import pandas as pd

            return pd.read_csv(path, dtype={"date": str, "code": str})
        return read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _load_run_artifacts(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"run_id": None, "run_dir": None, "run_request": {}, "equity_curve": None, "holdings": None,
                "metrics": {}, "factor_panel": None, "panel": None}
    run_dir = Path(run_dir)
    request = _read_optional(run_dir / "00_intake" / "run_request.json") or {}
    metrics_doc = _read_optional(run_dir / "05_backtest" / "metrics.json") or {}
    return {
        "run_id": request.get("run_id") or run_dir.name,
        "run_dir": run_dir,
        "run_request": request,
        "equity_curve": _read_optional(run_dir / "05_backtest" / "equity_curve.csv"),
        "holdings": _read_optional(run_dir / "05_backtest" / "holdings.parquet"),
        "metrics": metrics_doc if isinstance(metrics_doc, dict) else {},
        "factor_panel": _read_optional(run_dir / "03_factors" / "factor_panel.parquet"),
        "panel": _read_optional(run_dir / "02_data" / "panel.parquet"),
    }


def _load_snapshot(snapshot_path: Path | None) -> dict[str, Any]:
    if snapshot_path is None:
        return {"dir": None, "manifest": {}, "prices": None, "universe": None, "benchmark": None, "calendar": None}
    root = Path(snapshot_path)
    result: dict[str, Any] = {"dir": root, "manifest": {}, "prices": None, "universe": None, "benchmark": None, "calendar": None}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            payload = read_json(manifest_path)
            result["manifest"] = dict(payload) if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            result["manifest"] = {}
    for key, filename in (
        ("prices", "prices.parquet"),
        ("universe", "universe.parquet"),
        ("benchmark", "benchmark.parquet"),
        ("calendar", "calendar.parquet"),
    ):
        path = root / filename
        if path.exists():
            try:
                result[key] = read_parquet(path)
            except Exception:  # noqa: BLE001
                result[key] = None
    return result


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def _alert(alert_id: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"id": alert_id, "severity": severity, "message": message}
    payload.update(extra)
    return payload


def build_alerts(
    *,
    performance: Mapping[str, Any],
    drift_report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    previous_run_missing: bool,
) -> list[dict[str, Any]]:
    """Map metric readings to alerts using the configured thresholds."""

    alerts: list[dict[str, Any]] = []
    te = performance.get("realized_te")
    if isinstance(te, (int, float)):
        if te > thresholds["te_stop_level"]:
            alerts.append(
                _alert(
                    "tracking_error_stop",
                    "critical",
                    f"realized tracking error {te:.4f} exceeds the stop level "
                    f"{thresholds['te_stop_level']:.4f} ({thresholds['te_stop_multiple']:g}x the alert threshold)",
                    value=float(te),
                    threshold=thresholds["te_stop_level"],
                )
            )
        elif te > thresholds["te_alert"]:
            alerts.append(
                _alert(
                    "tracking_error_alert",
                    "warn",
                    f"realized tracking error {te:.4f} exceeds the alert threshold {thresholds['te_alert']:.4f}",
                    value=float(te),
                    threshold=thresholds["te_alert"],
                )
            )
    drawdown = performance.get("active_drawdown")
    if isinstance(drawdown, (int, float)) and drawdown <= -thresholds["excess_drawdown_alert"]:
        alerts.append(
            _alert(
                "excess_drawdown_alert",
                "warn",
                f"active drawdown {drawdown:.4f} breached -{thresholds['excess_drawdown_alert']:.4f}",
                value=float(drawdown),
                threshold=-thresholds["excess_drawdown_alert"],
            )
        )
    freshness = drift_report.get("data_freshness") or {}
    if freshness.get("severity") == "critical":
        alerts.append(
            _alert(
                "data_missing",
                "critical",
                "no usable frozen price data is available for the monitoring date",
                latest_price_date=freshness.get("latest_price_date"),
            )
        )
    elif freshness.get("stale"):
        alerts.append(
            _alert(
                "data_stale",
                "warn",
                f"frozen data is stale by {freshness.get('trading_days_stale')} trading day(s) "
                f"(limit {freshness.get('staleness_limit_days')})",
                value=freshness.get("trading_days_stale"),
                threshold=freshness.get("staleness_limit_days"),
            )
        )
    coverage = drift_report.get("coverage") or {}
    price_coverage = coverage.get("price_coverage")
    if isinstance(price_coverage, (int, float)) and price_coverage < thresholds["min_coverage"]:
        alerts.append(
            _alert(
                "coverage_low",
                "warn",
                f"price coverage {price_coverage:.3f} is below the minimum {thresholds['min_coverage']:.3f}",
                value=float(price_coverage),
                threshold=thresholds["min_coverage"],
            )
        )
    ic = drift_report.get("factor_ic") or {}
    recent_ic = ic.get("recent_mean")
    decay_value = ic.get("decay")
    if isinstance(recent_ic, (int, float)) and recent_ic < thresholds["ic_decay_alert"]:
        alerts.append(
            _alert(
                "factor_ic_negative",
                "warn",
                f"recent factor RankIC {recent_ic:.4f} is below the alert level {thresholds['ic_decay_alert']:.4f}",
                value=float(recent_ic),
                threshold=thresholds["ic_decay_alert"],
            )
        )
    if isinstance(decay_value, (int, float)) and decay_value > thresholds["ic_decay_absolute_alert"]:
        alerts.append(
            _alert(
                "factor_ic_decay",
                "warn",
                f"factor RankIC decayed by {decay_value:.4f} versus the baseline window "
                f"(limit {thresholds['ic_decay_absolute_alert']:.4f})",
                value=float(decay_value),
                threshold=thresholds["ic_decay_absolute_alert"],
            )
        )
    exposure = drift_report.get("exposure_drift") or {}
    if exposure.get("breach"):
        alerts.append(
            _alert(
                "exposure_drift",
                "warn",
                f"industry active weight breached the +/-{exposure.get('industry_limit'):g} limit "
                f"on {exposure.get('date')}",
                value=exposure.get("max_abs_industry_active"),
                threshold=exposure.get("industry_limit"),
            )
        )
    if previous_run_missing:
        alerts.append(
            _alert(
                "no_previous_run",
                "warn",
                "no completed research run was found; performance attribution uses fallback artifacts",
            )
        )
    return alerts


def _verdict(alerts: list[Mapping[str, Any]]) -> str:
    severities = {str(alert.get("severity")) for alert in alerts}
    if "critical" in severities:
        return "breach"
    if "warn" in severities:
        return "alert"
    return "ok"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_monitoring_report(
    *,
    monitor_id: str,
    as_of: str,
    verdict: str,
    alerts: list[Mapping[str, Any]],
    performance: Mapping[str, Any],
    drift_report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    trigger: Mapping[str, Any],
    run_id: str | None,
    previous_run_id: str | None,
) -> str:
    lines = [
        f"# Monthly monitoring - {monitor_id}",
        "",
        f"- **As of:** {as_of}",
        f"- **Verdict:** {verdict.upper()}",
        f"- **Latest research run:** `{run_id or 'n/a'}` (previous: `{previous_run_id or 'n/a'}`)",
        f"- **Trigger:** {trigger.get('type')} (new_data={trigger.get('new_data')})",
        f"- **Snapshot hash:** `{str(trigger.get('snapshot_hash'))[:16]}...`",
        "",
        "## Alerts",
        "",
    ]
    if alerts:
        lines.append("| id | severity | message |")
        lines.append("|---|---|---|")
        for alert in alerts:
            lines.append(f"| {alert.get('id')} | {alert.get('severity')} | {alert.get('message')} |")
    else:
        lines.append("_No alerts: all monitored metrics are inside their thresholds._")
    lines.extend(
        [
            "",
            "## Realized performance since the last rebalance",
            "",
            f"- Window: {performance.get('window_start')} -> {performance.get('window_end')} "
            f"({performance.get('n_days')} trading days, method={performance.get('method')})",
            f"- Portfolio return: {_fmt(performance.get('portfolio_return'))}",
            f"- Benchmark return: {_fmt(performance.get('benchmark_return'))}",
            f"- Cumulative excess (tracking difference): {_fmt(performance.get('tracking_difference'))}",
            f"- Realized tracking error (annualized): {_fmt(performance.get('realized_te'))}",
            f"- Realized active drawdown: {_fmt(performance.get('active_drawdown'))}",
            "",
            "## Factor RankIC drift",
            "",
        ]
    )
    factor_ic = drift_report.get("factor_ic") or {}
    lines.append(f"- Method: {factor_ic.get('method')}")
    lines.append(f"- Recent mean RankIC: {_fmt(factor_ic.get('recent_mean'))}")
    lines.append(f"- Baseline mean RankIC: {_fmt(factor_ic.get('baseline_mean'))}")
    lines.append(f"- Decay: {_fmt(factor_ic.get('decay'))}")
    lines.append(f"- Observations: {factor_ic.get('n_periods')}")
    if factor_ic.get("warning"):
        lines.append(f"- Warning: {factor_ic['warning']}")
    exposure = drift_report.get("exposure_drift") or {}
    lines.extend(
        [
            "",
            "## Exposures",
            "",
            f"- Latest holdings date: {exposure.get('date')}",
            f"- Names held: {exposure.get('n_names')}",
            f"- Max |name active weight|: {_fmt(exposure.get('max_abs_name_active'))}",
            f"- Max |industry active weight|: {_fmt(exposure.get('max_abs_industry_active'))} "
            f"(limit {_fmt(exposure.get('industry_limit'))})",
        ]
    )
    freshness = drift_report.get("data_freshness") or {}
    coverage_report = drift_report.get("coverage") or {}
    lines.extend(
        [
            "",
            "## Data freshness and coverage",
            "",
            f"- Latest frozen price date: {freshness.get('latest_price_date')}",
            f"- Trading days stale: {freshness.get('trading_days_stale')} "
            f"(limit {freshness.get('staleness_limit_days')})",
            f"- Price coverage: {_fmt(coverage_report.get('price_coverage'), 3)}",
            f"- Factor coverage: {_fmt(coverage_report.get('factor_coverage'), 3)}",
            "",
            "## Thresholds",
            "",
        ]
    )
    for key in sorted(thresholds):
        lines.append(f"- `{key}` = {thresholds[key]}")
    lines.append("")
    lines.append(
        "This note was generated deterministically from the frozen artifacts; no model output is used for numbers."
    )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_monitor(
    as_of: Any,
    llm_mode: str = "replay",
    monitor_id: str | None = None,
    *,
    run_dir: Path | str | None = None,
    previous_run_id: str | None = None,
    snapshot_dir: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
    out_root: Path | str | None = None,
    runs_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    previous_monitor_dir: Path | str | None = None,
    trigger: str | None = None,
    use_agent_text: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Run one monitoring cycle and write the monitoring artifact set."""

    as_of_iso = date_str(as_of)
    config_bundle = load_config_bundle(config)
    gates_config = dict(gates or config_bundle.get("gates") or {})
    thresholds = monitoring_thresholds(config_bundle, gates_config)
    root_out = Path(out_root) if out_root else DEFAULT_MONITOR_ROOT
    resolved_monitor_id = monitor_id or f"monitor-{as_of_iso}"
    out_dir = root_out / resolved_monitor_id

    snapshot_path = resolve_snapshot_dir(
        as_of=as_of_iso,
        snapshot_dir=snapshot_dir,
        config=config_bundle,
        snapshot_root=snapshot_root,
    )
    snapshot = _load_snapshot(snapshot_path)
    previous_monitor = resolve_previous_monitor(
        out_root=root_out, as_of=as_of_iso, previous_monitor_dir=previous_monitor_dir
    )
    previous_snapshot_hash = (previous_monitor or {}).get("snapshot_hash")
    snapshot_manifest = snapshot.get("manifest") or {}
    snapshot_hash = snapshot_manifest.get("snapshot_hash")
    trigger_report = drift.detect_new_data(snapshot_hash, previous_snapshot_hash)
    if trigger in {"new_data", "scheduled"}:
        trigger_report["type"] = trigger
        trigger_report["new_data"] = trigger == "new_data"

    # Registry-promoted runs are preferred so ad-hoc replay/verification runs do not
    # hijack the monthly monitoring attribution.
    registry_cfg: Mapping[str, Any] = {}
    for candidate in (
        (gates_config.get("registry") if isinstance(gates_config.get("registry"), Mapping) else None),
        (
            ((gates_config.get("change_control") or {}).get("registry"))
            if isinstance(gates_config.get("change_control"), Mapping)
            and isinstance((gates_config.get("change_control") or {}).get("registry"), Mapping)
            else None
        ),
    ):
        if candidate:
            registry_cfg = candidate
            break
    history_value = registry_cfg.get("versions_file") if registry_cfg else None
    history_path = None
    if history_value:
        history_path = Path(str(history_value))
        if not history_path.is_absolute():
            history_path = PROJECT_ROOT / history_path
    resolved_run = resolve_run_dir(
        as_of=as_of_iso,
        run_dir=run_dir,
        runs_root=runs_root,
        registered_run_ids=_registered_run_ids(history_path),
    )
    artifacts = _load_run_artifacts(resolved_run)
    previous_run_missing = artifacts.get("run_id") is None
    previous_run_id = previous_run_id or (previous_monitor or {}).get("run_id") or artifacts.get("run_id")

    initial_capital = None
    metrics_doc = artifacts.get("metrics") or {}
    if isinstance(metrics_doc, dict):
        initial_capital = metrics_doc.get("initial_capital") or (metrics_doc.get("metrics") or {}).get(
            "initial_capital"
        )
    capital = float(initial_capital) if initial_capital else 1.0

    prices_for_mark = snapshot.get("prices") if snapshot.get("prices") is not None else artifacts.get("panel")
    benchmark_for_mark = snapshot.get("benchmark") if snapshot.get("benchmark") is not None else None
    performance = drift.mark_to_market_performance(
        holdings=artifacts.get("holdings"),
        equity_curve=artifacts.get("equity_curve"),
        prices=prices_for_mark,
        benchmark=benchmark_for_mark,
        as_of=as_of_iso,
        initial_capital=capital,
    )
    factor_panel = artifacts.get("factor_panel")
    prices_for_ic = artifacts.get("panel") if artifacts.get("panel") is not None else prices_for_mark
    factor_ic = drift.factor_rank_ic(factor_panel, prices_for_ic, as_of=as_of_iso)
    exposure = drift.exposure_drift(
        artifacts.get("holdings"),
        as_of=as_of_iso,
        industry_limit=thresholds["industry_active_limit"],
    )
    freshness = drift.data_freshness(
        as_of=as_of_iso,
        snapshot_prices=snapshot.get("prices"),
        snapshot_manifest=snapshot_manifest,
        staleness_limit_days=int(thresholds["data_staleness_days"]),
        calendar=snapshot.get("calendar"),
    )
    coverage_report = drift.coverage(
        as_of=as_of_iso,
        panel=artifacts.get("panel"),
        snapshot_universe=snapshot.get("universe"),
        factor_panel=factor_panel,
    )
    coverage_report["min_coverage"] = thresholds["min_coverage"]

    drift_report = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "as_of": as_of_iso,
        "run_id": artifacts.get("run_id"),
        "snapshot_hash": snapshot_hash,
        "factor_ic": factor_ic,
        "exposure_drift": exposure,
        "data_freshness": freshness,
        "coverage": coverage_report,
    }
    performance_update = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "as_of": as_of_iso,
        "run_id": artifacts.get("run_id"),
        "previous_run_id": previous_run_id,
        "initial_capital": capital,
        "performance": performance,
    }
    alerts = build_alerts(
        performance=performance,
        drift_report=drift_report,
        thresholds=thresholds,
        previous_run_missing=previous_run_missing,
    )
    verdict = _verdict(alerts)

    request = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "monitor_id": resolved_monitor_id,
        "as_of": as_of_iso,
        "previous_run_id": previous_run_id,
        "run_id": artifacts.get("run_id"),
        "snapshot_hash": snapshot_hash,
        "previous_snapshot_hash": previous_snapshot_hash,
        "trigger": trigger_report,
        "llm_mode": llm_mode,
    }
    monitoring = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "monitor_id": resolved_monitor_id,
        "as_of": as_of_iso,
        "verdict": verdict,
        "alerts": alerts,
        "thresholds": {key: thresholds[key] for key in sorted(thresholds)},
        "trigger": trigger_report,
        "snapshot_hash": snapshot_hash,
        "previous_snapshot_hash": previous_snapshot_hash,
        "run_id": artifacts.get("run_id"),
        "counts": {
            "alerts": len(alerts),
            "critical": sum(1 for alert in alerts if alert.get("severity") == "critical"),
            "warn": sum(1 for alert in alerts if alert.get("severity") == "warn"),
        },
    }
    report_text = render_monitoring_report(
        monitor_id=resolved_monitor_id,
        as_of=as_of_iso,
        verdict=verdict,
        alerts=alerts,
        performance=performance,
        drift_report=drift_report,
        thresholds=thresholds,
        trigger=trigger_report,
        run_id=artifacts.get("run_id"),
        previous_run_id=previous_run_id,
    )
    result: dict[str, Any] = {
        "monitor_id": resolved_monitor_id,
        "as_of": as_of_iso,
        "verdict": verdict,
        "alerts": alerts,
        "thresholds": thresholds,
        "trigger": trigger_report,
        "snapshot_hash": snapshot_hash,
        "previous_snapshot_hash": previous_snapshot_hash,
        "run_id": artifacts.get("run_id"),
        "out_dir": str(out_dir),
        "paths": {
            "monitor_request": str(out_dir / "monitor_request.json"),
            "performance_update": str(out_dir / "performance_update.json"),
            "drift_report": str(out_dir / "drift_report.json"),
            "monitoring": str(out_dir / "monitoring.json"),
            "monitoring_report": str(out_dir / "monitoring_report.md"),
        },
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "monitor_request.json", request, indent=2, sort_keys=True)
        write_json(out_dir / "performance_update.json", performance_update, indent=2, sort_keys=True)
        write_json(out_dir / "drift_report.json", drift_report, indent=2, sort_keys=True)
        write_json(out_dir / "monitoring.json", monitoring, indent=2, sort_keys=True)
        write_text(out_dir / "monitoring_report.md", report_text)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aicwf.monitor", description="Monthly monitoring run")
    parser.add_argument("--as-of", required=True, help="monitoring date (ISO YYYY-MM-DD)")
    parser.add_argument("--run-dir", default=None, help="latest research run directory (default: newest run <= as-of)")
    parser.add_argument("--snapshot", default=None, help="frozen snapshot directory override")
    parser.add_argument("--monitor-id", default=None, help="monitor id (default: monitor-<as-of>)")
    parser.add_argument("--out-root", default=None, help="monitor output root (default: outputs/monitor)")
    parser.add_argument("--llm-mode", default="replay", choices=("replay", "null", "ollama", "openai_compat"))
    parser.add_argument("--trigger", choices=("auto", "new_data", "scheduled"), default="auto")
    parser.add_argument("--json", action="store_true", help="print the monitoring summary as JSON")
    args = parser.parse_args(argv)
    try:
        result = run_monitor(
            args.as_of,
            llm_mode=args.llm_mode,
            monitor_id=args.monitor_id,
            run_dir=args.run_dir,
            snapshot_dir=args.snapshot,
            out_root=args.out_root,
            trigger=None if args.trigger == "auto" else args.trigger,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"monitor failed: {type(exc).__name__}: {exc}\n")
        return 2
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    else:
        sys.stdout.write(
            f"monitor {result['monitor_id']} ({result['as_of']}): verdict={result['verdict'].upper()} "
            f"alerts={len(result['alerts'])} -> {result['out_dir']}\n"
        )
        for alert in result["alerts"]:
            sys.stdout.write(f"  [{alert['severity']}] {alert['id']}: {alert['message']}\n")
    return 0 if result["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
