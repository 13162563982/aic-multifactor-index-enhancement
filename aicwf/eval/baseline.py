"""Frozen evaluation baselines for change control.

A baseline is the immutable reference point an A/B evaluation compares a
candidate release against.  It pins the data snapshot, the configuration, the
prompt hashes and the expected headline metrics, plus the soft tolerances from
``config/gates.yaml: change_control.metric_acceptance``.

``python -m aicwf.eval --baseline-refresh --run <run_dir>`` regenerates the
baseline from a completed, verified run.  The Team Lead runs that once after
integration to replace the fixture baseline with the real-data baseline; the
mechanism itself is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import PROJECT_ROOT, date_str, relpath
from aicwf.util.artifact_hash import hash_json_artifact
from aicwf.util.hashing import hash_text
from aicwf.util.jsonio import read_json, read_yaml, write_json
from aicwf.util.timeutil import utc_now_iso

BASELINE_SCHEMA_VERSION = "1.0.0"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline_v1.0.0.json"
DEFAULT_GATES_PATH = PROJECT_ROOT / "config" / "gates.yaml"

#: Headline metrics compared against a baseline (soft change-control gates).
SOFT_METRIC_KEYS: tuple[str, ...] = (
    "annualized_excess_return",
    "information_ratio",
    "tracking_error",
    "max_drawdown",
    "turnover",
)
#: Additional metrics recorded in the decision record for context only.
INFO_METRIC_KEYS: tuple[str, ...] = (
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "cumulative_return",
    "benchmark_cumulative_return",
)

#: Tolerance semantics: how a candidate may differ from the baseline.
TOLERANCE_SEMANTICS: dict[str, dict[str, str]] = {
    "annualized_excess_return": {
        "tolerance": "annualized_excess_return_max_degradation",
        "direction": "higher_better",
        "description": "annualized excess return may not degrade by more than this",
    },
    "information_ratio": {
        "tolerance": "information_ratio_max_degradation",
        "direction": "higher_better",
        "description": "information ratio may not degrade by more than this",
    },
    "tracking_error": {
        "tolerance": "tracking_error_abs_tolerance",
        "direction": "absolute",
        "description": "tracking error must stay within this absolute band",
    },
    "max_drawdown": {
        "tolerance": "max_drawdown_max_degradation",
        "direction": "higher_better",
        "description": "max drawdown (negative) may not degrade by more than this",
    },
    "turnover": {
        "tolerance": "turnover_max_increase",
        "direction": "lower_better",
        "description": "turnover may not increase by more than this",
    },
}


def load_gates(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_GATES_PATH
    if not path.exists():
        return {}
    return dict(read_yaml(path) or {})


def metric_acceptance_tolerances(gates: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Soft tolerances from ``gates.change_control.metric_acceptance``."""

    change_control = (gates or {}).get("change_control", {}) or {}
    values = change_control.get("metric_acceptance", {}) or {}
    defaults = {
        "annualized_excess_return_max_degradation": 0.005,
        "information_ratio_max_degradation": 0.10,
        "tracking_error_abs_tolerance": 0.01,
        "max_drawdown_max_degradation": 0.02,
        "turnover_max_increase": 0.10,
    }
    tolerances: dict[str, float] = {}
    for key, default in defaults.items():
        try:
            tolerances[key] = float(values.get(key, default))
        except (TypeError, ValueError):
            tolerances[key] = default
    return tolerances


def hard_gates(gates: Mapping[str, Any] | None = None) -> list[str]:
    configured = ((gates or {}).get("change_control", {}) or {}).get("hard_gates")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return ["leakage_tests_pass", "verification_passes", "deterministic_replay", "budget_within_limit"]


def prompt_hashes(prompt_dir: Path | str | None) -> dict[str, str]:
    """Content hashes of every prompt file in a directory tree (posix relative paths)."""

    if not prompt_dir:
        return {}
    root = Path(prompt_dir)
    if not root.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}:
            hashes[path.relative_to(root).as_posix()] = hash_text(path.read_text(encoding="utf-8"))
    return hashes


def _run_config(run_dir: Path) -> dict[str, Any]:
    bundle_path = run_dir / "config_bundle.json"
    if bundle_path.exists():
        return dict(read_json(bundle_path))
    config: dict[str, Any] = {}
    for name in ("workflow", "strategy", "gates", "budgets", "model", "data"):
        path = PROJECT_ROOT / "config" / f"{name}.yaml"
        if path.exists():
            config[name] = read_yaml(path) or {}
    return config


def _read_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _select_metrics(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics[key] for key in keys if key in metrics and metrics[key] is not None}


def build_baseline_from_run(
    run_dir: Path | str,
    *,
    baseline_id: str = "baseline_v1.0.0",
    version: str | None = None,
    gates: Mapping[str, Any] | None = None,
    prompt_dir: Path | str | None = None,
    data_scope: str | None = None,
    evidence_mode: str | None = None,
    notes: list[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a baseline document from a completed run directory."""

    root = Path(run_dir)
    run_request = _read_optional(root / "00_intake" / "run_request.json") or {}
    metrics_doc = _read_optional(root / "05_backtest" / "metrics.json") or {}
    verification = _read_optional(root / "09_verification" / "verification.json")
    config = _run_config(root)
    gates = gates if gates is not None else (config.get("gates") or load_gates())
    metrics = dict(metrics_doc.get("metrics") or {})
    benchmark_metrics = dict(metrics_doc.get("benchmark_metrics") or {})
    if not metrics:
        raise ValueError(f"run {root} has no metrics.json; cannot freeze a baseline")
    prompt_dir_hashes = prompt_hashes(prompt_dir)
    snapshot_hash = run_request.get("snapshot_hash")
    if not snapshot_hash:
        snapshot_manifest = _read_optional(root.parent / "snapshot" / "manifest.json")
        if isinstance(snapshot_manifest, dict):
            snapshot_hash = snapshot_manifest.get("snapshot_hash")
    as_of = run_request.get("as_of")
    if not as_of:
        raise ValueError(f"run {root} has no as_of in run_request.json")
    resolved_version = version or run_request.get("candidate_version") or "v1.0.0"
    verification_verdict = (verification or {}).get("verdict")
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "version": str(resolved_version),
        "created_at_utc": created_at_utc or utc_now_iso(),
        "status": "active",
        "data_scope": data_scope or run_request.get("variant") or "unknown",
        "evidence_mode": evidence_mode or ("fixture" if run_request.get("variant") else "real"),
        "frozen": {
            "run_id": run_request.get("run_id"),
            "as_of": date_str(as_of),
            "snapshot_hash": snapshot_hash,
            "config_hash": run_request.get("config_hash") or metrics_doc.get("config_hash"),
            "prompt_hashes": prompt_dir_hashes,
            "llm_mode": run_request.get("llm_mode", "replay"),
            "seed": run_request.get("seed", 0),
            "workflow_version": run_request.get("workflow_version", "1.0.0"),
            "initial_capital": metrics_doc.get("initial_capital")
            or (metrics_doc.get("metrics") or {}).get("initial_capital"),
        },
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "headline_metrics": _select_metrics(metrics, SOFT_METRIC_KEYS + INFO_METRIC_KEYS),
        "verification": {
            "verdict": verification_verdict,
            "checks_failed": ((verification or {}).get("verifier") or {}).get("checks_failed"),
        },
        "tolerances": {
            **metric_acceptance_tolerances(gates),
            "tolerance_semantics": TOLERANCE_SEMANTICS,
        },
        "hard_gates": hard_gates(gates),
        "config": config,
        "notes": list(notes or []),
        "source": {
            "run_dir": relpath(root),
            "metrics_artifact_hash": None,
        },
    }
    metrics_path = root / "05_backtest" / "metrics.json"
    if metrics_path.exists():
        baseline["source"]["metrics_artifact_hash"] = hash_json_artifact(metrics_path)
    return baseline


def refresh_baseline(
    run_dir: Path | str,
    *,
    out_path: Path | str = DEFAULT_BASELINE_PATH,
    baseline_id: str = "baseline_v1.0.0",
    version: str | None = None,
    gates: Mapping[str, Any] | None = None,
    prompt_dir: Path | str | None = None,
    data_scope: str | None = None,
    evidence_mode: str | None = None,
    notes: list[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Freeze/refresh a baseline JSON file from a completed run."""

    baseline = build_baseline_from_run(
        run_dir,
        baseline_id=baseline_id,
        version=version,
        gates=gates,
        prompt_dir=prompt_dir,
        data_scope=data_scope,
        evidence_mode=evidence_mode,
        notes=notes,
        created_at_utc=created_at_utc,
    )
    write_json(Path(out_path), baseline, indent=2, sort_keys=True)
    return baseline


def load_baseline(path: Path | str = DEFAULT_BASELINE_PATH) -> dict[str, Any]:
    """Load and lightly validate a frozen baseline document."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"eval baseline not found: {path}; freeze one with "
            "`python -m aicwf.eval --baseline-refresh --run <run_dir>`"
        )
    baseline = read_json(path)
    if not isinstance(baseline, dict):
        raise ValueError(f"baseline {path} is not a JSON object")
    required = ("baseline_id", "version", "frozen", "metrics")
    missing = [key for key in required if key not in baseline]
    if missing:
        raise ValueError(f"baseline {path} is missing key(s): {', '.join(missing)}")
    baseline.setdefault("schema_version", BASELINE_SCHEMA_VERSION)
    baseline.setdefault("tolerances", metric_acceptance_tolerances())
    baseline.setdefault("hard_gates", hard_gates())
    baseline["_path"] = str(path)
    return baseline


def baseline_metrics(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Numeric headline metrics a candidate is compared against."""

    metrics = dict(baseline.get("metrics") or {})
    headline = dict(baseline.get("headline_metrics") or {})
    merged = {**metrics, **headline}
    return {key: value for key, value in merged.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def baseline_summary(baseline: Mapping[str, Any]) -> str:
    """One-line human summary of a baseline (used by CLI output)."""

    frozen = baseline.get("frozen") or {}
    metrics = baseline_metrics(baseline)
    return (
        f"{baseline.get('baseline_id')} version={baseline.get('version')} "
        f"as_of={frozen.get('as_of')} snapshot={str(frozen.get('snapshot_hash'))[:12]}... "
        f"excess={metrics.get('annualized_excess_return')} te={metrics.get('tracking_error')}"
    )


def baseline_config_hash(baseline: Mapping[str, Any]) -> str | None:
    frozen = baseline.get("frozen") or {}
    return frozen.get("config_hash")


def baseline_json_sha(path: Path | str) -> str:
    """Canonical content hash of the baseline file (recorded in decision records)."""

    import hashlib

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
