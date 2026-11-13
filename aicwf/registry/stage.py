"""Registry stage entry point (``12_registry``).

The controller executes this stage only after the ``approval`` stage has
recorded a human decision.  It promotes the verified run into the append-only
version registry, writes ``12_registry/version_entry.json`` and returns the
``{version, decision}`` digest for the run summary.

Public API used by the pipeline/CLI:
    run_registry(ctx) -> StageOutput
    promote(...) / promote_run(...) / rollback(...) / rollback_run(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import (
    PROJECT_ROOT,
    ApprovalRequired,
    ConfigError,
    RunContext,
    StageOutput,
    ValidationError,
    relpath,
)
from aicwf.registry.promote import collect_declared_hashes, promote
from aicwf.registry.store import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_RELEASES_DIR,
    history_summary,
    load_history,
    validate_history,
    verify_bundle,
)
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.jsonio import read_json, write_json


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def load_registry_paths(
    config: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Resolve ``(history_path, releases_dir)`` from config/gates.yaml."""

    resolved_gates = dict(gates or (config or {}).get("gates") or {})
    registry_cfg = resolved_gates.get("registry")
    if not isinstance(registry_cfg, Mapping):
        # config/gates.yaml nests these under change_control.registry
        registry_cfg = ((resolved_gates.get("change_control") or {}).get("registry") or {})
    registry_cfg = dict(registry_cfg or {})
    versions_file = Path(str(registry_cfg.get("versions_file") or DEFAULT_HISTORY_PATH))
    releases_path = Path(str(registry_cfg.get("releases_dir") or DEFAULT_RELEASES_DIR))
    if not versions_file.is_absolute():
        versions_file = PROJECT_ROOT / versions_file
    if not releases_path.is_absolute():
        releases_path = PROJECT_ROOT / releases_path
    return versions_file, releases_path


def _resolve_artifact_path(ctx: RunContext, stage: str, logical: str, run_relative: str) -> Path | None:
    try:
        return ctx.upstream_artifact(stage, logical)
    except Exception:  # noqa: BLE001 - fall back to the contract path
        candidate = Path(ctx.run_dir) / run_relative
        return candidate if candidate.exists() else None


def resolve_approval_artifact(ctx: RunContext) -> Path | None:
    """Find the approval artifact the human gate recorded."""

    path = _resolve_artifact_path(ctx, "approval", "approval", "11_approval/approval.json")
    if path is not None and path.exists():
        return path
    fallback = Path(ctx.run_dir) / "11_approval" / "approval.json"
    return fallback if fallback.exists() else None


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _declared_hashes_for_run(ctx: RunContext, approval: Mapping[str, Any]) -> dict[str, str]:
    """Recompute content hashes for every declared artifact and every approved hash.

    Artifacts are indexed under their declared project-relative path and under the
    ``<stage>:<logical>`` key form used by ``aicwf/pipeline.py``'s approval
    handler, so the approval binding can be checked with either spelling.
    """

    declared: dict[str, str] = {}
    for stage, output in (ctx.stage_outputs or {}).items():
        for logical, relative in output.artifacts.items():
            path = Path(str(relative))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not (path.exists() and path.is_file()):
                continue
            digest = hash_artifact(path)
            declared[str(relative)] = digest
            declared[f"{stage}:{logical}"] = digest
            try:
                declared[path.relative_to(PROJECT_ROOT).as_posix()] = digest
            except ValueError:
                pass
    approved_keys = list((approval.get("artifact_hashes") or {}).keys())
    resolved = collect_declared_hashes(ctx.run_dir, keys=approved_keys, project_root=PROJECT_ROOT)
    for key, value in resolved.items():
        declared.setdefault(key, value)
        if ":" in key:
            continue
        # Also keep the run-relative spelling so either form matches.
        try:
            relative = Path(key).relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = None
        if relative:
            declared.setdefault(relative, value)
    return declared


def _next_version_after(latest: Mapping[str, Any] | None) -> str | None:
    """Suggest the next monthly release version (latest vX.Y.Z -> vX.(Y+1).0)."""

    if not latest or not latest.get("version"):
        return None
    version = str(latest["version"]).strip()
    import re

    match = re.fullmatch(r"v(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    return f"v{major}.{minor + 1}.0"


def resolve_candidate_version(
    config: Mapping[str, Any] | None,
    approval: Mapping[str, Any] | None,
    run_request: Mapping[str, Any] | None,
    *,
    history_path: Path | str | None = None,
) -> str:
    """Determine the version to promote.

    Explicit configuration wins (``config['registry']['candidate_version']`` or
    the gates registry block), then the approval artifact, then the run request.
    If none is set the registry auto-selects the next monthly version (or
    ``v1.0.0`` for the first release) so the pipeline needs no extra wiring.
    """

    gates_config = (config or {}).get("gates") or {}
    change_control_registry = (gates_config.get("change_control") or {}).get("registry") or {}
    candidates = [
        ((config or {}).get("registry") or {}).get("candidate_version"),
        (gates_config.get("registry") or {}).get("candidate_version"),
        (approval or {}).get("version"),
        (run_request or {}).get("candidate_version"),
        change_control_registry.get("candidate_version"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    try:
        from aicwf.registry.store import latest_release

        latest = latest_release(history_path) if history_path is not None else latest_release()
    except Exception:  # noqa: BLE001 - empty/corrupt history: fall back to the first version
        latest = None
    suggested = _next_version_after(latest) or "v1.0.0"
    return suggested


# ---------------------------------------------------------------------------
# Stage handler
# ---------------------------------------------------------------------------


def run_registry(ctx: RunContext) -> StageOutput:
    """Promote the approved, verified run and write the version entry."""

    run_dir = Path(ctx.run_dir)
    config = dict(ctx.config or {})
    gates = dict(config.get("gates") or {})
    history_path, releases_path = load_registry_paths(config, gates)

    approval_path = resolve_approval_artifact(ctx)
    if approval_path is None:
        raise ApprovalRequired(
            f"missing approval artifact for run {ctx.run_id}; promotion requires a human-approved decision"
        )
    approval = _read_optional(approval_path)
    if not approval:
        raise ApprovalRequired(f"approval artifact {approval_path} is unreadable")
    run_request = _read_optional(run_dir / "00_intake" / "run_request.json") or {}
    version = resolve_candidate_version(config, approval, run_request, history_path=history_path)

    metrics_doc = _read_optional(run_dir / "05_backtest" / "metrics.json") or {}
    verification = _read_optional(run_dir / "09_verification" / "verification.json")
    if verification is None:
        try:
            summary = ctx.upstream_summary("verification")
        except Exception:  # noqa: BLE001
            summary = {}
        verification = {"verdict": summary.get("verdict"), "verifier": {"checks_failed": summary.get("checks_failed")}}
    verdict = str((verification or {}).get("verdict") or "").upper()
    if verdict != "PASS":
        raise ValidationError(
            f"refusing to promote run {ctx.run_id}: verification verdict is {verdict or 'unknown'}, expected PASS"
        )
    declared = _declared_hashes_for_run(ctx, approval)
    entry = promote(
        version=version,
        run_dir=run_dir,
        approval=approval,
        artifact_hashes=declared,
        metrics_doc=metrics_doc,
        verification=verification,
        config=config,
        gates=gates,
        history_path=history_path,
        releases=releases_path,
        rationale=approval.get("note")
        or f"Promoted by {approval.get('approver', 'unknown approver')} after PASS verification.",
    )
    stage_dir = ctx.stage_dir("registry")
    stage_dir.mkdir(parents=True, exist_ok=True)
    entry_path = ctx.artifact_path("registry", "version_entry.json")
    write_json(entry_path, entry, indent=2, sort_keys=True)
    try:
        ctx.event("version_promoted", version=entry.get("version"), decision=entry.get("decision"))
    except Exception:  # noqa: BLE001
        pass
    return StageOutput(
        artifacts={"version_entry": relpath(entry_path)},
        summary={"version": entry.get("version"), "decision": entry.get("decision")},
    )


# ---------------------------------------------------------------------------
# Read-only helpers used by CLI/doctor/tests
# ---------------------------------------------------------------------------


def registry_status(
    *,
    history_path: Path | str | None = None,
    releases: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the history chain and every release bundle; returns a status report."""

    path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
    releases_path = Path(releases) if releases else DEFAULT_RELEASES_DIR
    report: dict[str, Any] = {
        "history_path": path.as_posix(),
        "history_exists": path.exists(),
        "entries": 0,
        "valid_chain": False,
        "releases": [],
        "problems": [],
    }
    if not path.exists():
        report["problems"].append("registry history file does not exist")
        return report
    try:
        history = load_history(path)
        validate_history(history)
        report["valid_chain"] = True
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"history integrity: {type(exc).__name__}: {exc}")
        return report
    entries = history.get("entries") or []
    report["entries"] = len(entries)
    for entry in entries:
        if str(entry.get("kind")) != "release":
            continue
        version = str(entry.get("version"))
        bundle = releases_path / version
        problems = verify_bundle(bundle)
        report["releases"].append(
            {
                "version": version,
                "bundle": bundle.as_posix(),
                "bundle_intact": not problems,
                "problems": problems,
                "entry_hash": str(entry.get("entry_hash", ""))[:12],
            }
        )
        report["problems"].extend(f"{version}: {problem}" for problem in problems)
    report["versions"] = history_summary(path)
    return report
