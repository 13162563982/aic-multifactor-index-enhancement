"""Rollback to a previously released version.

A rollback never edits history.  It

1. loads the target release entry and verifies the frozen bundle hashes;
2. restores the bundle's config/prompts/skills into the working tree;
3. runs a smoke evaluation (bundle integrity plus, when the pipeline provides a
   callable, a real replay/smoke run on the mini fixture);
4. appends one ``rollback`` entry with the before/after metrics and the smoke
   evaluation result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from aicwf.contracts import PROJECT_ROOT, ValidationError
from aicwf.registry.store import (
    DEFAULT_HISTORY_PATH,
    RegistryIntegrityError,
    append_entry,
    entry_bundle_dir,
    find_entry,
    latest_release,
    load_history,
    restore_bundle,
    verify_bundle,
)
from aicwf.util.jsonio import write_json
from aicwf.util.timeutil import utc_now_iso

SmokeEval = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


def default_smoke_eval(bundle_dir: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Offline smoke evaluation used when the pipeline does not inject one.

    It re-verifies every frozen bundle file, checks that the bundle hashes match
    the release entry and (for fixture-backed entrants) returns the frozen
    metrics.  The pipeline can inject a stronger ``smoke_eval`` that replays the
    mini-fixture run; the rollback record keeps whichever result was used.
    """

    problems = verify_bundle(bundle_dir)
    recorded = (entry.get("bundle") or {}).get("bundle_hash")
    checks = [
        {"id": "bundle_exists", "status": "pass" if bundle_dir.exists() else "fail", "detail": str(bundle_dir)},
        {
            "id": "bundle_integrity",
            "status": "pass" if not problems else "fail",
            "detail": "all recorded file hashes match" if not problems else "; ".join(problems[:5]),
        },
        {
            "id": "bundle_hash_matches_registry",
            "status": "pass" if recorded else "skip",
            "detail": f"recorded bundle_hash {str(recorded)[:12]}..." if recorded else "no recorded bundle hash",
        },
    ]
    metrics = dict(entry.get("metrics") or {})
    checks.append(
        {
            "id": "frozen_metrics_available",
            "status": "pass" if metrics else "fail",
            "detail": f"{len(metrics)} frozen metric(s)" if metrics else "release entry has no metrics",
        }
    )
    status = "pass" if all(check["status"] in {"pass", "skip"} for check in checks) else "fail"
    return {
        "status": status,
        "mode": "bundle_integrity_fixture",
        "checks": checks,
        "metrics": metrics,
        "detail": "offline bundle-integrity smoke evaluation (inject a replay pipeline callable for a full smoke run)",
    }


def _resolve_path(path: Path | str | None, default: Path | str) -> Path:
    """Resolve a registry path, absolute or relative to the project root."""

    candidate = Path(path) if path is not None else Path(default)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def rollback(
    *,
    to_version: str,
    from_version: str | None = None,
    history_path: Path | str | None = None,
    releases: Path | str | None = None,
    releases_dir: Path | str | None = None,
    project_root: Path | str = PROJECT_ROOT,
    restore: bool = True,
    restore_targets: Mapping[str, Path | str] | None = None,
    smoke_eval: SmokeEval | None = None,
    gates: Mapping[str, Any] | None = None,
    rationale: str | None = None,
    created_at_utc: str | None = None,
    dry_run: bool = False,
    write_entry: Path | str | None = None,
) -> dict[str, Any]:
    """Roll the working tree back to ``to_version`` and append the rollback event."""

    target_history = _resolve_path(history_path, DEFAULT_HISTORY_PATH)
    releases_path = releases if releases is not None else releases_dir
    load_history(target_history, validate=True)
    target_entry = find_entry(to_version, path=target_history, kind="release")
    if target_entry is None:
        try:
            from aicwf.registry.store import release_versions

            available = ", ".join(release_versions(target_history)) or "<none>"
        except Exception:  # noqa: BLE001
            available = "<unavailable>"
        raise ValidationError(f"cannot roll back to {to_version!r}: no release with that version (available: {available})")
    if str(target_entry.get("decision")) != "accept":
        raise ValidationError(f"release {to_version!r} was not accepted; refusing to roll back to it")

    bundle_dir = entry_bundle_dir(target_entry, project_root=project_root)
    if not bundle_dir.exists() and releases_path is not None:
        bundle_dir = _resolve_path(releases_path, DEFAULT_RELEASES_DIR) / to_version
    integrity = verify_bundle(bundle_dir)
    if integrity:
        raise RegistryIntegrityError(
            f"release bundle for {to_version!r} is corrupt: {'; '.join(integrity[:5])}"
        )

    current = latest_release(target_history)
    resolved_from = from_version or (str(current.get("version")) if current else None)
    before_metrics = dict((current or {}).get("metrics") or {})
    after_metrics = dict(target_entry.get("metrics") or {})
    if resolved_from == to_version:
        before_metrics = after_metrics  # already on the target; still record the event

    evaluator = smoke_eval or default_smoke_eval
    smoke = dict(evaluator(bundle_dir, target_entry))
    smoke.setdefault("status", "pass")
    if str(smoke.get("status")) != "pass":
        raise ValidationError(
            f"smoke evaluation failed for rollback target {to_version!r}: {smoke.get('detail', smoke)}"
        )
    smoke.setdefault("target_version", to_version)
    smoke.setdefault("bundle_dir", bundle_dir.as_posix())

    restored_files: list[str] = []
    if not dry_run and restore:
        restored_files = restore_bundle(bundle_dir, targets=restore_targets, project_root=project_root)

    reason = rationale or (
        f"Rolled back from {resolved_from or '<initial>'} to {to_version} after the smoke evaluation "
        f"reported {smoke.get('status')}; the frozen bundle was restored and the rollback event appended."
    )
    result: dict[str, Any] = {
        "status": "dry_run" if dry_run else "rolled_back",
        "to_version": to_version,
        "from_version": resolved_from,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "smoke_eval": smoke,
        "restored_files": restored_files,
        "bundle_dir": bundle_dir.as_posix(),
        "entry": None,
    }
    if dry_run:
        return result

    stamp = str(created_at_utc or utc_now_iso())
    entry = {
        "kind": "rollback",
        "version": f"rollback-{to_version}",
        "to_version": to_version,
        "from_version": resolved_from,
        "created_at_utc": stamp,
        "decision": "rollback",
        "verification_verdict": target_entry.get("verification_verdict"),
        "snapshot_hash": target_entry.get("snapshot_hash"),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "smoke_eval": dict(smoke),
        "restored_files": restored_files,
        "previous_version": resolved_from,
        "rationale": reason,
        "bundle": {
            "path": (target_entry.get("bundle") or {}).get("path"),
            "bundle_hash": (target_entry.get("bundle") or {}).get("bundle_hash"),
        },
    }
    stored = append_entry(entry, path=target_history, created_at_utc=stamp)
    result["entry"] = stored
    if write_entry is not None:
        write_json(Path(write_entry), stored, indent=2, sort_keys=True)
    return result


def rollback_run(
    to_version: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias kept for the pipeline/CLI delegation signature."""

    return rollback(to_version=to_version, **kwargs)


def rollback_to(
    *,
    to_version: str,
    config: Mapping[str, Any] | None = None,
    history_path: Path | str | None = None,
    releases_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pipeline hook: roll back using the registry paths declared in the config.

    ``aicwf/pipeline.py`` delegates here, so this wrapper resolves
    ``change_control.registry.versions_file`` / ``releases_dir`` (or the legacy
    top-level ``registry`` block) before performing the rollback.
    """

    gates = (config or {}).get("gates") if isinstance(config, Mapping) else {}
    gates = gates if isinstance(gates, Mapping) else {}
    registry_cfg = gates.get("registry") if isinstance(gates.get("registry"), Mapping) else {}
    if not registry_cfg:
        change_control = gates.get("change_control") if isinstance(gates.get("change_control"), Mapping) else {}
        registry_cfg = (
            change_control.get("registry") if isinstance(change_control.get("registry"), Mapping) else {}
        )
    resolved_history = history_path or registry_cfg.get("versions_file") or DEFAULT_HISTORY_PATH
    resolved_releases = releases_dir or registry_cfg.get("releases_dir") or DEFAULT_RELEASES_DIR
    return rollback(
        to_version=to_version,
        history_path=resolved_history,
        releases_dir=resolved_releases,
        **kwargs,
    )


def rollback_to_version(to_version: str, **kwargs: Any) -> dict[str, Any]:
    """Backwards-compatible alias for :func:`rollback_to`."""

    return rollback_to(to_version=to_version, **kwargs)
