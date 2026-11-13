"""Promotion of a verified run into the version registry.

A promotion is a consequential action, so it requires a recorded human
approval whose artifact hashes still match the run's artifacts
(``config/gates.yaml: approval.bind_to_artifact_hashes``).  The promotion then

1. validates the approval document (action ``promote_version``, positive
   decision, run id, artifact-hash binding);
2. refuses to reuse an existing version (append-only history);
3. freezes ``registry/releases/<version>/`` with config + prompts + skills and
   a content-hash manifest;
4. appends one immutable ``release`` entry with the frozen metrics,
   verification verdict, approval digest and previous version.

Use :func:`promote_run` from the pipeline and :func:`promote` when the caller
already loaded the artifacts (CLI, tests, recovery).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import PROJECT_ROOT, ApprovalRequired, ValidationError
from aicwf.registry.store import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_RELEASES_DIR,
    RegistryConflict,
    RegistryError,
    append_entry,
    create_bundle,
    find_entry,
    latest_release,
    load_history,
    releases_dir,
)
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import read_json
from aicwf.util.timeutil import utc_now_iso

APPROVED_DECISIONS = {"approve", "approved", "accept", "accepted", "yes"}
PROMOTE_ACTION = "promote_version"
REQUIRED_GATES_ACTIONS = {"publish_report", "promote_version", "export_orders"}


# ---------------------------------------------------------------------------
# Approval handling
# ---------------------------------------------------------------------------


def load_approval(
    approval: Mapping[str, Any] | Path | str | None = None,
    *,
    run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve an approval document from a dict, a path or a run directory."""

    if isinstance(approval, Mapping):
        return dict(approval)
    if approval is not None:
        path = Path(approval)
        if not path.exists():
            raise ApprovalRequired(f"approval artifact not found: {path}")
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ApprovalRequired(f"approval artifact {path} is not a JSON object")
        return dict(payload)
    if run_dir is not None:
        path = Path(run_dir) / "11_approval" / "approval.json"
        if path.exists():
            payload = read_json(path)
            if isinstance(payload, dict):
                return dict(payload)
        raise ApprovalRequired(
            f"no approval artifact at {path}; a human must approve the run before promotion"
        )
    raise ApprovalRequired("promotion requires an approval artifact (human approval gate)")


def _decision_ok(value: Any) -> bool:
    return str(value or "").strip().lower() in APPROVED_DECISIONS


def validate_approval(
    approval: Mapping[str, Any],
    *,
    run_id: str | None = None,
    required_action: str = PROMOTE_ACTION,
    declared_hashes: Mapping[str, str] | None = None,
    bind_to_artifact_hashes: bool = True,
) -> None:
    """Raise :class:`ApprovalRequired` when the approval does not gate this promotion."""

    if not _decision_ok(approval.get("decision")):
        raise ApprovalRequired(
            f"approval decision is {approval.get('decision')!r}; promotion requires an approved decision"
        )
    if not str(approval.get("approver") or "").strip():
        raise ApprovalRequired("approval has no named approver")
    action = str(approval.get("action") or "").strip()
    scope = str(approval.get("scope") or "").strip().lower()
    if action != required_action and scope != "run":
        raise ApprovalRequired(
            f"approval action {action!r} does not cover {required_action!r} "
            "(expected the promote_version action or a run-scoped approval)"
        )
    if run_id and approval.get("run_id") and str(approval["run_id"]) != str(run_id):
        raise ApprovalRequired(
            f"approval was given for run {approval.get('run_id')!r}, not for run {run_id!r}"
        )
    if bind_to_artifact_hashes:
        approved_hashes = approval.get("artifact_hashes")
        if not isinstance(approved_hashes, Mapping) or not approved_hashes:
            raise ApprovalRequired("approval does not bind any artifact hashes")
        current = dict(declared_hashes or {})
        if not current:
            raise ApprovalRequired(
                "approval binds artifact hashes but the caller supplied none to compare against; "
                "provide the current run artifact hashes so changed artifacts force re-approval"
            )
        problems: list[str] = []
        for key, expected in approved_hashes.items():
            actual = _resolve_current_hash(str(key), current)
            if actual is None:
                problems.append(f"approved artifact {key!r} is no longer part of the run")
            elif str(actual) != str(expected):
                problems.append(
                    f"artifact {key!r} changed after approval "
                    f"(approved {str(expected)[:12]}..., now {str(actual)[:12]}...)"
                )
        if problems:
            raise ApprovalRequired("approval no longer matches the run: " + "; ".join(problems))


def _resolve_current_hash(approved_key: str, current: Mapping[str, str]) -> str | None:
    normalised = approved_key.replace("\\", "/").lstrip("./")
    if normalised in current:
        return str(current[normalised])
    for key, value in current.items():
        candidate = str(key).replace("\\", "/").lstrip("./")
        if candidate == normalised or candidate.endswith("/" + normalised):
            return str(value)
    return None


def collect_declared_hashes(
    run_dir: Path | str,
    *,
    keys: list[str] | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, str]:
    """Content hashes for the artifacts named by an approval document.

    Keys are resolved in this order:

    1. relative to the project root (the controller records project-relative
       artifact paths);
    2. relative to the run directory;
    3. ``<stage>:<logical>`` keys (as written by ``aicwf/pipeline.py``'s approval
       handler) by matching the artifact filename stem inside the run directory;
    4. unique basename inside the run directory.
    """

    root = Path(run_dir)
    root_project = Path(project_root)
    if keys is None:
        keys = []
        verification_path = root / "09_verification" / "verification.json"
        if verification_path.exists():
            verification = read_json(verification_path)
            keys.extend(list((verification.get("artifact_hashes") or {}).keys()))
    hashes: dict[str, str] = {}
    files_by_basename: dict[str, list[Path]] = {}
    files_by_stem: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            files_by_basename.setdefault(path.name, []).append(path)
            files_by_stem.setdefault(path.stem, []).append(path)
    for key in keys:
        normalised = str(key).replace("\\", "/")
        candidates = [root_project / normalised, root / normalised, root_project.parent / normalised]
        resolved: Path | None = next(
            (candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None
        )
        if resolved is None and ":" in normalised:
            # ``<stage>:<logical>`` key form used by the pipeline approval handler.
            logical = normalised.split(":", 1)[1]
            matches = files_by_stem.get(Path(logical).stem, [])
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                preferred = [match for match in matches if match.name.startswith(Path(logical).stem + ".")]
                if len(preferred) == 1:
                    resolved = preferred[0]
        if resolved is None:
            matches = files_by_basename.get(Path(normalised).name, [])
            if len(matches) == 1:
                resolved = matches[0]
            elif normalised.endswith(Path(normalised).name) and len(matches) > 1:
                relative = normalised.split("/", 2)[-1] if "/" in normalised else normalised
                candidates = [match for match in matches if match.as_posix().endswith(relative)]
                if len(candidates) == 1:
                    resolved = candidates[0]
        if resolved is not None:
            hashes[normalised] = hash_artifact(resolved)
    return hashes


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def _headline_metrics(metrics_doc: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if metrics is not None:
        return dict(metrics)
    if not metrics_doc:
        return {}
    return dict(metrics_doc.get("metrics") or {})


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _config_from_run(run_dir: Path) -> dict[str, Any] | None:
    bundle = _read_json_optional(run_dir / "config_bundle.json")
    if bundle:
        return bundle
    config: dict[str, Any] = {}
    for name in ("workflow", "strategy", "gates", "budgets", "model", "data"):
        path = PROJECT_ROOT / "config" / f"{name}.yaml"
        if path.exists():
            from aicwf.util.jsonio import read_yaml

            config[name] = read_yaml(path) or {}
    return config or None


def promote(
    *,
    version: str,
    kind: str = "release",
    decision: str = "accept",
    run_dir: Path | str | None = None,
    metrics: Mapping[str, Any] | None = None,
    metrics_doc: Mapping[str, Any] | None = None,
    verification: Mapping[str, Any] | None = None,
    approval: Mapping[str, Any] | Path | str | None = None,
    approval_path: Path | str | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    config_dir: Path | str | None = None,
    prompt_dir: Path | str | None = None,
    skills_dir: Path | str | None = None,
    snapshot_hash: str | None = None,
    config_hash: str | None = None,
    prompt_hashes: Mapping[str, str] | None = None,
    verification_verdict: str | None = None,
    history_path: Path | str | None = None,
    releases: Path | str | None = None,
    project_root: Path | str = PROJECT_ROOT,
    git_commit: str | None = None,
    git_clean: bool | None = None,
    enforce_git_clean: bool | None = None,
    gates: Mapping[str, Any] | None = None,
    rationale: str | None = None,
    created_at_utc: str | None = None,
    require_approval: bool = True,
    write_entry: Path | str | None = None,
) -> dict[str, Any]:
    """Promote one verified run into the registry; returns the appended entry."""

    root_run: Path | None = Path(run_dir) if run_dir else None
    run_request: dict[str, Any] = {}
    if root_run is not None and (root_run / "00_intake" / "run_request.json").exists():
        run_request = read_json(root_run / "00_intake" / "run_request.json")
    if metrics_doc is None and root_run is not None:
        metrics_doc = _read_json_optional(root_run / "05_backtest" / "metrics.json") or {}
    if verification is None and root_run is not None:
        verification = _read_json_optional(root_run / "09_verification" / "verification.json")
    if config is None and root_run is not None:
        config = _config_from_run(root_run)
    gates = dict(gates or {})
    if not gates and isinstance((config or {}).get("gates"), dict):
        gates = dict(config["gates"])
    require_git_clean_flag = bool(((gates.get("registry") or {}).get("require_git_clean", False)))
    if enforce_git_clean is None:
        enforce_git_clean = require_git_clean_flag and git_clean is not None
    if enforce_git_clean and git_clean is not True:
        raise RegistryError(
            "config/gates.yaml requires a clean git work tree before promotion and the caller "
            f"reported git_clean={git_clean!r}; commit the release state and retry"
        )

    resolved_run_id = str(run_request.get("run_id") or (root_run.name if root_run else "")) or None
    approval_doc: dict[str, Any] = {}
    if require_approval:
        approval_doc = load_approval(approval_path if approval_path is not None else approval, run_dir=root_run)
        if not artifact_hashes:
            approval_keys = list((approval_doc.get("artifact_hashes") or {}).keys())
            if root_run is not None and approval_keys:
                artifact_hashes = collect_declared_hashes(root_run, keys=approval_keys, project_root=project_root)
        validate_approval(
            approval_doc,
            run_id=resolved_run_id,
            declared_hashes=artifact_hashes,
            bind_to_artifact_hashes=bool(((gates.get("approval") or {}).get("bind_to_artifact_hashes", True))),
        )

    target_history = history_path if history_path is not None else DEFAULT_HISTORY_PATH
    load_history(target_history, validate=True)
    existing = find_entry(version, path=target_history, kind=kind)
    if existing is not None:
        raise RegistryConflict(
            f"version {version!r} already exists in the registry (sequence {existing.get('sequence')}); "
            "history is append-only, choose a new version"
        )
    previous = latest_release(target_history)
    previous_version = str(previous.get("version")) if previous else None

    resolved_metrics = _headline_metrics(metrics_doc, metrics)
    resolved_verification = dict(verification or {})
    verdict = verification_verdict or str(resolved_verification.get("verdict") or "UNKNOWN")
    if require_approval and verdict.upper() != "PASS":
        raise ValidationError(
            f"cannot promote a run whose verification verdict is {verdict!r}; a PASS is required"
        )
    snapshot = snapshot_hash or run_request.get("snapshot_hash")
    config_digest = config_hash or run_request.get("config_hash") or (metrics_doc or {}).get("config_hash")
    stamp = str(created_at_utc or utc_now_iso())

    target_releases = releases_dir(releases)
    bundle_manifest = create_bundle(
        version,
        config=config,
        config_dir=config_dir,
        prompt_dir=prompt_dir,
        skills_dir=skills_dir,
        destination=target_releases / version,
        created_at_utc=stamp,
    )
    resolved_prompt_hashes = dict(prompt_hashes or bundle_manifest.get("prompt_hashes") or {})
    approval_record: dict[str, Any] = {}
    if approval_doc:
        approval_record = {
            "approver": approval_doc.get("approver"),
            "decided_at_utc": approval_doc.get("decided_at_utc"),
            "action": approval_doc.get("action"),
            "scope": approval_doc.get("scope"),
            "decision": approval_doc.get("decision"),
            "note": approval_doc.get("note"),
            "n_artifact_hashes": len(approval_doc.get("artifact_hashes") or {}),
            "approval_hash": hash_json(approval_doc),
        }
    bundle_path = target_releases / version
    try:
        relative_bundle = bundle_path.relative_to(project_root).as_posix()
    except ValueError:
        relative_bundle = bundle_path.as_posix()
    entry = {
        "kind": kind,
        "version": version,
        "created_at_utc": stamp,
        "git_commit": git_commit,
        "git_clean": git_clean,
        "config_hash": config_digest,
        "prompt_hashes": resolved_prompt_hashes,
        "snapshot_hash": snapshot,
        "metrics": resolved_metrics,
        "verification_verdict": verdict.upper(),
        "approval": approval_record,
        "decision": decision,
        "rationale": rationale
        or (
            f"Promoted run {resolved_run_id} after verification verdict {verdict.upper()} and "
            f"human approval by {approval_record.get('approver', 'unknown')}."
        ),
        "previous_version": previous_version,
        "run_id": resolved_run_id,
        "artifact_hashes": dict(sorted((artifact_hashes or {}).items())),
        "bundle": {
            "path": relative_bundle,
            "bundle_hash": bundle_manifest.get("bundle_hash"),
            "config_hash": bundle_manifest.get("config_hash"),
            "files": bundle_manifest.get("files"),
        },
    }
    stored = append_entry(entry, path=target_history, created_at_utc=stamp)
    if write_entry is not None:
        from aicwf.util.jsonio import write_json

        write_json(Path(write_entry), stored, indent=2, sort_keys=True)
    return stored


def promote_run(
    run_dir: Path | str,
    *,
    version: str | None = None,
    rationale: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper that reads the version from the run artifacts."""

    root = Path(run_dir)
    request = _read_json_optional(root / "00_intake" / "run_request.json") or {}
    approval = kwargs.pop("approval", None)
    if approval is None:
        approval_file = root / "11_approval" / "approval.json"
        approval = approval_file if approval_file.exists() else None
    resolved = version or request.get("candidate_version")
    if resolved is None and isinstance(approval, Mapping):
        resolved = approval.get("version")
    if not resolved:
        raise ValidationError(
            "cannot determine the version to promote: pass version=... or record candidate_version "
            "in run_request.json / approval.json"
        )
    return promote(
        version=str(resolved),
        run_dir=root,
        approval=approval,
        rationale=rationale,
        **kwargs,
    )
