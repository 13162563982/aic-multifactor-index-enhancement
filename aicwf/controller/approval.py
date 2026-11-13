"""Human approval gate for stage execution and CLI actions.

Two things need a human decision before they may happen:

1. **Workflow gates** — stages marked ``human_gate: true`` in
   ``config/workflow.yaml`` (the approval stage and the registry stage) are not
   executed until an approved decision exists for the run.
2. **Consequential CLI actions** — publishing the report, promoting a version or
   exporting orders (``config/gates.yaml`` ``approval.actions``).

Decisions are persisted in SQLite (``approvals`` table) and can be bound to
artifact content hashes (``approval.bind_to_artifact_hashes``). If any bound
artifact changes after approval, the gate is no longer satisfied and the run must
be approved again — this is what makes a human approval auditable.

The approval stage in ``aicwf/pipeline.py`` should call
:func:`write_approval_artifact` so that all approval artifacts share one shape::

    {
      "run_id": ..., "action": ..., "approver": ..., "decided_at_utc": ...,
      "scope": ..., "artifact_hashes": {relpath: sha256}, "decision": "approve",
      "note": ..., "version": ...            # version optional
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from aicwf.contracts import PROJECT_ROOT, STAGE_DIRS, ApprovalRequired, ConfigError
from aicwf.util.jsonio import ensure_dir, write_json
from aicwf.util.timeutil import utc_now_iso

from .dag import StageSpec, Workflow
from .state import ApprovalRecord, StateStore

__all__ = [
    "APPROVAL_DECISIONS",
    "APPROVAL_SCOPES",
    "ApprovalGate",
    "ApprovalRequest",
    "EscalationRecord",
    "approval_payload",
    "build_escalation_record",
    "write_approval_artifact",
    "write_approval_request_artifact",
    "write_escalation_artifact",
]

APPROVAL_DECISIONS: tuple[str, ...] = ("approved", "rejected")
APPROVAL_SCOPES: tuple[str, ...] = ("run", "action")

_APPROVAL_WRITE_DECISION = {"approved": "approve", "rejected": "reject"}


def _normalise_decision(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"approved", "approve", "yes", "accepted"}:
        return "approved"
    if lowered in {"rejected", "reject", "no", "denied"}:
        return "rejected"
    raise ConfigError(f"invalid approval decision {value!r}; expected one of {APPROVAL_DECISIONS}")


def _normalise_scope(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered not in APPROVAL_SCOPES:
        raise ConfigError(f"invalid approval scope {value!r}; expected one of {APPROVAL_SCOPES}")
    return lowered


@dataclass(frozen=True)
class ApprovalRequest:
    """A pending human-approval request written next to the run while it waits."""

    run_id: str
    action: str
    stage: str
    requested_at_utc: str
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    scope: str = "run"
    instructions: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "action": self.action,
            "stage": self.stage,
            "requested_at_utc": self.requested_at_utc,
            "artifact_hashes": dict(self.artifact_hashes),
            "scope": self.scope,
            "instructions": self.instructions,
        }


@dataclass(frozen=True)
class EscalationRecord:
    """Machine-readable escalation written when a run becomes ``blocked``."""

    run_id: str
    stage: str
    category: str
    reason: str
    decided_at_utc: str
    error_type: str = ""
    error: str = ""
    attempts: int = 0
    required_action: str = "human_review"
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "stage": self.stage,
            "category": self.category,
            "reason": self.reason,
            "decided_at_utc": self.decided_at_utc,
            "error_type": self.error_type,
            "error": self.error,
            "attempts": self.attempts,
            "required_action": self.required_action,
            "context": dict(self.context),
        }
        return payload


class ApprovalGate:
    """Approval policy loaded from ``config/gates.yaml`` + persisted decisions.

    Parameters
    ----------
    state:
        Controller state store (source of truth for recorded decisions).
    config:
        Full merged config bundle; only the ``gates`` section is read.
    clock:
        Injectable UTC-ISO clock for deterministic tests.
    """

    def __init__(
        self,
        state: StateStore,
        config: Mapping[str, Any] | None,
        *,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.state = state
        self.config = dict(config or {})
        self.clock = clock
        gates = self.config.get("gates")
        gates = dict(gates) if isinstance(gates, Mapping) else {}
        approval = gates.get("approval")
        approval = dict(approval) if isinstance(approval, Mapping) else {}

        required = approval.get("required", False)
        self.required = bool(required)
        actions = approval.get("actions") or []
        if not isinstance(actions, (list, tuple)) or not all(isinstance(item, str) for item in actions):
            raise ConfigError("gates.approval.actions must be a list of strings")
        self.required_actions: tuple[str, ...] = tuple(actions)
        bind = approval.get("bind_to_artifact_hashes", True)
        self.bind_to_artifact_hashes = bool(bind)
        self.default_approver = str(approval.get("default_approver") or "unknown")
        stage_actions = approval.get("stage_actions") or {}
        if not isinstance(stage_actions, Mapping):
            raise ConfigError("gates.approval.stage_actions must be a mapping")
        self.stage_actions: dict[str, str] = {str(key): str(value) for key, value in stage_actions.items()}

    # -- policy --------------------------------------------------------
    def action_for_stage(self, spec: StageSpec) -> str:
        """The approval action required before ``spec`` may execute."""

        if spec.approval_action:
            return spec.approval_action
        if spec.name in self.stage_actions:
            return self.stage_actions[spec.name]
        return f"stage:{spec.name}"

    def human_gate_actions(self, workflow: Workflow) -> dict[str, str]:
        return {spec.name: self.action_for_stage(spec) for spec in workflow.human_gates()}

    def _validate_action(self, action: str) -> str:
        action = str(action or "").strip()
        if not action:
            raise ConfigError("approval action must be a non-empty string")
        if (
            self.required_actions
            and action not in self.required_actions
            and not action.startswith("stage:")
        ):
            raise ConfigError(
                f"unknown approval action {action!r}; configured actions: {list(self.required_actions)}"
            )
        return action

    # -- artifact binding ----------------------------------------------
    def current_artifact_hashes(self, run_id: str) -> dict[str, str]:
        """Declared artifact relpath -> canonical content hash for a run."""

        return self.state.artifact_hashes(run_id)

    @staticmethod
    def _covers(record: ApprovalRecord, action: str) -> bool:
        if record.decision != "approved":
            return False
        if record.scope == "run":
            return True
        return record.action in {action, "*", "all"}

    def _decisions(self, run_id: str) -> list[ApprovalRecord]:
        return self.state.list_approvals(run_id)

    def latest_decision(self, run_id: str) -> ApprovalRecord | None:
        decisions = self._decisions(run_id)
        return decisions[-1] if decisions else None

    def latest_approved_for(self, run_id: str, action: str) -> ApprovalRecord | None:
        """Latest approved decision whose scope covers ``action``."""

        matches = [record for record in self._decisions(run_id) if self._covers(record, action)]
        return matches[-1] if matches else None

    def is_rejected(self, run_id: str) -> bool:
        latest = self.latest_decision(run_id)
        return bool(latest and latest.decision == "rejected")

    def verify_binding(self, record: ApprovalRecord, artifact_hashes: Mapping[str, str] | None = None) -> bool:
        """Check that the artifacts a decision approved are still unchanged.

        Binding is a *subset* check: every hash recorded at approval time must
        still match the current hash of that artifact. Artifacts added after the
        approval (for example ``approval.json`` itself) do not invalidate it, but
        a re-run that changes an approved upstream artifact does.
        """

        if not self.bind_to_artifact_hashes:
            return True
        current = (
            dict(artifact_hashes)
            if artifact_hashes is not None
            else self.current_artifact_hashes(record.run_id)
        )
        return all(current.get(relpath) == digest for relpath, digest in record.artifact_hashes.items())

    def disk_artifact_hashes(self, run_id: str) -> dict[str, str]:
        """Re-hash declared artifacts from disk (tamper check for verification).

        Slower than :meth:`current_artifact_hashes` because it reads and
        canonically hashes every artifact file; use it at human gates or in the
        independent verification stage, not on every stage boundary.
        """

        from aicwf.util.artifact_hash import hash_artifact

        hashes: dict[str, str] = {}
        for artifact in self.state.list_artifacts(run_id):
            path = Path(artifact.relpath)
            if not path.is_absolute():
                from aicwf.contracts import PROJECT_ROOT

                path = PROJECT_ROOT / path
            if path.is_file():
                hashes[artifact.relpath] = hash_artifact(path)
            else:
                hashes[artifact.relpath] = "missing"
        return hashes

    def is_satisfied(
        self,
        run_id: str,
        action: str,
        *,
        artifact_hashes: Mapping[str, str] | None = None,
    ) -> bool:
        """True when an approved decision covers ``action`` and the binding holds."""

        if self.is_rejected(run_id):
            return False
        record = self.latest_approved_for(run_id, action)
        if record is None:
            return False
        return self.verify_binding(record, artifact_hashes)

    def require(
        self,
        run_id: str,
        action: str,
        *,
        artifact_hashes: Mapping[str, str] | None = None,
    ) -> ApprovalRecord:
        """Return the covering decision or raise :class:`ApprovalRequired`."""

        record = self.latest_approved_for(run_id, action)
        if record is None:
            raise ApprovalRequired(
                f"run {run_id} has no approved decision covering {action!r}; "
                f"approve with: python -m aicwf approve {run_id} --action {action}"
            )
        if not self.verify_binding(record, artifact_hashes):
            raise ApprovalRequired(
                f"approval for run {run_id} is stale: bound artifact hashes changed after "
                f"{record.decided_at_utc}; re-approve the current package"
            )
        return record

    def pending_gate_actions(self, workflow: Workflow, run_id: str) -> list[tuple[str, str]]:
        """Unsatisfied human gates as ``(stage, action)`` pairs, in workflow order."""

        pending: list[tuple[str, str]] = []
        for spec in workflow.human_gates():
            action = self.action_for_stage(spec)
            if not self.is_satisfied(run_id, action):
                pending.append((spec.name, action))
        return pending

    def pending_actions(self, run_id: str) -> list[str]:
        """Configured actions with no approved decision for the run yet."""

        return [action for action in self.required_actions if not self.is_satisfied(run_id, action)]

    # -- recording -----------------------------------------------------
    def record_decision(
        self,
        run_id: str,
        *,
        action: str,
        approver: str | None = None,
        decision: str = "approved",
        scope: str = "run",
        artifact_hashes: Mapping[str, str] | None = None,
        note: str = "",
    ) -> ApprovalRecord:
        """Persist a human decision and mirror it to the event log."""

        action = self._validate_action(action)
        normalised_decision = _normalise_decision(decision)
        normalised_scope = _normalise_scope(scope)
        if artifact_hashes is not None:
            hashes = {str(key): str(value) for key, value in artifact_hashes.items()}
        elif self.bind_to_artifact_hashes:
            hashes = self.current_artifact_hashes(run_id)
        else:
            hashes = {}
        record = ApprovalRecord(
            run_id=run_id,
            action=action,
            decision=normalised_decision,
            approver=str(approver or self.default_approver),
            scope=normalised_scope,
            artifact_hashes=hashes,
            note=str(note or ""),
            decided_at_utc=self.clock(),
        )
        self.state.add_approval(record)
        self.state.add_event(
            run_id,
            "approval_recorded",
            payload={
                "action": record.action,
                "decision": record.decision,
                "approver": record.approver,
                "scope": record.scope,
                "artifact_hashes": dict(record.artifact_hashes),
                "note": record.note,
            },
        )
        return record

    def build_request(
        self,
        run_id: str,
        *,
        action: str,
        stage: str,
        artifact_hashes: Mapping[str, str] | None = None,
        scope: str = "run",
    ) -> ApprovalRequest:
        """Build the request payload written to ``approval_request.json``."""

        action = self._validate_action(action)
        hashes = dict(artifact_hashes) if artifact_hashes is not None else self.current_artifact_hashes(run_id)
        return ApprovalRequest(
            run_id=run_id,
            action=action,
            stage=stage,
            requested_at_utc=self.clock(),
            artifact_hashes=hashes,
            scope=_normalise_scope(scope),
            instructions=(
                f"python -m aicwf approve {run_id} --action {action} "
                f"[--approver NAME] [--scope run|action]"
            ),
        )

    def resolve_payload_action(self, run_id: str, action: str | None = None) -> str:
        """Pick the action recorded in ``approval.json`` when not specified.

        Prefers the two consequential actions in dependency order, then any other
        configured action covered by the decisions, then the latest decision.
        """

        if action:
            return self._validate_action(action)
        preferred = ["promote_version", "publish_report", *self.required_actions]
        for candidate in preferred:
            if candidate and self.is_satisfied(run_id, candidate):
                return candidate
        latest = self.latest_decision(run_id)
        if latest is not None and latest.decision == "approved":
            return latest.action
        raise ApprovalRequired(f"run {run_id} has no approved decision to write into approval.json")

    def approval_payload(
        self,
        record: ApprovalRecord,
        *,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Canonical ``approval.json`` payload for one decision."""

        payload: dict[str, Any] = {
            "run_id": record.run_id,
            "action": record.action,
            "approver": record.approver,
            "decided_at_utc": record.decided_at_utc,
            "scope": record.scope,
            "artifact_hashes": dict(record.artifact_hashes),
            "decision": _APPROVAL_WRITE_DECISION.get(record.decision, record.decision),
            "note": record.note,
        }
        candidate_version = version
        if candidate_version is None:
            registry_cfg = self.config.get("registry")
            if isinstance(registry_cfg, Mapping):
                candidate_version = registry_cfg.get("candidate_version")
        if candidate_version is None:
            # Fallback used by this repository: candidate version lives next to
            # the change-control registry paths in config/gates.yaml.
            gates_cfg = self.config.get("gates")
            if isinstance(gates_cfg, Mapping):
                change_control = gates_cfg.get("change_control")
                if isinstance(change_control, Mapping):
                    registry_cfg = change_control.get("registry")
                    if isinstance(registry_cfg, Mapping):
                        candidate_version = registry_cfg.get("candidate_version")
        if candidate_version is not None:
            payload["version"] = str(candidate_version)
        return payload


# ---------------------------------------------------------------------------
# File writers used by the pipeline stages and the runner
# ---------------------------------------------------------------------------


def approval_payload(
    gate: ApprovalGate,
    run_id: str,
    *,
    action: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Build the approval artifact payload for the latest approved decision.

    When a run-scoped approval covers several consequential actions, the artifact
    records the action that was actually requested (``promote_version`` for the
    registry stage, ``publish_report`` for the approval stage) while carrying the
    approver and the artifact-hash binding of the original human decision.
    """

    resolved = gate.resolve_payload_action(run_id, action)
    record = gate.latest_approved_for(run_id, resolved)
    if record is None:
        raise ApprovalRequired(f"no approved decision covers action {resolved!r} for run {run_id}")
    gate.require(run_id, resolved)
    payload = gate.approval_payload(record, version=version)
    payload["action"] = resolved
    return payload


def _load_default_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load the merged config bundle from a config directory (pipeline fallback)."""

    from aicwf.util.jsonio import read_yaml

    directory = Path(config_dir) if config_dir is not None else PROJECT_ROOT / "config"
    bundle: dict[str, Any] = {}
    for name in ("workflow", "strategy", "gates", "budgets", "model", "data"):
        path = directory / f"{name}.yaml"
        if not path.is_file():
            continue
        content = read_yaml(path)
        if isinstance(content, Mapping) and isinstance(content.get(name), Mapping):
            content = content[name]
        bundle[name] = content
    return bundle


def _default_gate(
    *,
    state: StateStore | None = None,
    config: Mapping[str, Any] | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> ApprovalGate:
    """Build a gate for legacy/glue callers that do not pass one explicitly.

    Defaults follow the project layout *and* the same ``AICWF_*`` environment
    overrides the CLI documents (``AICWF_STATE_DB``, ``AICWF_CONFIG_DIR``), so the
    pipeline's approval handler keeps working when the CLI runs against an
    isolated config/state pair. Explicit arguments always win.
    """

    import os

    if state is None:
        from .state import default_db_path

        state_env = os.environ.get("AICWF_STATE_DB")
        state = StateStore(Path(state_env) if state_env else default_db_path(project_root))
    if config is None:
        config_env = os.environ.get("AICWF_CONFIG_DIR")
        config = _load_default_config(Path(config_env) if config_env else None)
    return ApprovalGate(state, config)


def write_approval_artifact(
    gate: ApprovalGate | None = None,
    run_id: str = "",
    run_dir: Path | str | None = None,
    *,
    action: str | None = None,
    version: str | None = None,
    filename: str = "approval.json",
    state: StateStore | None = None,
    config: Mapping[str, Any] | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> Path:
    """Write ``<run_dir>/11_approval/approval.json`` atomically and return its path.

    Typical use from the pipeline's ``approval`` stage handler::

        gate = getattr(ctx, "approval_gate", None)   # supplied by the Runner
        write_approval_artifact(gate, ctx.run_id, ctx.run_dir, action="publish_report")

    For robustness (and to keep older integration glue working) the leading
    ``gate`` argument may be omitted/anything-that-is-not-an-ApprovalGate, in
    which case a gate is built from ``state``/``config`` (defaults: the project
    state database and ``config/*.yaml``). Pass ``state``/``config`` explicitly
    when running against isolated paths.

    Raises :class:`aicwf.contracts.ApprovalRequired` if no covering decision
    exists or if the bound artifact hashes have changed since the approval.
    """

    if not isinstance(gate, ApprovalGate):
        gate = _default_gate(state=state, config=config, project_root=project_root)
    if not run_id:
        raise ConfigError("write_approval_artifact requires the run_id")
    if run_dir is None:
        raise ConfigError("write_approval_artifact requires the run_dir")
    payload = approval_payload(gate, run_id, action=action, version=version)
    path = Path(run_dir) / STAGE_DIRS["approval"] / filename
    ensure_dir(path.parent)
    write_json(path, payload, sort_keys=True)
    return path


def write_approval_request_artifact(
    request: ApprovalRequest,
    run_dir: Path | str,
    *,
    filename: str = "approval_request.json",
) -> Path:
    """Write the pending-approval request at the run-directory root."""

    path = Path(run_dir) / filename
    write_json(path, request.to_dict(), sort_keys=True)
    return path


def build_escalation_record(
    run_id: str,
    stage: str,
    *,
    category: str,
    reason: str,
    error_type: str = "",
    error: str = "",
    attempts: int = 0,
    required_action: str = "human_review",
    extra: Mapping[str, Any] | None = None,
    clock: Callable[[], str] = utc_now_iso,
) -> EscalationRecord:
    """Build the ``escalation.json`` record for a blocked run."""

    return EscalationRecord(
        run_id=run_id,
        stage=stage,
        category=category,
        reason=reason,
        decided_at_utc=clock(),
        error_type=error_type,
        error=error,
        attempts=int(attempts),
        required_action=required_action,
        context=dict(extra or {}),
    )


def write_escalation_artifact(
    record: EscalationRecord,
    run_dir: Path | str,
    *,
    filename: str = "escalation.json",
) -> Path:
    """Write ``<run_dir>/escalation.json`` atomically and return its path."""

    path = Path(run_dir) / filename
    write_json(path, record.to_dict(), sort_keys=True)
    return path
