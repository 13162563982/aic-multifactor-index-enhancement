"""The workflow runner: deterministic DAG execution with durable resume.

The runner walks stages in the topological order computed by
:mod:`aicwf.controller.dag`, calls the handler registered for each stage's
``handler`` name, records everything in :class:`aicwf.controller.state.StateStore`,
and stops in a well-defined state when something goes wrong:

``completed``
    all stages completed (or legitimately skipped);
``awaiting_approval``
    a ``human_gate`` stage is waiting for a recorded approval decision;
``blocked``
    human intervention required (validation/data-quality failure, retries
    exhausted, or a rejected approval); ``escalation.json`` is written;
``budget_exhausted``
    a budget limit was hit; the run can be resumed after limits are adjusted;
``failed``
    deterministic failure that automatic handling cannot fix.

Idempotency and resume
----------------------
Before a stage runs, the runner computes a deterministic *input hash* from the
stage definition, a stage-relevant config subset, the run's ``as_of`` and snapshot
hash, seed/LLM mode, agent prompt/skill/schema content hashes and the content
hashes of every upstream artifact. A completed stage whose stored input hash
matches is skipped (``skipped_completed``) and its recorded artifacts are reused;
if the hash differs, that stage and all its descendants are invalidated and
re-executed. No wall-clock value ever enters the hash.

Fault injection
---------------
:class:`aicwf.contracts.FaultSpec` supports ``crash_before`` / ``crash_after`` on
a chosen stage attempt. The runner records the attempt and then hard-exits the
process (``os._exit(42)``) so tests and ``aicwf fault-demo`` can prove that a
resumed run does not repeat completed stages. Faults are not charged to the retry
budget, so a crash at attempt 1 does not consume a stage's retries.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from aicwf.contracts import (
    PROJECT_ROOT,
    BudgetExceeded,
    BudgetLimits,
    ConfigError,
    FaultSpec,
    LLMClient,
    LLM_MODES,
    RunContext,
    StageFn,
    StageOutput,
    ValidationError,
    WorkflowError,
    parse_date,
    relpath,
)
from aicwf.util.hashing import hash_json, hash_text
from aicwf.util.jsonio import ensure_dir, read_json, write_json
from aicwf.util.timeutil import utc_now, utc_now_iso

from .approval import (
    ApprovalGate,
    build_escalation_record,
    write_approval_request_artifact,
    write_escalation_artifact,
)
from .budget import BudgetLedger, GuardedLLM
from .dag import StageSpec, Workflow, load_workflow
from .retry import (
    CATEGORY_APPROVAL,
    CATEGORY_BUDGET,
    CATEGORY_ESCALATE,
    CATEGORY_NON_RETRYABLE,
    CATEGORY_RETRYABLE,
    RetryDecision,
    RetryPolicy,
    StageTimeout,
    describe_exception,
    normalise_cancelled,
)
from .state import (
    ArtifactRecord,
    AttemptRecord,
    RunEventSink,
    RunRecord,
    StateStore,
    StepRecord,
)

__all__ = [
    "INPUT_HASH_SCHEMA",
    "STAGE_CONFIG_SECTIONS",
    "RunResult",
    "Runner",
    "compute_input_hash",
    "generate_run_id",
    "resolve_snapshot_identity",
    "snapshot_hash_for_dir",
    "stage_config_subset",
]

INPUT_HASH_SCHEMA = "aicwf-input-hash/1"

#: Config sections that can influence each stage's output. Only these sections
#: are hashed into ``input_hash``; operational sections (budgets, workflow retry
#: limits) are deliberately excluded so tightening a budget does not invalidate
#: already-computed research artifacts.
STAGE_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "intake": (),
    "hypothesis": ("strategy", "model"),
    "data_prep": ("data", "strategy"),
    "factor_build": ("strategy",),
    "factor_review": ("strategy", "model"),
    "backtest": ("strategy",),
    "robustness": ("strategy",),
    "robustness_review": ("model",),
    "portfolio": ("strategy",),
    "verification": ("gates",),
    "report": ("model",),
    "approval": ("gates",),
    "registry": ("gates",),
}

_FORCE_RESUME_STATUSES = frozenset({"failed", "blocked"})
_RESUME_ALLOWED = frozenset({"pending", "running", "awaiting_approval", "budget_exhausted", "stopped"})


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of ``Runner.run`` / ``Runner.resume``."""

    run_id: str
    status: str
    run_dir: str
    as_of: str
    error: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    @property
    def requires_human(self) -> bool:
        return self.status in {"awaiting_approval", "blocked"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "run_dir": self.run_dir,
            "as_of": self.as_of,
            "error": self.error,
            "steps": list(self.steps),
            "budget": dict(self.budget),
        }

    def step(self, stage: str) -> dict[str, Any] | None:
        for entry in self.steps:
            if entry.get("stage") == stage:
                return entry
        return None


@dataclass
class _StageOutcome:
    """Internal per-stage execution outcome."""

    status: str
    stage: str = ""
    attempt: int = 0
    category: str = ""
    reason: str = ""
    error: str | None = None
    error_type: str | None = None
    action: str | None = None
    decision: RetryDecision | None = None


# ---------------------------------------------------------------------------
# Deterministic identity helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


def generate_run_id(as_of: dt.date, *, now: dt.datetime | None = None) -> str:
    """Generate a filesystem-safe run id: ``run-<asof>-<hhmmss>-<token>``."""

    moment = now or utc_now()
    token = secrets.token_hex(3)
    return f"run-{as_of.strftime('%Y%m%d')}-{moment.strftime('%H%M%S')}-{token}"


def stage_config_subset(stage: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    """The part of the merged config bundle that can affect ``stage`` output."""

    config = dict(config or {})
    sections = STAGE_CONFIG_SECTIONS.get(stage, tuple(sorted(config)))
    return {name: config[name] for name in sections if name in config}


def snapshot_hash_for_dir(snapshot_dir: Path | str) -> str:
    """Canonical snapshot hash: manifest ``snapshot_hash`` else a file-hash digest."""

    path = Path(snapshot_dir)
    manifest = path / "manifest.json"
    if manifest.is_file():
        data = read_json(manifest)
        if isinstance(data, Mapping) and data.get("snapshot_hash"):
            return str(data["snapshot_hash"])
        return hash_json(data)
    files: dict[str, str] = {}
    for child in sorted(path.rglob("*")):
        if child.is_file():
            files[child.relative_to(path).as_posix()] = _hash_artifact_file(child)
    return hash_json({"files": files})


def _snapshot_as_of(snapshot_dir: Path) -> dt.date | None:
    """Best-effort ``as_of`` of a frozen snapshot (manifest first, dir name second)."""

    manifest = snapshot_dir / "manifest.json"
    if manifest.is_file():
        try:
            data = read_json(manifest)
        except (OSError, ValueError):
            data = None
        if isinstance(data, Mapping) and data.get("as_of"):
            try:
                return parse_date(str(data["as_of"]))
            except ConfigError:
                return None
    try:
        return parse_date(snapshot_dir.name)
    except ConfigError:
        return None


def resolve_snapshot_identity(
    as_of: dt.date | str,
    config: Mapping[str, Any] | None,
    *,
    project_root: Path | str = PROJECT_ROOT,
    snapshot_dir: Path | str | None = None,
) -> tuple[Path | None, str]:
    """Find the frozen data snapshot for ``as_of`` and return ``(dir, hash)``.

    Resolution order: explicit ``snapshot_dir`` argument, then
    ``data/snapshots/<as_of>``, then the ``strategy.data.snapshot`` path from
    config *only if its manifest as_of matches* — a snapshot for a different date
    must never be used (that would leak future information into the research).

    A run without a snapshot (offline unit tests) gets the deterministic marker
    hash of ``"missing-snapshot:<as_of>"`` instead of a wall-clock value.
    """

    root = Path(project_root)
    date = parse_date(as_of)
    candidate: Path | None = None
    if snapshot_dir is not None:
        candidate = Path(snapshot_dir)
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        options: list[Path] = [root / "data" / "snapshots" / date.isoformat()]
        strategy = config.get("strategy") if isinstance(config, Mapping) else None
        data_cfg = strategy.get("data") if isinstance(strategy, Mapping) else None
        configured = data_cfg.get("snapshot") if isinstance(data_cfg, Mapping) else None
        if configured:
            configured_path = Path(str(configured))
            if not configured_path.is_absolute():
                configured_path = root / configured_path
            if _snapshot_as_of(configured_path) == date:
                options.append(configured_path)
        candidate = next((path for path in options if path.is_dir()), None)
    if candidate is None or not candidate.is_dir():
        return None, hash_text(f"missing-snapshot:{date.isoformat()}")
    return candidate, snapshot_hash_for_dir(candidate)


def compute_input_hash(
    *,
    stage: str,
    handler: str,
    needs: Sequence[str],
    upstream_hashes: Mapping[str, str],
    config_subset: Mapping[str, Any],
    workflow_name: str,
    workflow_version: str,
    as_of: str,
    snapshot_hash: str,
    seed: int,
    llm_mode: str,
    prompt_hashes: Mapping[str, str] | None = None,
    spec_payload: Mapping[str, Any] | None = None,
) -> str:
    """Deterministic stage input hash — no wall-clock, no run id, no paths outside data.

    Two runs with identical inputs and config produce the same hash on any
    machine; that property is what makes skipping and resume safe.
    """

    payload = {
        "schema": INPUT_HASH_SCHEMA,
        "stage": stage,
        "handler": handler,
        "spec": dict(spec_payload or {}),
        "needs": sorted(str(item) for item in needs),
        "upstream": {str(key): str(value) for key, value in sorted(upstream_hashes.items())},
        "config": dict(config_subset),
        "workflow": {"name": workflow_name, "version": workflow_version},
        "as_of": str(as_of),
        "snapshot_hash": str(snapshot_hash),
        "seed": int(seed),
        "llm_mode": str(llm_mode),
        "prompts": {str(key): str(value) for key, value in sorted((prompt_hashes or {}).items())},
    }
    return hash_json(payload)


def _hash_artifact_file(path: Path) -> str:
    # Local import keeps pandas (imported by aicwf.util.artifact_hash) out of the
    # module import path of the lightweight controller CLI.
    from aicwf.util.artifact_hash import hash_artifact

    return hash_artifact(path)


def _artifact_fingerprint(path: Path) -> tuple[str, int]:
    """Return ``(canonical_content_hash, size_bytes)`` for a file or directory."""

    if path.is_dir():
        files = [child for child in sorted(path.rglob("*")) if child.is_file()]
        payload = {
            child.relative_to(path).as_posix(): _hash_artifact_file(child) for child in files
        }
        size = sum(child.stat().st_size for child in files)
        return hash_json({"dir": payload}), size
    return _hash_artifact_file(path), path.stat().st_size


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    """Execute a validated workflow DAG with durable, resumable state.

    Parameters
    ----------
    state:
        Open :class:`StateStore` (the runner never opens the DB itself so the CLI
        and tests control the path).
    registry:
        Mapping ``handler name -> StageFn`` supplied by ``aicwf/pipeline.py``.
    config:
        Full merged config bundle ``{workflow, strategy, gates, budgets, model, data}``.
        Exactly this mapping is handed to stages as ``RunContext.config``.
    runs_root:
        Directory holding ``<run_id>/`` run directories (default
        ``<project_root>/outputs/runs``).
    llm:
        Raw LLM backend; the runner wraps it in :class:`GuardedLLM` per stage.
    snapshot_dir / snapshot_hash:
        Optional snapshot override; otherwise resolved from config/``as_of``.
    budget_limits:
        Optional :class:`BudgetLimits` (or budgets-config mapping) override; by
        default loaded from ``config["budgets"]``.
    fault:
        Optional :class:`FaultSpec` or ``"crash_after:stage:1"`` string.
    project_root:
        Root used to resolve relative artifact paths and the default run dir.
    clock / sleep / exit_fn:
        Injectable time and hard-exit functions for deterministic tests.
    """

    def __init__(
        self,
        *,
        state: StateStore,
        registry: Mapping[str, StageFn],
        config: Mapping[str, Any],
        runs_root: Path | str | None = None,
        llm: LLMClient | None = None,
        snapshot_dir: Path | str | None = None,
        snapshot_hash: str | None = None,
        budget_limits: BudgetLimits | Mapping[str, Any] | None = None,
        fault: FaultSpec | str | None = None,
        project_root: Path | str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        exit_fn: Callable[[int], NoReturn] | None = None,
        echo_events: bool = False,
    ) -> None:
        if not isinstance(state, StateStore):
            raise ConfigError("Runner requires an aicwf.controller.state.StateStore instance")
        self.state = state
        self.config = dict(config or {})
        self.workflow: Workflow = load_workflow(self.config)
        self.registry: dict[str, StageFn] = dict(registry or {})
        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        default_runs_root = self.project_root / "outputs" / "runs"
        self.runs_root = Path(runs_root) if runs_root is not None else default_runs_root
        self.llm = llm
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None
        self.snapshot_hash = snapshot_hash
        if fault is None:
            self.fault: FaultSpec | None = None
        elif isinstance(fault, FaultSpec):
            self.fault = fault
        else:
            self.fault = FaultSpec.parse(str(fault))
        self.retry_policy = RetryPolicy.from_config(self.config.get("budgets"))
        self.gate = ApprovalGate(state, self.config)
        self.clock = clock
        self.sleep = sleep
        self.exit_fn = exit_fn or os._exit
        self.echo_events = echo_events

        if budget_limits is None:
            self._limits_override: BudgetLimits | None = None
        elif isinstance(budget_limits, BudgetLimits):
            self._limits_override = budget_limits
        elif isinstance(budget_limits, Mapping):
            self._limits_override = BudgetLimits.from_config(budget_limits)
        else:
            raise ConfigError("budget_limits must be BudgetLimits, a budgets mapping, or None")

        self._record: RunRecord | None = None
        self._outputs: dict[str, StageOutput] = {}
        self._events: RunEventSink | None = None
        self._ledger: BudgetLedger | None = None
        self._base_ctx: RunContext | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        as_of: str | dt.date,
        run_id: str | None = None,
        seed: int = 0,
        llm_mode: str = "replay",
        trigger: str = "",
    ) -> RunResult:
        """Start a new run for ``as_of`` (raises if ``run_id`` already exists)."""

        if llm_mode not in LLM_MODES:
            raise ConfigError(f"unknown llm_mode {llm_mode!r}; expected one of {list(LLM_MODES)}")
        as_of_date = parse_date(as_of)
        resolved_run_id = str(run_id) if run_id else generate_run_id(as_of_date)
        if self.state.get_run(resolved_run_id) is not None:
            raise WorkflowError(
                f"run {resolved_run_id!r} already exists; use 'aicwf resume {resolved_run_id}'"
            )
        snapshot_path, snapshot_digest = resolve_snapshot_identity(
            as_of_date,
            self.config,
            project_root=self.project_root,
            snapshot_dir=self.snapshot_dir,
        )
        if self.snapshot_hash:
            snapshot_digest = str(self.snapshot_hash)
        run_dir = self.runs_root / resolved_run_id
        ensure_dir(run_dir)
        now = utc_now_iso()
        record = RunRecord(
            run_id=resolved_run_id,
            as_of=as_of_date.isoformat(),
            status="running",
            run_dir=str(run_dir),
            llm_mode=str(llm_mode),
            seed=int(seed),
            workflow_version=self.workflow.version,
            config_hash=hash_json(self.config),
            snapshot_hash=snapshot_digest,
            snapshot_dir=str(snapshot_path) if snapshot_path is not None else "",
            trigger=str(trigger or ""),
            created_at_utc=now,
            updated_at_utc=now,
            started_at_utc=now,
        )
        self.state.create_run(record)
        self._events = RunEventSink(self.state, run_dir / "events.jsonl", resolved_run_id, echo=self.echo_events)
        self._events.emit(
            "run_created",
            payload={
                "as_of": record.as_of,
                "llm_mode": record.llm_mode,
                "seed": record.seed,
                "workflow_name": self.workflow.name,
                "workflow_version": self.workflow.version,
                "snapshot_hash": record.snapshot_hash,
                "snapshot_dir": record.snapshot_dir,
                "trigger": record.trigger,
            },
        )
        return self._execute(record)

    def resume(self, run_id: str, *, force: bool = False) -> RunResult:
        """Resume an interrupted/paused run without repeating completed stages."""

        record = self.state.get_run(run_id)
        if record is None:
            raise WorkflowError(f"unknown run {run_id!r}")
        if record.status == "completed":
            return self._build_result(record)
        if record.status in _FORCE_RESUME_STATUSES and not force:
            raise WorkflowError(
                f"run {run_id!r} is {record.status!r}; resolve the escalation then resume with --force"
            )
        if record.status not in _RESUME_ALLOWED and record.status not in _FORCE_RESUME_STATUSES:
            raise WorkflowError(f"run {run_id!r} cannot be resumed from status {record.status!r}")
        run_dir = Path(record.run_dir or (self.runs_root / run_id))
        ensure_dir(run_dir)
        now = utc_now_iso()
        previous_status = record.status
        record = self.state.update_run(
            run_id,
            status="running",
            current_stage=None,
            error=None,
            finished_at_utc=None,
            updated_at_utc=now,
        )
        reset = self.state.reset_running_steps(run_id)
        self._events = RunEventSink(self.state, run_dir / "events.jsonl", run_id, echo=self.echo_events)
        self._events.emit(
            "run_resumed",
            payload={"reset_steps": reset, "force": bool(force), "previous_status": previous_status},
        )
        return self._execute(record)

    def input_hash(self, run_id: str, stage: str) -> str | None:
        """Current deterministic input hash for one stage of an existing run."""

        record = self.state.get_run(run_id)
        if record is None:
            return None
        self._record = record
        spec = self.workflow.stage(stage)
        return self._compute_input_hash(spec)

    def invalidate_stages(
        self,
        run_id: str,
        stages: Sequence[str],
        *,
        reason: str = "",
    ) -> list[str]:
        """Plan a bounded revision: mark steps (and descendants) pending again.

        Used by the CLI's bounded revision loop after an independent verification
        FAIL. Only execution state is reset — the step rows become ``pending`` with
        cleared hashes/summary and a ``step_invalidated`` event is recorded — while
        the artifact registrations and the artifact files themselves are left
        untouched (they are overwritten by the re-run). The next
        ``resume(run_id, force=True)`` therefore re-executes exactly the named
        stages and everything downstream of them; unrelated completed stages are
        still skipped by their deterministic input hash.

        Returns the invalidated stage names in workflow order.
        """

        record = self.state.get_run(run_id)
        if record is None:
            raise WorkflowError(f"unknown run {run_id!r}")
        requested: list[str] = []
        for name in stages:
            requested.append(self.workflow.stage(str(name)).name)  # validates name
        if not requested:
            return []
        targets = set(self.workflow.downstream_closure(requested, include_self=True))

        event_sink = self._events
        if event_sink is None or getattr(event_sink, "run_id", run_id) != run_id:
            run_dir = Path(record.run_dir or (self.runs_root / run_id))
            ensure_dir(run_dir)
            event_sink = RunEventSink(
                self.state, run_dir / "events.jsonl", run_id, echo=self.echo_events
            )
            self._events = event_sink

        reset: list[str] = []
        for spec in self.workflow.order:
            if spec.name not in targets:
                continue
            step = self.state.get_step(run_id, spec.name)
            if step is None:
                continue
            step.status = "pending"
            step.input_hash = None
            step.output_hash = None
            step.summary = {}
            step.error = None
            step.error_type = None
            step.last_disposition = f"invalidated_revision:{reason}" if reason else "invalidated_revision"
            step.updated_at_utc = utc_now_iso()
            self.state.upsert_step(step)
            self._outputs.pop(spec.name, None)
            reset.append(spec.name)
            event_sink.emit(
                "step_invalidated",
                payload={
                    "stage": spec.name,
                    "reason": reason or "revision",
                    "source": "invalidate_stages",
                },
            )
        if reset:
            event_sink.emit(
                "revision_planned",
                payload={"stages": reset, "reason": reason, "requested": requested},
            )
        return reset

    # ------------------------------------------------------------------
    # Execution machinery
    # ------------------------------------------------------------------
    def _execute(self, record: RunRecord) -> RunResult:
        run_dir = Path(record.run_dir)
        self._record = record
        self._outputs = {}
        self._ledger = self._make_ledger()
        persisted_used = {budget.metric: budget.used_value for budget in self.state.list_budgets(record.run_id)}
        self._ledger.restore(persisted_used)
        self._ledger.start()
        if self._events is None:
            self._events = RunEventSink(
                self.state, run_dir / "events.jsonl", record.run_id, echo=self.echo_events
            )
        self._base_ctx = RunContext(
            run_id=record.run_id,
            as_of=parse_date(record.as_of),
            run_dir=run_dir,
            config=self.config,
            llm=None,
            seed=int(record.seed),
            llm_mode=record.llm_mode,
            fault=self.fault,
            snapshot_dir=Path(record.snapshot_dir) if record.snapshot_dir else None,
            stage_outputs=self._outputs,
            events=self._events,
            budget=self._ledger,
        )
        try:
            return self._run_loop()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # controller-level bug: stop cleanly, keep state
            normalise_cancelled(exc)
            error_type, message = describe_exception(exc)
            if self._events is not None:
                self._events.emit(
                    "controller_error",
                    level="error",
                    payload={"error_type": error_type, "error": message},
                )
            stage = None
            current = self.state.get_run(record.run_id)
            if current is not None and current.current_stage:
                stage = current.current_stage
            return self._finalize("failed", stage=stage, error=f"{error_type}: {message}")

    def _run_loop(self) -> RunResult:
        record = self._require_record()
        run_id = record.run_id
        for spec in self.workflow.order:
            self.state.update_run(run_id, current_stage=spec.name)
            step = self.state.get_step(run_id, spec.name)
            if step is not None and step.status == "completed":
                reason = self._completed_step_reason(spec, step)
                if reason is None:
                    self._mark_skipped(spec, step)
                    self._load_stage_output(spec, step)
                    continue
                self._invalidate_stage(spec, reason)
            else:
                # A step left blocked/failed/pending by an earlier interruption is
                # simply re-executed below; its attempt history is preserved.
                pass

            outcome = self._execute_stage(spec)
            if outcome.status != "completed":
                return self._finalize_failure(spec, outcome)
            self._persist_budgets()
            try:
                self._require_ledger().check()
            except BudgetExceeded as exc:
                self._events_emit("run_budget_exhausted", payload={"stage": spec.name, "error": str(exc)})
                return self._finalize("budget_exhausted", stage=spec.name, error=str(exc))
        return self._finalize("completed")

    # -- completed-step validation & invalidation ----------------------
    def _completed_step_reason(self, spec: StageSpec, step: StepRecord) -> str | None:
        """``None`` if the completed step can be reused, else the stale reason."""

        record = self._require_record()
        computed = self._compute_input_hash(spec)
        if step.input_hash != computed:
            return "input_hash_changed"
        for artifact in self.state.list_artifacts(record.run_id, spec.name):
            path = self._resolve_path(artifact.relpath)
            if not path.exists():
                return f"artifact_missing:{artifact.logical_name}"
            if artifact.size_bytes and path.stat().st_size != artifact.size_bytes:
                return f"artifact_size_changed:{artifact.logical_name}"
        return None

    def _mark_skipped(self, spec: StageSpec, step: StepRecord) -> None:
        step.skip_count += 1
        step.last_disposition = "skipped_completed"
        step.updated_at_utc = utc_now_iso()
        self.state.upsert_step(step)
        self._events_emit(
            "step_skipped_completed",
            payload={
                "stage": spec.name,
                "input_hash": step.input_hash,
                "skip_count": step.skip_count,
            },
        )

    def _load_stage_output(self, spec: StageSpec, step: StepRecord) -> None:
        artifacts = self.state.list_artifacts(step.run_id, spec.name)
        self._outputs[spec.name] = StageOutput(
            artifacts={artifact.logical_name: artifact.relpath for artifact in artifacts},
            summary=dict(step.summary),
        )

    def _invalidate_stage(self, spec: StageSpec, reason: str) -> None:
        record = self._require_record()
        targets = (spec.name,) + self.workflow.descendants(spec.name)
        for stage_name in targets:
            existing = self.state.get_step(record.run_id, stage_name)
            if existing is None:
                continue
            self.state.invalidate_step(record.run_id, stage_name, reason=reason)
            self._outputs.pop(stage_name, None)
            self._events_emit(
                "step_invalidated",
                payload={"stage": stage_name, "reason": reason, "caused_by": spec.name},
            )

    # -- gate check ----------------------------------------------------
    def _check_gate(self, spec: StageSpec) -> _StageOutcome | None:
        record = self._require_record()
        action = self.gate.action_for_stage(spec)
        if self.gate.is_satisfied(record.run_id, action):
            self._events_emit("stage_gate_satisfied", payload={"stage": spec.name, "action": action})
            return None
        latest = self.gate.latest_decision(record.run_id)
        if latest is not None and latest.decision == "rejected":
            error = f"approval rejected by {latest.approver or 'unknown'}: {latest.note}".strip()
            self._write_blocked_step(spec, disposition="approval_rejected")
            return _StageOutcome(
                status="blocked",
                stage=spec.name,
                category=CATEGORY_APPROVAL,
                reason="approval_rejected",
                error=error,
                error_type="ApprovalRejected",
            )
        self._write_blocked_step(spec, disposition="awaiting_approval")
        request = self.gate.build_request(record.run_id, action=action, stage=spec.name)
        write_approval_request_artifact(request, Path(record.run_dir))
        self._events_emit(
            "stage_awaiting_approval",
            payload={
                "stage": spec.name,
                "action": action,
                "instructions": request.instructions,
                "artifact_hashes": dict(request.artifact_hashes),
            },
        )
        self._persist_budgets()
        return _StageOutcome(
            status="awaiting_approval",
            stage=spec.name,
            category=CATEGORY_APPROVAL,
            reason="approval_required",
            action=action,
        )

    def _write_blocked_step(self, spec: StageSpec, *, disposition: str) -> StepRecord:
        record = self._require_record()
        step = self.state.get_step(record.run_id, spec.name) or StepRecord(
            run_id=record.run_id, stage=spec.name
        )
        step.status = "blocked"
        step.last_disposition = disposition
        step.updated_at_utc = utc_now_iso()
        self.state.upsert_step(step)
        return step

    # -- stage execution ----------------------------------------------
    def _execute_stage(self, spec: StageSpec) -> _StageOutcome:
        record = self._require_record()
        run_id = record.run_id
        if spec.human_gate:
            gate_outcome = self._check_gate(spec)
            if gate_outcome is not None:
                return gate_outcome

        handler = self.registry.get(spec.handler)
        if handler is not None and not callable(handler):
            return self._outcome_for_exception(
                spec,
                attempt=1,
                input_hash=self._compute_input_hash(spec),
                exc=ConfigError(f"registered handler for {spec.handler!r} is not callable"),
                started_iso=utc_now_iso(),
                finished_iso=utc_now_iso(),
                duration=0.0,
            )
        if handler is None:
            return self._outcome_for_exception(
                spec,
                attempt=1,
                input_hash=self._compute_input_hash(spec),
                exc=ConfigError(
                    f"no stage handler registered for handler {spec.handler!r} "
                    f"(stage {spec.name!r}); wire it in aicwf/pipeline.py build_registry()"
                ),
                started_iso=utc_now_iso(),
                finished_iso=utc_now_iso(),
                duration=0.0,
            )

        input_hash = self._compute_input_hash(spec)
        attempts = self.state.list_attempts(run_id, spec.name)
        failures = sum(1 for attempt in attempts if attempt.status == "failed")
        attempt_no = len(attempts) + 1
        max_retries = self.retry_policy.max_retries(spec.retries)
        existing = self.state.get_step(run_id, spec.name)
        skip_count = existing.skip_count if existing is not None else 0

        while True:
            if self._fault_matches("crash_before", spec.name, attempt_no):
                self._record_fault(spec, attempt_no, input_hash, kind="crash_before")
                self.exit_fn(42)  # pragma: no cover - process terminates
                raise AssertionError("exit_fn returned")  # pragma: no cover
            try:
                self._require_ledger().check()
            except BudgetExceeded as exc:
                return _StageOutcome(
                    status="budget_exhausted",
                    stage=spec.name,
                    attempt=attempt_no,
                    category=CATEGORY_BUDGET,
                    reason="budget_exhausted",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            started_iso = utc_now_iso()
            started_clock = self.clock()
            step = StepRecord(
                run_id=run_id,
                stage=spec.name,
                status="running",
                attempt=attempt_no,
                max_attempts=max_retries + 1,
                input_hash=input_hash,
                started_at_utc=started_iso,
                skip_count=skip_count,
                last_disposition="running",
                updated_at_utc=started_iso,
            )
            self.state.upsert_step(step)
            self.state.record_attempt(
                AttemptRecord(
                    run_id=run_id,
                    stage=spec.name,
                    attempt=attempt_no,
                    status="running",
                    started_at_utc=started_iso,
                )
            )
            self._events_emit(
                "stage_started",
                payload={
                    "stage": spec.name,
                    "handler": spec.handler,
                    "attempt": attempt_no,
                    "max_attempts": max_retries + 1,
                    "input_hash": input_hash,
                },
            )
            try:
                context = self._attempt_context(spec)
                output = self._invoke(
                    handler, context, timeout_s=spec.timeout_s, stage=spec.name
                )
                artifacts, summary = self._validate_stage_output(spec, output)
                duration = max(0.0, self.clock() - started_clock)
                if self._fault_matches("crash_after", spec.name, attempt_no):
                    self._record_fault(
                        spec, attempt_no, input_hash, kind="crash_after", artifacts=artifacts
                    )
                    self.exit_fn(42)  # pragma: no cover - process terminates
                    raise AssertionError("exit_fn returned")  # pragma: no cover
                self._complete_stage(
                    spec,
                    attempt_no=attempt_no,
                    input_hash=input_hash,
                    output=output,
                    artifacts=artifacts,
                    summary=summary,
                    started_iso=started_iso,
                    duration_s=duration,
                    skip_count=skip_count,
                    max_retries=max_retries,
                )
                return _StageOutcome(status="completed", stage=spec.name, attempt=attempt_no)
            except Exception as exc:
                normalise_cancelled(exc)
                duration = max(0.0, self.clock() - started_clock)
                finished_iso = utc_now_iso()
                return self._outcome_for_exception(
                    spec,
                    attempt=attempt_no,
                    input_hash=input_hash,
                    exc=exc,
                    started_iso=started_iso,
                    finished_iso=finished_iso,
                    duration=duration,
                    failures=failures,
                    max_retries=max_retries,
                )

    def _outcome_for_exception(
        self,
        spec: StageSpec,
        *,
        attempt: int,
        input_hash: str,
        exc: BaseException,
        started_iso: str,
        finished_iso: str,
        duration: float,
        failures: int = 0,
        max_retries: int = 0,
    ) -> _StageOutcome:
        record = self._require_record()
        decision = self.retry_policy.decide(
            exc,
            attempt=attempt,
            stage_retries=spec.retries,
            seed=f"{record.run_id}:{spec.name}",
        )
        will_retry = decision.retryable and failures < max_retries
        step_status = "failed"
        if decision.category == CATEGORY_APPROVAL:
            step_status = "blocked"
        elif decision.category == CATEGORY_ESCALATE:
            step_status = "blocked"
        elif decision.category == CATEGORY_RETRYABLE and will_retry:
            step_status = "failed"
        elif decision.category == CATEGORY_RETRYABLE:
            step_status = "blocked"
        existing = self.state.get_step(record.run_id, spec.name)
        step = StepRecord(
            run_id=record.run_id,
            stage=spec.name,
            status=step_status,
            attempt=attempt,
            max_attempts=max_retries + 1,
            input_hash=input_hash,
            started_at_utc=started_iso,
            finished_at_utc=finished_iso,
            duration_s=duration,
            error=decision.error_message,
            error_type=decision.error_type,
            skip_count=existing.skip_count if existing is not None else 0,
            last_disposition=decision.category,
            updated_at_utc=finished_iso,
        )
        self.state.fail_stage(
            step=step,
            attempt=AttemptRecord(
                run_id=record.run_id,
                stage=spec.name,
                attempt=attempt,
                status="failed",
                started_at_utc=started_iso,
                finished_at_utc=finished_iso,
                duration_s=duration,
                error_type=decision.error_type,
                error=decision.error_message,
            ),
        )
        self._events_emit(
            "stage_failed",
            payload={
                "stage": spec.name,
                "attempt": attempt,
                "category": decision.category,
                "will_retry": will_retry,
                "error_type": decision.error_type,
                "error": decision.error_message,
                "backoff_seconds": decision.backoff_seconds,
            },
        )
        if decision.retryable and will_retry:
            self._persist_budgets()
            self._events_emit(
                "stage_retry_scheduled",
                payload={
                    "stage": spec.name,
                    "next_attempt": attempt + 1,
                    "backoff_seconds": decision.backoff_seconds,
                    "error_type": decision.error_type,
                },
            )
            if decision.backoff_seconds > 0:
                self.sleep(decision.backoff_seconds)
            return self._execute_stage_retry_loop(spec, decision)

        if decision.category == CATEGORY_BUDGET:
            return _StageOutcome(
                status="budget_exhausted",
                stage=spec.name,
                attempt=attempt,
                category=CATEGORY_BUDGET,
                reason="budget_exhausted",
                error=decision.error_message,
                error_type=decision.error_type,
                decision=decision,
            )
        if decision.category == CATEGORY_APPROVAL:
            return _StageOutcome(
                status="awaiting_approval",
                stage=spec.name,
                attempt=attempt,
                category=CATEGORY_APPROVAL,
                reason="approval_required",
                error=decision.error_message,
                error_type=decision.error_type,
                decision=decision,
            )
        if decision.category == CATEGORY_RETRYABLE:
            return _StageOutcome(
                status="blocked",
                stage=spec.name,
                attempt=attempt,
                category=CATEGORY_RETRYABLE,
                reason="retries_exhausted",
                error=decision.error_message,
                error_type=decision.error_type,
                decision=decision,
            )
        if decision.category == CATEGORY_ESCALATE:
            return _StageOutcome(
                status="blocked",
                stage=spec.name,
                attempt=attempt,
                category=CATEGORY_ESCALATE,
                reason="escalation_required",
                error=decision.error_message,
                error_type=decision.error_type,
                decision=decision,
            )
        return _StageOutcome(
            status="failed",
            stage=spec.name,
            attempt=attempt,
            category=CATEGORY_NON_RETRYABLE,
            reason="deterministic_failure",
            error=decision.error_message,
            error_type=decision.error_type,
            decision=decision,
        )

    def _execute_stage_retry_loop(self, spec: StageSpec, previous: RetryDecision) -> _StageOutcome:
        """Continue the retry loop after a transient failure.

        Retries re-enter :meth:`_execute_stage` with the attempt history already
        recorded, which keeps attempt numbering and error logging in one place.
        """

        # The failed attempt was persisted above; re-enter the stage executor.
        # Guard against infinite recursion: _execute_stage recomputes the attempt
        # number from persisted attempts, so no explicit counter is needed.
        return self._execute_stage(spec)

    # -- handler invocation & validation -------------------------------
    def _invoke(
        self,
        handler: StageFn,
        context: RunContext,
        *,
        timeout_s: float,
        stage: str,
    ) -> StageOutput:
        if timeout_s is None or timeout_s <= 0:
            return handler(context)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"aicwf-{stage}")
        future = executor.submit(handler, context)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            raise StageTimeout(f"stage {stage!r} exceeded timeout_s={timeout_s:g}") from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_stage_output(
        self, spec: StageSpec, output: Any
    ) -> tuple[dict[str, tuple[str, str, int]], dict[str, Any]]:
        if not isinstance(output, StageOutput):
            raise ValidationError(
                f"stage {spec.name!r} handler returned {type(output).__name__}; expected StageOutput"
            )
        if not isinstance(output.artifacts, Mapping):
            raise ValidationError(f"stage {spec.name!r} StageOutput.artifacts must be a mapping")
        validated: dict[str, tuple[str, str, int]] = {}
        for logical, raw_path in output.artifacts.items():
            if not isinstance(logical, str) or not logical.strip():
                raise ValidationError(f"stage {spec.name!r} artifact names must be non-empty strings")
            if not isinstance(raw_path, (str, Path)):
                raise ValidationError(
                    f"stage {spec.name!r} artifact {logical!r} path must be a string, "
                    f"got {type(raw_path).__name__}"
                )
            path = self._resolve_path(raw_path)
            if not path.exists():
                raise ValidationError(
                    f"stage {spec.name!r} declared artifact {logical!r} at {raw_path!r} "
                    "which does not exist on disk"
                )
            digest, size = _artifact_fingerprint(path)
            validated[logical] = (self._store_relpath(path), digest, size)
        summary_raw = output.summary if output.summary is not None else {}
        if not isinstance(summary_raw, Mapping):
            raise ValidationError(f"stage {spec.name!r} StageOutput.summary must be a mapping")
        summary = {str(key): value for key, value in summary_raw.items()}
        try:
            json.dumps(summary, default=str)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"stage {spec.name!r} summary is not JSON-serialisable: {exc}") from None
        return validated, summary

    def _complete_stage(
        self,
        spec: StageSpec,
        *,
        attempt_no: int,
        input_hash: str,
        output: StageOutput,
        artifacts: Mapping[str, tuple[str, str, int]],
        summary: Mapping[str, Any],
        started_iso: str,
        duration_s: float,
        skip_count: int,
        max_retries: int,
    ) -> None:
        record = self._require_record()
        finished_iso = utc_now_iso()
        output_hash = hash_json(
            {
                "artifacts": {
                    str(logical): {"relpath": values[0], "content_hash": values[1]}
                    for logical, values in sorted(artifacts.items())
                },
                "summary": dict(summary),
            }
        )
        artifact_records = [
            ArtifactRecord(
                run_id=record.run_id,
                stage=spec.name,
                logical_name=logical,
                relpath=values[0],
                content_hash=values[1],
                size_bytes=values[2],
                created_at_utc=finished_iso,
            )
            for logical, values in sorted(artifacts.items())
        ]
        step = StepRecord(
            run_id=record.run_id,
            stage=spec.name,
            status="completed",
            attempt=attempt_no,
            max_attempts=max_retries + 1,
            input_hash=input_hash,
            output_hash=output_hash,
            started_at_utc=started_iso,
            finished_at_utc=finished_iso,
            duration_s=duration_s,
            summary=dict(summary),
            skip_count=skip_count,
            last_disposition="executed",
            updated_at_utc=finished_iso,
        )
        self.state.complete_stage(
            step=step,
            attempt=AttemptRecord(
                run_id=record.run_id,
                stage=spec.name,
                attempt=attempt_no,
                status="completed",
                started_at_utc=started_iso,
                finished_at_utc=finished_iso,
                duration_s=duration_s,
            ),
            artifacts=artifact_records,
        )
        self._outputs[spec.name] = StageOutput(
            artifacts={logical: values[0] for logical, values in artifacts.items()},
            summary=dict(summary),
        )
        self._events_emit(
            "stage_completed",
            payload={
                "stage": spec.name,
                "attempt": attempt_no,
                "duration_s": round(float(duration_s), 6),
                "input_hash": input_hash,
                "output_hash": output_hash,
                "artifacts": {
                    logical: {"relpath": values[0], "content_hash": values[1], "size_bytes": values[2]}
                    for logical, values in sorted(artifacts.items())
                },
            },
        )

    # -- fault injection ----------------------------------------------
    def _fault_matches(self, kind: str, stage: str, attempt: int) -> bool:
        fault = self.fault
        if fault is None:
            return False
        return fault.kind == kind and fault.stage == stage and int(fault.attempt) == int(attempt)

    def _record_fault(
        self,
        spec: StageSpec,
        attempt_no: int,
        input_hash: str,
        *,
        kind: str,
        artifacts: Mapping[str, tuple[str, str, int]] | None = None,
    ) -> None:
        record = self._require_record()
        now = utc_now_iso()
        existing = self.state.get_step(record.run_id, spec.name)
        step = existing or StepRecord(run_id=record.run_id, stage=spec.name)
        step.status = "running"
        step.attempt = attempt_no
        step.input_hash = input_hash
        step.last_disposition = f"fault:{kind}"
        step.updated_at_utc = now
        self.state.upsert_step(step)
        self.state.record_attempt(
            AttemptRecord(
                run_id=record.run_id,
                stage=spec.name,
                attempt=attempt_no,
                status="crashed",
                started_at_utc=step.started_at_utc or now,
                finished_at_utc=now,
            )
        )
        payload: dict[str, Any] = {
            "stage": spec.name,
            "attempt": attempt_no,
            "kind": kind,
            "exit_code": 42,
            "input_hash": input_hash,
        }
        if artifacts:
            payload["artifacts"] = {
                logical: {"relpath": values[0], "content_hash": values[1]}
                for logical, values in sorted(artifacts.items())
            }
        self._events_emit("fault_injected", level="warning", payload=payload)
        self._persist_budgets()

    # -- finalisation --------------------------------------------------
    def _finalize_failure(self, spec: StageSpec, outcome: _StageOutcome) -> RunResult:
        record = self._require_record()
        run_dir = Path(record.run_dir)
        if outcome.status == "awaiting_approval":
            self._events_emit(
                "run_awaiting_approval",
                payload={"stage": spec.name, "action": outcome.action},
            )
            return self._finalize("awaiting_approval", stage=spec.name, error=None)
        if outcome.status == "blocked":
            escalation = build_escalation_record(
                record.run_id,
                spec.name,
                category=outcome.category,
                reason=outcome.reason,
                error_type=outcome.error_type or "",
                error=outcome.error or "",
                attempts=outcome.attempt,
                extra={"retry_decision": outcome.decision.to_dict()} if outcome.decision else None,
            )
            write_escalation_artifact(escalation, run_dir)
            self._events_emit("run_blocked", level="error", payload=escalation.to_dict())
            return self._finalize("blocked", stage=spec.name, error=outcome.error)
        if outcome.status == "budget_exhausted":
            self._events_emit(
                "run_budget_exhausted",
                payload={"stage": spec.name, "error": outcome.error},
            )
            return self._finalize("budget_exhausted", stage=spec.name, error=outcome.error)
        self._events_emit(
            "run_failed",
            level="error",
            payload={"stage": spec.name, "category": outcome.category, "error": outcome.error},
        )
        return self._finalize("failed", stage=spec.name, error=outcome.error)

    def _finalize(self, status: str, *, stage: str | None = None, error: str | None = None) -> RunResult:
        record = self._require_record()
        run_dir = Path(record.run_dir)
        now = utc_now_iso()
        self._persist_budgets()
        updated = self.state.update_run(
            record.run_id,
            status=status,
            current_stage=stage,
            error=error,
            finished_at_utc=now if status not in {"running", "pending"} else None,
            updated_at_utc=now,
        )
        self._record = updated
        self._write_state_snapshot(run_dir)
        self._write_run_summary(run_dir)
        self._events_emit(
            "run_finished",
            payload={
                "status": status,
                "stage": stage,
                "error": error,
                "budget": self._require_ledger().to_dict(),
            },
        )
        return self._build_result(updated)

    # -- persistence helpers -------------------------------------------
    def _make_ledger(self) -> BudgetLedger:
        ledger = BudgetLedger.from_config(self.config.get("budgets"), clock=self.clock)
        if self._limits_override is not None:
            ledger.limits = self._limits_override
        return ledger

    def _persist_budgets(self) -> None:
        if self._ledger is None or self._record is None:
            return
        self.state.upsert_budgets(self._record.run_id, self._ledger.snapshot())

    def _write_state_snapshot(self, run_dir: Path) -> None:
        record = self._require_record()
        payload = self.state.export_run(record.run_id)
        write_json(run_dir / "state_snapshot.json", payload, sort_keys=True)

    def _write_run_summary(self, run_dir: Path) -> None:
        record = self._require_record()
        steps = self.state.list_steps(record.run_id)
        artifacts = self.state.list_artifacts(record.run_id)
        budgets = self.state.list_budgets(record.run_id)
        summary = {
            "run_id": record.run_id,
            "as_of": record.as_of,
            "status": record.status,
            "llm_mode": record.llm_mode,
            "seed": record.seed,
            "workflow_version": record.workflow_version,
            "config_hash": record.config_hash,
            "snapshot_hash": record.snapshot_hash,
            "trigger": record.trigger,
            "error": record.error,
            "steps": [
                {
                    "stage": step.stage,
                    "status": step.status,
                    "attempt": step.attempt,
                    "skip_count": step.skip_count,
                    "last_disposition": step.last_disposition,
                    "input_hash": step.input_hash,
                    "output_hash": step.output_hash,
                    "duration_s": step.duration_s,
                    "error_type": step.error_type,
                }
                for step in steps
            ],
            "artifacts": [
                {
                    "stage": artifact.stage,
                    "logical_name": artifact.logical_name,
                    "relpath": artifact.relpath,
                    "content_hash": artifact.content_hash,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in artifacts
            ],
            "budget": {
                budget.metric: {"limit": budget.limit_value, "used": budget.used_value}
                for budget in budgets
            },
            "next_action": self._next_action(record, steps),
        }
        write_json(run_dir / "run_summary.json", summary, sort_keys=True)

    def _next_action(self, record: RunRecord, steps: Sequence[StepRecord]) -> str:
        status = record.status
        if status == "completed":
            return "none"
        if status == "awaiting_approval":
            pending = self.gate.pending_gate_actions(self.workflow, record.run_id)
            if pending:
                _stage, action = pending[0]
                return f"python -m aicwf approve {record.run_id} --action {action}"
            return f"python -m aicwf resume {record.run_id}"
        if status == "blocked":
            return f"read {Path(record.run_dir) / 'escalation.json'} then 'python -m aicwf resume {record.run_id} --force'"
        if status == "budget_exhausted":
            return "raise limits in config/budgets.yaml then 'python -m aicwf resume <run_id>'"
        if status == "failed":
            return "inspect events.jsonl and run_summary.json; resume with --force after fixing"
        return f"python -m aicwf resume {record.run_id}"

    def _build_result(self, record: RunRecord) -> RunResult:
        steps = self.state.list_steps(record.run_id)
        budgets = self.state.list_budgets(record.run_id)
        return RunResult(
            run_id=record.run_id,
            status=record.status,
            run_dir=record.run_dir,
            as_of=record.as_of,
            error=record.error,
            steps=[step.to_dict() for step in steps],
            budget={
                budget.metric: {"limit": budget.limit_value, "used": budget.used_value}
                for budget in budgets
            },
        )

    # -- small utilities ------------------------------------------------
    def _require_record(self) -> RunRecord:
        if self._record is None:
            raise WorkflowError("runner has no active run; call run() or resume() first")
        return self._record

    def _require_ledger(self) -> BudgetLedger:
        if self._ledger is None:
            raise WorkflowError("runner has no active budget ledger")
        return self._ledger

    def _attempt_context(self, spec: StageSpec) -> RunContext:
        """Build the per-attempt context, including the live approval gate.

        ``RunContext`` has no approval field in the frozen contract, so the
        controller attaches the gate as an extra attribute (``ctx.approval_gate``).
        The pipeline's approval-stage handler can use it to write ``approval.json``
        without opening the state database itself.
        """

        record = self._require_record()
        ledger = self._require_ledger()
        llm: GuardedLLM | None = None
        if self.llm is not None:
            llm = GuardedLLM(
                self.llm,
                ledger,
                run_id=record.run_id,
                stage=spec.name,
            )
        base = self._base_ctx
        if base is None:  # pragma: no cover - defensive
            raise WorkflowError("runner base context was not initialised")
        context = RunContext(
            run_id=base.run_id,
            as_of=base.as_of,
            run_dir=base.run_dir,
            config=base.config,
            llm=llm,
            seed=base.seed,
            llm_mode=base.llm_mode,
            fault=base.fault,
            snapshot_dir=base.snapshot_dir,
            stage_outputs=dict(self._outputs),
            events=base.events,
            budget=ledger,
        )
        # Extra integration attribute (not part of the frozen RunContext schema).
        context.approval_gate = self.gate  # type: ignore[attr-defined]
        return context

    def _compute_input_hash(self, spec: StageSpec) -> str:
        record = self._require_record()
        upstream: dict[str, str] = {}
        for need in spec.needs:
            need_step = self.state.get_step(record.run_id, need)
            if need_step is None or need_step.status != "completed":
                raise WorkflowError(
                    f"cannot compute input hash for stage {spec.name!r}: "
                    f"upstream stage {need!r} is not completed"
                )
            for artifact in sorted(
                self.state.list_artifacts(record.run_id, need), key=lambda item: item.logical_name
            ):
                upstream[f"{need}/{artifact.logical_name}"] = artifact.content_hash
        return compute_input_hash(
            stage=spec.name,
            handler=spec.handler,
            needs=spec.needs,
            upstream_hashes=upstream,
            config_subset=stage_config_subset(spec.name, self.config),
            workflow_name=self.workflow.name,
            workflow_version=self.workflow.version,
            as_of=record.as_of,
            snapshot_hash=record.snapshot_hash,
            seed=record.seed,
            llm_mode=record.llm_mode,
            prompt_hashes=self._prompt_hashes(spec),
            spec_payload={
                "name": spec.name,
                "handler": spec.handler,
                "needs": list(spec.needs),
                "agent": dict(spec.agent),
            },
        )

    def _prompt_hashes(self, spec: StageSpec) -> dict[str, str]:
        if not spec.agent:
            return {}
        hashes: dict[str, str] = {}
        for key in ("prompt", "skill", "schema"):
            value = spec.agent.get(key)
            if not value:
                continue
            path = self._resolve_path(str(value))
            if path.is_file():
                hashes[key] = hash_text(path.read_text(encoding="utf-8"))
            else:
                hashes[key] = hash_text("")
        return hashes

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _store_relpath(self, path: Path) -> str:
        # Contract paths are relative to the repository root (aicwf.contracts.PROJECT_ROOT);
        # artifacts written outside it (tests with tmp_path) stay absolute so
        # RunContext.upstream_artifact() and later stages can find them.
        return relpath(path, PROJECT_ROOT)

    def _events_emit(self, kind: str, *, level: str = "info", payload: Mapping[str, Any] | None = None) -> None:
        if self._events is not None:
            self._events.emit(kind, level=level, payload=dict(payload or {}))
