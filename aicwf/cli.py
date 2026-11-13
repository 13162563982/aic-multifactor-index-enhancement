"""Command line interface for the AICWF multifactor index-enhancement workflow.

Usage overview::

    python -m aicwf run --as-of 2026-11-12 --llm replay
    python -m aicwf run --as-of 2026-11-12 --max-revisions 2   # bounded verification revisions
    python -m aicwf resume <run_id> [--force] [--max-revisions N]
    python -m aicwf status [<run_id>] [--json]
    python -m aicwf approve <run_id> --action publish_report
    python -m aicwf state-export [--run-id <run_id>] [--output state.json]
    python -m aicwf doctor
    python -m aicwf fault-demo --as-of 2026-11-12
    python -m aicwf monitor|eval|rollback|snapshot ...

Exit codes
----------
``0`` success, ``1`` failed/error, ``2`` usage, ``3`` awaiting approval,
``4`` blocked (escalation), ``5`` budget exhausted, ``6`` not implemented yet,
``42`` simulated crash (fault injection, from the child process).

The heavy workflow modules are imported lazily inside the command functions, so
``python -m aicwf --help`` stays fast and ``doctor`` works even when an optional
backend is missing. Everything the controller needs from the pipeline comes from
``aicwf/pipeline.py``:

* required: ``build_registry() -> Mapping[str, StageFn]``;
* optional: ``build_llm(mode, config)``, ``resolve_snapshot(as_of, config)``,
  ``run_monitor(*, as_of, llm_mode, config)``, ``run_eval(*, candidate, baseline,
  config)``, ``run_rollback(*, to, config)``, ``build_snapshot(*, as_of, config)``.

Paths are overridable through environment variables for tests and tooling:
``AICWF_PROJECT_ROOT``, ``AICWF_CONFIG_DIR``, ``AICWF_STATE_DB``,
``AICWF_RUNS_DIR``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicwf import __version__
from aicwf.contracts import (
    LLM_MODES,
    PROJECT_ROOT,
    STAGE_DIRS,
    ConfigError,
    FaultSpec,
    WorkflowError,
    parse_date,
)

__all__ = [
    "EXIT_OK",
    "EXIT_FAILURE",
    "EXIT_USAGE",
    "EXIT_AWAITING_APPROVAL",
    "EXIT_BLOCKED",
    "EXIT_BUDGET_EXHAUSTED",
    "EXIT_NOT_IMPLEMENTED",
    "EXIT_CRASHED",
    "STATUS_EXIT_CODES",
    "build_parser",
    "exit_code_for_status",
    "main",
]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_AWAITING_APPROVAL = 3
EXIT_BLOCKED = 4
EXIT_BUDGET_EXHAUSTED = 5
EXIT_NOT_IMPLEMENTED = 6
EXIT_CRASHED = 42

STATUS_EXIT_CODES: dict[str, int] = {
    "pending": EXIT_OK,
    "running": EXIT_OK,
    "completed": EXIT_OK,
    "awaiting_approval": EXIT_AWAITING_APPROVAL,
    "blocked": EXIT_BLOCKED,
    "budget_exhausted": EXIT_BUDGET_EXHAUSTED,
    "failed": EXIT_FAILURE,
    "stopped": EXIT_FAILURE,
}

_CONFIG_FILES = ("workflow", "strategy", "gates", "budgets", "model", "data")
_REQUIRED_CONFIG_FILES = ("workflow", "budgets", "gates")


def exit_code_for_status(status: str) -> int:
    """Map a run status to the CLI exit code."""

    return STATUS_EXIT_CODES.get(str(status), EXIT_FAILURE)


# ---------------------------------------------------------------------------
# Environment / path helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    env = os.environ.get("AICWF_PROJECT_ROOT")
    return Path(env).expanduser().resolve() if env else PROJECT_ROOT


def _config_dir() -> Path:
    env = os.environ.get("AICWF_CONFIG_DIR")
    return Path(env).expanduser().resolve() if env else _project_root() / "config"


def _state_db_path() -> Path:
    env = os.environ.get("AICWF_STATE_DB")
    return Path(env).expanduser().resolve() if env else _project_root() / "state" / "runs.db"


def _runs_root() -> Path:
    env = os.environ.get("AICWF_RUNS_DIR")
    return Path(env).expanduser().resolve() if env else _project_root() / "outputs" / "runs"


# ---------------------------------------------------------------------------
# Lazy subsystem access
# ---------------------------------------------------------------------------


def _open_state():
    from aicwf.controller.state import StateStore

    return StateStore(_state_db_path(), project_root=_project_root())


def _load_config(config_dir: Path | None = None) -> dict[str, Any]:
    from aicwf.util.jsonio import read_yaml

    directory = Path(config_dir) if config_dir is not None else _config_dir()
    bundle: dict[str, Any] = {}
    for name in _CONFIG_FILES:
        path = directory / f"{name}.yaml"
        if path.is_file():
            content = read_yaml(path)
            # Files such as workflow.yaml keep their section name as the YAML root
            # (``workflow:``); unwrap it so the merged bundle is
            # {workflow, strategy, gates, budgets, model, data}.
            if isinstance(content, Mapping) and isinstance(content.get(name), Mapping):
                content = content[name]
            bundle[name] = content
    missing = [name for name in _REQUIRED_CONFIG_FILES if name not in bundle]
    if missing:
        raise ConfigError(
            f"missing required config file(s) {missing} in {directory}; "
            "expected workflow.yaml, budgets.yaml, gates.yaml"
        )
    return bundle


def _import_pipeline():
    try:
        return importlib.import_module("aicwf.pipeline")
    except ImportError as exc:  # pragma: no cover - exercised by deployment failures
        raise ConfigError(
            f"could not import aicwf.pipeline ({exc}); the integration module is required "
            "for workflow execution"
        ) from exc


def _load_registry(pipeline: Any) -> dict[str, Callable[..., Any]]:
    builder = getattr(pipeline, "build_registry", None)
    if not callable(builder):
        raise ConfigError("aicwf.pipeline must expose build_registry() -> Mapping[str, StageFn]")
    registry = builder()
    if not isinstance(registry, Mapping):
        raise ConfigError("aicwf.pipeline.build_registry() must return a mapping handler -> StageFn")
    invalid = [key for key, value in registry.items() if not callable(value)]
    if invalid:
        raise ConfigError(f"aicwf.pipeline.build_registry() has non-callable handlers: {invalid}")
    return dict(registry)


def _load_llm(pipeline: Any, mode: str, config: Mapping[str, Any]) -> Any | None:
    builder = getattr(pipeline, "build_llm", None)
    if not callable(builder):
        return None
    return builder(mode, dict(config))


def _resolve_snapshot_hook(
    pipeline: Any, as_of: dt.date, config: Mapping[str, Any]
) -> tuple[Path | None, str | None]:
    resolver = getattr(pipeline, "resolve_snapshot", None)
    if not callable(resolver):
        return None, None
    resolved = resolver(as_of, dict(config))
    if resolved is None:
        return None, None
    if not isinstance(resolved, (tuple, list)) or len(resolved) != 2:
        raise ConfigError(
            "aicwf.pipeline.resolve_snapshot(as_of, config) must return (Path | None, str | None)"
        )
    snapshot_dir, snapshot_hash = resolved
    return (Path(snapshot_dir) if snapshot_dir is not None else None, snapshot_hash)


def _build_runner(
    config: Mapping[str, Any],
    *,
    mode: str,
    as_of: dt.date | None = None,
    fault: FaultSpec | str | None = None,
    echo_events: bool = False,
    state: Any | None = None,
):
    from aicwf.controller.runner import Runner

    pipeline = _import_pipeline()
    registry = _load_registry(pipeline)
    llm = _load_llm(pipeline, mode, config)
    snapshot_dir: Path | None = None
    snapshot_hash: str | None = None
    if as_of is not None:
        snapshot_dir, snapshot_hash = _resolve_snapshot_hook(pipeline, as_of, config)
    return Runner(
        state=state if state is not None else _open_state(),
        registry=registry,
        config=dict(config),
        runs_root=_runs_root(),
        llm=llm,
        snapshot_dir=snapshot_dir,
        snapshot_hash=snapshot_hash,
        fault=fault,
        project_root=_project_root(),
        echo_events=echo_events,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _read_run_summary(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "run_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _print_result(result: Any) -> None:
    print(f"run {result.run_id}: {result.status}")
    print(f"  as_of:  {result.as_of}")
    print(f"  run_dir: {result.run_dir}")
    for step in result.steps:
        if isinstance(step, Mapping):
            print(
                "  step {stage:<20} {status:<12} attempt={attempt} "
                "disposition={disposition}".format(
                    stage=str(step.get("stage", "")),
                    status=str(step.get("status", "")),
                    attempt=step.get("attempt", 0),
                    disposition=str(step.get("last_disposition", "")),
                )
            )
    if result.error:
        print(f"  error: {result.error}")
    summary = _read_run_summary(result.run_dir)
    if summary and summary.get("next_action") and summary["next_action"] != "none":
        print(f"next: {summary['next_action']}")


def _print_run_detail(run: Any, steps: Sequence[Any], budgets: Sequence[Any], approvals: Sequence[Any]) -> None:
    print(f"run: {run.run_id}")
    print(f"  status:        {run.status}")
    print(f"  as_of:         {run.as_of}")
    print(f"  llm_mode:      {run.llm_mode}")
    print(f"  seed:          {run.seed}")
    print(f"  workflow:      {run.workflow_version}")
    print(f"  snapshot_hash: {run.snapshot_hash}")
    print(f"  run_dir:       {run.run_dir}")
    if run.current_stage:
        print(f"  current_stage: {run.current_stage}")
    if run.error:
        print(f"  error:         {run.error}")
    print("  steps:")
    for step in steps:
        print(
            "    {stage:<20} {status:<12} attempt={attempt}/{max_attempts} "
            "skips={skip_count} {disposition}".format(
                stage=step.stage,
                status=step.status,
                attempt=step.attempt,
                max_attempts=step.max_attempts,
                skip_count=step.skip_count,
                disposition=step.last_disposition,
            )
        )
    if budgets:
        print("  budget:")
        for budget in budgets:
            print(f"    {budget.metric:<14} used={budget.used_value:.3f} limit={budget.limit_value:.3f}")
    if approvals:
        print("  approvals:")
        for approval in approvals:
            print(
                f"    {approval.action:<16} {approval.decision:<10} "
                f"approver={approval.approver!r} scope={approval.scope}"
            )
    summary = _read_run_summary(run.run_dir)
    if summary and summary.get("next_action") and summary["next_action"] != "none":
        print(f"next: {summary['next_action']}")


# ---------------------------------------------------------------------------
# Bounded verification revision loop
# ---------------------------------------------------------------------------

REVISIONS_FILENAME = "revisions.json"


def _utc_now_iso() -> str:
    from aicwf.util.timeutil import utc_now_iso

    return utc_now_iso()


def _verification_failure(run_dir: str | Path) -> tuple[bool, str]:
    """Return ``(failed, reason)`` when ``09_verification/verification.json`` says FAIL."""

    path = Path(run_dir) / STAGE_DIRS["verification"] / "verification.json"
    if not path.is_file():
        return False, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, ""
    if not isinstance(payload, Mapping):
        return False, ""
    if str(payload.get("verdict", "")).strip().upper() != "FAIL":
        return False, ""
    failed_checks: list[str] = []
    checks = payload.get("checks")
    if isinstance(checks, (list, tuple)):
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            status = str(check.get("status", "")).strip().upper()
            if status and status not in {"PASS", "OK"}:
                failed_checks.append(str(check.get("id") or "unknown_check"))
    reason = "verification FAIL"
    if failed_checks:
        reason += ": " + ", ".join(failed_checks[:5])
    return True, reason


def _revision_history(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read the persisted revision history for a run (``revisions.json``)."""

    path = Path(run_dir) / REVISIONS_FILENAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(payload, Mapping):
        history = payload.get("revisions")
        if isinstance(history, list):
            return [dict(entry) for entry in history if isinstance(entry, Mapping)]
    return []


def _record_revision(
    run_dir: str | Path,
    *,
    run_id: str,
    revision: int,
    max_revisions: int,
    stages: Sequence[str],
    reason: str,
) -> list[dict[str, Any]]:
    """Append one revision to ``<run_dir>/revisions.json`` (atomic write)."""

    from aicwf.util.jsonio import write_json

    history = _revision_history(run_dir)
    history.append(
        {
            "revision": int(revision),
            "max_revisions": int(max_revisions),
            "stages": list(stages),
            "reason": reason,
            "requested_at_utc": _utc_now_iso(),
        }
    )
    write_json(Path(run_dir) / REVISIONS_FILENAME, {"run_id": run_id, "revisions": history}, sort_keys=True)
    return history


def _planned_revision_stages(run_id: str, config: Mapping[str, Any]) -> list[str] | None:
    """Ask the pipeline which stages a verification-driven revision should re-run."""

    try:
        pipeline = _import_pipeline()
    except ConfigError:
        return None
    hook = getattr(pipeline, "plan_revision", None)
    if not callable(hook):
        return None
    plan = _call_with_supported_kwargs(hook, {"run_id": run_id, "config": dict(config)})
    if plan is None:
        return None
    if not isinstance(plan, (list, tuple)):
        raise ConfigError(
            "aicwf.pipeline.plan_revision(run_id, config) must return list[str] or None, "
            f"got {type(plan).__name__}"
        )
    return [str(stage) for stage in plan]


def _max_revisions(args: argparse.Namespace) -> int:
    value = int(getattr(args, "max_revisions", 0) or 0)
    if value < 0:
        raise ConfigError("--max-revisions must be >= 0")
    return value


def _apply_revision_loop(
    runner: Any,
    result: Any,
    config: Mapping[str, Any],
    *,
    max_revisions: int,
) -> Any:
    """Re-run failed checks up to ``max_revisions`` times, then stop.

    The pipeline owns the mapping from failed verification checks to affected
    stages (``aicwf.pipeline.plan_revision``); the controller only invalidates the
    planned stages and resumes with ``force=True``. Revision count persists in
    ``<run_dir>/revisions.json`` so a later CLI invocation cannot exceed the budget.
    """

    if max_revisions <= 0:
        return result
    run_dir = Path(result.run_dir)
    used = len(_revision_history(run_dir))
    while used < max_revisions and result.status in {"failed", "blocked"}:
        failed, reason = _verification_failure(run_dir)
        if not failed:
            break
        plan = _planned_revision_stages(result.run_id, config)
        if not plan:
            print(f"revision not planned ({plan!r}); stopping with run status {result.status}")
            break
        target_stages = [stage for stage in plan if stage in runner.workflow.stage_names]
        if not target_stages:
            print(f"revision not planned ({plan!r} contains no workflow stages); stopping")
            break
        invalidated = runner.invalidate_stages(result.run_id, target_stages, reason=reason)
        if not invalidated:
            print("revision could not invalidate any stage; stopping")
            break
        used += 1
        _record_revision(
            run_dir,
            run_id=result.run_id,
            revision=used,
            max_revisions=max_revisions,
            stages=invalidated,
            reason=reason,
        )
        print(f"revision {used}/{max_revisions}: re-running {', '.join(invalidated)} because {reason}")
        result = runner.resume(result.run_id, force=True)
    if used and result.status in {"failed", "blocked"}:
        print(
            f"revision budget exhausted ({used}/{max_revisions}); run is {result.status}: "
            "inspect verification.json and escalation state"
        )
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Start a new workflow run."""

    config = _load_config()
    as_of = parse_date(args.as_of)
    max_revisions = _max_revisions(args)
    fault = FaultSpec.parse(args.fault) if args.fault else None
    runner = _build_runner(
        config,
        mode=args.llm,
        as_of=as_of,
        fault=fault,
        echo_events=bool(args.verbose),
    )
    result = runner.run(
        as_of=as_of,
        run_id=args.run_id,
        seed=int(args.seed),
        llm_mode=args.llm,
        trigger=str(args.trigger or ""),
    )
    result = _apply_revision_loop(runner, result, config, max_revisions=max_revisions)
    _print_result(result)
    return exit_code_for_status(result.status)


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume an interrupted or paused run."""

    state = _open_state()
    record = state.get_run(args.run_id)
    if record is None:
        print(f"error: unknown run {args.run_id!r}", file=sys.stderr)
        return EXIT_FAILURE
    config = _load_config()
    max_revisions = _max_revisions(args)
    mode = args.llm or record.llm_mode or "replay"
    runner = _build_runner(
        config,
        mode=mode,
        as_of=None,
        echo_events=bool(args.verbose),
        state=state,
    )
    result = runner.resume(args.run_id, force=bool(args.force))
    result = _apply_revision_loop(runner, result, config, max_revisions=max_revisions)
    _print_result(result)
    return exit_code_for_status(result.status)


def cmd_status(args: argparse.Namespace) -> int:
    """Show one run in detail or list recent runs."""

    state = _open_state()
    if args.run_id:
        record = state.get_run(args.run_id)
        if record is None:
            print(f"error: unknown run {args.run_id!r}", file=sys.stderr)
            return EXIT_FAILURE
        if args.json:
            _print_json(state.export_run(record.run_id))
        else:
            _print_run_detail(
                record,
                state.list_steps(record.run_id),
                state.list_budgets(record.run_id),
                state.list_approvals(record.run_id),
            )
            events = state.recent_events(record.run_id, limit=int(args.events))
            if events:
                print("  recent events:")
                for event in events:
                    print(f"    {event.ts_utc} {event.level:<7} {event.kind}")
        return exit_code_for_status(record.status)
    runs = state.list_runs(limit=int(args.limit))
    if args.json:
        _print_json(state.export_state())
        return EXIT_OK
    if not runs:
        print("no runs recorded")
        return EXIT_OK
    print(f"{'run_id':<34} {'as_of':<12} {'status':<18} {'current_stage'}")
    for run in runs:
        print(
            f"{run.run_id:<34} {run.as_of:<12} {run.status:<18} {run.current_stage or ''}"
        )
    return EXIT_OK


def cmd_approve(args: argparse.Namespace) -> int:
    """Record a human approval decision and optionally resume the run."""

    from aicwf.controller.approval import ApprovalGate
    from aicwf.controller.dag import load_workflow

    config = _load_config()
    state = _open_state()
    record = state.get_run(args.run_id)
    if record is None:
        print(f"error: unknown run {args.run_id!r}", file=sys.stderr)
        return EXIT_FAILURE
    gate = ApprovalGate(state, config)
    workflow = load_workflow(config)
    action = args.action
    if not action:
        pending = gate.pending_gate_actions(workflow, record.run_id)
        if pending:
            action = pending[0][1]
        elif gate.required_actions:
            action = gate.required_actions[0]
        else:
            raise ConfigError("no --action given and the workflow/config defines no approval actions")
    decision = "rejected" if args.reject else "approved"
    scope = args.scope or "run"
    approval = gate.record_decision(
        record.run_id,
        action=action,
        approver=args.approver,
        decision=decision,
        scope=scope,
        note=args.note or "",
    )
    print(
        f"recorded {approval.decision} decision for action {approval.action!r} "
        f"on run {approval.run_id} (approver={approval.approver}, scope={approval.scope}, "
        f"artifact_hashes={len(approval.artifact_hashes)})"
    )
    if decision == "rejected":
        print("run stays paused; a rejection blocks the run until a new decision is recorded")
        return EXIT_BLOCKED
    if args.no_resume:
        return EXIT_OK
    if record.status == "completed":
        print("run is already completed; nothing to resume")
        return EXIT_OK
    if record.status in {"failed", "blocked"} and not args.force:
        print(
            f"run is {record.status!r}; fix the underlying issue and resume with "
            f"'python -m aicwf resume {record.run_id} --force'",
            file=sys.stderr,
        )
        return EXIT_BLOCKED
    runner = _build_runner(
        config,
        mode=getattr(args, "llm", None) or record.llm_mode or "replay",
        state=state,
    )
    result = runner.resume(record.run_id, force=bool(args.force))
    _print_result(result)
    return exit_code_for_status(result.status)


def cmd_state_export(args: argparse.Namespace) -> int:
    """Export controller state as JSON (stdout or a file)."""

    state = _open_state()
    payload = state.export_state(args.run_id)
    if args.output:
        from aicwf.util.jsonio import write_json

        path = write_json(args.output, payload, sort_keys=True)
        print(f"state exported to {path}")
    else:
        _print_json(payload)
    return EXIT_OK


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Check environment, config, snapshot and data availability."""

    results: list[tuple[str, str, str]] = []  # (status, name, detail)

    def record(status: str, name: str, detail: str) -> None:
        results.append((status, name, detail))

    # Python version
    version = sys.version_info
    if version >= (3, 11):
        record("PASS", "python", f"{version.major}.{version.minor}.{version.micro} (>= 3.11)")
    else:
        record("FAIL", "python", f"{version.major}.{version.minor} found; requires >= 3.11")

    # Config files and workflow DAG
    config: dict[str, Any] = {}
    try:
        config = _load_config()
        record("PASS", "config", f"loaded from {_config_dir()}")
    except Exception as exc:  # noqa: BLE001 - doctor reports every failure
        record("FAIL", "config", f"{type(exc).__name__}: {exc}")
    if config:
        try:
            from aicwf.controller.dag import load_workflow

            workflow = load_workflow(config)
            record(
                "PASS",
                "workflow",
                f"{workflow.name} v{workflow.version}, {len(workflow)} stages, "
                f"{len(workflow.human_gates())} human gates",
            )
        except Exception as exc:  # noqa: BLE001
            record("FAIL", "workflow", f"{type(exc).__name__}: {exc}")

    # State database writability
    try:
        state = _open_state()
        state.close()
        record("PASS", "state_db", f"SQLite ready at {_state_db_path()}")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "state_db", f"{type(exc).__name__}: {exc}")

    # Run directory writability
    try:
        runs_root = _runs_root()
        runs_root.mkdir(parents=True, exist_ok=True)
        probe = runs_root / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record("PASS", "runs_dir", f"writable at {runs_root}")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "runs_dir", f"{type(exc).__name__}: {exc}")

    # Snapshot availability
    snapshot_root = _project_root() / "data" / "snapshots"
    manifests = sorted(snapshot_root.glob("*/manifest.json")) if snapshot_root.is_dir() else []
    if manifests:
        record("PASS", "snapshot", f"{len(manifests)} frozen snapshot(s) under {snapshot_root}")
    else:
        record("WARN", "snapshot", f"no data/snapshots/*/manifest.json under {snapshot_root}")

    # Optional/heavy dependencies (presence only, never imported here)
    for module, required in (
        ("yaml", True),
        ("pandas", True),
        ("numpy", True),
        ("pyarrow", True),
        ("jsonschema", False),
        ("sklearn", False),
        ("matplotlib", False),
    ):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = module in sys.modules
        record("PASS" if found else ("FAIL" if required else "WARN"), f"package:{module}", "importable" if found else "missing")

    # Integration module
    try:
        pipeline_found = importlib.util.find_spec("aicwf.pipeline") is not None
    except (ImportError, ValueError):
        pipeline_found = "aicwf.pipeline" in sys.modules
    record(
        "PASS" if pipeline_found else "WARN",
        "pipeline",
        "aicwf.pipeline available" if pipeline_found else "aicwf/pipeline.py not found yet",
    )

    # gitignore hygiene for generated state
    gitignore = _project_root() / ".gitignore"
    if gitignore.is_file():
        text = gitignore.read_text(encoding="utf-8", errors="replace")
        ignored = all(token in text for token in ("state/", "outputs/runs/"))
        record("PASS" if ignored else "WARN", "gitignore", "state/ and outputs/runs/ ignored" if ignored else "generated state may be tracked")
    else:
        record("WARN", "gitignore", ".gitignore not found")

    failures = sum(1 for status, _, _ in results if status == "FAIL")
    warnings = sum(1 for status, _, _ in results if status == "WARN")
    for status, name, detail in results:
        print(f"[{status:<4}] {name:<22} {detail}")
    print(f"doctor: {len(results) - failures - warnings} passed, {warnings} warning(s), {failures} failed")
    return EXIT_FAILURE if failures else EXIT_OK


# -- fault demo --------------------------------------------------------


def cmd_fault_demo(args: argparse.Namespace) -> int:
    """Prove crash recovery: crash a run, resume it, verify no completed stage re-ran."""

    _import_pipeline()  # fail early with a clear message when integration is missing
    config = _load_config()
    from aicwf.controller.dag import load_workflow

    workflow = load_workflow(config)
    as_of = parse_date(args.as_of)
    stage = args.stage
    if stage is None:
        candidates = [spec.name for spec in workflow.order if spec.needs and not spec.human_gate]
        stage = candidates[min(1, len(candidates) - 1)] if candidates else workflow.order[-1].name
    if stage not in workflow:
        raise ConfigError(f"fault-demo stage {stage!r} is not part of the workflow")
    run_id = args.run_id or f"fault-demo-{as_of.isoformat()}"
    state = _open_state()
    if state.get_run(run_id) is not None:
        print(
            f"error: run {run_id!r} already exists; delete it or pass a different --run-id",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    fault = f"{args.fault_kind}:{stage}:1"
    base = [sys.executable, "-m", "aicwf"]
    project_root = _project_root()
    print(f"fault-demo: run {run_id} will crash {args.fault_kind} at stage {stage!r} (attempt 1)")
    first = subprocess.run(
        base + ["run", "--as-of", as_of.isoformat(), "--run-id", run_id, "--llm", args.llm, "--fault", fault],
        cwd=str(project_root),
    )
    if first.returncode != EXIT_CRASHED:
        print(
            f"error: expected crash exit code {EXIT_CRASHED}, child returned {first.returncode}",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    print(f"crashed as expected (exit {EXIT_CRASHED}); resuming {run_id} ...")
    second = subprocess.run(base + ["resume", run_id], cwd=str(project_root))
    if second.returncode != EXIT_OK:
        print(f"error: resume failed with exit code {second.returncode}", file=sys.stderr)
        return EXIT_FAILURE

    # Verify exactly-once behaviour from the recorded state.
    record = state.get_run(run_id)
    if record is None or record.status != "completed":
        print(f"error: run did not complete after resume (status={record.status if record else None})", file=sys.stderr)
        return EXIT_FAILURE
    steps = state.list_steps(run_id)
    attempts = state.list_attempts(run_id)
    attempt_counts: dict[str, int] = {}
    crashed: set[str] = set()
    for attempt in attempts:
        attempt_counts[attempt.stage] = attempt_counts.get(attempt.stage, 0) + 1
        if attempt.status == "crashed":
            crashed.add(attempt.stage)
    stage_order = list(workflow.stage_names)
    crash_index = stage_order.index(stage)
    ok = True
    print(f"{'stage':<20} {'status':<12} attempts crashed skipped")
    for index, stage_name in enumerate(stage_order):
        step = next((item for item in steps if item.stage == stage_name), None)
        if step is None:
            continue
        count = attempt_counts.get(stage_name, 0)
        crashed_here = stage_name in crashed
        expected_attempts = 2 if stage_name == stage else 1
        row_ok = count == expected_attempts and step.status == "completed"
        if index < crash_index:
            row_ok = row_ok and step.skip_count >= 1
        if not row_ok:
            ok = False
        print(
            f"{stage_name:<20} {step.status:<12} {count:<8} {str(crashed_here):<7} {step.skip_count}"
        )
    skipped = [step for step in steps if step.skip_count > 0]
    print(
        f"fault-demo: {len(skipped)} completed stage(s) were skipped after resume; "
        f"crash stage {stage!r} ran twice (crash + restart)"
    )
    if not ok:
        print("error: recovery verification failed", file=sys.stderr)
        return EXIT_FAILURE
    print("fault-demo: PASS - interrupted run resumed without re-running completed stages")
    return EXIT_OK


# -- pipeline placeholder commands ------------------------------------


def _pipeline_hook(name: str) -> Callable[..., Any] | None:
    try:
        pipeline = _import_pipeline()
    except ConfigError:
        return None
    hook = getattr(pipeline, name, None)
    return hook if callable(hook) else None


def _call_with_supported_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call ``func`` with the subset of kwargs its signature accepts."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # builtins / C callables
        return func(**dict(kwargs))
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return func(**dict(kwargs))
    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return func(**filtered)


def _handle_hook_result(result: Any) -> int:
    if result is None:
        return EXIT_OK
    if isinstance(result, bool):
        return EXIT_OK if result else EXIT_FAILURE
    if isinstance(result, int):
        return int(result)
    if isinstance(result, Path):
        print(str(result))
        return EXIT_OK
    if isinstance(result, Mapping):
        _print_json(result)
        return EXIT_OK
    print(str(result))
    return EXIT_OK


def _fallback_command(kind: str, args: argparse.Namespace, config: Mapping[str, Any]) -> int | None:
    """Delegate to the owning subsystem when ``aicwf.pipeline`` lacks a hook."""

    try:
        if kind == "monitor":
            module = importlib.import_module("aicwf.monitor.run")
            as_of = parse_date(args.as_of) if args.as_of else _default_as_of(config)
            argv = ["--as-of", as_of.isoformat(), "--llm", args.llm]
            return _handle_hook_result(module.main(argv))
        if kind == "eval":
            module = importlib.import_module("aicwf.eval.runner")
            argv = ["--candidate", args.candidate, "--baseline", args.baseline]
            return _handle_hook_result(module.main(argv))
        if kind == "rollback":
            module = importlib.import_module("aicwf.registry.rollback")
            gates = config.get("gates") if isinstance(config.get("gates"), Mapping) else {}
            change_control = gates.get("change_control") if isinstance(gates, Mapping) else {}
            registry_cfg = change_control.get("registry") if isinstance(change_control, Mapping) else {}
            registry_cfg = registry_cfg if isinstance(registry_cfg, Mapping) else {}
            result = _call_with_supported_kwargs(
                module.rollback,
                {
                    "to_version": args.to,
                    "history_path": registry_cfg.get("versions_file", "registry/versions.json"),
                    "releases_dir": registry_cfg.get("releases_dir", "registry/releases"),
                    "restore": True,
                },
            )
            return _handle_hook_result(result)
        if kind == "snapshot":
            script = _project_root() / "scripts" / "build_snapshot.py"
            if not script.is_file():
                print(f"error: snapshot builder not found at {script}", file=sys.stderr)
                return EXIT_NOT_IMPLEMENTED
            as_of = parse_date(args.as_of) if args.as_of else _default_as_of(config)
            completed = subprocess.run([sys.executable, str(script), "--as-of", as_of.isoformat()])
            return EXIT_OK if completed.returncode == 0 else EXIT_FAILURE
    except ImportError:
        return None
    return None


def _run_hook_or_fallback(
    hook_name: str,
    fallback_kind: str,
    kwargs: Mapping[str, Any],
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    hook = _pipeline_hook(hook_name)
    if hook is not None:
        return _handle_hook_result(_call_with_supported_kwargs(hook, kwargs))
    fallback = _fallback_command(fallback_kind, args, config)
    if fallback is not None:
        return fallback
    print(
        f"error: {hook_name} is not implemented yet "
        f"(add it to aicwf.pipeline or the {fallback_kind} subsystem)",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _default_as_of(config: Mapping[str, Any]) -> dt.date:
    strategy = config.get("strategy") if isinstance(config.get("strategy"), Mapping) else {}
    data_cfg = strategy.get("data") if isinstance(strategy, Mapping) else {}
    value = data_cfg.get("end_date") if isinstance(data_cfg, Mapping) else None
    if value:
        return parse_date(value)
    return dt.date.today()


def cmd_monitor(args: argparse.Namespace) -> int:
    """Run the monthly monitor workflow (pipeline hook or monitor subsystem)."""

    config = _load_config()
    as_of = parse_date(args.as_of) if args.as_of else _default_as_of(config)
    return _run_hook_or_fallback(
        "run_monitor",
        "monitor",
        {"as_of": as_of, "llm_mode": args.llm, "config": dict(config)},
        args,
        config,
    )


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a candidate against a baseline (pipeline hook or eval subsystem)."""

    config = _load_config()
    return _run_hook_or_fallback(
        "run_eval",
        "eval",
        {"candidate": args.candidate, "baseline": args.baseline, "config": dict(config)},
        args,
        config,
    )


def cmd_rollback(args: argparse.Namespace) -> int:
    """Roll the registry back to a released version."""

    config = _load_config()
    return _run_hook_or_fallback(
        "run_rollback",
        "rollback",
        {"to": args.to, "config": dict(config)},
        args,
        config,
    )


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Build or refresh a frozen data snapshot."""

    config = _load_config()
    as_of = parse_date(args.as_of) if args.as_of else _default_as_of(config)
    return _run_hook_or_fallback(
        "build_snapshot",
        "snapshot",
        {"as_of": as_of, "config": dict(config)},
        args,
        config,
    )


# ---------------------------------------------------------------------------
# Parser and entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aicwf`` argument parser (public for tests/introspection)."""

    parser = argparse.ArgumentParser(
        prog="aicwf",
        description="Long-running multi-factor CSI 300 index-enhancement research workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 ok, 1 failed, 2 usage, 3 awaiting approval, "
            "4 blocked, 5 budget exhausted, 6 not implemented, 42 simulated crash."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aicwf {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # run
    run_parser = subparsers.add_parser("run", help="start a new research run")
    run_parser.add_argument("--as-of", required=True, help="research as-of date (YYYY-MM-DD)")
    run_parser.add_argument("--run-id", default=None, help="explicit run id")
    run_parser.add_argument("--llm", "--llm-mode", dest="llm", choices=list(LLM_MODES), default="replay", help="LLM backend mode")
    run_parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    run_parser.add_argument("--fault", default=None, help="fault injection spec, e.g. crash_after:backtest:1")
    run_parser.add_argument("--trigger", default=None, help="trigger that caused this run (monthly/new_data/manual)")
    run_parser.add_argument(
        "--max-revisions",
        type=int,
        default=0,
        metavar="N",
        help="bounded verification revision loop: re-run up to N planned revisions (default 0 = off)",
    )
    run_parser.add_argument("--verbose", action="store_true", help="echo structured events to stderr")
    run_parser.set_defaults(func=cmd_run)

    # resume
    resume_parser = subparsers.add_parser("resume", help="resume an interrupted or paused run")
    resume_parser.add_argument("run_id", help="run id to resume")
    resume_parser.add_argument("--force", action="store_true", help="resume a failed/blocked run")
    resume_parser.add_argument("--llm", "--llm-mode", dest="llm", choices=list(LLM_MODES), default=None, help="override LLM mode")
    resume_parser.add_argument(
        "--max-revisions",
        type=int,
        default=0,
        metavar="N",
        help="bounded verification revision loop: re-run up to N planned revisions (default 0 = off)",
    )
    resume_parser.add_argument("--verbose", action="store_true", help="echo structured events to stderr")
    resume_parser.set_defaults(func=cmd_resume)

    # status
    status_parser = subparsers.add_parser("status", help="show a run or list recent runs")
    status_parser.add_argument("run_id", nargs="?", default=None, help="run id (omit to list runs)")
    status_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    status_parser.add_argument("--events", type=int, default=10, help="recent events to print")
    status_parser.add_argument("--limit", type=int, default=20, help="runs to list when no run id is given")
    status_parser.set_defaults(func=cmd_status)

    # approve
    approve_parser = subparsers.add_parser("approve", help="record a human approval decision")
    approve_parser.add_argument("run_id", help="run id to approve")
    approve_parser.add_argument("--action", default=None, help="approval action (default: pending gate action)")
    approve_parser.add_argument("--approver", default=None, help="approver name (default from config)")
    approve_parser.add_argument("--note", default="", help="decision note")
    approve_parser.add_argument("--scope", choices=["run", "action"], default=None, help="run-wide or action-scoped approval")
    approve_parser.add_argument("--reject", action="store_true", help="record a rejection instead of approval")
    approve_parser.add_argument("--no-resume", action="store_true", help="only record the decision")
    approve_parser.add_argument("--force", action="store_true", help="resume even if run is failed/blocked")
    approve_parser.add_argument("--llm", "--llm-mode", dest="llm", choices=list(LLM_MODES), default=None, help="override LLM mode on resume")
    approve_parser.set_defaults(func=cmd_approve)

    # state-export
    export_parser = subparsers.add_parser("state-export", help="export controller state as JSON")
    export_parser.add_argument("--run-id", default=None, help="export one run (default: all runs)")
    export_parser.add_argument("--output", default=None, help="write to this file instead of stdout")
    export_parser.set_defaults(func=cmd_state_export)

    # doctor
    doctor_parser = subparsers.add_parser("doctor", help="check environment, config, snapshot and data")
    doctor_parser.set_defaults(func=cmd_doctor)

    # fault-demo
    fault_parser = subparsers.add_parser("fault-demo", help="prove crash recovery with a fault-injected run")
    fault_parser.add_argument("--as-of", required=True, help="research as-of date (YYYY-MM-DD)")
    fault_parser.add_argument("--run-id", default=None, help="explicit run id for the demo run")
    fault_parser.add_argument("--llm", "--llm-mode", dest="llm", choices=list(LLM_MODES), default="replay", help="LLM backend mode")
    fault_parser.add_argument("--stage", default=None, help="stage to crash (default: a mid-run stage)")
    fault_parser.add_argument(
        "--fault-kind",
        choices=["crash_after", "crash_before"],
        default="crash_after",
        help="crash after (default) or before the stage runs",
    )
    fault_parser.set_defaults(func=cmd_fault_demo)

    # monitor
    monitor_parser = subparsers.add_parser("monitor", help="run the monthly monitoring workflow")
    monitor_parser.add_argument("--as-of", default=None, help="monitor date (default: config end_date)")
    monitor_parser.add_argument("--llm", "--llm-mode", dest="llm", choices=list(LLM_MODES), default="replay", help="LLM backend mode")
    monitor_parser.set_defaults(func=cmd_monitor)

    # eval
    eval_parser = subparsers.add_parser("eval", help="evaluate a candidate against a baseline")
    eval_parser.add_argument("--candidate", required=True, help="candidate version or bundle")
    eval_parser.add_argument("--baseline", required=True, help="baseline version or bundle")
    eval_parser.set_defaults(func=cmd_eval)

    # rollback
    rollback_parser = subparsers.add_parser("rollback", help="roll the registry back to a version")
    rollback_parser.add_argument("--to", required=True, help="target version")
    rollback_parser.set_defaults(func=cmd_rollback)

    # snapshot
    snapshot_parser = subparsers.add_parser("snapshot", help="build or refresh a frozen data snapshot")
    snapshot_parser.add_argument("--as-of", default=None, help="snapshot as-of date (default: config end_date)")
    snapshot_parser.set_defaults(func=cmd_snapshot)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code instead of calling ``sys.exit``."""

    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse --help / usage errors
        return int(exc.code or EXIT_OK)
    handler = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - required subparser prevents this
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(handler(args))
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - defensive top-level catch
        if os.environ.get("AICWF_DEBUG"):
            raise
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
