"""Controller subsystem: durable state, DAG execution, retries, budgets, approvals.

Public entry points (import from this package, not from private modules):

* :class:`~aicwf.controller.state.StateStore` — SQLite checkpoint at ``state/runs.db``;
* :class:`~aicwf.controller.dag.Workflow` / :func:`~aicwf.controller.dag.load_workflow`
  — validated, deterministically ordered stage graph;
* :class:`~aicwf.controller.runner.Runner` — idempotent, resumable stage executor;
* :class:`~aicwf.controller.retry.RetryPolicy` — failure classification and backoff;
* :class:`~aicwf.controller.budget.BudgetLedger` / ``GuardedLLM`` — run budgets;
* :class:`~aicwf.controller.approval.ApprovalGate` — human gates and ``approval.json``;
* trigger helpers in :mod:`aicwf.controller.scheduler`.

``aicwf/pipeline.py`` supplies the ``handler name -> StageFn`` registry; the CLI
calls ``pipeline.build_registry()`` and hands it to :class:`Runner`.
"""

from __future__ import annotations

from .approval import (
    ApprovalGate,
    ApprovalRequest,
    EscalationRecord,
    approval_payload,
    build_escalation_record,
    write_approval_artifact,
    write_approval_request_artifact,
    write_escalation_artifact,
)
from .budget import BUDGET_METRICS, BudgetExceeded, BudgetLedger, BudgetLimits, GuardedLLM
from .dag import (
    StageSpec,
    Workflow,
    handler_map,
    load_workflow,
    stage_order,
    topological_order,
)
from .retry import (
    CATEGORY_APPROVAL,
    CATEGORY_BUDGET,
    CATEGORY_ESCALATE,
    CATEGORY_NON_RETRYABLE,
    CATEGORY_RETRYABLE,
    RetryDecision,
    RetryPolicy,
    StageTimeout,
    classify_exception,
)
from .runner import (
    INPUT_HASH_SCHEMA,
    STAGE_CONFIG_SECTIONS,
    RunResult,
    Runner,
    compute_input_hash,
    generate_run_id,
    resolve_snapshot_identity,
    snapshot_hash_for_dir,
    stage_config_subset,
)
from .scheduler import (
    CronSpec,
    TriggerDecision,
    evaluate_monthly_trigger,
    evaluate_new_data_trigger,
    evaluate_triggers,
    write_trigger_record,
)
from .state import (
    ApprovalRecord,
    ArtifactRecord,
    AttemptRecord,
    BudgetRecord,
    EventRecord,
    RunEventSink,
    RunRecord,
    StateError,
    StateStore,
    StepRecord,
    default_db_path,
)

__all__ = [
    # state
    "StateStore",
    "StateError",
    "RunRecord",
    "StepRecord",
    "AttemptRecord",
    "ArtifactRecord",
    "BudgetRecord",
    "ApprovalRecord",
    "EventRecord",
    "RunEventSink",
    "default_db_path",
    # dag
    "StageSpec",
    "Workflow",
    "load_workflow",
    "topological_order",
    "handler_map",
    "stage_order",
    # runner
    "Runner",
    "RunResult",
    "compute_input_hash",
    "generate_run_id",
    "resolve_snapshot_identity",
    "snapshot_hash_for_dir",
    "stage_config_subset",
    "INPUT_HASH_SCHEMA",
    "STAGE_CONFIG_SECTIONS",
    # retry
    "RetryPolicy",
    "RetryDecision",
    "classify_exception",
    "StageTimeout",
    "CATEGORY_RETRYABLE",
    "CATEGORY_ESCALATE",
    "CATEGORY_NON_RETRYABLE",
    "CATEGORY_BUDGET",
    "CATEGORY_APPROVAL",
    # budget
    "BudgetLedger",
    "BudgetLimits",
    "BudgetExceeded",
    "GuardedLLM",
    "BUDGET_METRICS",
    # approval
    "ApprovalGate",
    "ApprovalRequest",
    "EscalationRecord",
    "approval_payload",
    "build_escalation_record",
    "write_approval_artifact",
    "write_approval_request_artifact",
    "write_escalation_artifact",
    # scheduler
    "CronSpec",
    "TriggerDecision",
    "evaluate_monthly_trigger",
    "evaluate_new_data_trigger",
    "evaluate_triggers",
    "write_trigger_record",
]
