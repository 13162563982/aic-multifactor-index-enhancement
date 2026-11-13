"""Shared contracts for the AICWF workflow runtime.

This module is dependency-free (standard library only) and is the single source of
truth for stage names, run/step statuses, artifact conventions, the LLM client
protocol, budget limits, fault injection and the per-run context object that is
handed to every stage.

Subsystems must not import each other's private modules; they communicate through
the dataclasses and helpers defined here and through files on disk whose layout
is documented in ``docs/artifact_contract.md``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

SCHEMA_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Stages and on-disk layout
# ---------------------------------------------------------------------------

STAGES: tuple[str, ...] = (
    "intake",
    "hypothesis",
    "data_prep",
    "factor_build",
    "factor_review",
    "backtest",
    "robustness",
    "robustness_review",
    "portfolio",
    "verification",
    "report",
    "approval",
    "registry",
)

STAGE_DIRS: dict[str, str] = {
    "intake": "00_intake",
    "hypothesis": "01_hypothesis",
    "data_prep": "02_data",
    "factor_build": "03_factors",
    "factor_review": "04_factor_review",
    "backtest": "05_backtest",
    "robustness": "06_robustness",
    "robustness_review": "07_robustness_review",
    "portfolio": "08_portfolio",
    "verification": "09_verification",
    "report": "10_report",
    "approval": "11_approval",
    "registry": "12_registry",
}

RUN_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "running",
        "awaiting_approval",
        "completed",
        "failed",
        "blocked",
        "budget_exhausted",
        "stopped",
    }
)
STEP_STATUSES: frozenset[str] = frozenset(
    {"pending", "running", "completed", "failed", "blocked", "skipped"}
)
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "failed", "blocked"})

LLM_MODES = ("ollama", "replay", "openai_compat", "null")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkflowError(Exception):
    """Base class for all workflow errors."""


class ConfigError(WorkflowError):
    """Configuration is missing or inconsistent."""


class ValidationError(WorkflowError):
    """An artifact or agent payload failed schema or domain validation."""


class DataQualityError(WorkflowError):
    """Input data fails a quality gate; never retried blindly."""


class BudgetExceeded(WorkflowError):
    """A run budget (time, calls, tokens, cost) was exhausted."""


class ApprovalRequired(WorkflowError):
    """A consequential action is blocked until a human approval is recorded."""


class RetryableError(WorkflowError):
    """Transient failure (network/timeout/5xx); the controller may retry."""


class NonRetryableError(WorkflowError):
    """Deterministic failure; retrying cannot help."""


class EscalationRequired(WorkflowError):
    """Automatic handling is exhausted; a human must intervene."""


# ---------------------------------------------------------------------------
# Stage and LLM contracts
# ---------------------------------------------------------------------------


@dataclass
class StageOutput:
    """What a stage produced.

    ``artifacts`` maps a logical name to a path relative to the project root.
    ``summary`` is a small JSON-serialisable digest stored in the run database so
    downstream stages and the CLI can read results without loading large files.
    """

    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False
    latency_ms: int = 0


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion contract implemented by all LLM backends."""

    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Budgets and fault injection
# ---------------------------------------------------------------------------


@dataclass
class BudgetLimits:
    max_wall_seconds: float = 900.0
    max_llm_calls: int = 40
    max_tokens_total: int = 200_000
    max_cost_usd: float = 0.0
    max_retries_per_step: int = 3
    max_revision_rounds: int = 2
    retry_backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "BudgetLimits":
        cfg = dict(cfg or {})
        run_cfg = cfg.get("run", cfg)
        backoff = run_cfg.get("retry_backoff_seconds") or [1.0, 2.0, 4.0]
        return cls(
            max_wall_seconds=float(run_cfg.get("max_wall_seconds", 900.0)),
            max_llm_calls=int(run_cfg.get("max_llm_calls", 40)),
            max_tokens_total=int(run_cfg.get("max_tokens_total", 200_000)),
            max_cost_usd=float(run_cfg.get("max_cost_usd", 0.0)),
            max_retries_per_step=int(run_cfg.get("max_retries_per_step", 3)),
            max_revision_rounds=int(run_cfg.get("max_revision_rounds", 2)),
            retry_backoff_seconds=tuple(float(x) for x in backoff),
        )


@dataclass
class FaultSpec:
    """Deliberate interruption used by the recovery tests and ``fault-demo``.

    ``crash_after`` exits after the stage artifact has been written but before the
    controller records the step as completed; ``crash_before`` exits before the
    stage runs. Both exercise the exact-once resume path.
    """

    kind: str  # "crash_after" | "crash_before"
    stage: str
    attempt: int = 1

    @classmethod
    def parse(cls, spec: str | None) -> "FaultSpec | None":
        if not spec:
            return None
        parts = [p.strip() for p in spec.split(":") if p.strip()]
        if len(parts) < 2:
            raise ConfigError(f"invalid fault spec {spec!r}; expected kind:stage[:attempt]")
        kind = parts[0]
        aliases = {
            "crash": "crash_after",
            "crash-after": "crash_after",
            "after": "crash_after",
            "crash-before": "crash_before",
            "before": "crash_before",
        }
        kind = aliases.get(kind, kind)
        if kind not in {"crash_after", "crash_before"}:
            raise ConfigError(f"unknown fault kind {kind!r}")
        stage = parts[1]
        attempt = int(parts[2]) if len(parts) > 2 else 1
        return cls(kind=kind, stage=stage, attempt=attempt)


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything a stage may read, plus where it may write.

    Stages receive exactly one of these. They must not read global mutable state;
    all inputs come from ``config`` or from upstream artifacts recorded in
    ``stage_outputs``.
    """

    run_id: str
    as_of: dt.date
    run_dir: Path
    config: dict[str, Any]
    llm: LLMClient | None = None
    seed: int = 0
    llm_mode: str = "replay"
    fault: FaultSpec | None = None
    snapshot_dir: Path | None = None
    stage_outputs: dict[str, StageOutput] = field(default_factory=dict)
    events: Any = None
    budget: Any = None

    # -- paths ---------------------------------------------------------
    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_DIRS:
            raise ConfigError(f"unknown stage {stage!r}")
        path = self.run_dir / STAGE_DIRS[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, stage: str, name: str) -> Path:
        path = self.stage_dir(stage) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register(self, stage: str, output: StageOutput) -> None:
        self.stage_outputs[stage] = output

    # -- upstream access ----------------------------------------------
    def upstream(self, stage: str) -> StageOutput:
        if stage not in self.stage_outputs:
            raise WorkflowError(f"stage {stage!r} has not produced output in run {self.run_id}")
        return self.stage_outputs[stage]

    def upstream_summary(self, stage: str) -> dict[str, Any]:
        return dict(self.upstream(stage).summary)

    def upstream_artifact(self, stage: str, logical: str) -> Path:
        rel = self.upstream(stage).artifacts.get(logical)
        if rel is None:
            raise WorkflowError(f"stage {stage!r} did not declare artifact {logical!r}")
        path = Path(rel)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise WorkflowError(f"declared artifact missing on disk: {path}")
        return path

    # -- bookkeeping ---------------------------------------------------
    def event(self, kind: str, level: str = "info", **payload: Any) -> None:
        if self.events is not None:
            self.events.emit(kind=kind, level=level, run_id=self.run_id, payload=payload)


StageFn = Callable[[RunContext], StageOutput]


# ---------------------------------------------------------------------------
# Small helpers shared by every subsystem
# ---------------------------------------------------------------------------


def parse_date(value: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ConfigError(f"cannot parse date {value!r}")


def date_str(value: str | dt.date | dt.datetime) -> str:
    return parse_date(value).isoformat()


def relpath(path: Path | str, root: Path = PROJECT_ROOT) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
