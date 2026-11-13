"""Failure classification and bounded retry policy.

The controller retries only failures that are *classified* as transient. The
classification is by exception type name so that cross-module exception classes
(e.g. ``HTTPError`` raised by ``requests`` inside the data client) can be listed
in ``config/budgets.yaml`` without the controller importing the raising library.

Categories
----------
``retryable``
    Transient failure: retried with deterministic exponential backoff while the
    per-stage retry budget lasts.
``escalate``
    Deterministic failure a human must resolve (validation / data-quality /
    explicit escalation requests). The stage becomes ``blocked``.
``non_retryable``
    Deterministic failure where retrying cannot help; the run ends ``failed``.
``budget``
    A run budget was exhausted; the run ends ``budget_exhausted`` and may be
    resumed after limits are adjusted.
``approval``
    A human approval is required; the run is ``awaiting_approval``.

Backoff jitter is derived from SHA-256 of ``(seed, retry_index)`` instead of a
random generator so replay runs on two machines sleep identically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from aicwf.contracts import (
    ApprovalRequired,
    BudgetExceeded,
    ConfigError,
    DataQualityError,
    EscalationRequired,
    NonRetryableError,
    RetryableError,
    ValidationError,
    WorkflowError,
)

__all__ = [
    "CATEGORY_APPROVAL",
    "CATEGORY_BUDGET",
    "CATEGORY_ESCALATE",
    "CATEGORY_NON_RETRYABLE",
    "CATEGORY_RETRYABLE",
    "DEFAULT_ESCALATE_CATEGORIES",
    "DEFAULT_NON_RETRYABLE_CATEGORIES",
    "DEFAULT_TRANSIENT_CATEGORIES",
    "RetryDecision",
    "RetryPolicy",
    "StageTimeout",
    "classify_exception",
    "describe_exception",
    "normalise_cancelled",
]

CATEGORY_RETRYABLE = "retryable"
CATEGORY_ESCALATE = "escalate"
CATEGORY_NON_RETRYABLE = "non_retryable"
CATEGORY_BUDGET = "budget"
CATEGORY_APPROVAL = "approval"

DEFAULT_TRANSIENT_CATEGORIES: tuple[str, ...] = (
    "RetryableError",
    "TimeoutError",
    "ConnectionError",
    "StageTimeout",
    "TransientError",
    "TransientFailure",
)
DEFAULT_ESCALATE_CATEGORIES: tuple[str, ...] = (
    "ValidationError",
    "DataQualityError",
    "EscalationRequired",
)
DEFAULT_NON_RETRYABLE_CATEGORIES: tuple[str, ...] = (
    "NonRetryableError",
    "ConfigError",
)

_MAX_MESSAGE_CHARS = 2000


class StageTimeout(RetryableError):
    """A stage exceeded its configured ``timeout_s`` (treated as transient)."""


def describe_exception(exc: BaseException) -> tuple[str, str]:
    """Return ``(error_type, message)`` with a bounded message length."""

    error_type = type(exc).__name__
    message = str(exc).strip()
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS] + "...(truncated)"
    return error_type, message or error_type


def _type_names(exc: BaseException) -> set[str]:
    return {klass.__name__ for klass in type(exc).__mro__}


def classify_exception(
    exc: BaseException,
    *,
    transient_categories: tuple[str, ...] | list[str] | None = None,
    escalate_categories: tuple[str, ...] | list[str] | None = None,
    non_retryable_categories: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Classify an exception into one of the five controller categories.

    Budget and approval exits are always recognised regardless of configuration;
    the configured type-name lists decide among retryable / escalate / failed.
    Unknown exceptions are treated as deterministic failures (never retried
    blindly). An unknown :class:`WorkflowError` subclass escalates, because it
    signals a workflow-level problem a human should look at.
    """

    if isinstance(exc, BudgetExceeded):
        return CATEGORY_BUDGET
    if isinstance(exc, ApprovalRequired):
        return CATEGORY_APPROVAL

    transient = tuple(transient_categories or DEFAULT_TRANSIENT_CATEGORIES)
    escalate = tuple(escalate_categories or DEFAULT_ESCALATE_CATEGORIES)
    non_retryable = tuple(non_retryable_categories or DEFAULT_NON_RETRYABLE_CATEGORIES)

    names = _type_names(exc)
    if names & set(transient):
        return CATEGORY_RETRYABLE
    if names & set(escalate):
        return CATEGORY_ESCALATE
    if names & set(non_retryable):
        return CATEGORY_NON_RETRYABLE
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return CATEGORY_NON_RETRYABLE
    if isinstance(exc, WorkflowError):
        return CATEGORY_ESCALATE
    return CATEGORY_NON_RETRYABLE


def normalise_cancelled(exc: BaseException) -> None:
    """Re-raise interpreter-level control flow that must never be swallowed."""

    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        raise exc


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryDecision:
    """Outcome of applying a :class:`RetryPolicy` to one failed stage attempt."""

    category: str
    retryable: bool
    attempt: int
    max_retries: int
    backoff_seconds: float = 0.0
    reason: str = ""
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "backoff_seconds": self.backoff_seconds,
            "reason": self.reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy loaded from ``config/budgets.yaml``.

    ``max_retries_per_step`` is the global ceiling; a stage may request fewer
    retries through ``workflow.yaml`` (``retries:``), never more.
    """

    max_retries_per_step: int = 3
    backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)
    jitter_fraction: float = 0.2
    transient_categories: tuple[str, ...] = DEFAULT_TRANSIENT_CATEGORIES
    escalate_categories: tuple[str, ...] = DEFAULT_ESCALATE_CATEGORIES
    non_retryable_categories: tuple[str, ...] = DEFAULT_NON_RETRYABLE_CATEGORIES

    # -- construction --------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "RetryPolicy":
        """Build a policy from a budgets bundle, ``run`` section, or ``None``.

        Accepted shapes::

            {"run": {..}, "retry": {..}}      # config/budgets.yaml bundle
            {"max_retries_per_step": .., ..}  # run section directly
            None                              # defaults
        """

        cfg = dict(cfg or {})
        run_cfg = cfg.get("run") if isinstance(cfg.get("run"), Mapping) else cfg
        run_cfg = dict(run_cfg or {})
        retry_cfg: Mapping[str, Any] = cfg.get("retry") if isinstance(cfg.get("retry"), Mapping) else {}

        backoff_raw = run_cfg.get("retry_backoff_seconds") or [1.0, 2.0, 4.0]
        backoff = tuple(max(0.0, float(value)) for value in backoff_raw)
        max_retries = int(run_cfg.get("max_retries_per_step", 3))
        if max_retries < 0:
            raise ConfigError("budgets.run.max_retries_per_step must be >= 0")
        jitter = float(run_cfg.get("retry_jitter_fraction", 0.2))
        if jitter < 0.0 or jitter > 1.0:
            raise ConfigError("budgets.run.retry_jitter_fraction must be within [0, 1]")

        def _categories(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            value = retry_cfg.get(key, list(default))
            if value is None:
                return default
            if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
                raise ConfigError(f"budgets.retry.{key} must be a list of exception type names")
            return tuple(value)

        return cls(
            max_retries_per_step=max_retries,
            backoff_seconds=backoff,
            jitter_fraction=jitter,
            transient_categories=_categories("transient_categories", DEFAULT_TRANSIENT_CATEGORIES),
            escalate_categories=_categories("escalate_categories", DEFAULT_ESCALATE_CATEGORIES),
            non_retryable_categories=_categories("non_retryable_categories", DEFAULT_NON_RETRYABLE_CATEGORIES),
        )

    # -- limits --------------------------------------------------------
    def max_retries(self, stage_retries: int) -> int:
        """Effective number of retries for a stage (never above the global cap)."""

        if stage_retries < 0:
            raise ConfigError(f"stage retries must be >= 0, got {stage_retries}")
        return min(int(stage_retries), self.max_retries_per_step)

    def max_attempts(self, stage_retries: int) -> int:
        return 1 + self.max_retries(stage_retries)

    # -- backoff -------------------------------------------------------
    def base_backoff(self, retry_index: int) -> float:
        """Backoff for the 1-based ``retry_index``; grows after the configured list."""

        if retry_index < 1:
            raise ConfigError(f"retry_index must be >= 1, got {retry_index}")
        if not self.backoff_seconds:
            return 0.0
        if retry_index <= len(self.backoff_seconds):
            return self.backoff_seconds[retry_index - 1]
        # Extend exponentially from the last configured value.
        extension = self.backoff_seconds[-1] * (2.0 ** (retry_index - len(self.backoff_seconds)))
        return extension

    def backoff(self, retry_index: int, *, seed: str = "") -> float:
        """Deterministic jittered backoff for replay-stable retry schedules."""

        base = self.base_backoff(retry_index)
        if base <= 0.0 or self.jitter_fraction <= 0.0:
            return base
        digest = hashlib.sha256(f"{seed}:{retry_index}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)  # [0, 1)
        factor = 1.0 + self.jitter_fraction * (2.0 * unit - 1.0)
        return max(0.0, round(base * factor, 6))

    # -- decisions -----------------------------------------------------
    def classify(
        self,
        exc: BaseException,
        *,
        transient_categories: tuple[str, ...] | None = None,
        escalate_categories: tuple[str, ...] | None = None,
        non_retryable_categories: tuple[str, ...] | None = None,
    ) -> str:
        return classify_exception(
            exc,
            transient_categories=transient_categories or self.transient_categories,
            escalate_categories=escalate_categories or self.escalate_categories,
            non_retryable_categories=non_retryable_categories or self.non_retryable_categories,
        )

    def decide(
        self,
        exc: BaseException,
        *,
        attempt: int,
        stage_retries: int,
        seed: str = "",
    ) -> RetryDecision:
        """Decide whether ``attempt`` (1-based) should be retried after ``exc``."""

        if attempt < 1:
            raise ConfigError(f"attempt must be >= 1, got {attempt}")
        category = self.classify(exc)
        max_retries = self.max_retries(stage_retries)
        failures_so_far = attempt  # attempt N failed => N failures so far
        can_retry = category == CATEGORY_RETRYABLE and failures_so_far <= max_retries
        error_type, message = describe_exception(exc)
        if category == CATEGORY_RETRYABLE:
            reason = "retrying transient failure" if can_retry else "retry budget exhausted"
        elif category == CATEGORY_ESCALATE:
            reason = "human intervention required"
        elif category == CATEGORY_BUDGET:
            reason = "run budget exhausted"
        elif category == CATEGORY_APPROVAL:
            reason = "approval required"
        else:
            reason = "deterministic failure"
        return RetryDecision(
            category=category,
            retryable=can_retry,
            attempt=attempt,
            max_retries=max_retries,
            backoff_seconds=self.backoff(failures_so_far, seed=seed) if can_retry else 0.0,
            reason=reason,
            error_type=error_type,
            error_message=message,
        )
