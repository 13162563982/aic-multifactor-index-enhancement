"""Tests for failure classification, retry limits, backoff and jitter."""

from __future__ import annotations

import pytest

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
from aicwf.controller import CATEGORY_APPROVAL, CATEGORY_BUDGET, CATEGORY_ESCALATE
from aicwf.controller import CATEGORY_NON_RETRYABLE, CATEGORY_RETRYABLE, RetryPolicy
from aicwf.controller.retry import (
    RetryDecision,
    StageTimeout,
    classify_exception,
    describe_exception,
    normalise_cancelled,
)
from test_controller_helpers import budgets_section


class CustomTransient(Exception):
    """Third-party-style transient error registered through config."""


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (RetryableError("flaky"), CATEGORY_RETRYABLE),
        (TimeoutError("slow"), CATEGORY_RETRYABLE),
        (ConnectionError("down"), CATEGORY_RETRYABLE),
        (StageTimeout("stage too slow"), CATEGORY_RETRYABLE),
        (CustomTransient("custom"), CATEGORY_RETRYABLE),
        (ValidationError("bad payload"), CATEGORY_ESCALATE),
        (DataQualityError("bad data"), CATEGORY_ESCALATE),
        (EscalationRequired("human"), CATEGORY_ESCALATE),
        (NonRetryableError("deterministic"), CATEGORY_NON_RETRYABLE),
        (ConfigError("bad config"), CATEGORY_NON_RETRYABLE),
        (RuntimeError("unexpected"), CATEGORY_NON_RETRYABLE),
        (WorkflowError("base workflow error"), CATEGORY_ESCALATE),
        (BudgetExceeded("no budget"), CATEGORY_BUDGET),
        (ApprovalRequired("need approval"), CATEGORY_APPROVAL),
    ],
)
def test_classify_exception(exception: BaseException, expected: str) -> None:
    assert (
        classify_exception(
            exception,
            transient_categories=("RetryableError", "TimeoutError", "ConnectionError", "StageTimeout", "CustomTransient"),
        )
        == expected
    )


def test_normalise_cancelled_reraises_control_flow() -> None:
    with pytest.raises(KeyboardInterrupt):
        normalise_cancelled(KeyboardInterrupt())
    with pytest.raises(SystemExit):
        normalise_cancelled(SystemExit(3))
    normalise_cancelled(ValueError("normal error"))  # does not raise


def test_describe_exception_bounds_message() -> None:
    error_type, message = describe_exception(ValueError("x" * 5000))
    assert error_type == "ValueError"
    assert message.endswith("...(truncated)")
    assert len(message) < 2100
    assert describe_exception(ValueError(""))[1] == "ValueError"


def test_policy_from_config_limits_and_backoff() -> None:
    policy = RetryPolicy.from_config(budgets_section())
    assert policy.max_retries_per_step == 3
    assert policy.max_retries(0) == 0
    assert policy.max_retries(2) == 2
    assert policy.max_retries(10) == 3  # global cap
    assert policy.max_attempts(1) == 2
    assert policy.base_backoff(1) == 0.5
    assert policy.base_backoff(3) == 2.0
    assert policy.base_backoff(4) == 4.0  # extend exponentially past the list
    assert policy.base_backoff(5) == 8.0
    assert policy.transient_categories[0] == "RetryableError"
    with pytest.raises(ConfigError):
        policy.max_retries(-1)


def test_policy_from_flat_run_section_and_none() -> None:
    policy = RetryPolicy.from_config({"max_retries_per_step": 1, "retry_backoff_seconds": [0.0]})
    assert policy.max_retries_per_step == 1
    assert policy.backoff_seconds == (0.0,)
    default_policy = RetryPolicy.from_config(None)
    assert default_policy.max_retries_per_step == 3


def test_policy_from_config_rejects_bad_values() -> None:
    with pytest.raises(ConfigError):
        RetryPolicy.from_config(budgets_section(retry_jitter_fraction=1.5))
    with pytest.raises(ConfigError):
        RetryPolicy.from_config({"run": {"max_retries_per_step": -2}})
    with pytest.raises(ConfigError):
        RetryPolicy.from_config({"retry": {"transient_categories": "RetryableError"}})


def test_backoff_jitter_is_deterministic_and_seeded() -> None:
    policy = RetryPolicy.from_config(
        budgets_section(retry_backoff_seconds=[1.0, 2.0], retry_jitter_fraction=0.2)
    )
    first = policy.backoff(1, seed="run-a:backtest")
    second = policy.backoff(1, seed="run-a:backtest")
    assert first == second
    assert first != policy.backoff(1, seed="run-b:backtest")
    assert 0.8 <= first <= 1.2
    no_jitter = RetryPolicy.from_config(
        budgets_section(retry_backoff_seconds=[1.0], retry_jitter_fraction=0.0)
    )
    assert no_jitter.backoff(1, seed="anything") == 1.0


def test_decide_retryable_until_budget_exhausted() -> None:
    policy = RetryPolicy.from_config(
        budgets_section(
            max_retries_per_step=2,
            retry_backoff_seconds=[0.5, 1.0],
            retry_jitter_fraction=0.0,
        )
    )
    decision = policy.decide(RetryableError("boom"), attempt=1, stage_retries=2, seed="run:stage")
    assert isinstance(decision, RetryDecision)
    assert decision.retryable is True
    assert decision.category == CATEGORY_RETRYABLE
    assert decision.max_retries == 2
    assert decision.backoff_seconds == 0.5
    assert decision.reason == "retrying transient failure"

    second = policy.decide(RetryableError("boom"), attempt=2, stage_retries=2, seed="run:stage")
    assert second.retryable is True
    assert second.backoff_seconds == 1.0

    exhausted = policy.decide(RetryableError("boom"), attempt=3, stage_retries=2, seed="run:stage")
    assert exhausted.retryable is False
    assert exhausted.reason == "retry budget exhausted"


def test_decide_stage_retries_capped_by_global_limit() -> None:
    policy = RetryPolicy.from_config(budgets_section(max_retries_per_step=1))
    decision = policy.decide(RetryableError("boom"), attempt=2, stage_retries=5)
    assert decision.max_retries == 1
    assert decision.retryable is False


@pytest.mark.parametrize(
    ("exception", "category", "reason"),
    [
        (ValidationError("bad"), CATEGORY_ESCALATE, "human intervention required"),
        (NonRetryableError("no"), CATEGORY_NON_RETRYABLE, "deterministic failure"),
        (BudgetExceeded("budget"), CATEGORY_BUDGET, "run budget exhausted"),
        (ApprovalRequired("approval"), CATEGORY_APPROVAL, "approval required"),
    ],
)
def test_decide_non_retryable_categories(exception: BaseException, category: str, reason: str) -> None:
    policy = RetryPolicy.from_config(budgets_section())
    decision = policy.decide(exception, attempt=1, stage_retries=3)
    assert decision.retryable is False
    assert decision.category == category
    assert decision.reason == reason
    assert decision.backoff_seconds == 0.0


def test_policy_accepts_custom_transient_categories() -> None:
    policy = RetryPolicy.from_config(
        {
            "run": {"max_retries_per_step": 1, "retry_backoff_seconds": [0.0]},
            "retry": {"transient_categories": ["CustomTransient"]},
        }
    )
    assert policy.classify(CustomTransient("custom")) == CATEGORY_RETRYABLE
    decision = policy.decide(CustomTransient("custom"), attempt=1, stage_retries=1)
    assert decision.retryable is True
