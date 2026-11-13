"""Tests for the budget ledger, GuardedLLM wrapper and persisted budget rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from aicwf.contracts import BudgetExceeded, BudgetLimits, LLMResponse
from aicwf.controller import BudgetLedger, GuardedLLM, StateStore
from aicwf.controller.state import RunRecord
from test_controller_helpers import budgets_section


class FakeClock:
    """Deterministic monotonic clock for wall-time budget tests."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeLLM:
    """Minimal LLMClient implementation counting calls."""

    model = "fake-model"

    def __init__(self, *, usage: Mapping[str, int] | None = None, text: str = "answer") -> None:
        self.calls: list[dict[str, Any]] = []
        self.usage = dict(usage) if usage is not None else {"prompt_tokens": 100, "completion_tokens": 50}
        self.text = text

    def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(dict(kwargs))
        return LLMResponse(text=self.text, model=self.model, usage=dict(self.usage))


def test_ledger_usage_and_remaining() -> None:
    ledger = BudgetLedger(limits=BudgetLimits(max_llm_calls=3, max_tokens_total=1000))
    ledger.start()
    assert ledger.llm_calls == 0
    assert ledger.tokens_total == 0
    ledger.record_llm_call(prompt_tokens=100, completion_tokens=50)
    ledger.record_llm_call(prompt_tokens=10, completion_tokens=5)
    used = ledger.used()
    assert used["llm_calls"] == 2.0
    assert used["tokens_total"] == 165.0
    assert used["wall_seconds"] >= 0.0
    remaining = ledger.remaining()
    assert remaining["llm_calls"] == 1.0
    assert remaining["tokens_total"] == 835.0
    assert ledger.exhausted() == []
    snapshot = ledger.snapshot()
    assert set(snapshot) == {"wall_seconds", "llm_calls", "tokens_total", "cost_usd"}
    assert snapshot["llm_calls"] == (3.0, 2.0)


def test_ledger_enforces_call_and_token_limits() -> None:
    ledger = BudgetLedger(limits=BudgetLimits(max_llm_calls=1, max_tokens_total=100))
    ledger.start()
    ledger.record_llm_call(prompt_tokens=50, completion_tokens=0)
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        ledger.ensure_capacity(calls=1)
    with pytest.raises(BudgetExceeded, match="tokens_total"):
        ledger.ensure_capacity(tokens=100)
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        ledger.record_llm_call(prompt_tokens=10)  # second call exceeds max_llm_calls


def test_ledger_wall_clock_limit() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(limits=BudgetLimits(max_wall_seconds=10.0), clock=clock)
    ledger.start()
    clock.advance(5.0)
    ledger.check()  # 5s < 10s
    clock.advance(6.0)
    with pytest.raises(BudgetExceeded, match="wall_seconds"):
        ledger.check()
    assert ledger.exhausted() == ["wall_seconds"]


def test_ledger_restore_from_persisted_usage() -> None:
    clock = FakeClock(100.0)
    ledger = BudgetLedger(limits=BudgetLimits(max_wall_seconds=60.0), clock=clock)
    ledger.restore({"wall_seconds": 50.0, "llm_calls": 2, "tokens_total": 0, "cost_usd": 0.0})
    ledger.start()
    clock.advance(5.0)
    assert ledger.wall_seconds() == pytest.approx(55.0)
    clock.advance(10.0)
    with pytest.raises(BudgetExceeded, match="wall_seconds"):
        ledger.check()


def test_ledger_cost_pricing_and_from_config() -> None:
    config = budgets_section(max_cost_usd=100.0)
    config["llm"] = {"price_per_1k_input_usd": 1.0, "price_per_1k_output_usd": 2.0}
    ledger = BudgetLedger.from_config(config)
    ledger.start()
    ledger.record_llm_call(prompt_tokens=1000, completion_tokens=500)
    # 1000/1000 * 1.0 + 500/1000 * 2.0 = 2.0
    assert ledger.used()["cost_usd"] == pytest.approx(2.0)
    assert ledger.estimated_cost(prompt_tokens=1000, completion_tokens=0) == pytest.approx(1.0)


def test_ledger_cost_limit_enforced_when_priced() -> None:
    config = budgets_section(max_cost_usd=0.5)
    config["llm"] = {"price_per_1k_input_usd": 1.0, "price_per_1k_output_usd": 1.0}
    ledger = BudgetLedger.from_config(config)
    ledger.start()
    with pytest.raises(BudgetExceeded, match="cost_usd"):
        ledger.ensure_capacity(cost_usd=1.0)


def test_guarded_llm_counts_calls_tokens_cost() -> None:
    fake = FakeLLM(usage={"prompt_tokens": 100, "completion_tokens": 50})
    config = budgets_section(max_cost_usd=100.0)
    config["llm"] = {"price_per_1k_input_usd": 1.0, "price_per_1k_output_usd": 2.0}
    ledger = BudgetLedger.from_config(config)
    ledger.start()
    guarded = GuardedLLM(fake, ledger, run_id="run-1", stage="hypothesis")
    assert guarded.model == "fake-model"
    response = guarded.complete(system="system", user="user")
    assert response.text == "answer"
    assert len(fake.calls) == 1
    assert ledger.llm_calls == 1
    assert ledger.tokens_total == 150
    assert ledger.cost_usd == pytest.approx(0.1 * 1.0 + 0.05 * 2.0)


def test_guarded_llm_blocks_before_exceeding_call_budget() -> None:
    fake = FakeLLM()
    ledger = BudgetLedger(limits=BudgetLimits(max_llm_calls=1))
    ledger.start()
    guarded = GuardedLLM(fake, ledger)
    guarded.complete(system="s", user="u")
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        guarded.complete(system="s", user="u")
    assert len(fake.calls) == 1  # the second backend call never happened
    assert ledger.llm_calls == 1


def test_guarded_llm_estimates_tokens_when_usage_missing() -> None:
    fake = FakeLLM(usage={}, text="a" * 400)
    ledger = BudgetLedger(limits=BudgetLimits(max_tokens_total=10_000))
    ledger.start()
    GuardedLLM(fake, ledger).complete(system="s" * 40, user="u" * 40)
    assert ledger.prompt_tokens > 0
    assert ledger.completion_tokens == 100  # 400 chars / 4
    assert ledger.tokens_total > 0


def test_guarded_llm_precheck_rejects_oversized_prompt() -> None:
    fake = FakeLLM()
    ledger = BudgetLedger(limits=BudgetLimits(max_tokens_total=10))
    ledger.start()
    with pytest.raises(BudgetExceeded, match="tokens_total"):
        GuardedLLM(fake, ledger).complete(system="s" * 400, user="u" * 400)
    assert fake.calls == []


def test_budget_snapshot_persists_in_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "runs.db")
    try:
        store.create_run(RunRecord(run_id="run-budget", as_of="2024-12-31", status="running"))
        ledger = BudgetLedger(limits=BudgetLimits(max_llm_calls=5))
        ledger.start()
        ledger.record_llm_call(prompt_tokens=100, completion_tokens=50)
        store.upsert_budgets("run-budget", ledger.snapshot())
        rows = {row.metric: row for row in store.list_budgets("run-budget")}
        assert rows["llm_calls"].used_value == 1.0
        assert rows["llm_calls"].limit_value == 5.0
        assert rows["tokens_total"].used_value == 150.0
    finally:
        store.close()
