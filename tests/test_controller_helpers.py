"""Shared helpers for the controller/CLI test suites.

This module contains no test functions; it is imported by
``test_controller_*.py`` / ``test_cli*.py`` and exists so every test builds the
same tiny offline workflow with fake stages.

All paths must point into ``tmp_path``; the tests never touch the real
``state/runs.db`` or ``outputs/runs/``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicwf.contracts import RunContext, StageOutput
from aicwf.controller import Runner, StateStore

TEST_AS_OF = dt.date(2024, 12, 31)

DEFAULT_RETRY_SECTION: dict[str, Any] = {
    "transient_categories": [
        "RetryableError",
        "TimeoutError",
        "ConnectionError",
        "StageTimeout",
    ],
    "escalate_categories": ["ValidationError", "DataQualityError", "EscalationRequired"],
    "non_retryable_categories": ["NonRetryableError", "ConfigError"],
}


def stage_spec(
    name: str,
    *,
    handler: str | None = None,
    needs: Sequence[str] = (),
    retries: int = 0,
    timeout_s: float = 30.0,
    human_gate: bool = False,
    approval_action: str | None = None,
    produces: Sequence[str] = (),
    agent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``workflow.stages`` entry."""

    spec: dict[str, Any] = {
        "name": name,
        "handler": handler or name,
        "needs": list(needs),
        "timeout_s": timeout_s,
        "retries": retries,
        "produces": list(produces),
    }
    if human_gate:
        spec["human_gate"] = True
    if approval_action:
        spec["approval_action"] = approval_action
    if agent is not None:
        spec["agent"] = dict(agent)
    return spec


def workflow_section(
    stages: Sequence[Mapping[str, Any]],
    *,
    name: str = "test-workflow",
    version: str = "0.1.0",
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "execution": {"mode": "deterministic_topological", "stop_on_first_failure": True},
        "stages": [dict(item) for item in stages],
        "triggers": {},
    }


def budgets_section(**run_overrides: Any) -> dict[str, Any]:
    run = {
        "max_wall_seconds": 900.0,
        "max_llm_calls": 10,
        "max_tokens_total": 10_000,
        "max_cost_usd": 0.0,
        "max_retries_per_step": 3,
        "retry_backoff_seconds": [0.5, 1.0, 2.0],
        "retry_jitter_fraction": 0.0,
        "max_revision_rounds": 2,
        "stage_default_timeout_s": 60.0,
    }
    run.update(run_overrides)
    return {
        "run": run,
        "retry": json.loads(json.dumps(DEFAULT_RETRY_SECTION)),
        "llm": {"price_per_1k_input_usd": 0.0, "price_per_1k_output_usd": 0.0},
    }


def gates_section(**approval_overrides: Any) -> dict[str, Any]:
    approval = {
        "required": True,
        "actions": ["publish_report", "promote_version"],
        "bind_to_artifact_hashes": True,
        "default_approver": "tester",
    }
    approval.update(approval_overrides)
    return {"approval": approval}


def make_config(
    stages: Sequence[Mapping[str, Any]],
    *,
    workflow_version: str = "0.1.0",
    strategy: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    budget_run: Mapping[str, Any] | None = None,
    gates_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full merged config bundle shaped like ``load_config_bundle()`` output."""

    return {
        "workflow": workflow_section(list(stages), version=workflow_version),
        "strategy": dict(strategy) if strategy is not None else {"x": 1},
        "data": dict(data) if data is not None else {"quality": {"min_rows": 1}},
        "model": dict(model) if model is not None else {"default": {"backend": "null"}},
        "budgets": budgets_section(**dict(budget_run or {})),
        "gates": gates_section(**dict(gates_approval or {})),
    }


def fake_exit(code: int) -> None:
    """Stand-in for ``os._exit`` so a misconfigured fault cannot kill pytest."""

    raise SystemExit(code)


class FakeStages:
    """Registry of deterministic fake stage handlers with failure injection."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.contexts: list[RunContext] = []
        self.failures: dict[str, list[BaseException]] = {}
        self.overrides: dict[str, Callable[[RunContext], StageOutput]] = {}

    # -- handler factory ------------------------------------------------
    def handler(self, stage_name: str) -> Callable[[RunContext], StageOutput]:
        def run(ctx: RunContext) -> StageOutput:
            self.calls.append(stage_name)
            self.contexts.append(ctx)
            queue = self.failures.get(stage_name)
            if queue:
                raise queue.pop(0)
            override = self.overrides.get(stage_name)
            if override is not None:
                return override(ctx)
            return self.write_artifact(ctx, stage_name)

        return run

    @staticmethod
    def write_artifact(
        ctx: RunContext,
        stage_name: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> StageOutput:
        body = dict(payload) if payload is not None else {
            "stage": stage_name,
            "as_of": ctx.as_of.isoformat(),
        }
        path = ctx.artifact_path(stage_name, "artifact.json")
        path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        return StageOutput(
            artifacts={"artifact": str(path)},
            summary={"stage": stage_name, "as_of": ctx.as_of.isoformat()},
        )

    # -- helpers -------------------------------------------------------
    def registry(self, stage_names: Sequence[str]) -> dict[str, Callable[[RunContext], StageOutput]]:
        return {name: self.handler(name) for name in stage_names}

    def registry_for(self, config: Mapping[str, Any]) -> dict[str, Callable[[RunContext], StageOutput]]:
        stages = config["workflow"]["stages"]
        return {str(spec["handler"]): self.handler(str(spec["name"])) for spec in stages}

    def count(self, stage_name: str) -> int:
        return self.calls.count(stage_name)

    def fail_next(self, stage_name: str, error: BaseException) -> None:
        self.failures.setdefault(stage_name, []).append(error)

    def override(self, stage_name: str, handler: Callable[[RunContext], StageOutput]) -> None:
        self.overrides[stage_name] = handler


class Harness:
    """Per-test controller fixture: tmp DB, tmp runs root, fake stages."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = Path(tmp_path)
        self.state = StateStore(self.tmp_path / "runs.db")
        self.runs_root = self.tmp_path / "runs"
        self.project_root = self.tmp_path
        self.sleeps: list[float] = []
        self.stages = FakeStages()

    def close(self) -> None:
        self.state.close()

    def runner(
        self,
        config: Mapping[str, Any],
        *,
        registry: Mapping[str, Callable[[RunContext], StageOutput]] | None = None,
        **kwargs: Any,
    ) -> Runner:
        if registry is None:
            registry = self.stages.registry_for(config)
        kwargs.setdefault("sleep", self.sleeps.append)
        kwargs.setdefault("exit_fn", fake_exit)
        kwargs.setdefault("project_root", self.project_root)
        kwargs.setdefault("runs_root", self.runs_root)
        return Runner(
            state=self.state,
            registry=registry,
            config=dict(config),
            **kwargs,
        )

    # -- state shortcuts ----------------------------------------------
    def steps_by_name(self, run_id: str) -> dict[str, Any]:
        return {step.stage: step for step in self.state.list_steps(run_id)}

    def event_kinds(self, run_id: str) -> list[str]:
        return [event.kind for event in self.state.list_events(run_id)]

    def artifact_paths(self, run_id: str) -> list[Path]:
        return [Path(record.relpath) for record in self.state.list_artifacts(run_id)]


def run_config_three_stages() -> dict[str, Any]:
    """intake -> factor_build -> backtest, the standard small DAG for tests."""

    return make_config(
        [
            stage_spec("intake"),
            stage_spec("factor_build", needs=["intake"]),
            stage_spec("backtest", needs=["factor_build"]),
        ]
    )
