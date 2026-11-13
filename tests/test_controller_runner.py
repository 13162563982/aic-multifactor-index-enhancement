"""End-to-end tests for the resumable DAG runner.

Covers: happy path, deterministic input hashing, idempotent skip, crash
injection (in-process and a real ``os._exit`` subprocess), downstream
invalidation, retries/backoff, escalation, budget exhaustion and human approval
gating. Everything uses ``tmp_path`` and fake stages; no network is touched.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from aicwf.contracts import (
    BudgetLimits,
    ConfigError,
    LLMResponse,
    RetryableError,
    RunContext,
    StageOutput,
    ValidationError,
    WorkflowError,
)
from aicwf.controller import (
    ApprovalGate,
    Runner,
    StateStore,
    compute_input_hash,
    load_workflow,
    resolve_snapshot_identity,
    write_approval_artifact,
)
from test_controller_helpers import (
    TEST_AS_OF,
    FakeStages,
    Harness,
    make_config,
    run_config_three_stages,
    stage_spec,
)

RUN_ID = "run-runner-1"


@pytest.fixture()
def harness(tmp_path: Path):
    harness = Harness(tmp_path)
    try:
        yield harness
    finally:
        harness.close()


class FakeLLM:
    """Minimal offline LLM client for budget tests."""

    model = "fake-llm"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="{}",
            model=self.model,
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_completes_and_records_everything(harness: Harness) -> None:
    config = run_config_three_stages()
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID, seed=7)

    assert result.status == "completed" and result.succeeded
    assert result.as_of == "2024-12-31"
    assert harness.stages.calls == ["intake", "factor_build", "backtest"]

    steps = harness.steps_by_name(RUN_ID)
    assert set(steps) == {"intake", "factor_build", "backtest"}
    for step in steps.values():
        assert step.status == "completed"
        assert step.attempt == 1
        assert step.last_disposition == "executed"
        assert step.input_hash and step.output_hash
        assert step.duration_s >= 0.0

    artifacts = harness.state.list_artifacts(RUN_ID)
    assert len(artifacts) == 3
    for artifact in artifacts:
        assert artifact.size_bytes > 0
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.content_hash)
        assert Path(artifact.relpath).is_file()

    run_dir = Path(result.run_dir)
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "state_snapshot.json").is_file()
    assert (run_dir / "run_summary.json").is_file()
    snapshot = json.loads((run_dir / "state_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["run"]["status"] == "completed"
    assert len(snapshot["steps"]) == 3
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["next_action"] == "none"
    assert [item["stage"] for item in summary["steps"]] == ["intake", "factor_build", "backtest"]
    assert summary["budget"]["llm_calls"]["used"] == 0.0

    events = harness.state.list_events(RUN_ID)
    kinds = [event.kind for event in events]
    assert kinds.count("stage_completed") == 3
    assert "run_created" in kinds and "run_finished" in kinds
    assert [event.kind for event in harness.state.recent_events(RUN_ID, limit=1)] == ["run_finished"]

    # every stage saw the full merged config bundle and a budget ledger
    for ctx in harness.stages.contexts:
        assert ctx.run_id == RUN_ID
        assert ctx.as_of == TEST_AS_OF
        assert ctx.seed == 7
        assert ctx.llm_mode == "replay"
        assert ctx.config == config
        assert ctx.budget is not None
        assert ctx.events is not None

    run = harness.state.get_run(RUN_ID)
    assert run is not None and run.status == "completed" and run.finished_at_utc
    assert run.snapshot_hash and run.config_hash and run.workflow_version == "0.1.0"


def test_resume_completed_run_is_noop(harness: Harness) -> None:
    config = run_config_three_stages()
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    calls_before = list(harness.stages.calls)
    events_before = len(harness.state.list_events(RUN_ID))

    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed"
    assert harness.stages.calls == calls_before
    assert len(harness.state.list_events(RUN_ID)) == events_before


def test_upstream_context_is_available_to_downstream_stages(harness: Harness) -> None:
    config = run_config_three_stages()
    seen: dict[str, object] = {}

    def downstream(ctx: RunContext) -> StageOutput:
        upstream = ctx.upstream("intake")
        path = ctx.upstream_artifact("intake", "artifact")
        seen["summary"] = dict(upstream.summary)
        seen["artifact_exists"] = path.is_file()
        seen["stage_outputs"] = sorted(ctx.stage_outputs)
        return FakeStages.write_artifact(ctx, "factor_build")

    harness.stages.override("factor_build", downstream)
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "completed"
    assert seen["summary"] == {"stage": "intake", "as_of": "2024-12-31"}
    assert seen["artifact_exists"] is True
    assert seen["stage_outputs"] == ["intake"]


def test_duplicate_run_id_and_unknown_resume_raise(harness: Harness) -> None:
    config = run_config_three_stages()
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    with pytest.raises(WorkflowError, match="already exists"):
        harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    with pytest.raises(WorkflowError, match="unknown run"):
        harness.runner(config).resume("does-not-exist")


def test_run_id_is_generated_when_omitted(harness: Harness) -> None:
    result = harness.runner(run_config_three_stages()).run(as_of=TEST_AS_OF)
    assert re.fullmatch(r"run-20241231-\d{6}-[0-9a-f]{6}", result.run_id)
    assert harness.state.get_run(result.run_id) is not None


# ---------------------------------------------------------------------------
# Deterministic input hashing
# ---------------------------------------------------------------------------


def test_input_hash_is_stable_across_runs_and_has_no_wall_clock(harness: Harness) -> None:
    config = run_config_three_stages()
    runner_a = harness.runner(config)
    runner_a.run(as_of=TEST_AS_OF, run_id="hash-a", seed=3)
    runner_b = harness.runner(config)
    runner_b.run(as_of=TEST_AS_OF, run_id="hash-b", seed=3)

    for stage in ("intake", "factor_build", "backtest"):
        hash_a = harness.state.get_step("hash-a", stage).input_hash  # type: ignore[union-attr]
        hash_b = harness.state.get_step("hash-b", stage).input_hash  # type: ignore[union-attr]
        assert hash_a == hash_b
        assert re.fullmatch(r"[0-9a-f]{64}", hash_a)

    # a strategy change invalidates the stages that read strategy config, but not
    # the intake stage (whose config subset is empty)
    changed = run_config_three_stages()
    changed["strategy"] = {"x": 2}
    runner_c = harness.runner(changed)
    runner_c.run(as_of=TEST_AS_OF, run_id="hash-c", seed=3)
    assert (
        harness.state.get_step("hash-c", "intake").input_hash  # type: ignore[union-attr]
        == harness.state.get_step("hash-a", "intake").input_hash  # type: ignore[union-attr]
    )
    assert (
        harness.state.get_step("hash-c", "factor_build").input_hash  # type: ignore[union-attr]
        != harness.state.get_step("hash-a", "factor_build").input_hash  # type: ignore[union-attr]
    )


def test_compute_input_hash_is_pure_and_sensitive() -> None:
    base = dict(
        stage="factor_build",
        handler="factor_build",
        needs=["intake"],
        upstream_hashes={"intake/run_request": "hash-upstream"},
        config_subset={"strategy": {"x": 1}},
        workflow_name="workflow",
        workflow_version="1.0.0",
        as_of="2024-12-31",
        snapshot_hash="snapshot-hash",
        seed=0,
        llm_mode="replay",
        prompt_hashes={"prompt": "prompt-hash"},
        spec_payload={"name": "factor_build", "handler": "factor_build"},
    )
    baseline = compute_input_hash(**base)
    assert baseline == compute_input_hash(**base)
    assert re.fullmatch(r"[0-9a-f]{64}", baseline)

    for override in (
        {"config_subset": {"strategy": {"x": 2}}},
        {"upstream_hashes": {"intake/run_request": "changed"}},
        {"as_of": "2024-12-30"},
        {"snapshot_hash": "other-snapshot"},
        {"seed": 1},
        {"llm_mode": "ollama"},
        {"prompt_hashes": {"prompt": "changed"}},
        {"spec_payload": {"name": "factor_build", "handler": "other"}},
        {"needs": ["intake", "data_prep"]},
    ):
        assert compute_input_hash(**{**base, **override}) != baseline


def test_agent_prompt_hash_changes_invalidate_agent_stage(harness: Harness, tmp_path: Path) -> None:
    prompt = tmp_path / "prompts" / "researcher.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("version one", encoding="utf-8")
    config = make_config(
        [
            stage_spec("intake"),
            stage_spec("hypothesis", needs=["intake"], agent={"prompt": str(prompt)}),
        ]
    )
    harness.runner(config).run(as_of=TEST_AS_OF, run_id="agent-a")
    prompt.write_text("version two", encoding="utf-8")
    harness.runner(config).run(as_of=TEST_AS_OF, run_id="agent-b")
    hash_a = harness.state.get_step("agent-a", "hypothesis").input_hash  # type: ignore[union-attr]
    hash_b = harness.state.get_step("agent-b", "hypothesis").input_hash  # type: ignore[union-attr]
    assert hash_a != hash_b


def test_snapshot_resolution_never_uses_a_wrong_date_snapshot(tmp_path: Path) -> None:
    def _write_snapshot(day: str, digest: str) -> Path:
        directory = tmp_path / "data" / "snapshots" / day
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"as_of": day, "snapshot_hash": digest}), encoding="utf-8"
        )
        return directory

    target = _write_snapshot("2024-12-31", "hash-2024")
    other = _write_snapshot("2026-11-12", "hash-2026")
    config = run_config_three_stages()
    config["strategy"] = {"data": {"snapshot": str(other)}}

    resolved, digest = resolve_snapshot_identity(__import__("datetime").date(2024, 12, 31), config, project_root=tmp_path)
    assert resolved == target and digest == "hash-2024"
    # a date with no matching snapshot must not fall back to the 2026 snapshot
    missing, missing_hash = resolve_snapshot_identity(
        __import__("datetime").date(2024, 6, 30), config, project_root=tmp_path
    )
    assert missing is None
    assert missing_hash and missing_hash != "hash-2026"
    # explicit snapshot_dir always wins (used by the pipeline resolver/tests)
    explicit, explicit_hash = resolve_snapshot_identity(
        __import__("datetime").date(2000, 1, 1), config, project_root=tmp_path, snapshot_dir=target
    )
    assert explicit == target and explicit_hash == "hash-2024"


def test_invalidate_stages_keeps_artifact_rows_and_reruns_on_resume(harness: Harness) -> None:
    config = run_config_three_stages()
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    artifacts_before = harness.state.list_artifacts(RUN_ID)
    assert len(artifacts_before) == 3

    runner = harness.runner(config)
    reset = runner.invalidate_stages(RUN_ID, ["factor_build"], reason="weights_bounds")
    assert reset == ["factor_build", "backtest"]

    steps = harness.steps_by_name(RUN_ID)
    assert steps["intake"].status == "completed"
    assert steps["factor_build"].status == "pending"
    assert steps["backtest"].status == "pending"
    assert steps["factor_build"].input_hash is None
    assert steps["factor_build"].last_disposition == "invalidated_revision:weights_bounds"
    # artifact registrations are intentionally preserved for the re-run to replace
    assert harness.state.list_artifacts(RUN_ID) == artifacts_before
    kinds = harness.event_kinds(RUN_ID)
    assert kinds.count("step_invalidated") == 2
    assert "revision_planned" in kinds

    empty = runner.invalidate_stages(RUN_ID, [])
    assert empty == []
    with pytest.raises(ConfigError, match="unknown stage"):
        runner.invalidate_stages(RUN_ID, ["not_a_stage"])
    with pytest.raises(WorkflowError, match="unknown run"):
        runner.invalidate_stages("missing-run", ["factor_build"])

    # a revision resume re-runs exactly the invalidated stages and completes
    harness.state.update_run(RUN_ID, status="failed", error="verification FAIL")
    resumed = harness.runner(config).resume(RUN_ID, force=True)
    assert resumed.status == "completed"
    steps = harness.steps_by_name(RUN_ID)
    assert steps["intake"].attempt == 1
    assert steps["factor_build"].attempt == 2
    assert steps["backtest"].attempt == 2
    assert [step["stage"] for step in resumed.steps if step["stage"] == "factor_build"] == ["factor_build"]


# ---------------------------------------------------------------------------
# Crash injection and resume
# ---------------------------------------------------------------------------


def test_crash_after_then_resume_skips_completed_stages(harness: Harness) -> None:
    config = run_config_three_stages()
    with pytest.raises(SystemExit) as excinfo:
        harness.runner(config, fault="crash_after:backtest:1").run(
            as_of=TEST_AS_OF, run_id=RUN_ID
        )
    assert excinfo.value.code == 42
    assert harness.stages.calls == ["intake", "factor_build", "backtest"]

    # crash leaves the run 'running' with the interrupted stage still running
    record = harness.state.get_run(RUN_ID)
    assert record is not None and record.status == "running"
    steps = harness.steps_by_name(RUN_ID)
    assert steps["intake"].status == "completed"
    assert steps["factor_build"].status == "completed"
    assert steps["backtest"].status == "running"
    crash_attempts = harness.state.list_attempts(RUN_ID, "backtest")
    assert [attempt.status for attempt in crash_attempts] == ["crashed"]
    assert "fault_injected" in harness.event_kinds(RUN_ID)
    # the stage artifact was written before the crash, but not registered
    assert harness.state.list_artifacts(RUN_ID, "backtest") == []

    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed"
    # only the interrupted stage re-executed
    assert harness.stages.calls == ["intake", "factor_build", "backtest", "backtest"]
    steps = harness.steps_by_name(RUN_ID)
    assert steps["intake"].attempt == 1 and steps["intake"].skip_count == 1
    assert steps["factor_build"].attempt == 1 and steps["factor_build"].skip_count == 1
    assert steps["backtest"].attempt == 2 and steps["backtest"].skip_count == 0
    assert [attempt.status for attempt in harness.state.list_attempts(RUN_ID, "backtest")] == [
        "crashed",
        "completed",
    ]
    kinds = harness.event_kinds(RUN_ID)
    assert kinds.count("step_skipped_completed") == 2
    assert kinds.count("run_resumed") == 1


def test_crash_before_does_not_execute_stage(harness: Harness) -> None:
    config = run_config_three_stages()
    with pytest.raises(SystemExit):
        harness.runner(config, fault="crash_before:backtest:1").run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert harness.stages.calls == ["intake", "factor_build"]
    assert [attempt.status for attempt in harness.state.list_attempts(RUN_ID, "backtest")] == ["crashed"]
    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed"
    assert harness.stages.calls == ["intake", "factor_build", "backtest"]


def test_crash_on_retry_attempt_does_not_consume_retry_budget(harness: Harness) -> None:
    config = make_config(
        [stage_spec("intake"), stage_spec("backtest", needs=["intake"], retries=1)]
    )
    harness.stages.fail_next("backtest", RetryableError("transient"))
    with pytest.raises(SystemExit):
        harness.runner(config, fault="crash_after:backtest:2").run(as_of=TEST_AS_OF, run_id=RUN_ID)
    attempts = harness.state.list_attempts(RUN_ID, "backtest")
    assert [attempt.status for attempt in attempts] == ["failed", "crashed"]

    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed"
    attempts = harness.state.list_attempts(RUN_ID, "backtest")
    assert [attempt.status for attempt in attempts] == ["failed", "crashed", "completed"]


def test_downstream_invalidation_after_config_change(harness: Harness) -> None:
    config = run_config_three_stages()
    harness.stages.override(
        "factor_build",
        lambda ctx: FakeStages.write_artifact(
            ctx, "factor_build", payload={"x": ctx.config["strategy"]["x"]}
        ),
    )
    with pytest.raises(SystemExit):
        harness.runner(config, fault="crash_before:backtest:1").run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert harness.stages.calls == ["intake", "factor_build"]

    changed = run_config_three_stages()
    changed["strategy"] = {"x": 2}
    result = harness.runner(changed).resume(RUN_ID)
    assert result.status == "completed"
    steps = harness.steps_by_name(RUN_ID)
    assert steps["intake"].attempt == 1 and steps["intake"].skip_count == 1
    assert steps["factor_build"].attempt == 2
    assert steps["backtest"].attempt == 2
    kinds = harness.event_kinds(RUN_ID)
    assert kinds.count("step_invalidated") >= 2
    artifact = harness.state.list_artifacts(RUN_ID, "factor_build")[0]
    payload = json.loads(Path(artifact.relpath).read_text(encoding="utf-8"))
    assert payload["x"] == 2


def test_real_long_running_subprocess_is_stopped_by_timeout(tmp_path: Path) -> None:
    """A stage that blocks forever is timed out, not allowed to hang the run."""

    config = make_config(
        [
            stage_spec("intake"),
            stage_spec("backtest", needs=["intake"], retries=1, timeout_s=0.05),
        ]
    )
    harness = Harness(tmp_path)
    try:
        def slow(ctx: RunContext) -> StageOutput:
            time.sleep(0.25)
            return FakeStages.write_artifact(ctx, "backtest")

        harness.stages.override("backtest", slow)
        result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
        assert result.status == "blocked"
        attempts = harness.state.list_attempts(RUN_ID, "backtest")
        assert [attempt.status for attempt in attempts] == ["failed", "failed"]
        assert attempts[0].error_type == "StageTimeout"
        escalation = json.loads((Path(result.run_dir) / "escalation.json").read_text(encoding="utf-8"))
        assert escalation["reason"] == "retries_exhausted"
    finally:
        harness.close()


def test_real_os_exit_crash_then_resume(tmp_path: Path) -> None:
    """Prove exactly-once resume with a real ``os._exit(42)`` process."""

    project_root = Path(__file__).resolve().parents[1]
    config = run_config_three_stages()
    base = tmp_path / "subprocess"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    script = base / "crash_child.py"
    script.write_text(
        textwrap.dedent(
            """
            import datetime as dt
            import json
            import sys
            from pathlib import Path

            from aicwf.contracts import StageOutput
            from aicwf.controller import Runner, StateStore

            base = Path(sys.argv[1])
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))


            def handler(stage_name):
                def run(ctx):
                    path = ctx.artifact_path(stage_name, "artifact.json")
                    path.write_text(json.dumps({"stage": stage_name}), encoding="utf-8")
                    with open(base / "executed.log", "a", encoding="utf-8") as handle:
                        handle.write(stage_name + "\\n")
                    return StageOutput(artifacts={"artifact": str(path)}, summary={"stage": stage_name})

                return run


            registry = {spec["handler"]: handler(spec["name"]) for spec in config["workflow"]["stages"]}
            state = StateStore(base / "runs.db")
            Runner(
                state=state,
                registry=registry,
                config=config,
                runs_root=base / "runs",
                project_root=base,
                fault="crash_after:backtest:1",
            ).run(as_of=dt.date(2024, 12, 31), run_id="subprocess-crash")
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [sys.executable, str(script), str(base)],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 42, completed.stderr
    executed = (base / "executed.log").read_text(encoding="utf-8").splitlines()
    assert executed == ["intake", "factor_build", "backtest"]

    state = StateStore(base / "runs.db")
    record = state.get_run("subprocess-crash")
    assert record is not None and record.status == "running"
    assert [attempt.status for attempt in state.list_attempts("subprocess-crash", "backtest")] == ["crashed"]

    def parent_handler(stage_name: str):
        def run(ctx: RunContext) -> StageOutput:
            path = ctx.artifact_path(stage_name, "artifact.json")
            path.write_text(json.dumps({"stage": stage_name}), encoding="utf-8")
            with open(base / "executed.log", "a", encoding="utf-8") as handle:
                handle.write(stage_name + "\n")
            return StageOutput(artifacts={"artifact": str(path)}, summary={"stage": stage_name})

        return run

    registry = {spec["handler"]: parent_handler(spec["name"]) for spec in config["workflow"]["stages"]}
    try:
        runner = Runner(
            state=state,
            registry=registry,
            config=config,
            runs_root=base / "runs",
            project_root=base,
            sleep=lambda seconds: None,
        )
        result = runner.resume("subprocess-crash")
        assert result.status == "completed"
    finally:
        state.close()

    executed = (base / "executed.log").read_text(encoding="utf-8").splitlines()
    assert executed.count("intake") == 1
    assert executed.count("factor_build") == 1
    assert executed.count("backtest") == 2


# ---------------------------------------------------------------------------
# Retry / escalation / budget
# ---------------------------------------------------------------------------


def test_retry_then_success_records_attempts_and_backoff(harness: Harness) -> None:
    from aicwf.contracts import RetryableError

    config = make_config(
        [stage_spec("intake"), stage_spec("backtest", needs=["intake"], retries=2)]
    )
    harness.stages.fail_next("backtest", RetryableError("network 1"))
    harness.stages.fail_next("backtest", RetryableError("network 2"))
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "completed"
    attempts = harness.state.list_attempts(RUN_ID, "backtest")
    assert [attempt.status for attempt in attempts] == ["failed", "failed", "completed"]
    assert [attempt.attempt for attempt in attempts] == [1, 2, 3]
    # config backoff [0.5, 1.0], jitter disabled in the test budgets
    assert harness.sleeps == [0.5, 1.0]
    kinds = harness.event_kinds(RUN_ID)
    assert kinds.count("stage_retry_scheduled") == 2
    assert kinds.count("stage_failed") == 2
    step = harness.steps_by_name(RUN_ID)["backtest"]
    assert step.attempt == 3 and step.status == "completed"


def test_validation_error_blocks_with_escalation(harness: Harness) -> None:
    config = make_config(
        [stage_spec("intake"), stage_spec("backtest", needs=["intake"], retries=2)]
    )
    harness.stages.fail_next("backtest", ValidationError("artifact schema invalid"))
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "blocked"
    assert "artifact schema invalid" in (result.error or "")
    # non-retryable: exactly one attempt even though retries were configured
    assert len(harness.state.list_attempts(RUN_ID, "backtest")) == 1
    escalation = json.loads((Path(result.run_dir) / "escalation.json").read_text(encoding="utf-8"))
    assert escalation["category"] == "escalate"
    assert escalation["reason"] == "escalation_required"
    assert escalation["error_type"] == "ValidationError"
    assert harness.state.get_run(RUN_ID).status == "blocked"  # type: ignore[union-attr]
    with pytest.raises(WorkflowError, match="--force"):
        harness.runner(config).resume(RUN_ID)


def test_missing_handler_fails_run_cleanly(harness: Harness) -> None:
    config = make_config([stage_spec("intake")])
    result = harness.runner(config, registry={}).run(as_of=TEST_AS_OF, run_id="missing-handler")
    assert result.status == "failed"
    assert "no stage handler registered" in (result.error or "")
    assert "intake" in (result.error or "")


def test_retries_exhausted_blocks_and_force_resume_retries(harness: Harness) -> None:
    from aicwf.contracts import RetryableError

    config = make_config(
        [stage_spec("intake"), stage_spec("backtest", needs=["intake"], retries=1)]
    )

    def always_fail(ctx: RunContext) -> StageOutput:
        raise RetryableError("endpoint down")

    harness.stages.override("backtest", always_fail)
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "blocked"
    assert len(harness.state.list_attempts(RUN_ID, "backtest")) == 2
    escalation = json.loads((Path(result.run_dir) / "escalation.json").read_text(encoding="utf-8"))
    assert escalation["reason"] == "retries_exhausted"
    assert escalation["attempts"] == 2

    forced = harness.runner(config).resume(RUN_ID, force=True)
    assert forced.status == "blocked"  # still failing, but it did retry once more


def test_stage_returning_wrong_type_blocks(harness: Harness) -> None:
    config = make_config([stage_spec("intake")])
    harness.stages.override("intake", lambda ctx: {"not": "a stage output"})  # type: ignore[return-value]
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "blocked"
    escalation = json.loads((Path(result.run_dir) / "escalation.json").read_text(encoding="utf-8"))
    assert escalation["error_type"] == "ValidationError"


def test_stage_declaring_missing_artifact_blocks(harness: Harness) -> None:
    config = make_config([stage_spec("intake")])
    harness.stages.override(
        "intake",
        lambda ctx: StageOutput(artifacts={"artifact": str(Path(ctx.run_dir) / "nope.json")}),
    )
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "blocked"
    assert "does not exist" in (result.error or "")


def test_budget_exhaustion_sets_run_status_and_persists_ledger(harness: Harness) -> None:
    config = make_config(
        [stage_spec("intake"), stage_spec("hypothesis", needs=["intake"])]
    )
    llm = FakeLLM()

    def uses_llm(ctx: RunContext) -> StageOutput:
        ctx.llm.complete(system="system", user="user")
        ctx.llm.complete(system="system", user="user")
        return FakeStages.write_artifact(ctx, "hypothesis")

    harness.stages.override("hypothesis", uses_llm)
    result = harness.runner(config, llm=llm, budget_limits=BudgetLimits(max_llm_calls=1)).run(
        as_of=TEST_AS_OF, run_id=RUN_ID
    )
    assert result.status == "budget_exhausted"
    assert "llm_calls" in (result.error or "")
    assert llm.calls == 1
    run = harness.state.get_run(RUN_ID)
    assert run is not None and run.status == "budget_exhausted"
    budgets = {row.metric: row for row in harness.state.list_budgets(RUN_ID)}
    assert budgets["llm_calls"].used_value == 1.0
    assert budgets["llm_calls"].limit_value == 1.0
    assert "run_budget_exhausted" in harness.event_kinds(RUN_ID)
    # resuming without more budget fails fast again
    resumed = harness.runner(
        config, llm=FakeLLM(), budget_limits=BudgetLimits(max_llm_calls=1)
    ).resume(RUN_ID)
    assert resumed.status == "budget_exhausted"


# ---------------------------------------------------------------------------
# Human approval gate
# ---------------------------------------------------------------------------


def _approval_config() -> dict:
    return make_config(
        [
            stage_spec("intake"),
            stage_spec("report", needs=["intake"]),
            stage_spec("approval", needs=["report"], human_gate=True, approval_action="publish_report"),
            stage_spec("registry", needs=["approval"], human_gate=True, approval_action="promote_version"),
        ]
    )


def _wire_approval_stage(harness: Harness) -> None:
    def approval_stage(ctx: RunContext) -> StageOutput:
        gate = ApprovalGate(harness.state, ctx.config)
        path = write_approval_artifact(gate, ctx.run_id, ctx.run_dir, action="publish_report")
        return StageOutput(artifacts={"approval": str(path)}, summary={"decision": "approve"})

    harness.stages.override("approval", approval_stage)


def test_approval_gate_blocks_then_approve_and_resume(harness: Harness) -> None:
    config = _approval_config()
    _wire_approval_stage(harness)

    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "awaiting_approval"
    assert result.requires_human
    assert harness.stages.calls == ["intake", "report"]
    record = harness.state.get_run(RUN_ID)
    assert record is not None and record.status == "awaiting_approval"
    steps = harness.steps_by_name(RUN_ID)
    assert steps["report"].status == "completed"
    assert steps["approval"].status == "blocked"
    assert steps["approval"].last_disposition == "awaiting_approval"
    request_path = Path(result.run_dir) / "approval_request.json"
    assert request_path.is_file()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["action"] == "publish_report"
    assert request["stage"] == "approval"
    assert "approve" in request["instructions"]

    # resuming without a decision stays paused and does not execute the gate stage
    still_paused = harness.runner(config).resume(RUN_ID)
    assert still_paused.status == "awaiting_approval"
    assert harness.stages.calls == ["intake", "report"]

    gate = ApprovalGate(harness.state, config)
    workflow = load_workflow(config)
    assert gate.pending_gate_actions(workflow, RUN_ID) == [
        ("approval", "publish_report"),
        ("registry", "promote_version"),
    ]
    gate.record_decision(
        RUN_ID, action="publish_report", approver="alice", scope="run", note="approved for release"
    )
    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed", result.error
    assert harness.stages.calls == ["intake", "report", "approval", "registry"]
    steps = harness.steps_by_name(RUN_ID)
    assert steps["approval"].status == "completed"
    assert steps["registry"].status == "completed"
    approval_path = Path(result.run_dir) / "11_approval" / "approval.json"
    assert approval_path.is_file()
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    assert approval["decision"] == "approve"
    assert approval["approver"] == "alice"
    assert approval["action"] == "publish_report"
    assert len(approval["artifact_hashes"]) >= 2
    assert gate.pending_gate_actions(workflow, RUN_ID) == []
    assert harness.state.list_approvals(RUN_ID)[0].scope == "run"


def test_action_scoped_approval_requires_second_decision(harness: Harness) -> None:
    config = _approval_config()
    _wire_approval_stage(harness)
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)

    gate = ApprovalGate(harness.state, config)
    gate.record_decision(RUN_ID, action="publish_report", approver="alice", scope="action")
    result = harness.runner(config).resume(RUN_ID)
    # approval stage runs, then the registry gate still waits for promote_version
    assert result.status == "awaiting_approval"
    assert harness.stages.calls == ["intake", "report", "approval"]
    request = json.loads((Path(result.run_dir) / "approval_request.json").read_text(encoding="utf-8"))
    assert request["action"] == "promote_version"
    gate.record_decision(RUN_ID, action="promote_version", approver="alice", scope="action")
    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "completed"


def test_rejected_approval_blocks_run(harness: Harness) -> None:
    config = _approval_config()
    _wire_approval_stage(harness)
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    gate = ApprovalGate(harness.state, config)
    gate.record_decision(RUN_ID, action="publish_report", approver="alice", decision="rejected", note="wrong numbers")
    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "blocked"
    escalation = json.loads((Path(result.run_dir) / "escalation.json").read_text(encoding="utf-8"))
    assert escalation["reason"] == "approval_rejected"
    assert "alice" in escalation["error"]


def test_approval_binding_breaks_when_upstream_artifact_changes(harness: Harness) -> None:
    config = _approval_config()
    _wire_approval_stage(harness)
    harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    gate = ApprovalGate(harness.state, config)
    gate.record_decision(RUN_ID, action="publish_report", approver="alice", scope="run")
    assert gate.is_satisfied(RUN_ID, "publish_report")

    # simulate a stage re-run producing a different report artifact hash
    artifact = harness.state.list_artifacts(RUN_ID, "report")[0]
    from aicwf.controller.state import ArtifactRecord

    harness.state.replace_artifacts(
        RUN_ID,
        "report",
        [
            ArtifactRecord(
                run_id=RUN_ID,
                stage="report",
                logical_name=artifact.logical_name,
                relpath=artifact.relpath,
                content_hash="0" * 64,
                size_bytes=artifact.size_bytes,
            )
        ],
    )
    assert not gate.is_satisfied(RUN_ID, "publish_report")
    result = harness.runner(config).resume(RUN_ID)
    assert result.status == "awaiting_approval"  # gate must be granted again


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_run_summary_contains_step_and_artifact_digests(harness: Harness) -> None:
    config = run_config_three_stages()
    result = harness.runner(config).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    summary = json.loads((Path(result.run_dir) / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == RUN_ID
    assert summary["config_hash"]
    assert len(summary["artifacts"]) == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["content_hash"]) for item in summary["artifacts"])
    assert all(item["status"] == "completed" for item in summary["steps"])
    # run summary carries no wall-clock timestamps (only state/events do)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", json.dumps(summary))


def test_runner_wires_guarded_llm_per_stage(harness: Harness) -> None:
    from aicwf.controller import GuardedLLM

    config = make_config([stage_spec("intake"), stage_spec("hypothesis", needs=["intake"])])
    seen: list[object] = []

    def handler(ctx: RunContext) -> StageOutput:
        seen.append(ctx.llm)
        assert isinstance(ctx.llm, GuardedLLM)
        assert ctx.llm.model == "fake-llm"
        # integration hook for the pipeline approval stage
        assert isinstance(ctx.approval_gate, ApprovalGate)  # type: ignore[attr-defined]
        return FakeStages.write_artifact(ctx, "hypothesis")

    harness.stages.override("hypothesis", handler)
    result = harness.runner(config, llm=FakeLLM()).run(as_of=TEST_AS_OF, run_id=RUN_ID)
    assert result.status == "completed"
    assert len(seen) == 1
