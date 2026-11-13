"""Tests for the SQLite state store (schema, persistence, export, transitions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicwf.controller import StateError, StateStore
from aicwf.controller.state import (
    SCHEMA_VERSION,
    ApprovalRecord,
    ArtifactRecord,
    AttemptRecord,
    BudgetRecord,
    RunEventSink,
    RunRecord,
    StepRecord,
    default_db_path,
)

RUN_ID = "run-state-1"


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "runs.db")


def _run(run_id: str = RUN_ID, status: str = "running") -> RunRecord:
    return RunRecord(run_id=run_id, as_of="2024-12-31", status=status, run_dir=str(Path("outputs") / "runs" / run_id))


def _step(run_id: str = RUN_ID, stage: str = "intake", status: str = "completed") -> StepRecord:
    return StepRecord(
        run_id=run_id,
        stage=stage,
        status=status,
        attempt=1,
        max_attempts=1,
        input_hash="in-hash",
        output_hash="out-hash",
        duration_s=0.25,
        summary={"rows": 3},
        last_disposition="executed",
    )


def test_schema_has_required_tables_and_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        rows = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        tables = {row["name"] for row in rows}
        assert {
            "runs",
            "steps",
            "attempts",
            "artifacts",
            "events",
            "budgets",
            "approvals",
        } <= tables
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        # WAL is enabled for crash-safe resume.
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        store.close()


def test_default_db_path_is_state_runs_db(tmp_path: Path) -> None:
    assert default_db_path(tmp_path) == tmp_path / "state" / "runs.db"


def test_run_roundtrip_and_listing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        record = store.create_run(_run())
        assert record.created_at_utc and record.updated_at_utc
        fetched = store.get_run(RUN_ID)
        assert fetched is not None
        assert fetched.as_of == "2024-12-31"
        assert fetched.status == "running"
        updated = store.update_run(RUN_ID, status="completed", error=None, current_stage="registry")
        assert updated.status == "completed"
        assert updated.current_stage == "registry"
        assert store.get_run("missing") is None
        assert [run.run_id for run in store.list_runs()] == [RUN_ID]
        assert store.latest_run().run_id == RUN_ID  # type: ignore[union-attr]
        store.create_run(
            RunRecord(
                run_id="run-state-2",
                as_of="2024-12-31",
                status="failed",
                run_dir="outputs/runs/run-state-2",
            )
        )
        assert {run.run_id for run in store.list_runs(status="failed")} == {"run-state-2"}
    finally:
        store.close()


def test_state_guards_reject_invalid_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        with pytest.raises(StateError):
            store.create_run(_run(status="not-a-status"))
        store.create_run(_run())
        with pytest.raises(StateError):
            store.create_run(_run())  # duplicate
        with pytest.raises(StateError):
            store.update_run(RUN_ID, bogus_field=1)
        with pytest.raises(StateError):
            store.update_run("unknown-run", status="completed")
        with pytest.raises(StateError):
            store.upsert_step(StepRecord(run_id=RUN_ID, stage="intake", status="nope"))
        with pytest.raises(StateError):
            store.export_run("unknown-run")
    finally:
        store.close()


def test_complete_stage_and_artifact_replacement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_run())
        artifact_file = tmp_path / "outputs" / "runs" / RUN_ID / "00_intake" / "run_request.json"
        artifact_file.parent.mkdir(parents=True)
        artifact_file.write_text("{}", encoding="utf-8")
        artifact = ArtifactRecord(
            run_id=RUN_ID,
            stage="intake",
            logical_name="run_request",
            relpath=str(artifact_file),
            content_hash="hash-1",
            size_bytes=2,
        )
        store.complete_stage(
            step=_step(),
            attempt=AttemptRecord(run_id=RUN_ID, stage="intake", attempt=1, status="completed"),
            artifacts=[artifact],
        )
        step = store.get_step(RUN_ID, "intake")
        assert step is not None and step.status == "completed" and step.summary == {"rows": 3}
        records = store.list_artifacts(RUN_ID, "intake")
        assert len(records) == 1 and records[0].content_hash == "hash-1"
        assert store.artifact_hashes(RUN_ID) == {str(artifact_file): "hash-1"}
        assert store.get_artifact(RUN_ID, "intake", "run_request") is not None
        assert store.get_artifact(RUN_ID, "intake", "missing") is None

        replaced = ArtifactRecord(
            run_id=RUN_ID,
            stage="intake",
            logical_name="run_request_v2",
            relpath=str(artifact_file),
            content_hash="hash-2",
            size_bytes=2,
        )
        store.replace_artifacts(RUN_ID, "intake", [replaced])
        assert [item.logical_name for item in store.list_artifacts(RUN_ID, "intake")] == ["run_request_v2"]

        store.fail_stage(
            step=_step(stage="hypothesis", status="failed"),
            attempt=AttemptRecord(
                run_id=RUN_ID,
                stage="hypothesis",
                attempt=1,
                status="failed",
                error_type="RetryableError",
                error="boom",
            ),
        )
        assert store.get_step(RUN_ID, "hypothesis").status == "failed"  # type: ignore[union-attr]
        attempts = store.list_attempts(RUN_ID, "hypothesis")
        assert attempts and attempts[0].error_type == "RetryableError"
        store.update_attempt(RUN_ID, "hypothesis", 1, status="crashed")
        assert store.list_attempts(RUN_ID, "hypothesis")[0].status == "crashed"
    finally:
        store.close()


def test_invalidate_and_reset_running_steps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_run())
        store.upsert_step(_step())
        store.replace_artifacts(
            RUN_ID,
            "intake",
            [
                ArtifactRecord(
                    run_id=RUN_ID,
                    stage="intake",
                    logical_name="x",
                    relpath="some/path.json",
                    content_hash="h",
                )
            ],
        )
        store.invalidate_step(RUN_ID, "intake", reason="input_hash_changed")
        step = store.get_step(RUN_ID, "intake")
        assert step is not None
        assert step.status == "pending" and step.input_hash is None and step.output_hash is None
        assert step.last_disposition == "invalidated:input_hash_changed"
        assert store.list_artifacts(RUN_ID, "intake") == []

        store.upsert_step(_step(stage="factor_build", status="running"))
        reset = store.reset_running_steps(RUN_ID)
        assert reset == ["factor_build"]
        assert store.get_step(RUN_ID, "factor_build").status == "pending"  # type: ignore[union-attr]
    finally:
        store.close()


def test_events_dual_sink(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_run())
        run_dir = tmp_path / "run-dir"
        sink = RunEventSink(store, run_dir / "events.jsonl", RUN_ID)
        sink.emit("run_created", payload={"as_of": "2024-12-31"})
        sink.emit("stage_started", level="warning", payload={"stage": "intake"})
        db_events = store.list_events(RUN_ID)
        assert [event.kind for event in db_events] == ["run_created", "stage_started"]
        assert db_events[1].level == "warning"
        lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["kind"] == "stage_started"
        recent = store.recent_events(RUN_ID, limit=1)
        assert [event.kind for event in recent] == ["stage_started"]
    finally:
        store.close()


def test_budgets_and_approvals_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_run())
        store.upsert_budgets(RUN_ID, {"llm_calls": (10.0, 3.0), "cost_usd": (0.0, 0.0)})
        budgets = {item.metric: item for item in store.list_budgets(RUN_ID)}
        assert budgets["llm_calls"].limit_value == 10.0
        assert budgets["llm_calls"].used_value == 3.0
        store.upsert_budgets(RUN_ID, [BudgetRecord(run_id=RUN_ID, metric="llm_calls", limit_value=20.0, used_value=4.0)])
        budgets = {item.metric: item for item in store.list_budgets(RUN_ID)}
        assert budgets["llm_calls"].used_value == 4.0

        approved = store.add_approval(
            ApprovalRecord(
                run_id=RUN_ID,
                action="publish_report",
                decision="approved",
                approver="tester",
                scope="run",
                artifact_hashes={"a/b.json": "hash"},
                note="ok",
            )
        )
        assert approved.approval_id
        store.add_approval(
            ApprovalRecord(run_id=RUN_ID, action="promote_version", decision="rejected", approver="tester")
        )
        assert len(store.list_approvals(RUN_ID)) == 2
        assert len(store.list_approvals(RUN_ID, action="publish_report")) == 1
        assert len(store.list_approvals(RUN_ID, decision="rejected")) == 1
    finally:
        store.close()


def test_export_run_and_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_run())
        store.upsert_step(_step())
        store.record_attempt(AttemptRecord(run_id=RUN_ID, stage="intake", attempt=1, status="completed"))
        store.replace_artifacts(
            RUN_ID,
            "intake",
            [
                ArtifactRecord(
                    run_id=RUN_ID,
                    stage="intake",
                    logical_name="artifact",
                    relpath="outputs/runs/x/00_intake/a.json",
                    content_hash="hash",
                    size_bytes=2,
                )
            ],
        )
        store.upsert_budgets(RUN_ID, {"llm_calls": (10.0, 1.0)})
        store.add_approval(ApprovalRecord(run_id=RUN_ID, action="promote_version", approver="tester"))
        store.add_event(RUN_ID, "run_created", payload={"a": 1})

        exported = store.export_run(RUN_ID)
        assert exported["schema_version"] == SCHEMA_VERSION
        assert exported["run"]["run_id"] == RUN_ID
        assert len(exported["steps"]) == 1
        assert len(exported["attempts"]) == 1
        assert len(exported["artifacts"]) == 1
        assert len(exported["budgets"]) == 1
        assert len(exported["approvals"]) == 1
        assert len(exported["events"]) == 1

        all_state = store.export_state()
        assert len(all_state["runs"]) == 1
        assert all_state["exported_at_utc"]
        assert store.export_state(RUN_ID)["runs"][0]["run"]["run_id"] == RUN_ID

        path = store.write_state_export(tmp_path / "state_snapshot.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["runs"][0]["run"]["run_id"] == RUN_ID
    finally:
        store.close()
