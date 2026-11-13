"""Tests for the human approval gate, artifact binding and approval artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicwf.contracts import ApprovalRequired, ConfigError
from aicwf.controller import (
    ApprovalGate,
    StateStore,
    build_escalation_record,
    load_workflow,
    write_approval_request_artifact,
    write_escalation_artifact,
)
from aicwf.controller.approval import write_approval_artifact
from aicwf.controller.state import ApprovalRecord, ArtifactRecord, RunRecord
from test_controller_helpers import gates_section, make_config, stage_spec

RUN_ID = "run-approval"


def _setup(tmp_path: Path, config: dict | None = None):
    config = config or make_config(
        [
            stage_spec("intake"),
            stage_spec("approval", needs=["intake"], human_gate=True, approval_action="publish_report"),
            stage_spec("registry", needs=["approval"], human_gate=True, approval_action="promote_version"),
        ]
    )
    store = StateStore(tmp_path / "runs.db")
    store.create_run(
        RunRecord(run_id=RUN_ID, as_of="2024-12-31", status="running", run_dir=str(tmp_path / "run"))
    )
    return store, ApprovalGate(store, config), config


def _add_artifacts(store: StateStore, mapping: dict[str, str]) -> None:
    store.replace_artifacts(
        RUN_ID,
        "backtest",
        [
            ArtifactRecord(
                run_id=RUN_ID,
                stage="backtest",
                logical_name=f"art{i}",
                relpath=relpath,
                content_hash=digest,
                size_bytes=10,
            )
            for i, (relpath, digest) in enumerate(mapping.items())
        ],
    )


def test_gate_reads_policy_from_config(tmp_path: Path) -> None:
    config = make_config(
        [stage_spec("intake")],
        gates_approval={
            "required": True,
            "actions": ["publish_report", "export_orders"],
            "bind_to_artifact_hashes": False,
            "default_approver": "alice",
        },
    )
    store, gate, _ = _setup(tmp_path, config)
    try:
        assert gate.required is True
        assert gate.required_actions == ("publish_report", "export_orders")
        assert gate.bind_to_artifact_hashes is False
        assert gate.default_approver == "alice"
    finally:
        store.close()


def test_action_for_stage_explicit_and_fallback(tmp_path: Path) -> None:
    store, gate, config = _setup(tmp_path)
    try:
        workflow = load_workflow(config)
        assert gate.action_for_stage(workflow.stage("approval")) == "publish_report"
        assert gate.action_for_stage(workflow.stage("registry")) == "promote_version"
        assert workflow.human_gates()
    finally:
        store.close()

    fallback_config = make_config([stage_spec("intake"), stage_spec("approval", needs=["intake"], human_gate=True)])
    store2 = StateStore(tmp_path / "runs2.db")
    try:
        store2.create_run(RunRecord(run_id="fallback", as_of="2024-12-31", status="running"))
        gate2 = ApprovalGate(store2, fallback_config)
        workflow2 = load_workflow(fallback_config)
        assert gate2.action_for_stage(workflow2.stage("approval")) == "stage:approval"
        decision = gate2.record_decision("fallback", action="stage:approval", note="explicit stage approval")
        assert decision.action == "stage:approval"
        assert gate2.is_satisfied("fallback", "stage:approval")
    finally:
        store2.close()


def test_scope_run_satisfies_all_actions_action_scope_does_not(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        gate.record_decision(RUN_ID, action="publish_report", approver="tester", scope="run")
        assert gate.is_satisfied(RUN_ID, "publish_report")
        assert gate.is_satisfied(RUN_ID, "promote_version")
    finally:
        store.close()

    store2 = StateStore(tmp_path / "action-scope.db")
    try:
        store2.create_run(RunRecord(run_id="r2", as_of="2024-12-31", status="running"))
        gate2 = ApprovalGate(store2, make_config([stage_spec("intake")]))
        gate2.record_decision("r2", action="publish_report", approver="tester", scope="action")
        assert gate2.is_satisfied("r2", "publish_report")
        assert not gate2.is_satisfied("r2", "promote_version")
    finally:
        store2.close()


def test_rejection_blocks_even_after_previous_approval(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        gate.record_decision(RUN_ID, action="publish_report", approver="tester", scope="run")
        assert gate.is_satisfied(RUN_ID, "publish_report")
        gate.record_decision(RUN_ID, action="publish_report", approver="tester", decision="rejected", note="numbers look wrong")
        assert gate.is_rejected(RUN_ID)
        assert not gate.is_satisfied(RUN_ID, "publish_report")
        assert gate.latest_decision(RUN_ID).decision == "rejected"  # type: ignore[union-attr]
    finally:
        store.close()


def test_artifact_hash_binding_detects_changed_upstream(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        _add_artifacts(store, {"outputs/run/05_backtest/metrics.json": "hash-a"})
        decision = gate.record_decision(RUN_ID, action="publish_report", approver="tester")
        assert decision.artifact_hashes == {"outputs/run/05_backtest/metrics.json": "hash-a"}
        assert gate.is_satisfied(RUN_ID, "publish_report")

        # change the recorded content hash as a re-run would
        _add_artifacts(store, {"outputs/run/05_backtest/metrics.json": "hash-b"})
        assert not gate.is_satisfied(RUN_ID, "publish_report")
        with pytest.raises(ApprovalRequired, match="stale"):
            gate.require(RUN_ID, "publish_report")

        # an added artifact after approval does not invalidate the bound subset
        gate.record_decision(RUN_ID, action="publish_report", approver="tester")
        store.replace_artifacts(
            RUN_ID,
            "approval",
            [
                ArtifactRecord(
                    run_id=RUN_ID,
                    stage="approval",
                    logical_name="approval",
                    relpath="outputs/run/11_approval/approval.json",
                    content_hash="hash-approval",
                )
            ],
        )
        assert gate.is_satisfied(RUN_ID, "publish_report")
    finally:
        store.close()


def test_require_and_disk_hashes(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        with pytest.raises(ApprovalRequired, match="no approved decision"):
            gate.require(RUN_ID, "promote_version")
        with pytest.raises(ApprovalRequired):
            gate.resolve_payload_action(RUN_ID)

        artifact_file = tmp_path / "run" / "05_backtest" / "metrics.json"
        artifact_file.parent.mkdir(parents=True)
        artifact_file.write_text('{"x": 1}', encoding="utf-8")
        store.replace_artifacts(
            RUN_ID,
            "backtest",
            [
                ArtifactRecord(
                    run_id=RUN_ID,
                    stage="backtest",
                    logical_name="metrics",
                    relpath=str(artifact_file),
                    content_hash="db-hash",
                )
            ],
        )
        disk_hashes = gate.disk_artifact_hashes(RUN_ID)
        assert disk_hashes[str(artifact_file)] != "db-hash"
        # unbound/nonexistent paths are reported as missing rather than raising
        store.replace_artifacts(
            RUN_ID,
            "backtest",
            [
                ArtifactRecord(
                    run_id=RUN_ID,
                    stage="backtest",
                    logical_name="gone",
                    relpath=str(tmp_path / "does-not-exist.json"),
                    content_hash="h",
                )
            ],
        )
        assert gate.disk_artifact_hashes(RUN_ID) == {str(tmp_path / "does-not-exist.json"): "missing"}
    finally:
        store.close()


def test_pending_gate_actions_and_validation(tmp_path: Path) -> None:
    store, gate, config = _setup(tmp_path)
    try:
        workflow = load_workflow(config)
        assert gate.pending_gate_actions(workflow, RUN_ID) == [
            ("approval", "publish_report"),
            ("registry", "promote_version"),
        ]
        gate.record_decision(RUN_ID, action="publish_report", scope="run")
        assert gate.pending_gate_actions(workflow, RUN_ID) == []
        assert gate.pending_actions(RUN_ID) == []

        with pytest.raises(ConfigError, match="unknown approval action"):
            gate.record_decision(RUN_ID, action="delete_everything")
        with pytest.raises(ConfigError, match="invalid approval decision"):
            gate.record_decision(RUN_ID, action="publish_report", decision="maybe")
        with pytest.raises(ConfigError, match="invalid approval scope"):
            gate.record_decision(RUN_ID, action="publish_report", scope="global")
        with pytest.raises(ConfigError, match="non-empty"):
            gate.record_decision(RUN_ID, action="")
    finally:
        store.close()


def test_write_approval_artifact_payload(tmp_path: Path) -> None:
    config = make_config(
        [
            stage_spec("intake"),
            stage_spec("approval", needs=["intake"], human_gate=True, approval_action="publish_report"),
            stage_spec("registry", needs=["approval"], human_gate=True, approval_action="promote_version"),
        ]
    )
    config["registry"] = {"candidate_version": "v1.1.0"}
    store, gate, _ = _setup(tmp_path, config)
    try:
        run_dir = tmp_path / "run"
        _add_artifacts(store, {"outputs/run/05_backtest/metrics.json": "hash-a"})
        with pytest.raises(ApprovalRequired):
            write_approval_artifact(gate, RUN_ID, run_dir)
        gate.record_decision(RUN_ID, action="publish_report", approver="alice", scope="run", note="looks good")
        path = write_approval_artifact(gate, RUN_ID, run_dir, action="publish_report")
        assert path == run_dir / "11_approval" / "approval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == RUN_ID
        assert payload["action"] == "publish_report"
        assert payload["decision"] == "approve"
        assert payload["approver"] == "alice"
        assert payload["scope"] == "run"
        assert payload["artifact_hashes"] == {"outputs/run/05_backtest/metrics.json": "hash-a"}
        assert payload["version"] == "v1.1.0"

        # the default payload action prefers the consequential promote action
        default_path = write_approval_artifact(gate, RUN_ID, run_dir, filename="approval_default.json")
        default_payload = json.loads(default_path.read_text(encoding="utf-8"))
        assert default_payload["action"] == "promote_version"
    finally:
        store.close()


def test_request_and_escalation_artifacts(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        request = gate.build_request(RUN_ID, action="publish_report", stage="approval")
        assert request.artifact_hashes == {}
        assert "python -m aicwf approve" in request.instructions
        path = write_approval_request_artifact(request, tmp_path / "run")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == RUN_ID
        assert payload["action"] == "publish_report"
        assert payload["stage"] == "approval"

        escalation = build_escalation_record(
            RUN_ID,
            "backtest",
            category="escalate",
            reason="escalation_required",
            error_type="DataQualityError",
            error="bad coverage",
            attempts=2,
        )
        escalation_path = write_escalation_artifact(escalation, tmp_path / "run")
        assert escalation_path.name == "escalation.json"
        escalation_payload = json.loads(escalation_path.read_text(encoding="utf-8"))
        assert escalation_payload["category"] == "escalate"
        assert escalation_payload["required_action"] == "human_review"
        assert escalation_payload["attempts"] == 2
    finally:
        store.close()


def test_write_approval_artifact_can_build_its_own_gate(tmp_path: Path) -> None:
    """Integration glue may omit the gate argument and pass state/config instead."""

    config = make_config(
        [
            stage_spec("intake"),
            stage_spec("approval", needs=["intake"], human_gate=True, approval_action="publish_report"),
        ]
    )
    store, gate, _ = _setup(tmp_path, config)
    try:
        run_dir = tmp_path / "run"
        _add_artifacts(store, {"outputs/run/05_backtest/metrics.json": "hash-a"})
        gate.record_decision(RUN_ID, action="publish_report", approver="alice", scope="run")
        path = write_approval_artifact(
            run_id=RUN_ID, run_dir=run_dir, state=store, config=config, action="publish_report"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == RUN_ID and payload["decision"] == "approve"
        # legacy call shape: a non-ApprovalGate first argument is tolerated; the
        # default payload action prefers the consequential promote_version action
        legacy = write_approval_artifact(
            "approval", RUN_ID, run_dir, state=store, config=config, filename="legacy.json"
        )
        assert json.loads(legacy.read_text(encoding="utf-8"))["action"] == "promote_version"
        with pytest.raises(ConfigError):
            write_approval_artifact(run_dir=run_dir, state=store, config=config)
    finally:
        store.close()


def test_approval_artifact_version_falls_back_to_gates_config(tmp_path: Path) -> None:
    config = make_config(
        [
            stage_spec("intake"),
            stage_spec("approval", needs=["intake"], human_gate=True, approval_action="publish_report"),
        ]
    )
    gates = config["gates"]
    gates.setdefault("change_control", {})["registry"] = {"candidate_version": "v9.9.9"}
    store, gate, _ = _setup(tmp_path, config)
    try:
        run_dir = tmp_path / "run"
        gate.record_decision(RUN_ID, action="publish_report", approver="alice", scope="run")
        path = write_approval_artifact(gate, RUN_ID, run_dir, action="publish_report")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == "v9.9.9"
    finally:
        store.close()


def test_approval_record_roundtrip_through_state(tmp_path: Path) -> None:
    store, gate, _ = _setup(tmp_path)
    try:
        record = gate.record_decision(RUN_ID, action="publish_report", approver="alice", note="ok")
        stored = store.list_approvals(RUN_ID)
        assert len(stored) == 1
        assert isinstance(stored[0], ApprovalRecord)
        assert stored[0].to_dict()["decision"] == "approved"
        assert stored[0].artifact_hashes == record.artifact_hashes
    finally:
        store.close()
