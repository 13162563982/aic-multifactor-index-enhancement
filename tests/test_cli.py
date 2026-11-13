"""Tests for the ``aicwf`` CLI: commands, exit codes and lazy imports.

The pipeline is replaced with a tiny in-memory fake module so the CLI can be
exercised offline; paths are redirected to ``tmp_path`` through the documented
``AICWF_*`` environment variables so the real project state is never touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from aicwf import cli
from aicwf.contracts import NonRetryableError, StageOutput
from test_controller_helpers import budgets_section, gates_section, make_config, stage_spec, workflow_section

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVAL_STAGES = [
    stage_spec("intake"),
    stage_spec("report", needs=["intake"]),
    stage_spec("approval", needs=["report"], human_gate=True, approval_action="publish_report"),
    stage_spec("registry", needs=["approval"], human_gate=True, approval_action="promote_version"),
]


def _fake_handler(stage_name: str):
    def run(ctx) -> StageOutput:
        path = ctx.artifact_path(stage_name, "artifact.json")
        path.write_text(json.dumps({"stage": stage_name}), encoding="utf-8")
        return StageOutput(artifacts={"artifact": str(path)}, summary={"stage": stage_name})

    return run


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, stages: Sequence[Mapping[str, Any]]) -> types.ModuleType:
    module = types.ModuleType("aicwf.pipeline")
    registry = {str(spec["handler"]): _fake_handler(str(spec["name"])) for spec in stages}
    module.build_registry = lambda: dict(registry)  # type: ignore[attr-defined]
    module.build_llm = lambda mode, config: None  # type: ignore[attr-defined]
    module.resolve_snapshot = lambda as_of, config: (None, None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aicwf.pipeline", module)
    return module


def _install_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stages: Sequence[Mapping[str, Any]],
    *,
    gates_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = make_config(stages, gates_approval=gates_approval)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "workflow.yaml").write_text(
        yaml.safe_dump({"workflow": config["workflow"]}, sort_keys=False), encoding="utf-8"
    )
    (config_dir / "budgets.yaml").write_text(
        yaml.safe_dump(config["budgets"], sort_keys=False), encoding="utf-8"
    )
    (config_dir / "gates.yaml").write_text(
        yaml.safe_dump(config["gates"], sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("AICWF_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AICWF_STATE_DB", str(tmp_path / "state" / "runs.db"))
    monkeypatch.setenv("AICWF_RUNS_DIR", str(tmp_path / "outputs" / "runs"))
    monkeypatch.setenv("AICWF_PROJECT_ROOT", str(tmp_path))
    _install_fake_pipeline(monkeypatch, stages)
    return config


def _fake_artifact_stage(ctx, stage_name: str, payload: Mapping[str, Any] | None = None) -> StageOutput:
    path = ctx.artifact_path(stage_name, "artifact.json")
    path.write_text(json.dumps(payload or {"stage": stage_name}), encoding="utf-8")
    return StageOutput(artifacts={"artifact": str(path)}, summary={"stage": stage_name})


def _install_revision_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    always_fail: bool = False,
) -> dict[str, Any]:
    """Fake pipeline whose verification fails on the first round and can pass on a revision."""

    stages = [
        stage_spec("intake"),
        stage_spec("factor_build", needs=["intake"]),
        stage_spec("backtest", needs=["factor_build"]),
        stage_spec("verification", needs=["backtest"]),
        stage_spec("report", needs=["verification"]),
    ]
    config = _install_environment(monkeypatch, tmp_path, stages)
    rounds = {"factor_build": 0}

    def intake(ctx) -> StageOutput:
        return _fake_artifact_stage(ctx, "intake")

    def factor_build(ctx) -> StageOutput:
        rounds["factor_build"] += 1
        return _fake_artifact_stage(ctx, "factor_build", {"round": rounds["factor_build"]})

    def backtest(ctx) -> StageOutput:
        return _fake_artifact_stage(ctx, "backtest")

    def verification(ctx) -> StageOutput:
        round_no = json.loads(ctx.upstream_artifact("factor_build", "artifact").read_text(encoding="utf-8"))["round"]
        verdict = "FAIL" if (always_fail or round_no == 1) else "PASS"
        path = ctx.artifact_path("verification", "verification.json")
        payload = {
            "verdict": verdict,
            "checks": [
                {
                    "id": "weights_bounds",
                    "status": "fail" if verdict == "FAIL" else "pass",
                    "detail": "weights sum",
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return StageOutput(artifacts={"verification": str(path)}, summary={"verdict": verdict})

    def report(ctx) -> StageOutput:
        verdict = json.loads(
            ctx.upstream_artifact("verification", "verification").read_text(encoding="utf-8")
        )["verdict"]
        if verdict == "FAIL":
            raise NonRetryableError("report stage cannot publish with FAIL verification")
        return _fake_artifact_stage(ctx, "report")

    registry = {
        "intake": intake,
        "factor_build": factor_build,
        "backtest": backtest,
        "verification": verification,
        "report": report,
    }
    fake = sys.modules["aicwf.pipeline"]
    fake.build_registry = lambda: dict(registry)  # type: ignore[attr-defined]
    fake.plan_revision = lambda run_id, config: ["factor_build"]  # type: ignore[attr-defined]
    return config





def _state(tmp_path: Path):
    from aicwf.controller import StateStore

    return StateStore(tmp_path / "state" / "runs.db")


@pytest.fixture()
def simple_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _install_environment(
        monkeypatch,
        tmp_path,
        [stage_spec("intake"), stage_spec("factor_build", needs=["intake"])],
    )
    return tmp_path


@pytest.fixture()
def gated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _install_environment(monkeypatch, tmp_path, APPROVAL_STAGES)
    return tmp_path


# ---------------------------------------------------------------------------
# run / resume / status exit codes
# ---------------------------------------------------------------------------


def test_run_completes_with_exit_code_0(
    simple_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-run"])
    captured = capsys.readouterr()
    assert rc == cli.EXIT_OK
    assert "cli-run: completed" in captured.out
    state = _state(simple_env)
    try:
        record = state.get_run("cli-run")
        assert record is not None and record.status == "completed"
        assert len(state.list_steps("cli-run")) == 2
        assert (Path(record.run_dir) / "events.jsonl").is_file()
    finally:
        state.close()


def test_run_duplicate_id_fails(simple_env: Path) -> None:
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "dup"]) == cli.EXIT_OK
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "dup"]) == cli.EXIT_FAILURE


def test_llm_mode_alias_is_accepted(simple_env: Path) -> None:
    assert (
        cli.main(
            ["run", "--as-of", "2024-12-31", "--run-id", "alias", "--llm-mode", "replay"]
        )
        == cli.EXIT_OK
    )


def test_run_invalid_inputs_fail(simple_env: Path) -> None:
    assert cli.main(["run", "--as-of", "not-a-date", "--run-id", "bad"]) == cli.EXIT_FAILURE
    assert cli.main(["run", "--as-of", "2024-12-31", "--fault", "bogus"]) == cli.EXIT_FAILURE
    assert cli.main(["run", "--as-of", "2024-12-31", "--llm", "nope"]) == cli.EXIT_USAGE
    assert cli.main(["run", "--as-of", "2024-12-31", "--max-revisions", "-1"]) == cli.EXIT_FAILURE


def test_gated_run_exits_3_and_approve_completes(
    gated_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-gated"])
    assert rc == cli.EXIT_AWAITING_APPROVAL
    assert "awaiting_approval" in capsys.readouterr().out
    state = _state(gated_env)
    try:
        assert state.get_run("cli-gated").status == "awaiting_approval"  # type: ignore[union-attr]
    finally:
        state.close()

    # resume without approval stays paused
    assert cli.main(["resume", "cli-gated"]) == cli.EXIT_AWAITING_APPROVAL

    # --no-resume records the decision but leaves the run paused
    rc = cli.main(["approve", "cli-gated", "--action", "publish_report", "--no-resume"])
    assert rc == cli.EXIT_OK
    state = _state(gated_env)
    try:
        assert state.get_run("cli-gated").status == "awaiting_approval"  # type: ignore[union-attr]
        approvals = state.list_approvals("cli-gated")
        assert approvals and approvals[0].action == "publish_report"
        assert approvals[0].approver == "tester"  # from gates.yaml default_approver
    finally:
        state.close()

    rc = cli.main(["approve", "cli-gated", "--action", "publish_report", "--note", "ship it"])
    assert rc == cli.EXIT_OK
    state = _state(gated_env)
    try:
        assert state.get_run("cli-gated").status == "completed"  # type: ignore[union-attr]
        assert len(state.list_steps("cli-gated")) == 4
    finally:
        state.close()


def test_reject_returns_blocked(gated_env: Path) -> None:
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-reject"]) == cli.EXIT_AWAITING_APPROVAL
    rc = cli.main(["approve", "cli-reject", "--action", "publish_report", "--reject"])
    assert rc == cli.EXIT_BLOCKED


def test_status_exit_codes(simple_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-status"]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["status", "cli-status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "cli-status" in out and "completed" in out

    assert cli.main(["status", "cli-status", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["run_id"] == "cli-status"
    assert payload["run"]["status"] == "completed"

    assert cli.main(["status"]) == cli.EXIT_OK
    assert "cli-status" in capsys.readouterr().out

    assert cli.main(["status", "does-not-exist"]) == cli.EXIT_FAILURE
    assert "unknown run" in capsys.readouterr().err


def test_status_exit_code_for_awaiting_run(gated_env: Path) -> None:
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-await"]) == cli.EXIT_AWAITING_APPROVAL
    assert cli.main(["status", "cli-await"]) == cli.EXIT_AWAITING_APPROVAL


def test_resume_unknown_and_completed(simple_env: Path) -> None:
    assert cli.main(["resume", "nope"]) == cli.EXIT_FAILURE
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-resume"]) == cli.EXIT_OK
    assert cli.main(["resume", "cli-resume"]) == cli.EXIT_OK


# ---------------------------------------------------------------------------
# approve validation / state-export / doctor
# ---------------------------------------------------------------------------


def test_approve_unknown_run_and_action(gated_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["approve", "missing-run", "--action", "publish_report"]) == cli.EXIT_FAILURE
    assert "unknown run" in capsys.readouterr().err
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-badapp"]) == cli.EXIT_AWAITING_APPROVAL
    assert cli.main(["approve", "cli-badapp", "--action", "delete_everything"]) == cli.EXIT_FAILURE
    assert "unknown approval action" in capsys.readouterr().err


def test_state_export_stdout_and_file(
    simple_env: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["run", "--as-of", "2024-12-31", "--run-id", "cli-export"]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["state-export", "--run-id", "cli-export"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["run"]["run_id"] == "cli-export"
    assert payload["runs"][0]["steps"]

    output = tmp_path / "export.json"
    assert cli.main(["state-export", "--output", str(output)]) == cli.EXIT_OK
    assert output.is_file()
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["schema_version"] >= 1
    assert len(exported["runs"]) == 1


def test_doctor_ok_and_failure(simple_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor"]) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert "[PASS]" in output
    assert "doctor:" in output
    assert "failed" in output

    # break the workflow DAG: doctor must report FAIL and a non-zero exit code
    workflow_path = simple_env / "config" / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump({"workflow": {"name": "broken", "stages": []}}, sort_keys=False),
        encoding="utf-8",
    )
    assert cli.main(["doctor"]) == cli.EXIT_FAILURE


def test_doctor_missing_config_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AICWF_CONFIG_DIR", str(tmp_path / "empty-config"))
    monkeypatch.setenv("AICWF_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("AICWF_STATE_DB", str(tmp_path / "state" / "runs.db"))
    monkeypatch.setenv("AICWF_RUNS_DIR", str(tmp_path / "runs"))
    assert cli.main(["doctor"]) == cli.EXIT_FAILURE


# ---------------------------------------------------------------------------
# Pipeline placeholders, fault-demo, lazy imports, usage
# ---------------------------------------------------------------------------


def test_placeholder_monitor_hook_and_not_implemented(
    simple_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # simulate "no pipeline hook and no monitor subsystem yet"
    monkeypatch.setitem(sys.modules, "aicwf.monitor.run", None)
    assert cli.main(["monitor"]) == cli.EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err

    fake_pipeline = sys.modules["aicwf.pipeline"]
    fake_pipeline.run_monitor = lambda as_of, llm_mode, config: 0  # type: ignore[attr-defined]
    assert cli.main(["monitor"]) == cli.EXIT_OK

    # hooks are called with only the keyword arguments they accept
    fake_pipeline.run_monitor = lambda as_of: {"due": True}  # type: ignore[attr-defined]
    assert cli.main(["monitor"]) == cli.EXIT_OK
    assert "due" in capsys.readouterr().out

    monkeypatch.setitem(sys.modules, "aicwf.pipeline", types.ModuleType("aicwf.pipeline"))
    assert cli.main(["monitor"]) == cli.EXIT_NOT_IMPLEMENTED


def test_eval_and_rollback_hooks(simple_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_pipeline = sys.modules["aicwf.pipeline"]
    fake_pipeline.run_eval = lambda candidate, baseline, config: 1  # type: ignore[attr-defined]
    assert cli.main(["eval", "--candidate", "v2", "--baseline", "v1"]) == 1
    fake_pipeline.run_rollback = lambda to, config: {"rolled_back_to": to}  # type: ignore[attr-defined]
    assert cli.main(["rollback", "--to", "v1"]) == cli.EXIT_OK
    assert "rolled_back_to" in capsys.readouterr().out


def test_fault_demo_requires_pipeline(
    simple_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "aicwf.pipeline", None)
    rc = cli.main(["fault-demo", "--as-of", "2024-12-31"])
    assert rc == cli.EXIT_FAILURE
    assert "aicwf.pipeline" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Bounded verification revision loop
# ---------------------------------------------------------------------------


def test_bounded_revision_loop_recovers_from_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_revision_pipeline(monkeypatch, tmp_path)
    rc = cli.main(
        [
            "run",
            "--as-of",
            "2024-12-31",
            "--run-id",
            "revision-ok",
            "--llm",
            "null",
            "--max-revisions",
            "2",
        ]
    )
    output = capsys.readouterr().out
    assert rc == cli.EXIT_OK, output
    assert "revision 1/2: re-running" in output
    assert "because verification FAIL: weights_bounds" in output
    assert "revision-ok: completed" in output

    state = _state(tmp_path)
    try:
        run = state.get_run("revision-ok")
        assert run is not None and run.status == "completed"
        steps = {step.stage: step for step in state.list_steps("revision-ok")}
        assert steps["factor_build"].attempt == 2
        assert steps["backtest"].attempt == 2
        assert steps["verification"].attempt == 2
        assert steps["report"].attempt == 2
        revisions = json.loads((Path(run.run_dir) / "revisions.json").read_text(encoding="utf-8"))
        assert len(revisions["revisions"]) == 1
        assert revisions["revisions"][0]["stages"] == [
            "factor_build",
            "backtest",
            "verification",
            "report",
        ]
        assert "verification FAIL: weights_bounds" in revisions["revisions"][0]["reason"]
    finally:
        state.close()


def test_bounded_revision_loop_exhausts_budget_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_revision_pipeline(monkeypatch, tmp_path, always_fail=True)
    rc = cli.main(
        [
            "run",
            "--as-of",
            "2024-12-31",
            "--run-id",
            "revision-exhausted",
            "--llm",
            "null",
            "--max-revisions",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert rc == cli.EXIT_FAILURE, output
    assert "revision 1/1: re-running" in output
    assert "revision budget exhausted (1/1)" in output
    assert "revision-exhausted: failed" in output

    state = _state(tmp_path)
    try:
        run = state.get_run("revision-exhausted")
        assert run is not None and run.status == "failed"
        steps = {step.stage: step for step in state.list_steps("revision-exhausted")}
        assert steps["verification"].attempt == 2  # initial attempt + one exhausted revision
        revisions = json.loads((Path(run.run_dir) / "revisions.json").read_text(encoding="utf-8"))
        assert len(revisions["revisions"]) == 1
    finally:
        state.close()


def test_usage_errors_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_USAGE
    assert cli.main(["run"]) == cli.EXIT_USAGE
    assert cli.main(["unknown-command"]) == cli.EXIT_USAGE
    assert cli.main(["--version"]) == cli.EXIT_OK
    assert "aicwf" in capsys.readouterr().out


def test_cli_import_is_light() -> None:
    """Importing the CLI must not pull in the pipeline or the heavy controller."""

    code = (
        "import sys\n"
        "import aicwf.cli\n"
        "assert 'aicwf.pipeline' not in sys.modules, 'pipeline imported eagerly'\n"
        "assert 'aicwf.controller.runner' not in sys.modules, 'runner imported eagerly'\n"
        "assert 'pandas' not in sys.modules, 'pandas imported eagerly'\n"
        "print('light-import-ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "light-import-ok" in completed.stdout
