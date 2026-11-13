"""Offline tests for the append-only version registry, promotion and rollback."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

from aicwf.contracts import ApprovalRequired, RunContext, ValidationError
from aicwf.eval.fixtures import DEFAULT_SNAPSHOT_DIR, build_demo_run, demo_config_from_run
from aicwf.registry.promote import load_approval, promote, promote_run, validate_approval
from aicwf.registry.rollback import rollback
from aicwf.registry.stage import registry_status, run_registry
from aicwf.registry.store import (
    RegistryConflict,
    RegistryIntegrityError,
    append_entry,
    load_history,
    verify_bundle,
)
from aicwf.util.jsonio import read_json, write_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("registry_baseline")
    return build_demo_run(root / "base", variant="baseline", replay_evidence=False)


@pytest.fixture(scope="module")
def accepted_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("registry_accepted")
    return build_demo_run(root / "accepted", variant="accepted", replay_evidence=False)


@pytest.fixture()
def registry(tmp_path: Path) -> tuple[Path, Path]:
    history = tmp_path / "registry" / "versions.json"
    releases = tmp_path / "registry" / "releases"
    return history, releases


def _config(run: Path) -> dict:
    return demo_config_from_run(run)


def _promote(run: Path, version: str, registry: tuple[Path, Path], **kwargs) -> dict:
    from aicwf.contracts import PROJECT_ROOT

    history, releases = registry
    return promote_run(
        run,
        version=version,
        history_path=history,
        releases=releases,
        project_root=kwargs.pop("project_root", PROJECT_ROOT),
        created_at_utc="2025-01-01T00:00:00+00:00",
        **kwargs,
    )


def _copy_run(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run, target)
    return target


# ---------------------------------------------------------------------------
# Append-only store
# ---------------------------------------------------------------------------


def test_append_only_history_digest_chain(tmp_path: Path) -> None:
    history_path = tmp_path / "versions.json"
    first = append_entry({"kind": "release", "version": "v0.1.0", "decision": "accept", "metrics": {"x": 1}}, path=history_path)
    second = append_entry({"kind": "release", "version": "v0.2.0", "decision": "accept", "metrics": {"x": 2}}, path=history_path)
    history = load_history(history_path)
    assert [entry["version"] for entry in history["entries"]] == ["v0.1.0", "v0.2.0"]
    assert second["previous_entry_hash"] == first["entry_hash"]
    assert [entry["sequence"] for entry in history["entries"]] == [1, 2]

    # Tampering with an appended entry breaks the digest chain.
    tampered = json.loads(history_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["decision"] = "reject"
    write_json(history_path, tampered)
    with pytest.raises(RegistryIntegrityError):
        load_history(history_path)


def test_promote_requires_approval(tmp_path: Path, baseline_run: Path, registry) -> None:
    run = _copy_run(baseline_run, tmp_path)
    (run / "11_approval" / "approval.json").unlink()
    with pytest.raises(ApprovalRequired):
        _promote(run, "v1.0.0", registry)


def test_promote_rejects_wrong_artifact_hashes(tmp_path: Path, baseline_run: Path, registry) -> None:
    run = _copy_run(baseline_run, tmp_path)
    history, releases = registry
    approval = read_json(run / "11_approval" / "approval.json")
    approval["artifact_hashes"]["05_backtest/ledger.parquet"] = "0" * 64
    with pytest.raises(ApprovalRequired):
        promote(
            version="v1.0.0",
            run_dir=run,
            approval=approval,
            history_path=history,
            releases=releases,
            created_at_utc="2025-01-01T00:00:00+00:00",
        )


def test_promote_rejects_artifact_changed_after_approval(tmp_path: Path, baseline_run: Path, registry) -> None:
    run = _copy_run(baseline_run, tmp_path)
    metrics_path = run / "05_backtest" / "metrics.json"
    metrics = read_json(metrics_path)
    metrics["metrics"]["annualized_excess_return"] = 0.9
    write_json(metrics_path, metrics)
    with pytest.raises(ApprovalRequired):
        _promote(run, "v1.0.0", registry)


def test_promote_rejects_failing_verification(tmp_path: Path, baseline_run: Path, registry) -> None:
    from aicwf.util.artifact_hash import hash_artifact

    run = _copy_run(baseline_run, tmp_path)
    verification_path = run / "09_verification" / "verification.json"
    verification = read_json(verification_path)
    verification["verdict"] = "FAIL"
    write_json(verification_path, verification)
    approval = read_json(run / "11_approval" / "approval.json")
    # Re-approve the (now failing) artifact so binding passes and the verdict is the rejection reason.
    approval["artifact_hashes"]["09_verification/verification.json"] = hash_artifact(verification_path)
    with pytest.raises(ValidationError):
        promote(
            version="v1.0.0",
            run_dir=run,
            approval=approval,
            history_path=registry[0],
            releases=registry[1],
        )


def test_promote_appends_release_and_frozen_bundle(baseline_run: Path, registry) -> None:
    history, releases = registry
    entry = _promote(baseline_run, "v1.0.0", registry)
    run_request = read_json(baseline_run / "00_intake" / "run_request.json")
    assert entry["kind"] == "release"
    assert entry["decision"] == "accept"
    assert entry["verification_verdict"] == "PASS"
    assert entry["previous_version"] is None
    assert entry["sequence"] == 1
    assert entry["snapshot_hash"] == run_request["snapshot_hash"]
    assert entry["metrics"]["annualized_excess_return"] is not None
    assert entry["approval"]["approver"]
    bundle_dir = releases / "v1.0.0"
    assert bundle_dir.exists()
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "config" / "strategy.yaml").exists()
    assert verify_bundle(bundle_dir) == []
    assert load_history(history)["entries"][-1]["entry_hash"] == entry["entry_hash"]


def test_promote_is_append_only(baseline_run: Path, registry) -> None:
    _promote(baseline_run, "v1.0.0", registry)
    with pytest.raises(RegistryConflict):
        _promote(baseline_run, "v1.0.0", registry)


def test_approval_scope_run_covers_promotion() -> None:
    approval = {
        "run_id": "run-1",
        "action": "publish_report",
        "scope": "run",
        "approver": "Zhengjia Zhu",
        "decision": "approve",
        "artifact_hashes": {"a": "b"},
    }
    validate_approval(approval, run_id="run-1", declared_hashes={"a": "b"})
    with pytest.raises(ApprovalRequired):
        validate_approval({**approval, "decision": "reject"}, run_id="run-1", declared_hashes={"a": "b"})
    with pytest.raises(ApprovalRequired):
        validate_approval(approval, run_id="other-run", declared_hashes={"a": "b"})


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_round_trip(baseline_run: Path, accepted_run: Path, tmp_path: Path, registry) -> None:
    history, releases = registry
    baseline_entry = _promote(baseline_run, "v1.0.0", registry)
    accepted_entry = _promote(accepted_run, "v1.1.0", registry)
    history_before = json.loads(history.read_text(encoding="utf-8"))["entries"]

    restore_root = tmp_path / "restored"
    result = rollback(
        to_version="v1.0.0",
        history_path=history,
        releases=releases,
        project_root=tmp_path / "project",
        restore_targets={
            "config": restore_root / "config",
            "prompts": restore_root / "prompts",
            "skills": restore_root / "skills",
        },
        created_at_utc="2025-01-02T00:00:00+00:00",
    )
    assert result["status"] == "rolled_back"
    assert result["from_version"] == "v1.1.0"
    assert result["to_version"] == "v1.0.0"
    assert result["smoke_eval"]["status"] == "pass"
    assert result["before_metrics"]["annualized_excess_return"] == pytest.approx(
        accepted_entry["metrics"]["annualized_excess_return"]
    )
    assert result["after_metrics"]["annualized_excess_return"] == pytest.approx(
        baseline_entry["metrics"]["annualized_excess_return"]
    )
    assert result["restored_files"]

    restored_strategy = restore_root / "config" / "strategy.yaml"
    assert restored_strategy.exists()
    import yaml

    restored = yaml.safe_load(restored_strategy.read_text(encoding="utf-8"))
    demo = (restored.get("demo") or (restored.get("strategy") or {}).get("demo") or {})
    assert demo.get("variant") == "baseline"

    history_after = load_history(history)["entries"]
    assert len(history_after) == len(history_before) + 1
    assert history_after[: len(history_before)] == history_before  # history never rewritten
    rollback_entry = history_after[-1]
    assert rollback_entry["kind"] == "rollback"
    assert rollback_entry["from_version"] == "v1.1.0"
    assert rollback_entry["to_version"] == "v1.0.0"
    assert rollback_entry["before_metrics"] == accepted_entry["metrics"]
    assert rollback_entry["after_metrics"] == baseline_entry["metrics"]
    assert rollback_entry["smoke_eval"]["status"] == "pass"


def test_rollback_detects_corrupt_bundle(baseline_run: Path, tmp_path: Path, registry) -> None:
    history, releases = registry
    _promote(baseline_run, "v1.0.0", registry)
    config_path = releases / "v1.0.0" / "config" / "strategy.yaml"
    config_path.write_text("# tampered after release\n", encoding="utf-8")
    assert verify_bundle(releases / "v1.0.0")
    with pytest.raises(RegistryIntegrityError):
        rollback(to_version="v1.0.0", history_path=history, releases=releases)


def test_rollback_unknown_version(registry) -> None:
    history, releases = registry
    with pytest.raises(ValidationError):
        rollback(to_version="v9.9.9", history_path=history, releases=releases)


def test_rollback_dry_run_does_not_append(baseline_run: Path, registry) -> None:
    _promote(baseline_run, "v1.0.0", registry)
    history, releases = registry
    before = len(load_history(history)["entries"])
    result = rollback(to_version="v1.0.0", history_path=history, releases=releases, dry_run=True)
    assert result["status"] == "dry_run"
    assert len(load_history(history)["entries"]) == before


# ---------------------------------------------------------------------------
# Registry stage
# ---------------------------------------------------------------------------


def _stage_context(run: Path, tmp_path: Path) -> RunContext:
    config = _config(run)
    registry_cfg = config["gates"].setdefault("registry", {})
    registry_cfg["versions_file"] = str(tmp_path / "registry" / "versions.json")
    registry_cfg["releases_dir"] = str(tmp_path / "registry" / "releases")
    return RunContext(
        run_id="demo-baseline-2024-12-31",
        as_of=dt.date(2024, 12, 31),
        run_dir=run,
        config=config,
        llm_mode="replay",
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
    )


def test_registry_stage_promotes_approved_run(baseline_run: Path, tmp_path: Path) -> None:
    ctx = _stage_context(baseline_run, tmp_path)
    output = run_registry(ctx)
    assert output.summary["decision"] == "accept"
    assert output.summary["version"] == "v1.0.0"
    entry_path = Path(output.artifacts["version_entry"])
    if not entry_path.is_absolute():
        from aicwf.contracts import PROJECT_ROOT

        entry_path = PROJECT_ROOT / entry_path
    assert entry_path.exists()
    written = read_json(entry_path)
    assert written["version"] == "v1.0.0"
    history, releases = tmp_path / "registry" / "versions.json", tmp_path / "registry" / "releases"
    assert load_history(history)["entries"][-1]["version"] == "v1.0.0"
    assert verify_bundle(releases / "v1.0.0") == []


def test_registry_stage_requires_approval(baseline_run: Path, tmp_path: Path) -> None:
    run = _copy_run(baseline_run, tmp_path)
    (run / "11_approval" / "approval.json").unlink()
    ctx = _stage_context(run, tmp_path)
    with pytest.raises(ApprovalRequired):
        run_registry(ctx)


def test_registry_stage_refuses_failed_verification(baseline_run: Path, tmp_path: Path) -> None:
    from aicwf.util.artifact_hash import hash_artifact

    run = _copy_run(baseline_run, tmp_path)
    verification_path = run / "09_verification" / "verification.json"
    verification = read_json(verification_path)
    verification["verdict"] = "FAIL"
    write_json(verification_path, verification)
    approval_path = run / "11_approval" / "approval.json"
    approval = read_json(approval_path)
    approval["artifact_hashes"]["09_verification/verification.json"] = hash_artifact(verification_path)
    write_json(approval_path, approval)
    ctx = _stage_context(run, tmp_path)
    with pytest.raises(ValidationError):
        run_registry(ctx)


def test_registry_status_reports_bundles(baseline_run: Path, registry) -> None:
    history, releases = registry
    _promote(baseline_run, "v1.0.0", registry)
    report = registry_status(history_path=history, releases=releases)
    assert report["valid_chain"] is True
    assert report["entries"] == 1
    assert report["problems"] == []
    assert report["releases"][0]["version"] == "v1.0.0"
    assert report["releases"][0]["bundle_intact"] is True


# ---------------------------------------------------------------------------
# Pipeline integration aliases / key forms
# ---------------------------------------------------------------------------


def test_promote_accepts_stage_logical_approval_keys(tmp_path: Path, baseline_run: Path, registry) -> None:
    """The pipeline approval handler records ``<stage>:<logical>`` hash keys."""

    from aicwf.util.artifact_hash import hash_artifact

    run = _copy_run(baseline_run, tmp_path)
    approval_path = run / "11_approval" / "approval.json"
    approval = read_json(approval_path)
    approval["artifact_hashes"] = {
        "verification:verification": hash_artifact(run / "09_verification" / "verification.json"),
        "backtest:metrics": hash_artifact(run / "05_backtest" / "metrics.json"),
        "portfolio:target_weights": hash_artifact(run / "08_portfolio" / "target_weights.parquet"),
    }
    write_json(approval_path, approval)
    entry = _promote(run, "v1.0.0", registry)
    assert entry["version"] == "v1.0.0"
    assert entry["approval"]["n_artifact_hashes"] == 3


def test_rollback_to_pipeline_hook(baseline_run: Path, registry) -> None:
    from aicwf.registry.rollback import rollback_to

    history, releases = registry
    _promote(baseline_run, "v1.0.0", registry)
    config = {
        "gates": {
            "change_control": {
                "registry": {"versions_file": str(history), "releases_dir": str(releases)}
            }
        }
    }
    result = rollback_to(to_version="v1.0.0", config=config, restore=False)
    assert result["status"] == "rolled_back"
    assert result["to_version"] == "v1.0.0"


def test_registry_stage_auto_selects_version(tmp_path: Path, accepted_run: Path) -> None:
    """With no version configured anywhere, the stage picks v1.0.0 then v1.1.0."""

    from aicwf.util.artifact_hash import hash_artifact

    run = _copy_run(accepted_run, tmp_path)
    run_request_path = run / "00_intake" / "run_request.json"
    run_request = read_json(run_request_path)
    run_request.pop("candidate_version", None)
    write_json(run_request_path, run_request)
    approval_path = run / "11_approval" / "approval.json"
    approval = read_json(approval_path)
    approval.pop("version", None)
    # The artifact changed after the run was built; re-approve its new content.
    approval["artifact_hashes"]["00_intake/run_request.json"] = hash_artifact(run_request_path)
    write_json(approval_path, approval)
    config = _config(run)
    config["gates"].setdefault("registry", {})
    config["gates"]["registry"]["versions_file"] = str(tmp_path / "registry" / "versions.json")
    config["gates"]["registry"]["releases_dir"] = str(tmp_path / "registry" / "releases")
    # Simulate "no version configured anywhere": drop the project-level candidate_version.
    change_control_registry = config["gates"].setdefault("change_control", {}).setdefault("registry", {})
    change_control_registry.pop("candidate_version", None)
    ctx = RunContext(
        run_id="demo-accepted-2024-12-31",
        as_of=dt.date(2024, 12, 31),
        run_dir=run,
        config=config,
        llm_mode="replay",
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
    )
    first = run_registry(ctx)
    assert first.summary["version"] == "v1.0.0"
    # A second promotion of a fresh run auto-bumps to the next monthly version.
    run2 = _copy_run(accepted_run, tmp_path / "second")
    run_request2_path = run2 / "00_intake" / "run_request.json"
    run_request2 = read_json(run_request2_path)
    run_request2.pop("candidate_version", None)
    write_json(run_request2_path, run_request2)
    approval2_path = run2 / "11_approval" / "approval.json"
    approval2 = read_json(approval2_path)
    approval2.pop("version", None)
    approval2["artifact_hashes"]["00_intake/run_request.json"] = hash_artifact(run_request2_path)
    write_json(approval2_path, approval2)
    ctx2 = RunContext(
        run_id="demo-accepted-2024-12-31-second",
        as_of=dt.date(2024, 12, 31),
        run_dir=run2,
        config=config,
        llm_mode="replay",
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
    )
    second = run_registry(ctx2)
    assert second.summary["version"] == "v1.1.0"
