"""Offline tests for the eval harness: frozen baseline, hard gates, accept/reject."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aicwf.contracts import PROJECT_ROOT
from aicwf.eval.baseline import (
    baseline_metrics,
    load_baseline,
    metric_acceptance_tolerances,
    refresh_baseline,
)
from aicwf.eval.cases import list_cases, load_case
from aicwf.eval.fixtures import build_demo_run, demo_config_from_run
from aicwf.eval.runner import (
    compare_soft_metric,
    main,
    resolve_baseline_path,
    resolve_candidate_run,
    run_eval,
)
from aicwf.util.jsonio import read_json, write_json

FIXED_TIME = "2025-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("eval")
    baseline_run = build_demo_run(root / "baseline", variant="baseline")
    accepted_run = build_demo_run(root / "accepted", variant="accepted")
    rejected_run = build_demo_run(root / "rejected", variant="rejected")
    baseline_path = root / "baseline.json"
    refresh_baseline(
        baseline_run,
        out_path=baseline_path,
        baseline_id="baseline_v1.0.0",
        version="v1.0.0",
        data_scope="mini_fixture",
        evidence_mode="fixture",
        created_at_utc=FIXED_TIME,
    )
    prompt_dir = root / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "researcher.md").write_text("# candidate researcher prompt\n", encoding="utf-8")
    return {
        "root": root,
        "baseline_run": baseline_run,
        "accepted_run": accepted_run,
        "rejected_run": rejected_run,
        "baseline_path": baseline_path,
        "prompt_dir": prompt_dir,
    }


def _copy_run(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run, target)
    return target


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_baseline_refresh_pins_frozen_state(workspace: dict) -> None:
    baseline = load_baseline(workspace["baseline_path"])
    run_request = read_json(workspace["baseline_run"] / "00_intake" / "run_request.json")
    metrics = read_json(workspace["baseline_run"] / "05_backtest" / "metrics.json")["metrics"]
    assert baseline["baseline_id"] == "baseline_v1.0.0"
    assert baseline["version"] == "v1.0.0"
    assert baseline["frozen"]["snapshot_hash"] == run_request["snapshot_hash"]
    assert baseline["frozen"]["config_hash"] == run_request["config_hash"]
    assert baseline["frozen"]["llm_mode"] == "replay"
    assert baseline["metrics"]["annualized_excess_return"] == pytest.approx(metrics["annualized_excess_return"])
    assert set(baseline["tolerances"]) >= {
        "annualized_excess_return_max_degradation",
        "information_ratio_max_degradation",
        "tracking_error_abs_tolerance",
        "max_drawdown_max_degradation",
        "turnover_max_increase",
    }
    assert set(baseline["hard_gates"]) == {
        "leakage_tests_pass",
        "verification_passes",
        "deterministic_replay",
        "budget_within_limit",
    }
    assert baseline.get("config")
    assert baseline_metrics(baseline)["annualized_excess_return"] == pytest.approx(
        metrics["annualized_excess_return"]
    )


def test_baseline_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "nope.json")


def test_metric_tolerances_from_gates() -> None:
    tolerances = metric_acceptance_tolerances()
    assert tolerances["annualized_excess_return_max_degradation"] == pytest.approx(0.005)
    assert tolerances["turnover_max_increase"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Soft metric comparison semantics
# ---------------------------------------------------------------------------


def test_soft_metric_directions() -> None:
    tolerances = metric_acceptance_tolerances()
    assert compare_soft_metric("annualized_excess_return", 0.10, 0.0955, tolerances)["status"] == "pass"
    assert compare_soft_metric("annualized_excess_return", 0.10, 0.094, tolerances)["status"] == "fail"
    assert compare_soft_metric("information_ratio", 1.0, 0.95, tolerances)["status"] == "pass"
    assert compare_soft_metric("information_ratio", 1.0, 0.80, tolerances)["status"] == "fail"
    assert compare_soft_metric("tracking_error", 0.05, 0.055, tolerances)["status"] == "pass"
    assert compare_soft_metric("tracking_error", 0.05, 0.07, tolerances)["status"] == "fail"
    assert compare_soft_metric("max_drawdown", -0.10, -0.115, tolerances)["status"] == "pass"
    assert compare_soft_metric("max_drawdown", -0.10, -0.13, tolerances)["status"] == "fail"
    assert compare_soft_metric("turnover", 0.30, 0.38, tolerances)["status"] == "pass"
    assert compare_soft_metric("turnover", 0.30, 0.50, tolerances)["status"] == "fail"
    assert compare_soft_metric("turnover", 0.30, None, tolerances)["status"] == "skip"


# ---------------------------------------------------------------------------
# Accept / reject decisions
# ---------------------------------------------------------------------------


def _eval(workspace: dict, run: Path, tmp_path: Path, **kwargs) -> dict:
    return run_eval(
        run_dir=run,
        baseline_path=workspace["baseline_path"],
        out_path=tmp_path / "decision.json",
        created_at_utc=FIXED_TIME,
        **kwargs,
    )


def test_eval_accepts_candidate_within_tolerances(workspace: dict, tmp_path: Path) -> None:
    record = _eval(
        workspace,
        workspace["accepted_run"],
        tmp_path,
        case_id="accepted_v1.1.0",
        candidate_prompt_dir=workspace["prompt_dir"],
    )
    assert record["decision"] == "accept"
    assert all(gate["status"] == "pass" for gate in record["hard_gates"])
    assert record["metric_diffs"] and all(diff["status"] == "pass" for diff in record["metric_diffs"])
    assert record["candidate"]["prompt_hashes"]
    changed_keys = {change["key"] for change in record["config_changes"]}
    assert "strategy.demo.lookback" in changed_keys
    decision_path = Path(record["decision_path"])
    if not decision_path.is_absolute():
        decision_path = PROJECT_ROOT / decision_path
    assert decision_path.exists()
    assert record["rationale"]


def test_eval_rejects_metric_regression(workspace: dict, tmp_path: Path) -> None:
    record = _eval(workspace, workspace["rejected_run"], tmp_path, case_id="rejected_v1.2.0")
    assert record["decision"] == "reject"
    assert all(gate["status"] == "pass" for gate in record["hard_gates"])  # rejected on metrics, not gates
    failed = {diff["metric"] for diff in record["metric_diffs"] if diff["status"] == "fail"}
    assert "turnover" in failed
    assert "annualized_excess_return" in failed
    assert any("turnover" in reason for reason in record["reasons"])


def test_eval_rejects_failed_verification(workspace: dict, tmp_path: Path) -> None:
    run = _copy_run(workspace["accepted_run"], tmp_path)
    verification_path = run / "09_verification" / "verification.json"
    verification = read_json(verification_path)
    verification["verdict"] = "FAIL"
    write_json(verification_path, verification)
    record = _eval(workspace, run, tmp_path / "eval")
    assert record["decision"] == "reject"
    gate = next(gate for gate in record["hard_gates"] if gate["id"] == "verification_passes")
    assert gate["status"] == "fail"


def test_eval_requires_replay_evidence(workspace: dict, tmp_path: Path) -> None:
    run = _copy_run(workspace["accepted_run"], tmp_path)
    (run / "replay_evidence.json").unlink()
    record = _eval(workspace, run, tmp_path / "eval")
    assert record["decision"] == "reject"
    gate = next(gate for gate in record["hard_gates"] if gate["id"] == "deterministic_replay")
    assert gate["status"] == "fail"
    overridden = run_eval(
        run_dir=run,
        baseline_path=workspace["baseline_path"],
        out_path=tmp_path / "decision_override.json",
        hard_gate_overrides={"deterministic_replay": True},
        created_at_utc=FIXED_TIME,
    )
    assert overridden["decision"] in {"accept", "reject"}


def test_eval_rejects_missing_leakage_evidence(workspace: dict, tmp_path: Path) -> None:
    run = _copy_run(workspace["accepted_run"], tmp_path)
    (run / "02_data" / "pit_audit.json").unlink()
    record = _eval(workspace, run, tmp_path / "eval")
    assert record["decision"] == "reject"
    gate = next(gate for gate in record["hard_gates"] if gate["id"] == "leakage_tests_pass")
    assert gate["status"] == "fail"


def test_eval_rejects_budget_breach_evidence(workspace: dict, tmp_path: Path) -> None:
    run = _copy_run(workspace["accepted_run"], tmp_path)
    summary = read_json(run / "run_summary.json")
    summary["status"] = "failed"
    write_json(run / "run_summary.json", summary)
    record = _eval(workspace, run, tmp_path / "eval")
    gate = next(gate for gate in record["hard_gates"] if gate["id"] == "budget_within_limit")
    assert gate["status"] == "fail"


def test_decision_record_is_deterministic(workspace: dict, tmp_path: Path) -> None:
    out = tmp_path / "decision.json"
    first = run_eval(
        run_dir=workspace["accepted_run"],
        baseline_path=workspace["baseline_path"],
        out_path=out,
        created_at_utc=FIXED_TIME,
    )
    bytes_first = out.read_bytes()
    second = run_eval(
        run_dir=workspace["accepted_run"],
        baseline_path=workspace["baseline_path"],
        out_path=out,
        created_at_utc=FIXED_TIME,
    )
    assert first["decision"] == second["decision"] == "accept"
    assert bytes_first == out.read_bytes()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_eval_exit_codes(workspace: dict, tmp_path: Path) -> None:
    accepted_out = tmp_path / "accepted.json"
    code = main(
        [
            "--run",
            str(workspace["accepted_run"]),
            "--baseline",
            str(workspace["baseline_path"]),
            "--out",
            str(accepted_out),
            "--expect",
            "accept",
            "--created-at",
            FIXED_TIME,
        ]
    )
    assert code == 0
    assert json.loads(accepted_out.read_text(encoding="utf-8"))["decision"] == "accept"

    rejected_out = tmp_path / "rejected.json"
    code = main(
        [
            "--run",
            str(workspace["rejected_run"]),
            "--baseline",
            str(workspace["baseline_path"]),
            "--out",
            str(rejected_out),
            "--expect",
            "accept",
            "--created-at",
            FIXED_TIME,
        ]
    )
    assert code == 3  # expected decision mismatch
    assert json.loads(rejected_out.read_text(encoding="utf-8"))["decision"] == "reject"


def test_cli_baseline_refresh(workspace: dict, tmp_path: Path) -> None:
    target = tmp_path / "refreshed_baseline.json"
    code = main(
        [
            "--baseline-refresh",
            "--run",
            str(workspace["baseline_run"]),
            "--baseline",
            str(target),
            "--data-scope",
            "mini_fixture",
            "--evidence-mode",
            "fixture",
            "--created-at",
            FIXED_TIME,
        ]
    )
    assert code == 0
    baseline = load_baseline(target)
    assert baseline["data_scope"] == "mini_fixture"
    assert baseline["evidence_mode"] == "fixture"


# ---------------------------------------------------------------------------
# Committed case fixtures
# ---------------------------------------------------------------------------


def test_committed_cases_load() -> None:
    cases_root = PROJECT_ROOT / "eval" / "cases"
    if not cases_root.exists() or not list(cases_root.glob("*/case.json")):
        pytest.skip("committed eval cases have not been generated yet (run eval/make_fixture_evidence.py)")
    cases = {case["case_id"]: case for case in list_cases(cases_root)}
    assert "accepted_v1.1.0" in cases
    assert "rejected_v1.2.0" in cases
    accepted = load_case(cases_root / "accepted_v1.1.0")
    assert accepted["expected_decision"] == "accept"
    assert accepted["_prompt_dir"]
    rejected = load_case(cases_root / "rejected_v1.2.0")
    assert rejected["expected_decision"] == "reject"
    for case in (accepted, rejected):
        decision_path = Path(case["_dir"]) / "decision.json"
        if decision_path.exists():
            decision = read_json(decision_path)
            assert decision["decision"] == case["expected_decision"]


# ---------------------------------------------------------------------------
# Pipeline integration aliases
# ---------------------------------------------------------------------------


def test_pipeline_style_run_eval_aliases(workspace: dict, tmp_path: Path) -> None:
    """``aicwf.pipeline.run_eval(candidate=..., baseline=..., config=...)`` must work."""

    config = demo_config_from_run(workspace["accepted_run"])
    record = run_eval(
        candidate=workspace["accepted_run"],
        baseline=str(workspace["baseline_path"]),
        config=config,
        out_path=tmp_path / "pipeline_decision.json",
        created_at_utc=FIXED_TIME,
    )
    assert record["decision"] == "accept"
    assert Path(record["decision_path"]).exists() or (PROJECT_ROOT / str(record["decision_path"])).exists()


def test_resolve_candidate_and_baseline_targets(workspace: dict, tmp_path: Path) -> None:
    assert resolve_candidate_run(workspace["accepted_run"]) == workspace["accepted_run"]
    case_dir = tmp_path / "case"
    (case_dir / "prompts").mkdir(parents=True)
    shutil.copytree(workspace["accepted_run"], case_dir / "candidate_run")
    write_json(
        case_dir / "case.json",
        {
            "case_id": "tmp_case",
            "candidate_version": "v1.1.0",
            "expected_decision": "accept",
            "candidate_run": "candidate_run",
        },
    )
    assert resolve_candidate_run(case_dir) == case_dir / "candidate_run"
    with pytest.raises(FileNotFoundError):
        resolve_candidate_run(tmp_path / "does-not-exist")
    assert resolve_baseline_path(workspace["baseline_path"]) == workspace["baseline_path"]
    assert resolve_baseline_path(str(workspace["baseline_path"])) == workspace["baseline_path"]
    with pytest.raises(FileNotFoundError):
        resolve_baseline_path("no-such-baseline")


def test_cli_candidate_alias(workspace: dict, tmp_path: Path) -> None:
    code = main(
        [
            "--candidate",
            str(workspace["accepted_run"]),
            "--baseline",
            str(workspace["baseline_path"]),
            "--out",
            str(tmp_path / "alias_decision.json"),
            "--created-at",
            FIXED_TIME,
        ]
    )
    assert code == 0
    record = read_json(tmp_path / "alias_decision.json")
    assert record["decision"] == "accept"


def test_committed_baseline_loads() -> None:
    """The committed baseline must be a valid, meaningfully frozen baseline.

    It is now the REAL-data baseline (frozen from outputs/runs/final-2026-11-12,
    snapshot 8ab22713...); the earlier fixture-backed placeholder is preserved as
    eval/baseline_v1.0.0.fixture.json for offline regression.
    """

    path = PROJECT_ROOT / "eval" / "baseline_v1.0.0.json"
    if not path.exists():
        pytest.skip("committed baseline not generated yet")
    baseline = load_baseline(path)
    frozen = baseline["frozen"]
    assert baseline["baseline_id"] == "baseline_v1.0.0"
    assert baseline["version"] == "v1.0.0"
    assert baseline["evidence_mode"] in {"fixture", "real"}
    assert frozen["as_of"]
    assert frozen["snapshot_hash"]
    assert frozenset(baseline["hard_gates"]) == frozenset(
        {"leakage_tests_pass", "verification_passes", "deterministic_replay", "budget_within_limit"}
    )
    assert set(baseline["tolerances"]) >= {
        "annualized_excess_return_max_degradation",
        "information_ratio_max_degradation",
        "tracking_error_abs_tolerance",
        "max_drawdown_max_degradation",
        "turnover_max_increase",
    }
    assert baseline["metrics"]["annualized_excess_return"] is not None
    assert baseline["metrics"]["tracking_error"] is not None
    assert (baseline.get("verification") or {}).get("verdict") == "PASS"
    if baseline["evidence_mode"] == "real":
        assert baseline["data_scope"] == "real_csi300"
        assert frozen["snapshot_hash"] == (
            "8ab22713074fcfef89f9e1fd25116536c53e3a8f385adbadcf4e41c154c125c8"
        )
        assert frozen["as_of"] == "2026-11-12"
        assert frozen["prompt_hashes"]  # real prompts are pinned
    fixture_path = PROJECT_ROOT / "eval" / "baseline_v1.0.0.fixture.json"
    if fixture_path.exists():
        fixture_baseline = load_baseline(fixture_path)
        assert fixture_baseline["evidence_mode"] == "fixture"
        assert fixture_baseline["metrics"]["annualized_excess_return"] is not None
