"""Offline tests for independent verification (clean run + tamper detection).

Fixtures: deterministic demo runs built on the frozen mini snapshot.  Each test
tampers one artifact and checks that verification fails closed with the expected
check id, so the suite proves the verifier actually recomputes instead of
trusting the run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from aicwf.contracts import RunContext
from aicwf.eval.fixtures import (
    DEFAULT_SNAPSHOT_DIR,
    build_demo_run,
    demo_config_from_run,
)
from aicwf.util.hashing import hash_json
from aicwf.verify.stage import (
    build_verification,
    generate_verifier_note_text,
    is_blocking,
    render_verifier_note,
    run_verification,
)

AS_OF = "2024-12-31"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(verification: dict, check_id: str) -> dict:
    for check in verification["checks"]:
        if check["id"] == check_id:
            return check
    raise AssertionError(f"check {check_id!r} missing from verification: {[check['id'] for check in verification['checks']]}")


def _copy_run(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run, target)
    return target


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _verify(run: Path, snapshot: Path = DEFAULT_SNAPSHOT_DIR, **kwargs) -> dict:
    config = demo_config_from_run(run)
    kwargs.setdefault("as_of", AS_OF)
    return build_verification(run_dir=run, config=config, snapshot_dir=snapshot, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("verify_clean")
    return build_demo_run(root / "base", variant="baseline", with_verification=False, replay_evidence=False)


@pytest.fixture(scope="module")
def clean_verification(clean_run: Path) -> dict:
    return _verify(clean_run)


# ---------------------------------------------------------------------------
# Clean run
# ---------------------------------------------------------------------------


def test_clean_run_passes_every_required_check(clean_verification: dict) -> None:
    verification = clean_verification
    assert verification["verdict"] == "PASS"
    required = verification["verifier"]["required_checks"]
    assert set(required) == {
        "artifact_schema",
        "artifact_hashes",
        "ledger_reconcile",
        "weights_bounds",
        "industry_neutrality",
        "forbidden_securities",
        "pit_fundamentals",
        "pit_prices",
        "determinism_summary",
    }
    for check_id in required:
        check = _check(verification, check_id)
        assert check["status"] == "pass", f"{check_id}: {check['detail']}"
    assert verification["discrepancies"] == []
    assert verification["verifier"]["checks_failed"] == 0
    assert len(verification["artifact_hashes"]) >= 15
    assert verification["recomputed"]["metrics"]["annualized_excess_return"] is not None
    assert is_blocking(verification) is False


def test_verification_is_deterministic(clean_run: Path) -> None:
    first = _verify(clean_run)
    second = _verify(clean_run)
    assert hash_json(first) == hash_json(second)


def test_verifier_note_is_deterministic_and_cites_verdict(clean_verification: dict) -> None:
    note = render_verifier_note(clean_verification)
    assert "PASS" in note
    assert "Independent" in note
    assert note == render_verifier_note(clean_verification)
    fallback = generate_verifier_note_text(clean_verification, None)
    assert "PASS" in fallback


def test_independent_module_does_not_import_backtest() -> None:
    """The verifier must not import the implementation it verifies (static import scan)."""

    import ast

    verify_root = Path(__file__).resolve().parents[1] / "aicwf" / "verify"
    forbidden_prefixes = ("aicwf.backtest", "aicwf.portfolio", "aicwf.factors")
    offenders: list[str] = []
    for path in verify_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [node.module or ""]
            else:
                continue
            for target in targets:
                if any(target == prefix or target.startswith(prefix + ".") for prefix in forbidden_prefixes):
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', '?')} imports {target}")
    assert not offenders, "verification imports the implementation under test: " + "; ".join(offenders)


def test_verification_without_explicit_as_of_uses_run_request(clean_run: Path) -> None:
    verification = _verify(clean_run, as_of=None)
    assert verification["as_of"] == AS_OF
    assert verification["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampered_ledger_cash_flow_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    ledger_path = run / "05_backtest" / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    ledger.loc[0, "cash_flow"] = float(ledger.loc[0, "cash_flow"]) + 1000.0
    _write_frame(ledger_path, ledger)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "ledger_reconcile")["status"] == "fail"
    assert is_blocking(verification) is True


def test_missing_artifact_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    (run / "02_data" / "panel.parquet").unlink()
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    schema = _check(verification, "artifact_schema")
    assert schema["status"] == "fail"
    assert "panel" in schema["detail"]


def test_tampered_artifact_hash_fails(clean_run: Path, clean_verification: dict, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    declared = dict(clean_verification["artifact_hashes"])
    ledger_path = run / "05_backtest" / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    ledger.loc[0, "price"] = float(ledger.loc[0, "price"]) * 1.01
    _write_frame(ledger_path, ledger)
    verification = _verify(run, declared_hashes=declared)
    assert verification["verdict"] == "FAIL"
    hashes = _check(verification, "artifact_hashes")
    assert hashes["status"] == "fail"
    assert "declared" in hashes["detail"]


def test_declared_hash_for_unknown_artifact_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    declared = {"09_verification/does_not_exist.json": hashlib.sha256(b"x").hexdigest()}
    verification = _verify(run, declared_hashes=declared)
    assert _check(verification, "artifact_hashes")["status"] == "fail"


def test_pit_price_date_violation_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    panel_path = run / "02_data" / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    future = panel.iloc[[-1]].copy()
    future["date"] = "2025-01-06"
    panel = pd.concat([panel, future], ignore_index=True)
    _write_frame(panel_path, panel)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "pit_prices")["status"] == "fail"


def test_pit_audit_violation_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    audit_path = run / "02_data" / "pit_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["violations"] = [{"id": "future_fundamental", "detail": "used a notice_date after as_of"}]
    _write_json(audit_path, audit)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "pit_fundamentals")["status"] == "fail"


def test_weight_bound_violation_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    holdings_path = run / "05_backtest" / "holdings.parquet"
    holdings = pd.read_parquet(holdings_path)
    holdings.loc[0, "weight"] = 0.99
    holdings.loc[0, "active_weight"] = 0.99 - float(holdings.loc[0, "benchmark_weight"])
    _write_frame(holdings_path, holdings)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "weights_bounds")["status"] == "fail"


def test_suspended_trade_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    ledger = pd.read_parquet(run / "05_backtest" / "ledger.parquet")
    trade = ledger.iloc[0]
    panel_path = run / "02_data" / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    mask = (panel["date"].astype(str) == str(trade["date"])) & (panel["code"].astype(str) == str(trade["code"]))
    assert mask.any(), "expected the first ledger trade to have a panel row"
    panel.loc[mask, "suspended"] = True
    _write_frame(panel_path, panel)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "forbidden_securities")["status"] == "fail"


def test_tampered_metrics_are_reported_as_discrepancies(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    metrics_path = run / "05_backtest" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["annualized_excess_return"] = float(metrics["metrics"]["annualized_excess_return"]) + 0.10
    _write_json(metrics_path, metrics)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "metrics_recompute")["status"] == "fail"
    discrepancies = verification["discrepancies"]
    assert any(record["field"].endswith("annualized_excess_return") for record in discrepancies)


def test_tampered_snapshot_is_detected(clean_run: Path, tmp_path: Path) -> None:
    snapshot_copy = tmp_path / "snapshot"
    shutil.copytree(DEFAULT_SNAPSHOT_DIR, snapshot_copy)
    prices_path = snapshot_copy / "prices.parquet"
    prices = pd.read_parquet(prices_path)
    prices.loc[0, "close"] = float(prices.loc[0, "close"]) * 1.05
    _write_frame(prices_path, prices)
    run = _copy_run(clean_run, tmp_path / "run_dir")
    verification = _verify(run, snapshot=snapshot_copy)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "artifact_hashes")["status"] == "fail"


def test_wallclock_timestamp_in_numeric_artifact_fails(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    metrics_path = run / "05_backtest" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.setdefault("metadata", {})["generated_at"] = "2024-12-31T00:00:00Z"
    _write_json(metrics_path, metrics)
    verification = _verify(run)
    assert verification["verdict"] == "FAIL"
    assert _check(verification, "determinism_summary")["status"] == "fail"


# ---------------------------------------------------------------------------
# Stage integration
# ---------------------------------------------------------------------------


def test_run_verification_stage_writes_artifacts(clean_run: Path, tmp_path: Path) -> None:
    run = _copy_run(clean_run, tmp_path)
    config = demo_config_from_run(run)
    ctx = RunContext(
        run_id="demo-baseline-2024-12-31",
        as_of=dt.date(2024, 12, 31),
        run_dir=run,
        config=config,
        llm_mode="replay",
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
    )
    output = run_verification(ctx)
    assert output.summary["verdict"] == "PASS"
    assert output.summary["checks_failed"] == 0
    assert set(output.artifacts) == {"verification", "verifier_note"}
    verification_path = Path(output.artifacts["verification"])
    note_path = Path(output.artifacts["verifier_note"])
    if not verification_path.is_absolute():
        from aicwf.contracts import PROJECT_ROOT

        verification_path = PROJECT_ROOT / verification_path
        note_path = PROJECT_ROOT / note_path
    assert verification_path.exists()
    assert note_path.exists()
    written = json.loads(verification_path.read_text(encoding="utf-8"))
    assert written["verdict"] == "PASS"
    assert "PASS" in note_path.read_text(encoding="utf-8")
