"""Offline tests for monthly monitoring: alert thresholds and deterministic output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from aicwf.eval.fixtures import DEFAULT_SNAPSHOT_DIR, build_demo_run, demo_config_from_run
from aicwf.monitor.drift import data_freshness, detect_new_data, exposure_drift
from aicwf.monitor.run import build_alerts, monitoring_thresholds, run_monitor
from aicwf.util.jsonio import read_json, write_json

AS_OF = "2024-12-31"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("monitor")
    return build_demo_run(
        root / "base", variant="baseline", with_verification=False, replay_evidence=False
    )


@pytest.fixture(scope="module")
def demo_config(demo_run: Path) -> dict:
    return demo_config_from_run(demo_run)


def _thresholds(overrides: dict | None = None) -> dict:
    thresholds = {
        "te_alert": 0.08,
        "te_stop_multiple": 2.0,
        "excess_drawdown_alert": 0.10,
        "data_staleness_days": 3,
        "min_coverage": 0.90,
        "ic_decay_alert": -0.05,
        "industry_active_limit": 0.02,
        "te_stop_level": 0.16,
        "ic_decay_absolute_alert": 0.05,
    }
    if overrides:
        thresholds.update(overrides)
        if "te_alert" in overrides and "te_stop_level" not in overrides:
            thresholds["te_stop_level"] = thresholds["te_alert"] * thresholds["te_stop_multiple"]
        if "ic_decay_alert" in overrides and "ic_decay_absolute_alert" not in overrides:
            thresholds["ic_decay_absolute_alert"] = abs(thresholds["ic_decay_alert"])
    return thresholds


def _performance(**overrides) -> dict:
    payload = {
        "realized_te": 0.02,
        "active_drawdown": -0.01,
        "tracking_difference": 0.005,
    }
    payload.update(overrides)
    return payload


def _drift_report(**overrides) -> dict:
    payload = {
        "factor_ic": {"recent_mean": 0.05, "baseline_mean": 0.05, "decay": 0.0, "n_periods": 12},
        "data_freshness": {
            "latest_price_date": AS_OF,
            "trading_days_stale": 0,
            "staleness_limit_days": 3,
            "stale": False,
            "severity": "ok",
        },
        "coverage": {"price_coverage": 1.0, "factor_coverage": 1.0},
        "exposure_drift": {"breach": False, "max_abs_industry_active": 0.01, "industry_limit": 0.02},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Alert threshold matrix
# ---------------------------------------------------------------------------


def test_alerts_tracking_error_thresholds() -> None:
    thresholds = _thresholds()
    assert build_alerts(performance=_performance(), drift_report=_drift_report(), thresholds=thresholds, previous_run_missing=False) == []
    warn = build_alerts(
        performance=_performance(realized_te=0.10), drift_report=_drift_report(), thresholds=thresholds, previous_run_missing=False
    )
    assert [alert["id"] for alert in warn] == ["tracking_error_alert"]
    assert warn[0]["severity"] == "warn"
    critical = build_alerts(
        performance=_performance(realized_te=0.20), drift_report=_drift_report(), thresholds=thresholds, previous_run_missing=False
    )
    assert critical[0]["id"] == "tracking_error_stop"
    assert critical[0]["severity"] == "critical"


def test_alerts_drawdown_freshness_coverage_ic_exposure() -> None:
    thresholds = _thresholds()
    alerts = build_alerts(
        performance=_performance(active_drawdown=-0.15),
        drift_report=_drift_report(
            data_freshness={
                "latest_price_date": "2024-12-20",
                "trading_days_stale": 8,
                "staleness_limit_days": 3,
                "stale": True,
                "severity": "warn",
            },
            coverage={"price_coverage": 0.5, "factor_coverage": 1.0},
            factor_ic={"recent_mean": -0.10, "baseline_mean": 0.10, "decay": 0.20, "n_periods": 12},
            exposure_drift={"breach": True, "date": AS_OF, "max_abs_industry_active": 0.05, "industry_limit": 0.02},
        ),
        thresholds=thresholds,
        previous_run_missing=True,
    )
    ids = {alert["id"] for alert in alerts}
    assert {"excess_drawdown_alert", "data_stale", "coverage_low", "factor_ic_negative", "factor_ic_decay", "exposure_drift", "no_previous_run"} <= ids


def test_alerts_data_missing_is_critical() -> None:
    alerts = build_alerts(
        performance=_performance(),
        drift_report=_drift_report(
            data_freshness={
                "latest_price_date": None,
                "trading_days_stale": None,
                "staleness_limit_days": 3,
                "stale": True,
                "severity": "critical",
            }
        ),
        thresholds=_thresholds(),
        previous_run_missing=False,
    )
    assert alerts[0]["id"] == "data_missing"
    assert alerts[0]["severity"] == "critical"


def test_thresholds_come_from_strategy_config(demo_config: dict) -> None:
    thresholds = monitoring_thresholds(demo_config)
    strategy_monitoring = demo_config["strategy"]["monitoring"]
    assert thresholds["te_alert"] == pytest.approx(strategy_monitoring["te_alert"])
    assert thresholds["ic_decay_alert"] == pytest.approx(strategy_monitoring["ic_decay_alert"])
    assert thresholds["industry_active_limit"] == pytest.approx(
        demo_config["strategy"]["backtest"]["constraints"]["industry_active_limit"]
    )


def test_detect_new_data_trigger() -> None:
    assert detect_new_data("aaa", "aaa")["type"] == "scheduled"
    assert detect_new_data("aaa", "bbb")["type"] == "new_data"
    assert detect_new_data("aaa", None)["type"] == "scheduled"  # first run has no previous snapshot


def test_exposure_drift_reports_limit_breach() -> None:
    holdings = pd.DataFrame(
        {
            "date": ["2024-12-31", "2024-12-31"],
            "code": ["A", "B"],
            "industry": ["x", "y"],
            "active_weight": [0.05, -0.01],
            "weight": [0.2, 0.2],
        }
    )
    report = exposure_drift(holdings, as_of=AS_OF, industry_limit=0.02)
    assert report["breach"] is True
    assert report["max_abs_industry_active"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Full monitor runs
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, demo_run: Path, config: dict, **kwargs) -> dict:
    kwargs.setdefault("snapshot_dir", DEFAULT_SNAPSHOT_DIR)
    kwargs.setdefault("out_root", tmp_path / "monitor")
    kwargs.setdefault("as_of", AS_OF)
    return run_monitor(run_dir=demo_run, config=config, **kwargs)


def test_run_monitor_writes_contract_artifacts(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    result = _run(tmp_path, demo_run, demo_config)
    out = Path(result["out_dir"])
    for filename in (
        "monitor_request.json",
        "performance_update.json",
        "drift_report.json",
        "monitoring.json",
        "monitoring_report.md",
    ):
        assert (out / filename).exists(), f"missing {filename}"
    request = read_json(out / "monitor_request.json")
    assert request["monitor_id"] == f"monitor-{AS_OF}"
    assert request["as_of"] == AS_OF
    assert request["run_id"] == "demo-baseline-2024-12-31"
    assert request["snapshot_hash"]
    monitoring = read_json(out / "monitoring.json")
    assert monitoring["verdict"] in {"ok", "alert", "breach"}
    assert isinstance(monitoring["alerts"], list)
    assert monitoring["thresholds"]["te_alert"] == pytest.approx(demo_config["strategy"]["monitoring"]["te_alert"])
    drift = read_json(out / "drift_report.json")
    assert drift["factor_ic"]["n_periods"] >= 10
    assert drift["coverage"]["price_coverage"] == pytest.approx(1.0)
    assert drift["coverage"]["universe_size"] == 6
    performance = read_json(out / "performance_update.json")["performance"]
    assert performance["method"] in {"mark_to_market", "equity_curve"}
    assert performance["realized_te"] is not None
    report = (out / "monitoring_report.md").read_text(encoding="utf-8")
    assert monitoring["verdict"].upper() in report
    assert "Factor RankIC drift" in report


def test_run_monitor_output_is_deterministic(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    first = _run(tmp_path, demo_run, demo_config)
    out = Path(first["out_dir"])
    first_bytes = {
        filename: (out / filename).read_bytes()
        for filename in ("monitor_request.json", "performance_update.json", "drift_report.json", "monitoring.json")
    }
    first_report = (out / "monitoring_report.md").read_text(encoding="utf-8")
    second = _run(tmp_path, demo_run, demo_config)
    assert second["monitor_id"] == first["monitor_id"]
    for filename, content in first_bytes.items():
        assert (out / filename).read_bytes() == content, f"{filename} is not deterministic"
    assert (out / "monitoring_report.md").read_text(encoding="utf-8") == first_report


def test_run_monitor_alerts_on_high_tracking_error(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    config = json.loads(json.dumps(demo_config))
    config["strategy"]["monitoring"]["te_alert"] = 0.01
    config["strategy"]["monitoring"]["te_stop_multiple"] = 2.0
    result = _run(tmp_path, demo_run, config, monitor_id="high-te")
    assert result["verdict"] == "breach"
    ids = {alert["id"] for alert in result["alerts"]}
    assert "tracking_error_stop" in ids
    assert result["thresholds"]["te_stop_level"] == pytest.approx(0.02)


def test_run_monitor_ok_with_relaxed_thresholds(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    config = json.loads(json.dumps(demo_config))
    config["strategy"]["monitoring"] = {
        "te_alert": 10.0,
        "te_stop_multiple": 2.0,
        "excess_drawdown_alert": 10.0,
        "data_staleness_days": 10_000,
        "min_coverage": 0.0,
        "ic_decay_alert": -100.0,
    }
    config["strategy"]["backtest"]["constraints"]["industry_active_limit"] = 10.0
    result = _run(tmp_path, demo_run, config, monitor_id="relaxed")
    assert result["verdict"] == "ok"
    assert result["alerts"] == []


def test_run_monitor_detects_stale_data(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    stale_snapshot = tmp_path / "stale_snapshot"
    shutil.copytree(DEFAULT_SNAPSHOT_DIR, stale_snapshot)
    cutoff = "2024-11-01"
    for filename in ("prices.parquet", "benchmark.parquet", "calendar.parquet"):
        path = stale_snapshot / filename
        frame = pd.read_parquet(path)
        frame = frame[frame["date"].astype(str) <= cutoff].reset_index(drop=True)
        frame.to_parquet(path, index=False)
    result = _run(tmp_path, demo_run, demo_config, snapshot_dir=stale_snapshot, monitor_id="stale")
    ids = {alert["id"] for alert in result["alerts"]}
    assert "data_stale" in ids, result["alerts"]
    drift = read_json(Path(result["out_dir"]) / "drift_report.json")
    assert drift["data_freshness"]["stale"] is True
    assert drift["data_freshness"]["trading_days_stale"] > demo_config["strategy"]["monitoring"]["data_staleness_days"]


def test_run_monitor_new_data_trigger(tmp_path: Path, demo_run: Path, demo_config: dict) -> None:
    first = _run(tmp_path, demo_run, demo_config, monitor_id="first")
    assert first["trigger"]["type"] == "scheduled"
    changed_snapshot = tmp_path / "changed_snapshot"
    shutil.copytree(DEFAULT_SNAPSHOT_DIR, changed_snapshot)
    manifest_path = changed_snapshot / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["snapshot_hash"] = "0" * 64
    write_json(manifest_path, manifest)
    second = _run(
        tmp_path,
        demo_run,
        demo_config,
        snapshot_dir=changed_snapshot,
        monitor_id="second",
        previous_monitor_dir=first["out_dir"],
    )
    assert second["trigger"]["type"] == "new_data"
    assert second["trigger"]["new_data"] is True
    assert second["previous_snapshot_hash"] == first["snapshot_hash"]


def test_run_monitor_offline_without_a_run(tmp_path: Path, demo_config: dict) -> None:
    result = run_monitor(
        AS_OF,
        llm_mode="null",
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
        out_root=tmp_path / "monitor",
        runs_root=tmp_path / "no_runs",
        config=demo_config,
        monitor_id="no-run",
    )
    ids = {alert["id"] for alert in result["alerts"]}
    assert "no_previous_run" in ids
    assert result["verdict"] == "alert"
    assert Path(result["out_dir"]).joinpath("monitoring.json").exists()


def test_data_freshness_helper() -> None:
    freshness = data_freshness(
        as_of=AS_OF,
        snapshot_prices=pd.DataFrame({"date": ["2024-12-20"], "code": ["A"], "close": [10.0]}),
        snapshot_manifest={"as_of": AS_OF},
        staleness_limit_days=3,
    )
    assert freshness["stale"] is True
    assert freshness["trading_days_stale"] > 3
    fresh = data_freshness(
        as_of=AS_OF,
        snapshot_prices=pd.DataFrame({"date": [AS_OF], "code": ["A"], "close": [10.0]}),
        snapshot_manifest={"as_of": AS_OF},
        staleness_limit_days=3,
    )
    assert fresh["stale"] is False
