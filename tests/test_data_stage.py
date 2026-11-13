"""Data-prep stage tests on the deterministic mini snapshot (offline, fast)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from aicwf.contracts import DataQualityError, RunContext
from aicwf.data.stage import PANEL_COLUMNS, build_panel, resolve_snapshot_dir, run_data_prep

AS_OF = dt.date(2024, 12, 31)


def make_context(tmp_path: Path, mini_snapshot_dir: Path, *, as_of: dt.date = AS_OF, config: dict | None = None) -> RunContext:
    return RunContext(
        run_id="data-test-run",
        as_of=as_of,
        run_dir=tmp_path / "run",
        config=config if config is not None else {},
        llm_mode="null",
        snapshot_dir=mini_snapshot_dir,
    )


@pytest.fixture
def stage_output(tmp_path: Path, mini_snapshot_dir: Path):
    ctx = make_context(tmp_path, mini_snapshot_dir)
    output = run_data_prep(ctx)
    return ctx, output


def test_stage_writes_all_artifacts(stage_output):
    ctx, output = stage_output
    stage_dir = ctx.run_dir / "02_data"
    for name in ("panel.parquet", "benchmark.parquet", "data_quality.json", "pit_audit.json"):
        assert (stage_dir / name).exists(), name
    assert set(output.artifacts) == {"panel", "benchmark", "data_quality", "pit_audit"}
    for relative in output.artifacts.values():
        assert Path(relative).exists()
    assert ctx.stage_outputs["data_prep"] is output


def test_panel_schema_and_pit_cutoff(stage_output):
    ctx, output = stage_output
    panel = pd.read_parquet(ctx.run_dir / "02_data" / "panel.parquet")
    assert list(panel.columns) == list(PANEL_COLUMNS)
    # ISO date strings (contract) and sorted by (date, code)
    assert panel["date"].map(type).eq(str).all()
    assert panel["date"].max() <= AS_OF.isoformat()
    ordered = panel.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    assert panel[["date", "code"]].equals(ordered[["date", "code"]])
    assert not panel.duplicated(subset=["date", "code"]).any()
    # universe/summary consistency
    assert panel["code"].nunique() == output.summary["universe_size"] == 6
    assert output.summary["rows"] == len(panel)
    assert output.summary["max_date"] == AS_OF.isoformat()
    # contract dtypes
    assert panel["suspended"].dtype == bool
    assert panel["in_universe"].dtype == bool
    assert pd.api.types.is_numeric_dtype(panel["ret"])
    assert pd.api.types.is_numeric_dtype(panel["log_size"])
    # first return of each code is NaN (no prior close inside the frozen window)
    first_rows = panel.sort_values(["code", "date"], kind="mergesort").groupby("code").head(1)
    assert first_rows["ret"].isna().all()


def test_suspended_days_are_flagged_and_flat(stage_output):
    ctx, _ = stage_output
    panel = pd.read_parquet(ctx.run_dir / "02_data" / "panel.parquet")
    suspended = panel.loc[panel["suspended"]]
    assert len(suspended) == 4  # the fixture deliberately drops four bars
    assert (suspended["ret"] == 0).all()
    assert (suspended["volume"] == 0).all()
    assert (suspended["amount"] == 0).all()
    assert suspended["close"].notna().all()  # previous close forward-filled


def test_benchmark_output(stage_output):
    ctx, _ = stage_output
    benchmark = pd.read_parquet(ctx.run_dir / "02_data" / "benchmark.parquet")
    assert list(benchmark.columns) == ["date", "close", "ret"]
    assert benchmark["date"].max() <= AS_OF.isoformat()
    assert benchmark["date"].is_monotonic_increasing
    assert benchmark["ret"].iloc[0] != benchmark["ret"].iloc[0]  # NaN first return
    assert benchmark["ret"].iloc[1:].notna().all()


def test_data_quality_artifact(stage_output, mini_snapshot_dir: Path):
    ctx, output = stage_output
    quality = json.loads((ctx.run_dir / "02_data" / "data_quality.json").read_text(encoding="utf-8"))
    for key in ("snapshot_hash", "rows", "universe_size", "coverage", "checks", "warnings"):
        assert key in quality
    manifest = json.loads((mini_snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert quality["snapshot_hash"] == manifest["snapshot_hash"]
    assert len(quality["checks"]) >= 8
    statuses = {check["id"]: check["status"] for check in quality["checks"]}
    assert all(status == "pass" for status in statuses.values()), statuses
    assert quality["rows"] == len(pd.read_parquet(ctx.run_dir / "02_data" / "panel.parquet"))
    assert quality["universe_size"] == 6
    assert quality["coverage"] == pytest.approx(1.0)
    assert quality["max_date"] == AS_OF.isoformat()


def test_pit_audit_artifact(stage_output):
    ctx, _ = stage_output
    audit = json.loads((ctx.run_dir / "02_data" / "pit_audit.json").read_text(encoding="utf-8"))
    for key in ("as_of", "violations", "max_panel_date", "fundamentals_available_as_of", "checks"):
        assert key in audit
    assert audit["as_of"] == AS_OF.isoformat()
    assert audit["violations"] == []
    assert audit["max_panel_date"] <= AS_OF.isoformat()
    assert audit["fundamentals_available_as_of"] == 66  # 72 rows minus six 2024 annual reports
    assert audit["fundamentals_estimated_rows"] >= 1
    assert all(check["status"] in {"pass", "warn"} for check in audit["checks"])


def test_stage_logs_through_context_events(tmp_path: Path, mini_snapshot_dir: Path):
    events: list[str] = []

    class Recorder:
        def emit(self, kind, level="info", **payload):
            events.append(kind)

    ctx = make_context(tmp_path, mini_snapshot_dir)
    ctx.events = Recorder()
    run_data_prep(ctx)
    assert "data_prep_snapshot" in events
    assert "data_prep_completed" in events


def test_config_based_snapshot_resolution(tmp_path: Path, mini_snapshot_dir: Path, project_root: Path):
    relative = mini_snapshot_dir.relative_to(project_root).as_posix()
    ctx = RunContext(
        run_id="resolve-test",
        as_of=AS_OF,
        run_dir=tmp_path / "run",
        config={"strategy": {"data": {"snapshot": relative}}},
        llm_mode="null",
    )
    assert resolve_snapshot_dir(ctx) == mini_snapshot_dir
    output = run_data_prep(ctx)
    assert output.summary["rows"] > 0


def test_snapshot_later_than_run_as_of_is_rejected(tmp_path: Path, mini_snapshot_dir: Path):
    ctx = make_context(tmp_path, mini_snapshot_dir, as_of=dt.date(2024, 6, 28))
    with pytest.raises(DataQualityError, match="later than the run decision date"):
        run_data_prep(ctx)


def test_missing_snapshot_raises_data_quality_error(tmp_path: Path):
    ctx = RunContext(
        run_id="missing-test",
        as_of=AS_OF,
        run_dir=tmp_path / "run",
        config={},
        llm_mode="null",
        snapshot_dir=tmp_path / "does-not-exist",
    )
    with pytest.raises(DataQualityError, match="not found"):
        run_data_prep(ctx)


def test_build_panel_is_deterministic(mini_snapshot_dir: Path):
    from aicwf.data.snapshot import load_snapshot

    snapshot = load_snapshot(mini_snapshot_dir)
    first = build_panel(snapshot, AS_OF)
    second = build_panel(snapshot, AS_OF)
    pd.testing.assert_frame_equal(first, second)
