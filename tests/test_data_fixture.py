"""Deterministic mini-fixture generator tests (subprocess, no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aicwf.data.snapshot import FRAME_SPECS, load_snapshot


def run_make_fixture(project_root: Path, out_dir: Path, *extra: str) -> dict:
    script = project_root / "scripts" / "make_fixture.py"
    assert script.exists()
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), *extra],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(f"make_fixture failed ({result.returncode}): {result.stdout}\n{result.stderr}")
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_generator_is_deterministic_and_matches_committed_fixture(project_root: Path, tmp_path: Path):
    first = run_make_fixture(project_root, tmp_path / "mini_a")
    second = run_make_fixture(project_root, tmp_path / "mini_b")
    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert first["files"] == second["files"]
    assert first["content_hashes"] == second["content_hashes"]

    committed_manifest = project_root / "tests" / "fixtures" / "mini_snapshot" / "manifest.json"
    if committed_manifest.exists():
        committed = json.loads(committed_manifest.read_text(encoding="utf-8"))
        assert committed["snapshot_hash"] == first["snapshot_hash"], (
            "committed mini snapshot is stale; rerun python scripts/make_fixture.py"
        )


def test_generated_layout_matches_contract(project_root: Path, tmp_path: Path):
    run_make_fixture(project_root, tmp_path / "mini_layout")
    snapshot = load_snapshot(tmp_path / "mini_layout")
    assert len(snapshot.universe) == 6
    assert snapshot.calendar["date"].nunique() == len(snapshot.calendar)
    for name, spec in FRAME_SPECS.items():
        assert list(snapshot.frames[name].columns) == list(spec.columns), name
    # 36 months ending 2024-12-31
    assert snapshot.as_of.isoformat() == "2024-12-31"
    assert snapshot.prices["date"].min() == "2022-01-04"
    assert snapshot.prices["date"].max() == "2024-12-31"
    # prices never start before the synthetic listing date
    for row in snapshot.universe.itertuples():
        assert snapshot.prices.loc[snapshot.prices["code"] == row.code, "date"].min() >= row.listed_date
    # fundamentals carry the 90-day fallback for the deliberately missing notices
    estimated = snapshot.fundamentals.loc[snapshot.fundamentals["notice_date_estimated"]]
    assert len(estimated) == 3
    assert estimated["notice_date"].notna().all()


def test_manifest_notes_mark_fixture_as_synthetic(project_root: Path, tmp_path: Path):
    manifest = run_make_fixture(project_root, tmp_path / "mini_notes")
    notes = " ".join(manifest["notes"]).lower()
    assert "synthetic" in notes
    assert "no network" in notes
    assert manifest["provenance"]["builder"] == "scripts/make_fixture.py"
