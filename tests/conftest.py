"""Shared pytest fixtures.

Subsystem test files may add their own local fixtures, but must not modify this file
without the Team Lead; keep shared fixtures here small and dependency-free.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def mini_snapshot_dir(project_root: Path) -> Path:
    path = project_root / "tests" / "fixtures" / "mini_snapshot"
    if not path.exists():
        pytest.skip("mini snapshot fixture has not been generated yet")
    return path


@pytest.fixture
def run_context(tmp_path: Path):
    from aicwf.contracts import RunContext

    return RunContext(
        run_id="test-run",
        as_of=dt.date(2024, 12, 31),
        run_dir=tmp_path / "run",
        config={},
        llm_mode="null",
    )
