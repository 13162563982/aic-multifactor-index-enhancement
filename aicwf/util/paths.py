"""Filesystem helpers for project, snapshot and run locations."""

from __future__ import annotations

from pathlib import Path

from aicwf.contracts import PROJECT_ROOT, parse_date


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def snapshot_root() -> Path:
    return project_path("data", "snapshots")


def snapshot_dir(as_of: str | Path) -> Path:
    return snapshot_root() / parse_date(str(as_of)).isoformat()


def runs_root() -> Path:
    return project_path("outputs", "runs")


def monitor_root() -> Path:
    return project_path("outputs", "monitor")


def evidence_root() -> Path:
    return project_path("evidence")


def config_dir() -> Path:
    return project_path("config")


def load_config_bundle(config_paths: dict[str, Path] | None = None) -> dict:
    """Load every YAML config file into one nested dict keyed by file stem."""

    from aicwf.util.jsonio import read_yaml

    bundle: dict = {}
    paths = config_paths or {
        name: project_path("config", f"{name}.yaml")
        for name in ("workflow", "strategy", "gates", "budgets", "model", "data")
    }
    for name, path in paths.items():
        path = Path(path)
        if path.exists():
            bundle[name] = read_yaml(path)
    return bundle
