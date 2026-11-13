"""Deterministic-replay evidence for a real pipeline run.

The eval hard gate ``deterministic_replay`` needs proof that a run's numeric
artifacts can be reproduced from its own inputs.  Agent markdown is inherently
model-dependent, so this helper replays the **deterministic numeric stages** —
``data_prep -> factor_build -> backtest -> portfolio`` — from the identical frozen
snapshot and configuration into a throwaway run directory and compares the
content hashes of every numeric artifact with the original run.  The result is
written to ``<run_dir>/replay_evidence.json`` (the format the eval runner reads),
including the replay scope so the claim cannot be overstated.

Used by ``aicwf.eval.runner`` (hard gate) and by the change-control evidence
pipeline.  Robustness is intentionally excluded from the replay scope because
re-running its ~15 engine variants takes minutes and is not needed to reproduce
the headline metrics that change control compares.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import PROJECT_ROOT, STAGE_DIRS, RunContext, StageOutput, date_str, relpath
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import ensure_dir, read_json, write_json
from aicwf.util.paths import load_config_bundle

REPLAY_SCHEMA_VERSION = "1.0.0"

#: deterministic stage -> numeric artifacts compared by the replay check.
REPLAY_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "data_prep": ("panel.parquet", "benchmark.parquet", "data_quality.json", "pit_audit.json"),
    "factor_build": ("factor_panel.parquet", "factor_report.json"),
    "backtest": ("ledger.parquet", "holdings.parquet", "metrics.json", "equity_curve.csv"),
    "portfolio": ("target_weights.parquet", "orders.csv", "risk_report.json"),
}


def _load_config_bundle_like_cli(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load the config exactly like the CLI does (unwrap section-root YAML files).

    The intake/metrics ``config_hash`` is computed over the CLI-normalised bundle,
    so replay evidence must use the same projection or ``metrics.json`` will differ
    on its ``config_hash`` field alone.
    """

    from aicwf.util.jsonio import read_yaml

    directory = Path(config_dir) if config_dir is not None else PROJECT_ROOT / "config"
    bundle: dict[str, Any] = {}
    for name in ("workflow", "strategy", "gates", "budgets", "model", "data"):
        path = directory / f"{name}.yaml"
        if not path.is_file():
            continue
        content = read_yaml(path)
        if isinstance(content, Mapping) and isinstance(content.get(name), Mapping):
            content = content[name]
        bundle[name] = content
    return bundle


def _resolve_snapshot(run_dir: Path, snapshot_dir: Path | str | None) -> Path | None:
    if snapshot_dir:
        return Path(snapshot_dir)
    request = read_json(run_dir / "00_intake" / "run_request.json")
    declared = request.get("snapshot")
    if declared:
        candidate = Path(str(declared))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
    candidate = PROJECT_ROOT / "data" / "snapshots" / date_str(request.get("as_of"))
    return candidate if candidate.exists() else None


def _replayed_artifact(staging: Path, stage: str, filename: str) -> Path:
    return staging / STAGE_DIRS[stage] / filename


def build_replay_evidence(
    run_dir: Path | str,
    *,
    config: Mapping[str, Any] | None = None,
    snapshot_dir: Path | str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Replay the deterministic numeric stages and compare artifact content hashes."""

    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    request = read_json(run / "00_intake" / "run_request.json")
    as_of = date_str(request.get("as_of"))
    config_bundle = dict(config) if config is not None else _load_config_bundle_like_cli()
    snapshot = _resolve_snapshot(run, snapshot_dir)
    staging = PROJECT_ROOT / "outputs" / "runs" / f".replay-{run.name}-{uuid.uuid4().hex[:8]}"
    ensure_dir(staging)
    try:
        ctx = RunContext(
            run_id=f"{run.name}-replay",
            as_of=request.get("as_of"),
            run_dir=staging,
            config=config_bundle,
            llm_mode="null",
            snapshot_dir=snapshot,
        )
        # Agent/LLM-independent inputs the numeric stages consume.
        ctx.register(
            "hypothesis",
            StageOutput(
                artifacts={"hypothesis": str(run / "01_hypothesis" / "hypothesis.json")},
                summary={},
            ),
        )
        from aicwf.backtest.stage import run_backtest
        from aicwf.data.stage import run_data_prep
        from aicwf.factors.stage import run_factor_build
        from aicwf.portfolio.stage import run_portfolio

        functions = {
            "data_prep": run_data_prep,
            "factor_build": run_factor_build,
            "backtest": run_backtest,
            "portfolio": run_portfolio,
        }
        for stage, function in functions.items():
            output = function(ctx)
            ctx.register(stage, output)

        artifact_hashes: dict[str, str] = {}
        replay_hashes: dict[str, str] = {}
        missing: list[str] = []
        for stage, filenames in REPLAY_ARTIFACTS.items():
            for filename in filenames:
                original = run / STAGE_DIRS[stage] / filename
                replayed = _replayed_artifact(staging, stage, filename)
                name = f"{STAGE_DIRS[stage]}/{filename}"
                if not original.exists() or not replayed.exists():
                    missing.append(name)
                    continue
                artifact_hashes[name] = hash_artifact(original)
                replay_hashes[name] = hash_artifact(replayed)
        mismatched = sorted(
            name
            for name in artifact_hashes
            if artifact_hashes[name] != replay_hashes.get(name)
        )
        identical = bool(artifact_hashes) and not mismatched
        evidence: dict[str, Any] = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "method": (
                "independent numeric-stage replay (data_prep -> factor_build -> backtest -> portfolio) "
                "from the identical frozen snapshot and configuration; agent-generated markdown excluded"
            ),
            "run_id": request.get("run_id") or run.name,
            "as_of": as_of,
            "snapshot_hash": request.get("snapshot_hash"),
            "replay_scope": {stage: list(filenames) for stage, filenames in REPLAY_ARTIFACTS.items()},
            "excluded_from_scope": ["robustness", "agent-generated markdown (rationale/reviews/report)"],
            "config_hash_of_replay": hash_json(
                {key: config_bundle.get(key) for key in ("workflow", "strategy", "gates", "budgets")}
            ),
            "n_artifacts": len(artifact_hashes),
            "artifact_hashes": artifact_hashes,
            "replay_artifact_hashes": replay_hashes,
            "identical": identical,
        }
        if mismatched:
            evidence["mismatched_artifacts"] = mismatched
        if missing:
            evidence["missing_artifacts"] = missing
        if write:
            write_json(run / "replay_evidence.json", evidence, indent=2, sort_keys=True)
        return evidence
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_replay_evidence(run_dir: Path | str) -> dict[str, Any] | None:
    path = Path(run_dir) / "replay_evidence.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return dict(payload) if isinstance(payload, dict) else None
