"""Intake stage: validate the run, pin the data snapshot, and record provenance.

This is the only stage that may create the run request. It is deterministic: the
request contains no wall-clock timestamp (those live in the controller event log),
so two runs with the same run id and snapshot produce byte-identical artifacts.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from aicwf.contracts import (
    STAGES,
    ConfigError,
    RunContext,
    StageOutput,
    ValidationError,
    date_str,
)
from aicwf.util.hashing import hash_file, hash_json
from aicwf.util.jsonio import read_json, write_json
from aicwf.util.paths import PROJECT_ROOT, project_path

REQUIRED_SECTIONS = ("workflow", "strategy", "gates", "budgets", "model", "data")
HASHED_SECTIONS = ("workflow", "strategy", "gates", "budgets")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def workflow_section(config: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow mapping from either config shape."""

    section = (config or {}).get("workflow") or {}
    if isinstance(section, dict) and isinstance(section.get("workflow"), dict):
        return section["workflow"]
    return section


def validate_config_bundle(config: dict[str, Any]) -> list[str]:
    """Validate the merged config bundle; return non-fatal warnings."""

    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ConfigError(f"missing config sections: {', '.join(missing)}")

    strategy = config["strategy"] or {}
    workflow = workflow_section(config)
    stage_names = [stage.get("name") for stage in workflow.get("stages", [])]
    _require(
        set(stage_names) == set(STAGES),
        f"workflow stages {sorted(stage_names)} do not match the contract {list(STAGES)}",
    )
    known = set(STAGES)
    for stage in workflow["stages"]:
        for need in stage.get("needs", []):
            _require(need in known, f"stage {stage['name']} depends on unknown stage {need}")
        _require(
            stage["name"] not in stage.get("needs", []),
            f"stage {stage['name']} depends on itself",
        )

    catalog = (strategy.get("factors") or {}).get("catalog") or {}
    _require(bool(catalog), "factor catalog is empty")
    default_weights = (strategy.get("factors") or {}).get("composite", {}).get("default_weights") or {}
    unknown = sorted(set(default_weights) - set(catalog))
    _require(not unknown, f"composite weights reference unknown factors: {unknown}")

    constraints = (strategy.get("backtest") or {}).get("constraints") or {}
    for key in ("max_weight", "max_active_weight", "te_target", "turnover_limit"):
        _require(key in constraints, f"strategy.backtest.constraints.{key} is required")
    _require(
        0 < constraints["te_target"] <= 0.20,
        "te_target must be a small positive annualised number (e.g. 0.05)",
    )
    _require(
        constraints["max_active_weight"] <= constraints["max_weight"],
        "max_active_weight must not exceed max_weight",
    )

    start = strategy.get("backtest", {}).get("start_date")
    end = strategy.get("backtest", {}).get("end_date")
    _require(bool(start and end), "backtest start_date/end_date are required")
    warnings: list[str] = []
    if date_str(start) >= date_str(end):
        raise ConfigError("backtest start_date must be before end_date")

    model_cfg = (config.get("model") or {}).get("default") or {}
    if model_cfg.get("backend") == "ollama":
        warnings.append("LLM backend is local Ollama; runs are reproducible only with the replay cache")
    return warnings


def resolve_snapshot(ctx: RunContext) -> tuple[Path, dict[str, Any]]:
    """Locate the frozen snapshot and verify its manifest and file hashes."""

    if ctx.snapshot_dir is not None:
        candidate = Path(ctx.snapshot_dir).expanduser().resolve()
    else:
        configured = (ctx.config.get("strategy", {}).get("data") or {}).get("snapshot")
        candidate = Path(configured).expanduser() if configured else project_path("data", "snapshots", date_str(ctx.as_of))
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()

    if not candidate.exists():
        fallback = project_path("data", "snapshots", date_str(ctx.as_of))
        if fallback.exists():
            candidate = fallback
        else:
            raise ConfigError(f"data snapshot not found: {candidate}")

    manifest_path = candidate / "manifest.json"
    _require(manifest_path.exists(), f"snapshot manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    _require(
        date_str(manifest.get("as_of", "")) == date_str(ctx.as_of),
        f"snapshot as_of {manifest.get('as_of')} does not match run as_of {date_str(ctx.as_of)}",
    )
    for required in ("prices.parquet", "universe.parquet", "benchmark.parquet", "fundamentals.parquet"):
        _require((candidate / required).exists(), f"snapshot file missing: {required}")
    declared = manifest.get("files") or {}
    for name, digest in declared.items():
        path = candidate / name
        _require(path.exists(), f"manifest lists missing file: {name}")
        actual = hash_file(path)
        _require(actual == digest, f"snapshot file hash mismatch for {name}")
    return candidate, manifest


def _prompt_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory in ("prompts", "skills"):
        root = project_path(directory)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}:
                hashes[f"{directory}/{path.relative_to(root).as_posix()}"] = hash_file(path)
    return hashes


def run_intake(ctx: RunContext) -> StageOutput:
    """Validate configuration and snapshot, then write ``run_request.json``."""

    warnings = validate_config_bundle(ctx.config)
    snapshot, manifest = resolve_snapshot(ctx)
    ctx.snapshot_dir = snapshot

    strategy = ctx.config["strategy"]
    universe = read_universe_summary(snapshot)
    budget_cfg = (ctx.config.get("budgets") or {}).get("run") or {}
    config_hash = hash_json({name: ctx.config.get(name) for name in HASHED_SECTIONS})

    manifest_start, manifest_end = _manifest_price_range(manifest)
    data_start = strategy.get("data", {}).get("start_date") or manifest_start
    data_end = strategy.get("data", {}).get("end_date") or manifest_end
    if not data_start or not data_end:
        raise ConfigError("cannot determine the data window from config or snapshot manifest")

    request = {
        "run_id": ctx.run_id,
        "as_of": date_str(ctx.as_of),
        "snapshot": snapshot.relative_to(PROJECT_ROOT).as_posix(),
        "snapshot_hash": manifest.get("snapshot_hash"),
        "config_hash": config_hash,
        "workflow_version": workflow_section(ctx.config).get("version"),
        "llm_mode": ctx.llm_mode,
        "seed": ctx.seed,
        "data_window": {"start": date_str(data_start), "end": date_str(data_end)},
        "benchmark": strategy.get("backtest", {}).get("benchmark"),
        "universe": universe,
        "budget": {
            "max_wall_seconds": budget_cfg.get("max_wall_seconds"),
            "max_llm_calls": budget_cfg.get("max_llm_calls"),
            "max_tokens_total": budget_cfg.get("max_tokens_total"),
            "max_cost_usd": budget_cfg.get("max_cost_usd"),
        },
        "prompt_hashes": _prompt_hashes(),
        "warnings": warnings,
        "information_cutoff": date_str(ctx.as_of),
    }

    out = ctx.artifact_path("intake", "run_request.json")
    write_json(out, request, sort_keys=True)
    summary = {
        "as_of": date_str(ctx.as_of),
        "snapshot_hash": manifest.get("snapshot_hash"),
        "config_hash": config_hash,
        "universe_size": universe.get("size"),
        "warnings": warnings,
    }
    ctx.event("intake.completed", snapshot=str(snapshot), universe_size=universe.get("size"))
    return StageOutput(
        artifacts={"run_request": out.relative_to(PROJECT_ROOT).as_posix()},
        summary=summary,
    )


def _manifest_price_range(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read the price date range from either manifest shape.

    The snapshot builder records ``date_ranges.prices`` as ``[min, max]``; earlier
    drafts used ``{"min": ..., "max": ...}``. Accept both.
    """

    ranges = manifest.get("date_ranges") or {}
    prices = ranges.get("prices")
    if isinstance(prices, dict):
        return prices.get("min"), prices.get("max")
    if isinstance(prices, (list, tuple)) and len(prices) >= 2:
        return str(prices[0]), str(prices[1])
    return None, None


def read_universe_summary(snapshot: Path) -> dict[str, Any]:
    """Small digest of the universe file (size, industry count, weight sum)."""

    import pandas as pd

    path = snapshot / "universe.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    industries = frame["industry"].dropna().nunique() if "industry" in frame else None
    weight_sum = float(frame["weight"].sum()) if "weight" in frame else None
    return {
        "size": int(len(frame)),
        "industries": int(industries) if industries is not None else None,
        "weight_sum": round(weight_sum, 10) if weight_sum is not None else None,
    }


def expected_run_dir(run_id: str) -> Path:
    return project_path("outputs", "runs", run_id)
