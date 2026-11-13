"""Integration layer used by the CLI.

The CLI imports this module lazily. Everything here is glue: configuration
loading, backend construction, the handler registry that maps workflow stage
names to callables, the approval-stage handler, and thin wrappers around the
monitoring / evaluation / rollback / snapshot utilities owned by other modules.

Keeping the glue in one place makes the dependencies between subsystems explicit
and lets the test suite import any subsystem without importing the entire
workflow.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import subprocess
import sys
from pathlib import Path

from aicwf.contracts import (
    PROJECT_ROOT,
    ConfigError,
    RunContext,
    StageFn,
    StageOutput,
    WorkflowError,
    date_str,
)
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import read_json, write_json
from aicwf.util.paths import load_config_bundle, project_path

CONFIG_NAMES = ("workflow", "strategy", "gates", "budgets", "model", "data")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_config_paths(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root is not None else PROJECT_ROOT
    return {name: root / "config" / f"{name}.yaml" for name in CONFIG_NAMES}


def load_config(
    *,
    overrides: dict[str, Path] | None = None,
    root: Path | None = None,
) -> dict:
    """Load and lightly validate the merged configuration bundle."""

    paths = default_config_paths(root)
    if overrides:
        paths.update(overrides)
    missing = [name for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise ConfigError(f"missing config files: {missing}")
    bundle = load_config_bundle(paths)
    if "workflow" not in bundle or "strategy" not in bundle:
        raise ConfigError("workflow.yaml and strategy.yaml are required")
    return bundle


def config_hash(config: dict) -> str:
    sections = {name: config.get(name) for name in ("workflow", "strategy", "gates", "budgets")}
    return hash_json(sections)


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------


def build_llm(
    mode: str,
    config: dict,
    *,
    cache_dir: Path | str | None = None,
    run_id: str | None = None,
):
    """Construct the raw LLM backend (the controller wraps it with a budget guard)."""

    from aicwf.agents.llm import make_llm

    return make_llm(mode=mode, config=config, cache_dir=cache_dir, run_id=run_id)


# ---------------------------------------------------------------------------
# Snapshot resolution
# ---------------------------------------------------------------------------


def resolve_snapshot(as_of: dt.date, config: dict) -> tuple[Path | None, str | None]:
    """Best-effort snapshot resolution used by CLI and doctor checks."""

    from aicwf.intake import resolve_snapshot as _resolve

    ctx = RunContext(
        run_id="resolve",
        as_of=as_of if isinstance(as_of, dt.date) else dt.date.fromisoformat(str(as_of)),
        run_dir=Path("."),
        config=config,
    )
    try:
        path, manifest = _resolve(ctx)
    except WorkflowError:
        return None, None
    return Path(path), manifest.get("snapshot_hash")


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


def build_registry() -> dict[str, StageFn]:
    """Map ``handler:`` names in workflow.yaml to stage callables.

    Imports happen here, not at module import, so the CLI stays light and a
    partially implemented subsystem does not break unrelated commands.
    """

    from aicwf.agents import stages as agent_stages
    from aicwf.backtest import stage as backtest_stage
    from aicwf.data import stage as data_stage
    from aicwf.factors import stage as factor_stage
    from aicwf.intake import run_intake
    from aicwf.portfolio import stage as portfolio_stage
    from aicwf.registry import stage as registry_stage
    from aicwf.verify import stage as verify_stage

    registry: dict[str, StageFn] = {
        "intake": run_intake,
        "hypothesis": agent_stages.run_hypothesis,
        "data_prep": data_stage.run_data_prep,
        "factor_build": factor_stage.run_factor_build,
        "factor_review": agent_stages.run_factor_review,
        "backtest": backtest_stage.run_backtest,
        "robustness": backtest_stage.run_robustness,
        "robustness_review": agent_stages.run_risk_review,
        "portfolio": portfolio_stage.run_portfolio,
        "verification": verify_stage.run_verification,
        "report": agent_stages.run_report,
        "approval": run_approval,
        "registry": registry_stage.run_registry,
    }
    return registry


# ---------------------------------------------------------------------------
# Approval stage handler
# ---------------------------------------------------------------------------


def _approval_action(ctx: RunContext, stage: str) -> str:
    stages = (ctx.config.get("workflow") or {}).get("workflow", {}).get("stages", [])
    for spec in stages:
        if spec.get("name") == stage:
            return spec.get("approval_action") or "publish_report"
    return "publish_report"


def _verified_artifact_hashes(ctx: RunContext) -> dict[str, str]:
    """Content hashes of the package a human is asked to approve."""

    wanted = {
        "report": ("report",),
        "verification": ("verification",),
        "portfolio": ("target_weights", "orders", "risk_report"),
        "backtest": ("metrics",),
        "robustness": ("robustness",),
    }
    hashes: dict[str, str] = {}
    for stage, logical_names in wanted.items():
        output = ctx.stage_outputs.get(stage)
        if output is None:
            continue
        for logical in logical_names:
            relative = output.artifacts.get(logical)
            if not relative:
                continue
            path = Path(relative)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                hashes[f"{stage}:{logical}"] = hash_artifact(path)
    return hashes


def run_approval(ctx: RunContext) -> StageOutput:
    """Approval stage: write the approval artifact once a human decision exists.

    The runner blocks the stage before it executes unless a matching decision is
    recorded; reaching this handler means the gate has been satisfied.
    """

    action = _approval_action(ctx, "approval")
    artifact_hashes = _verified_artifact_hashes(ctx)
    payload = {
        "run_id": ctx.run_id,
        "action": action,
        "decision": "approved",
        "artifact_hashes": artifact_hashes,
        "verification_verdict": (ctx.stage_outputs.get("verification").summary.get("verdict") if "verification" in ctx.stage_outputs else None),
        "information_cutoff": date_str(ctx.as_of),
    }
    try:
        from aicwf.controller import approval as approval_mod

        writer = getattr(approval_mod, "write_approval_artifact", None)
        if writer is not None:
            path = writer(gate="approval", run_id=ctx.run_id, run_dir=ctx.run_dir)
            if path is None:
                path = ctx.artifact_path("approval", "approval.json")
                write_json(path, payload, sort_keys=True)
        else:  # pragma: no cover - compatibility with an older controller
            path = ctx.artifact_path("approval", "approval.json")
            write_json(path, payload, sort_keys=True)
    except ImportError:  # pragma: no cover - controller not available in isolated tests
        path = ctx.artifact_path("approval", "approval.json")
        write_json(path, payload, sort_keys=True)

    path = Path(path)
    if not path.exists():  # pragma: no cover - defensive
        raise WorkflowError("approval handler did not produce approval.json")
    ctx.event("approval.recorded", action=action)
    return StageOutput(
        artifacts={"approval": path.relative_to(PROJECT_ROOT).as_posix()},
        summary={"decision": "approved", "action": action, "artifact_hashes": artifact_hashes},
    )


# ---------------------------------------------------------------------------
# Thin delegations owned by other subsystems
# ---------------------------------------------------------------------------


def run_monitor(*, as_of, llm_mode: str, config: dict, **kwargs):
    from aicwf.monitor.run import run_monitor as _run

    return _run(as_of=as_of, llm_mode=llm_mode, config=config, **kwargs)


def run_eval(*, candidate, baseline, config: dict, **kwargs):
    from aicwf.eval.runner import run_eval as _run

    return _run(candidate=candidate, baseline=baseline, config=config, **kwargs)


def run_rollback(*, to, config: dict, **kwargs):
    from aicwf.registry.rollback import rollback_to as _rollback

    return _rollback(to_version=to, config=config, **kwargs)


def build_snapshot(*, as_of, config: dict, **kwargs) -> int:
    script = project_path("scripts", "build_snapshot.py")
    command = [sys.executable, str(script), "--as-of", date_str(as_of)]
    return subprocess.call(command)


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


def run_doctor(config_paths: dict[str, Path] | None = None) -> int:
    """Check the local environment, configuration, snapshot and model."""

    issues: list[str] = []
    notes: list[str] = []

    print("== environment ==")
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    for module in ("pandas", "numpy", "scipy", "sklearn", "pyarrow", "matplotlib", "yaml", "jsonschema", "requests"):
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "?")
            print(f"  {module}: {version}")
        except Exception as exc:  # pragma: no cover - environment dependent
            issues.append(f"missing dependency {module}: {exc}")
            print(f"  {module}: MISSING")

    print("== configuration ==")
    try:
        config = load_config(overrides=config_paths)
        from aicwf.intake import validate_config_bundle

        warnings = validate_config_bundle(config)
        print(f"  config loaded: {', '.join(sorted(config))}")
        for warning in warnings:
            print(f"  warning: {warning}")
    except Exception as exc:
        issues.append(f"config error: {exc}")
        config = {}
        print(f"  ERROR: {exc}")

    if config:
        print("== data snapshot ==")
        as_of = date_str((config.get("strategy", {}).get("data") or {}).get("end_date") or "2026-11-12")
        snapshot, snapshot_hash = resolve_snapshot(dt.date.fromisoformat(as_of), config)
        if snapshot is None:
            issues.append(f"snapshot for as_of {as_of} is missing or invalid")
            print(f"  ERROR: snapshot for {as_of} not found")
        else:
            try:
                from aicwf.data.snapshot import verify_snapshot

                manifest = verify_snapshot(snapshot, deep=False)
                print(f"  snapshot: {snapshot.relative_to(PROJECT_ROOT).as_posix()}")
                print(f"  snapshot hash: {snapshot_hash}")
                counts = manifest.get("row_counts", {})
                print(f"  row counts: {json.dumps(counts, sort_keys=True)}")
                notes.append(f"deep verification deferred; run build_snapshot --verify for a full hash check")
            except Exception as exc:
                issues.append(f"snapshot verification failed: {exc}")
                print(f"  ERROR: {exc}")

        print("== local model ==")
        model_cfg = (config.get("model", {}) or {}).get("default", {}) or {}
        base_url = model_cfg.get("base_url") or "http://127.0.0.1:11434"
        model = model_cfg.get("model") or "qwen3.5:9b"
        try:
            import urllib.request

            with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [item.get("name") for item in payload.get("models", [])]
            if model in names:
                print(f"  ollama: reachable, model {model} present")
            else:
                print(f"  ollama: reachable; model {model} not in {names}")
                issues.append(f"model {model} is not available in Ollama")
        except Exception as exc:
            print(f"  ollama: not reachable ({exc}); replay mode still works")
            notes.append("Ollama is offline; use --llm replay for deterministic runs")

        cache_dir = project_path((config.get("model", {}) or {}).get("replay", {}).get("cache_dir", "evidence/llm_cache"))
        entries = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        print(f"== replay cache ==\n  {len(entries)} entries in {cache_dir.relative_to(PROJECT_ROOT).as_posix()}")

    print("== run state ==")
    state_dir = project_path("state")
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        probe = state_dir / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print(f"  writable: {state_dir.relative_to(PROJECT_ROOT).as_posix()}")
    except Exception as exc:
        issues.append(f"state directory is not writable: {exc}")
        print(f"  ERROR: {exc}")

    if notes:
        print("== notes ==")
        for note in notes:
            print(f"  - {note}")
    if issues:
        print("== problems ==")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("doctor: OK")
    return 0
