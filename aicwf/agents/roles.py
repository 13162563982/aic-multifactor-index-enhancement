"""Role registry: prompt, skill and schema assets for every agent role.

The prompts in ``prompts/`` and the reusable skills in ``skills/`` are plain Markdown on
disk so that they can be reviewed, diffed, hashed and versioned independently of the
code. This module maps a role name to its assets, applies the per-stage overrides from
``config/workflow.yaml`` (prompt file, skill file, schema, ``max_tokens``), and resolves
the LLM client for a :class:`aicwf.contracts.RunContext`.

The prompt and skill directories may be overridden through ``config.model.prompts`` or
``AICWF_PROMPT_DIR`` / ``AICWF_SKILL_DIR``; the evaluation harness uses that to run a
candidate prompt set against a fixed baseline without touching the working tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from aicwf.agents.llm import make_llm
from aicwf.agents.validate import config_section, load_schema, schema_path
from aicwf.contracts import (
    LLMClient,
    PROJECT_ROOT,
    RunContext,
    WorkflowError,
)
from aicwf.util.hashing import hash_text
from aicwf.util.jsonio import read_text

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

AGENT_LOOP_DEFAULTS = {
    "max_tool_rounds": 3,
    "max_retries_invalid_json": 2,
    "repair_temperature": 0.0,
    "max_read_chars": 12_000,
}


@dataclass(frozen=True)
class RoleSpec:
    """Static (default) contract for one agent role."""

    role: str
    prompt_file: str
    skill_file: str
    schema: str | None
    output_kind: str
    max_tokens: int
    min_cited_numbers: int = 0
    min_chars: int = 200
    required_phrases: tuple[str, ...] = ()


ROLE_SPECS: dict[str, RoleSpec] = {
    "researcher": RoleSpec(
        role="researcher",
        prompt_file="researcher.md",
        skill_file="hypothesis_spec/SKILL.md",
        schema="hypothesis",
        output_kind="json",
        max_tokens=1800,
        min_chars=0,
    ),
    "factor_analyst": RoleSpec(
        role="factor_analyst",
        prompt_file="factor_analyst.md",
        skill_file="factor_diagnostics/SKILL.md",
        schema="factor_flags",
        output_kind="json",
        max_tokens=1400,
        min_chars=0,
    ),
    "risk_critic": RoleSpec(
        role="risk_critic",
        prompt_file="risk_critic.md",
        skill_file="risk_review/SKILL.md",
        schema=None,
        output_kind="markdown",
        max_tokens=1600,
        min_cited_numbers=3,
        min_chars=500,
        required_phrases=(
            "Verdict on robustness",
            "What would change our mind",
        ),
    ),
    "report_writer": RoleSpec(
        role="report_writer",
        prompt_file="report_writer.md",
        skill_file="report_template/SKILL.md",
        schema=None,
        output_kind="markdown",
        max_tokens=2200,
        min_cited_numbers=5,
        min_chars=900,
        required_phrases=(
            "Executive summary",
            "Independent verification",
            "Limitations and information cutoff",
        ),
    ),
    "verifier": RoleSpec(
        role="verifier",
        prompt_file="verifier.md",
        skill_file="pit_and_cutoff/SKILL.md",
        schema=None,
        output_kind="markdown",
        max_tokens=1200,
        min_cited_numbers=2,
        min_chars=400,
        required_phrases=(
            "Verdict",
            "Checks performed",
            "Discrepancies",
            "Information cutoff",
        ),
    ),
}


@dataclass
class RoleAssets:
    """Everything needed to instantiate an :class:`AgentRuntime` for one role."""

    spec: RoleSpec
    prompt_text: str
    skill_text: str
    schema: dict[str, Any] | None
    prompt_path: Path
    skill_path: Path

    @property
    def role(self) -> str:
        return self.spec.role


# ---------------------------------------------------------------------------
# Directory resolution (supports evaluation-time prompt overrides)
# ---------------------------------------------------------------------------


def prompts_dir(config: Mapping[str, Any] | None = None) -> Path:
    override = os.environ.get("AICWF_PROMPT_DIR") or _prompt_setting(config, "dir")
    if override:
        path = Path(override)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROMPTS_DIR


def skills_dir(config: Mapping[str, Any] | None = None) -> Path:
    override = os.environ.get("AICWF_SKILL_DIR") or _prompt_setting(config, "skill_dir")
    if override:
        path = Path(override)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return SKILLS_DIR


def _prompt_setting(config: Mapping[str, Any] | None, key: str) -> str | None:
    model_cfg = config_section(config, "model")
    prompts_cfg = model_cfg.get("prompts")
    if isinstance(prompts_cfg, Mapping) and prompts_cfg.get(key):
        return str(prompts_cfg[key])
    return None


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


def _workflow_agent_config(
    config: Mapping[str, Any] | None, role: str
) -> dict[str, Any]:
    workflow = config_section(config, "workflow")
    for stage in workflow.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        agent = stage.get("agent") or {}
        if isinstance(agent, Mapping) and agent.get("role") == role:
            return dict(agent)
    return {}


def _strip_prefix(value: str, prefix: str) -> str:
    value = str(value).replace("\\", "/")
    if value.startswith(prefix):
        return value[len(prefix) :].lstrip("/")
    return value


def resolve_role_spec(role: str, config: Mapping[str, Any] | None = None) -> RoleSpec:
    """Return the role spec, applying ``config/workflow.yaml`` stage overrides."""

    key = str(role)
    if key not in ROLE_SPECS:
        raise WorkflowError(
            f"unknown agent role {role!r}; known roles: {sorted(ROLE_SPECS)}"
        )
    spec = ROLE_SPECS[key]
    agent = _workflow_agent_config(config, key)
    if not agent:
        return spec
    prompt_file = _strip_prefix(agent["prompt"], "prompts/") if agent.get("prompt") else spec.prompt_file
    skill_file = _strip_prefix(agent["skill"], "skills/") if agent.get("skill") else spec.skill_file
    schema_name: str | None
    if "schema" in agent:
        schema_name = agent.get("schema")  # None means Markdown-only output
    else:
        schema_name = spec.schema
    max_tokens = int(agent.get("max_tokens", spec.max_tokens))
    return replace(
        spec,
        prompt_file=Path(prompt_file).name,
        skill_file=skill_file,
        schema=schema_name,
        max_tokens=max_tokens,
    )


def load_prompt(role: str, config: Mapping[str, Any] | None = None) -> str:
    spec = resolve_role_spec(role, config)
    path = prompts_dir(config) / spec.prompt_file
    if not path.exists():
        raise WorkflowError(f"prompt file for role {role!r} is missing: {path}")
    return read_text(path)


def load_skill(skill_file: str, config: Mapping[str, Any] | None = None) -> str:
    relative = _strip_prefix(skill_file, "skills/")
    path = skills_dir(config) / relative
    if not path.exists():
        raise WorkflowError(f"skill file is missing: {path}")
    return read_text(path)


def load_role_assets(role: str, config: Mapping[str, Any] | None = None) -> RoleAssets:
    """Load the prompt, skill and schema for a role in one call."""

    spec = resolve_role_spec(role, config)
    prompt_path = prompts_dir(config) / spec.prompt_file
    skill_relative = _strip_prefix(spec.skill_file, "skills/")
    skill_path = skills_dir(config) / skill_relative
    if not prompt_path.exists():
        raise WorkflowError(f"prompt file for role {role!r} is missing: {prompt_path}")
    if not skill_path.exists():
        raise WorkflowError(f"skill file for role {role!r} is missing: {skill_path}")
    schema = None
    if spec.schema and spec.output_kind == "json":
        schema = load_schema(spec.schema)
    return RoleAssets(
        spec=spec,
        prompt_text=read_text(prompt_path),
        skill_text=read_text(skill_path),
        schema=schema,
        prompt_path=prompt_path,
        skill_path=skill_path,
    )


# ---------------------------------------------------------------------------
# Prompt hashing (used by the registry / eval baseline)
# ---------------------------------------------------------------------------


def compute_prompt_hashes(
    *,
    config: Mapping[str, Any] | None = None,
    prompt_dir: Path | str | None = None,
    skill_dir: Path | str | None = None,
) -> dict[str, str]:
    """Hash every prompt and skill file, keyed by project-relative posix path.

    The registry stores this mapping so a release can be reproduced and a candidate
    prompt set can be compared against the frozen baseline.
    """

    pdir = Path(prompt_dir) if prompt_dir is not None else prompts_dir(config)
    sdir = Path(skill_dir) if skill_dir is not None else skills_dir(config)
    hashes: dict[str, str] = {}
    for base, prefix in ((pdir, "prompts"), (sdir, "skills")):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            if not path.is_file():
                continue
            key = f"{prefix}/{path.relative_to(base).as_posix()}"
            hashes[key] = hash_text(read_text(path))
    return hashes


def prompt_hashes_by_role(config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Role name -> sha256 of its prompt file (convenience for diagnostics)."""

    return {
        role: hash_text(load_prompt(role, config))
        for role in ROLE_SPECS
    }


# ---------------------------------------------------------------------------
# Agent loop limits and LLM resolution
# ---------------------------------------------------------------------------


def agent_loop_limits(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolved agent-loop limits from ``config/model.yaml`` ``agent_loop``."""

    model_cfg = config_section(config, "model")
    raw = dict(model_cfg.get("agent_loop") or {})
    limits = dict(AGENT_LOOP_DEFAULTS)
    limits.update({key: raw[key] for key in limits if key in raw})
    limits["max_tool_rounds"] = int(limits["max_tool_rounds"])
    limits["max_retries_invalid_json"] = int(limits["max_retries_invalid_json"])
    limits["repair_temperature"] = float(limits["repair_temperature"])
    limits["max_read_chars"] = int(limits["max_read_chars"])
    return limits


def stage_agent_config(config: Mapping[str, Any] | None, role: str) -> dict[str, Any]:
    """The ``agent:`` block of the workflow stage that runs ``role`` (may be empty)."""

    return _workflow_agent_config(config, role)


def resolve_llm(ctx: RunContext) -> LLMClient:
    """Return ``ctx.llm`` or build the configured backend for this run."""

    if ctx.llm is not None:
        return ctx.llm
    return make_llm(
        ctx.llm_mode,
        ctx.config,
        cache_dir=None,
        run_id=ctx.run_id,
        recorder=make_event_recorder(ctx),
    )


def make_event_recorder(ctx: RunContext):
    """Adapter from LLM recording events to ``RunContext.event`` (never raises)."""

    def recorder(record: dict[str, Any]) -> None:
        payload = dict(record or {})
        kind = str(payload.pop("kind", "agent_llm_event"))
        try:
            ctx.event(kind, **payload)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            return

    return recorder


def iter_prompt_skill_files(
    config: Mapping[str, Any] | None = None,
) -> Iterable[tuple[str, Path]]:
    """Yield ``(project-relative posix path, path)`` for every prompt/skill file."""

    for base, prefix in ((prompts_dir(config), "prompts"), (skills_dir(config), "skills")):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            if path.is_file():
                yield f"{prefix}/{path.relative_to(base).as_posix()}", path


__all__ = [
    "ROLE_SPECS",
    "RoleAssets",
    "RoleSpec",
    "agent_loop_limits",
    "compute_prompt_hashes",
    "iter_prompt_skill_files",
    "load_prompt",
    "load_role_assets",
    "load_skill",
    "make_event_recorder",
    "prompt_hashes_by_role",
    "prompts_dir",
    "resolve_llm",
    "resolve_role_spec",
    "schema_path",
    "skills_dir",
    "stage_agent_config",
]
