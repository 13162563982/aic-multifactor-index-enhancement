"""Agent subsystem: LLM backends, deterministic agent runtime, roles and stages.

Public entry points used by the controller / pipeline:

* :func:`aicwf.agents.llm.make_llm` - build the LLM backend for a run.
* :class:`aicwf.agents.runtime.AgentRuntime` - bounded tool loop + validated output.
* :func:`aicwf.agents.stages.run_hypothesis`,
  :func:`aicwf.agents.stages.run_factor_review`,
  :func:`aicwf.agents.stages.run_risk_review`,
  :func:`aicwf.agents.stages.run_report` - stage handlers.
* :func:`aicwf.agents.stages.generate_verifier_note` - verification helper.
* :func:`aicwf.agents.roles.compute_prompt_hashes` - prompt/skill hashes for the registry.
"""

from aicwf.agents.llm import (
    NullLLM,
    OllamaLLM,
    OpenAICompatLLM,
    ReplayCache,
    ReplayCacheMiss,
    ReplayLLM,
    cache_key,
    make_llm,
    resolve_cache_dir,
)
from aicwf.agents.roles import (
    agent_loop_limits,
    compute_prompt_hashes,
    load_prompt,
    load_role_assets,
    load_skill,
    resolve_llm,
    resolve_role_spec,
    stage_agent_config,
)
from aicwf.agents.runtime import AgentRuntime, JsonParseError, extract_json_value
from aicwf.agents.stages import (
    generate_verifier_note,
    run_factor_review,
    run_hypothesis,
    run_report,
    run_risk_review,
)
from aicwf.agents.tools import TOOL_WHITELIST, Toolbox, ToolRejected
from aicwf.agents.validate import (
    check_cited_numbers,
    check_markdown_contract,
    load_schema,
    validate_approval_payload,
    validate_factor_flags,
    validate_hypothesis,
    validate_payload,
    validate_verification_dict,
    validate_version_payload,
)

__all__ = [
    "TOOL_WHITELIST",
    "AgentRuntime",
    "JsonParseError",
    "NullLLM",
    "OllamaLLM",
    "OpenAICompatLLM",
    "ReplayCache",
    "ReplayCacheMiss",
    "ReplayLLM",
    "ToolRejected",
    "Toolbox",
    "agent_loop_limits",
    "cache_key",
    "check_cited_numbers",
    "check_markdown_contract",
    "compute_prompt_hashes",
    "extract_json_value",
    "generate_verifier_note",
    "load_prompt",
    "load_role_assets",
    "load_schema",
    "load_skill",
    "make_llm",
    "resolve_cache_dir",
    "resolve_llm",
    "resolve_role_spec",
    "run_factor_review",
    "run_hypothesis",
    "run_report",
    "run_risk_review",
    "stage_agent_config",
    "validate_approval_payload",
    "validate_factor_flags",
    "validate_hypothesis",
    "validate_payload",
    "validate_verification_dict",
    "validate_version_payload",
]
