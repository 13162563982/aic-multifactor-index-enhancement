"""Tests for the deterministic agent runtime (tool loop, whitelists, repair, guards)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicwf.agents.llm import NullLLM, ReplayCacheMiss
from aicwf.agents.runtime import AgentRuntime, extract_json_value
from aicwf.agents.tools import Toolbox
from aicwf.agents.validate import build_allowed_numbers, load_schema
from aicwf.contracts import ConfigError, NonRetryableError, ValidationError
from test_agents_support import (
    AS_OF,
    DEFAULT_WEIGHTS,
    ScriptedLLM,
    hypothesis_payload,
    make_ctx,
    mini_config,
)

SIMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "integer"}},
}


def make_runtime(
    llm,
    *,
    schema=SIMPLE_SCHEMA,
    output_kind="json",
    toolbox=None,
    max_tool_rounds=3,
    max_repairs=2,
    validators=(),
    text_validators=(),
    allowed_numbers=None,
    min_cited=0,
    min_chars=10,
    required_phrases=(),
) -> AgentRuntime:
    return AgentRuntime(
        role="tester",
        system_prompt="You are a deterministic test agent.",
        skill_text="Test skill text: cite only supplied values.",
        schema=schema,
        llm=llm,
        allowed_paths=(),
        max_tool_rounds=max_tool_rounds,
        toolbox=toolbox,
        output_kind=output_kind,
        as_of=AS_OF,
        max_repairs=max_repairs,
        validators=validators,
        text_validators=text_validators,
        allowed_numbers=allowed_numbers,
        min_cited_numbers=min_cited,
        min_chars=min_chars,
        required_phrases=required_phrases,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_system_prompt_states_cutoff_and_guards() -> None:
    runtime = make_runtime(ScriptedLLM(['{"answer": 1}']))
    system = runtime.system_message
    assert "Information cutoff (as_of): 2024-12-31" in system
    assert "No network" in system
    assert "No invented numbers" in system
    assert "untrusted" in system.lower()
    assert "submit_json" in system
    assert "answer" in system  # schema is embedded
    assert "repair" in system.lower()


def test_runtime_requires_llm_and_schema() -> None:
    with pytest.raises(ConfigError):
        AgentRuntime("t", "p", "s", SIMPLE_SCHEMA, None, ())
    with pytest.raises(ConfigError):
        AgentRuntime("t", "p", "s", None, NullLLM(), (), output_kind="json")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_value_handles_fences_and_prose() -> None:
    assert extract_json_value('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_value('Here is the payload: {"a": {"b": [1, 2]}} thanks') == {
        "a": {"b": [1, 2]}
    }
    assert extract_json_value('{"a": "brace } inside string"}') == {"a": "brace } inside string"}
    from aicwf.agents.runtime import JsonParseError

    with pytest.raises(JsonParseError):
        extract_json_value("no json here")


# ---------------------------------------------------------------------------
# Basic interaction
# ---------------------------------------------------------------------------


def test_bare_payload_is_accepted_as_submission() -> None:
    llm = ScriptedLLM(['{"answer": 7}'])
    runtime = make_runtime(llm)
    assert runtime.run("give an answer") == {"answer": 7}
    assert len(llm.calls) == 1
    assert llm.calls[0]["system"] == runtime.system_message


def test_tool_directives_then_submit() -> None:
    llm = ScriptedLLM(
        [
            {"tool": "list_artifacts"},
            {"tool": "read_artifact", "name": "panel"},
            {"tool": "run_check", "name": "coverage"},
            {"tool": "submit_json", "content": {"answer": 3}},
        ]
    )
    toolbox = Toolbox(
        blobs={"panel": {"rows": 3000}},
        checks={"coverage": lambda args: {"coverage": 0.98}},
        read_roots=[],
    )
    runtime = make_runtime(llm, toolbox=toolbox)
    assert runtime.run("do the work") == {"answer": 3}
    assert len(llm.calls) == 4
    assert runtime.stats["tool_rounds"] == 3
    # Observations are carried into the next user message.
    assert "0.98" in llm.calls[3]["user"]
    assert "rows" in llm.calls[2]["user"] and "3000" in llm.calls[2]["user"]


def test_unknown_tool_is_rejected_and_reported_to_model(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            {"tool": "write_file", "args": {"path": "evil.json", "content": "x"}},
            {"tool": "submit_json", "content": {"answer": 1}},
        ]
    )
    runtime = make_runtime(llm)
    assert runtime.run("try to write a file") == {"answer": 1}
    assert runtime.stats["tool_rejections"] == 1
    assert "not permitted" in llm.calls[1]["user"]
    assert not list(tmp_path.rglob("evil.json"))


def test_read_artifact_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": 1}', encoding="utf-8")
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    toolbox = Toolbox(files={"outside": outside}, read_roots=[allowed_dir])
    llm = ScriptedLLM(
        [
            {"tool": "read_artifact", "name": "outside"},
            {"tool": "submit_json", "content": {"answer": 2}},
        ]
    )
    runtime = make_runtime(llm, toolbox=toolbox)
    assert runtime.run("read the secret") == {"answer": 2}
    assert "read allowlist" in llm.calls[1]["user"]
    assert '"secret": 1' not in llm.calls[1]["user"]


def test_binary_artifact_is_refused_but_listed(tmp_path: Path) -> None:
    binary = tmp_path / "panel.parquet"
    binary.write_bytes(b"PAR1")
    toolbox = Toolbox(files={"panel": binary}, read_roots=[tmp_path])
    llm = ScriptedLLM(
        [
            {"tool": "read_artifact", "name": "panel"},
            {"tool": "submit_json", "content": {"answer": 3}},
        ]
    )
    runtime = make_runtime(llm, toolbox=toolbox)
    assert runtime.run("read binary") == {"answer": 3}
    assert "binary" in llm.calls[1]["user"]


def test_tool_loop_is_bounded() -> None:
    read = {"tool": "list_artifacts"}
    llm = ScriptedLLM([read] * 20)
    runtime = make_runtime(llm, max_tool_rounds=2, max_repairs=0)
    with pytest.raises(NonRetryableError) as info:
        runtime.run("loop forever")
    assert "max_tool_rounds=2" in str(info.value)
    # One tool execution per round and one directive that breaks the bound.
    assert len(llm.calls) == 3


def test_run_check_whitelist_only() -> None:
    toolbox = Toolbox(checks={"known": lambda args: {"ok": True}}, read_roots=[])
    llm = ScriptedLLM(
        [
            {"tool": "run_check", "name": "unknown", "args": {}},
            {"tool": "run_check", "name": "known", "args": {}},
            {"tool": "submit_json", "content": {"answer": 4}},
        ]
    )
    runtime = make_runtime(llm, toolbox=toolbox)
    assert runtime.run("run checks") == {"answer": 4}
    assert "unknown check" in llm.calls[1]["user"]
    assert '"ok": true' in llm.calls[2]["user"]


# ---------------------------------------------------------------------------
# Repair paths
# ---------------------------------------------------------------------------


def test_schema_repair_path_uses_temperature_zero() -> None:
    llm = ScriptedLLM([{"answer": "not an integer"}, {"answer": 5}])
    runtime = make_runtime(llm)
    assert runtime.run("answer") == {"answer": 5}
    assert runtime.stats["repairs"] == 1
    assert llm.calls[0]["temperature"] == 0.1
    assert llm.calls[1]["temperature"] == 0.0
    assert "Repair required" in llm.calls[1]["user"]
    assert "answer" in llm.calls[1]["user"]


def test_domain_validator_repair_path() -> None:
    def validator(payload):
        return [] if payload.get("answer") == 4 else ["answer must be 4"]

    llm = ScriptedLLM([{"answer": 3}, {"answer": 4}])
    runtime = make_runtime(llm, validators=(validator,))
    assert runtime.run("answer") == {"answer": 4}
    assert llm.calls[1]["temperature"] == 0.0
    assert "must be 4" in llm.calls[1]["user"]


def test_invalid_json_self_correction_then_success() -> None:
    llm = ScriptedLLM(["this is not json", '{"answer": 6}'])
    runtime = make_runtime(llm)
    assert runtime.run("answer") == {"answer": 6}
    assert len(llm.calls) == 2
    assert "not a parseable JSON object" in llm.calls[1]["user"]


def test_persistent_failure_raises_non_retryable_with_transcript() -> None:
    llm = ScriptedLLM(default={"answer": "wrong"})
    runtime = make_runtime(llm, max_repairs=2)
    with pytest.raises(NonRetryableError) as info:
        runtime.run("answer")
    error = info.value
    assert "failed validation after 3 attempt" in str(error)
    assert len(llm.calls) == 3
    transcript = getattr(error, "transcript")
    assert transcript["role"] == "tester"
    assert transcript["errors"]
    assert len(transcript["attempts"]) == 3


def test_invalid_submission_shape_raises_non_retryable() -> None:
    llm = ScriptedLLM(['{"tool": "submit_json", "content": "not an object"}'])
    runtime = make_runtime(llm, max_repairs=0)
    with pytest.raises(NonRetryableError):
        runtime.run("submit a bad shape")


def test_repair_budget_is_bounded() -> None:
    llm = ScriptedLLM(default={"answer": "wrong"})
    runtime = make_runtime(llm, max_repairs=1)
    with pytest.raises(NonRetryableError):
        runtime.run("answer")
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# Markdown output and citation guards
# ---------------------------------------------------------------------------


def test_plain_markdown_is_accepted() -> None:
    text = "## Summary\n\nInformation cutoff: 2024-12-31. Values 0.05 and 12.5 are supplied."
    llm = ScriptedLLM([text])
    runtime = make_runtime(
        llm,
        schema=None,
        output_kind="markdown",
        allowed_numbers=build_allowed_numbers({"x": 0.05, "y": 12.5}),
        min_cited=2,
        min_chars=20,
    )
    assert runtime.run("write a note") == text


def test_markdown_cites_invented_number_then_repairs() -> None:
    invented = "## Note\n\nInformation cutoff: 2024-12-31. Invented value 0.987654 is bad."
    valid = "## Note\n\nInformation cutoff: 2024-12-31. Values 0.05 and 12.5 are supplied."
    llm = ScriptedLLM([invented, valid])
    runtime = make_runtime(
        llm,
        schema=None,
        output_kind="markdown",
        allowed_numbers=build_allowed_numbers({"x": 0.05, "y": 12.5}),
        min_cited=2,
        min_chars=20,
    )
    assert runtime.run("write a note") == valid
    assert "0.987654" in llm.calls[1]["user"]
    assert llm.calls[1]["temperature"] == 0.0


def test_markdown_requires_cutoff_and_phrases() -> None:
    missing_cutoff = "## Summary\n\nValues 0.05 and 12.5 are supplied."
    missing_phrase = "Information cutoff: 2024-12-31. Values 0.05 and 12.5 are supplied."
    valid = (
        "Information cutoff: 2024-12-31. Values 0.05 and 12.5 are supplied.\n"
        "## Verdict on robustness\n\nWe are cautious."
    )
    llm = ScriptedLLM([missing_cutoff, missing_phrase, valid])
    runtime = make_runtime(
        llm,
        schema=None,
        output_kind="markdown",
        allowed_numbers=build_allowed_numbers({"x": 0.05, "y": 12.5}),
        min_cited=2,
        min_chars=20,
        required_phrases=("Verdict on robustness",),
        max_repairs=2,
    )
    assert runtime.run("write a note") == valid
    assert llm.calls[1]["temperature"] == 0.0
    assert "information cutoff" in llm.calls[1]["user"].lower()
    assert "Verdict on robustness" in llm.calls[2]["user"]


# ---------------------------------------------------------------------------
# Write allowlist and replay miss
# ---------------------------------------------------------------------------


def test_write_allowlist(tmp_path: Path, run_context) -> None:
    stage_dir = run_context.stage_dir("report")
    toolbox = Toolbox(
        write_paths={"note": stage_dir / "note.md", "data": stage_dir / "data.json"},
        write_roots=[run_context.run_dir],
        read_roots=[run_context.run_dir],
    )
    runtime = make_runtime(ScriptedLLM(['{"answer": 1}']), toolbox=toolbox)
    path = runtime.write_markdown("note", "hello")
    assert path.read_text(encoding="utf-8").strip() == "hello"
    runtime.write_json("data", {"ok": True})
    with pytest.raises(ValidationError):
        runtime.write_json("undeclared", {"ok": False})


def test_write_allowlist_blocks_escaping_paths(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    toolbox = Toolbox(
        write_paths={"data": outside},
        write_roots=[tmp_path / "run"],
        read_roots=[],
    )
    runtime = make_runtime(ScriptedLLM(['{"answer": 1}']), toolbox=toolbox)
    with pytest.raises(ValidationError):
        runtime.write_json("data", {"ok": True})
    assert not outside.exists()


def test_replay_miss_fails_fast_without_repair(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=None, llm_mode="replay")
    llm = __import__("aicwf.agents.llm", fromlist=["make_llm"]).make_llm("replay", ctx.config)
    runtime = make_runtime(llm, max_repairs=2)
    with pytest.raises(ReplayCacheMiss):
        runtime.run("nothing recorded")


def test_stats_summary_tracks_usage() -> None:
    llm = ScriptedLLM(['{"answer": 1}'])
    runtime = make_runtime(llm)
    runtime.run("answer")
    stats = runtime.stats_summary()
    assert stats["role"] == "tester"
    assert stats["calls"] == 1
    assert stats["prompt_tokens"] == 100
    assert stats["completion_tokens"] == 50
    assert stats["attempts"] == 1


def test_hypothesis_schema_runtime_end_to_end() -> None:
    schema = load_schema("hypothesis")
    llm = ScriptedLLM([hypothesis_payload()])
    runtime = AgentRuntime(
        role="researcher",
        system_prompt="research prompt",
        skill_text="skill",
        schema=schema,
        llm=llm,
        allowed_paths=(),
        as_of=AS_OF,
    )
    payload = runtime.run("produce the hypothesis")
    assert payload["factor_weights"] == DEFAULT_WEIGHTS
    assert payload["information_cutoff"] == "2024-12-31"
