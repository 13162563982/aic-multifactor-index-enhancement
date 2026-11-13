"""Offline tests for the agent-driven stages on hand-built RunContexts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from aicwf.agents.llm import NullLLM, ReplayCacheMiss
from aicwf.agents.roles import (
    ROLE_SPECS,
    compute_prompt_hashes,
    load_role_assets,
    load_prompt,
    load_skill,
)
from aicwf.agents.stages import (
    generate_verifier_note,
    run_factor_review,
    run_hypothesis,
    run_report,
    run_risk_review,
)
from aicwf.agents.validate import (
    default_factor_weights,
    factor_catalog,
    load_schema,
    strategy_constraints,
    validate_factor_flags,
    validate_hypothesis,
    validate_payload,
    validate_verification_dict,
)
from aicwf.contracts import NonRetryableError, ValidationError
from aicwf.util.jsonio import read_json
from test_agents_support import (
    AS_OF,
    AS_OF_STR,
    DEFAULT_WEIGHTS,
    FACTOR_IDS,
    FakeEvents,
    ScriptedLLM,
    hypothesis_payload,
    make_ctx,
    mini_config,
    populate_upstream,
    valid_report_markdown,
    valid_risk_markdown,
    verification_payload,
)


def valid_flags_payload(as_of: dt.date = AS_OF) -> dict:
    return {
        "assessment_markdown": (
            f"Information cutoff: {as_of.isoformat()}. Value EP RankIC is 0.041 with "
            "t-statistic 3.1, so the value factor still carries usable cross-sectional "
            "information; momentum RankIC is 0.019 with a lower t-statistic."
        ),
        "summary": "Value and quality are healthy; momentum is marginal.",
        "flags": [
            {
                "id": "value_ic_positive",
                "severity": "info",
                "message": "Value RankIC is positive in the supplied factor report.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Role assets and prompts
# ---------------------------------------------------------------------------


def test_role_assets_load_for_every_role() -> None:
    for role in ROLE_SPECS:
        assets = load_role_assets(role)
        assert assets.prompt_text.strip()
        assert assets.skill_text.strip()
        assert "cutoff" in assets.prompt_text.lower()
        assert "cutoff" in assets.skill_text.lower() or "information cutoff" in assets.skill_text.lower()
        assert assets.spec.output_kind in {"json", "markdown"}
        if assets.spec.output_kind == "json":
            assert assets.schema is not None


def test_prompt_and_skill_hashes_are_stable() -> None:
    first = compute_prompt_hashes()
    second = compute_prompt_hashes()
    assert first == second
    assert len([key for key in first if key.startswith("prompts/")]) == 5
    assert len([key for key in first if key.startswith("skills/")]) == 5
    assert all(len(value) == 64 for value in first.values())


def test_real_project_config_and_catalog_are_compatible() -> None:
    from aicwf.util.paths import load_config_bundle

    bundle = load_config_bundle()
    catalog = factor_catalog(bundle)
    assert "value_ep" in catalog and "momentum_12_1" in catalog
    payload = hypothesis_payload(as_of=dt.date(2026, 11, 12))
    # The hypothesis contract allows 3-8 factors; use the four largest default weights,
    # renormalised, rather than the full 13-factor default composite.
    default_weights = default_factor_weights(bundle)
    top = dict(sorted(default_weights.items(), key=lambda pair: -pair[1])[:4])
    total = sum(top.values())
    payload["factor_weights"] = {key: round(value / total, 6) for key, value in top.items()}
    payload["constraints"] = strategy_constraints(bundle)
    errors = validate_hypothesis(payload, config=bundle, as_of=dt.date(2026, 11, 12))
    assert errors == []
    assert validate_payload(payload, load_schema("hypothesis")) == []


# ---------------------------------------------------------------------------
# Hypothesis stage
# ---------------------------------------------------------------------------


def test_run_hypothesis_null_mode(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    output = run_hypothesis(ctx)
    hypothesis_path = Path(output.artifacts["hypothesis"])
    rationale_path = Path(output.artifacts["rationale"])
    assert hypothesis_path.exists()
    assert rationale_path.exists()
    payload = read_json(hypothesis_path)
    assert validate_payload(payload, load_schema("hypothesis")) == []
    assert validate_hypothesis(payload, config=ctx.config, as_of=ctx.as_of) == []
    assert payload["information_cutoff"] == AS_OF_STR
    assert abs(sum(payload["factor_weights"].values()) - 1.0) <= 1e-6
    rationale = rationale_path.read_text(encoding="utf-8")
    assert AS_OF_STR in rationale
    assert "Falsification criteria" in rationale
    assert "value_ep" in rationale
    summary = output.summary
    assert set(summary) >= {
        "thesis",
        "factor_weights",
        "rebalance",
        "constraints",
        "information_cutoff",
    }
    assert summary["information_cutoff"] == AS_OF_STR
    assert summary["agent"]["calls"] >= 1


def test_run_hypothesis_repairs_unknown_factor_id(tmp_path: Path) -> None:
    bad = hypothesis_payload()
    bad["factor_weights"] = {"unknown_factor": 1.0}
    llm = ScriptedLLM([bad, hypothesis_payload()])
    ctx = make_ctx(tmp_path, llm=llm)
    output = run_hypothesis(ctx)
    payload = read_json(Path(output.artifacts["hypothesis"]))
    assert payload["factor_weights"] == DEFAULT_WEIGHTS
    assert len(llm.calls) == 2
    assert "unknown factor ids" in llm.calls[1]["user"]
    assert llm.calls[1]["temperature"] == 0.0


def test_run_hypothesis_renormalises_rounded_weights(tmp_path: Path) -> None:
    payload = hypothesis_payload()
    # Normal presentation rounding: sums to 1.000001, not exactly 1.0.
    payload["factor_weights"] = {
        "value_ep": 0.250001,
        "quality_roe": 0.25,
        "momentum_12_1": 0.25,
        "low_vol_60d": 0.25,
    }
    llm = ScriptedLLM([payload])
    ctx = make_ctx(tmp_path, llm=llm)
    output = run_hypothesis(ctx)
    written_payload = read_json(Path(output.artifacts["hypothesis"]))
    written = written_payload["factor_weights"]
    assert abs(sum(written.values()) - 1.0) < 1e-9
    assert set(written) == set(payload["factor_weights"])
    assert validate_hypothesis(written_payload, config=ctx.config, as_of=ctx.as_of) == []
    assert abs(written["value_ep"] - 0.250001) < 1e-6


def test_run_hypothesis_persistent_invalid_fails(tmp_path: Path) -> None:
    bad = hypothesis_payload()
    bad["factor_weights"] = {"unknown_factor": 1.0}
    llm = ScriptedLLM(default=bad)
    ctx = make_ctx(tmp_path, llm=llm)
    with pytest.raises(NonRetryableError) as info:
        run_hypothesis(ctx)
    assert "failed validation" in str(info.value)
    assert not Path(ctx.artifact_path("hypothesis", "hypothesis.json")).exists()
    assert len(llm.calls) == 3


def test_run_hypothesis_replay_miss_fails_fast(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=None, llm_mode="replay")
    with pytest.raises(ReplayCacheMiss):
        run_hypothesis(ctx)


# ---------------------------------------------------------------------------
# Factor review stage
# ---------------------------------------------------------------------------


def test_run_factor_review_null_mode(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    populate_upstream(ctx)
    output = run_factor_review(ctx)
    flags = read_json(Path(output.artifacts["flags"]))
    assert validate_payload(flags, load_schema("factor_flags")) == []
    assert validate_factor_flags(flags) == []
    assessment = Path(output.artifacts["assessment"]).read_text(encoding="utf-8")
    assert AS_OF_STR in assessment
    assert "## Flags" in assessment
    assert output.summary["n_flags"] == len(flags["flags"])
    assert "factor_weights" in output.summary


def test_run_factor_review_repairs_warn_without_number(tmp_path: Path) -> None:
    bad = valid_flags_payload()
    bad["flags"] = [
        {
            "id": "momentum_weak",
            "severity": "warn",
            "message": "Momentum looks weak and should be watched carefully.",
        }
    ]
    llm = ScriptedLLM([bad, valid_flags_payload()])
    ctx = make_ctx(tmp_path, llm=llm)
    populate_upstream(ctx)
    output = run_factor_review(ctx)
    flags = read_json(Path(output.artifacts["flags"]))
    assert flags["flags"][0]["id"] == "value_ic_positive"
    assert len(llm.calls) == 2
    assert "diagnostic number" in llm.calls[1]["user"]


def test_run_factor_review_requires_upstream_report(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    from aicwf.contracts import WorkflowError

    with pytest.raises(WorkflowError):
        run_factor_review(ctx)


# ---------------------------------------------------------------------------
# Risk review stage
# ---------------------------------------------------------------------------


def test_run_risk_review_null_mode(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    populate_upstream(ctx)
    output = run_risk_review(ctx)
    critique_path = Path(output.artifacts["critique"])
    critique = critique_path.read_text(encoding="utf-8")
    assert AS_OF_STR in critique
    assert "Verdict on robustness" in critique
    assert "What would change our mind" in critique
    assert output.summary["information_cutoff"] == AS_OF_STR
    assert output.summary["failed_checks"] == 1


def test_run_risk_review_repairs_invented_number(tmp_path: Path) -> None:
    invented = valid_risk_markdown() + "\nAn invented robustness value 0.987654 is quoted here.\n"
    llm = ScriptedLLM([invented, valid_risk_markdown()])
    ctx = make_ctx(tmp_path, llm=llm)
    populate_upstream(ctx)
    output = run_risk_review(ctx)
    critique = Path(output.artifacts["critique"]).read_text(encoding="utf-8")
    assert "0.987654" not in critique
    assert len(llm.calls) == 2
    assert "0.987654" in llm.calls[1]["user"]


def test_run_risk_review_requires_robustness(tmp_path: Path) -> None:
    from aicwf.contracts import WorkflowError

    ctx = make_ctx(tmp_path, llm=NullLLM())
    populate_upstream(ctx)
    del ctx.stage_outputs["robustness"]
    with pytest.raises(WorkflowError):
        run_risk_review(ctx)


# ---------------------------------------------------------------------------
# Report stage
# ---------------------------------------------------------------------------


def test_run_report_null_mode_writes_note_table_and_charts(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    populate_upstream(ctx)
    output = run_report(ctx)
    note = Path(output.artifacts["research_note"]).read_text(encoding="utf-8")
    table = Path(output.artifacts["metrics_table"]).read_text(encoding="utf-8")
    assert AS_OF_STR in note
    assert "PASS" in note
    assert "metrics_table.md" in note
    assert AS_OF_STR in table
    assert "annualized_excess_return" in table
    assert "0.0234" in table
    assert "paths" in output.summary and output.summary["paths"]
    assert output.summary["verdict"] == "PASS"
    for key, rel in output.artifacts.items():
        assert Path(rel).exists(), key


def test_run_report_requires_pass_verdict(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    populate_upstream(ctx, verdict="FAIL")
    with pytest.raises(NonRetryableError) as info:
        run_report(ctx)
    assert "FAIL" in str(info.value)


def test_run_report_repairs_unverified_citation(tmp_path: Path) -> None:
    invented = valid_report_markdown() + "\nAn invented metric 0.987654 appears here.\n"
    llm = ScriptedLLM([invented, valid_report_markdown()])
    ctx = make_ctx(tmp_path, llm=llm)
    populate_upstream(ctx)
    output = run_report(ctx)
    note = Path(output.artifacts["research_note"]).read_text(encoding="utf-8")
    assert "0.987654" not in note
    assert len(llm.calls) == 2
    assert "0.987654" in llm.calls[1]["user"]


def test_run_report_requires_verification_stage(tmp_path: Path) -> None:
    from aicwf.contracts import WorkflowError

    ctx = make_ctx(tmp_path, llm=NullLLM())
    with pytest.raises(WorkflowError):
        run_report(ctx)


# ---------------------------------------------------------------------------
# Verifier note helper
# ---------------------------------------------------------------------------


def test_generate_verifier_note_null_mode(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    verification = verification_payload()
    note = generate_verifier_note(ctx, verification)
    assert isinstance(note, str)
    assert "PASS" in note
    assert AS_OF_STR in note
    assert "## Checks performed" in note
    assert "## Discrepancies" in note
    assert "Information cutoff" in note


def test_generate_verifier_note_accepts_fail_verdict(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    verification = verification_payload(verdict="FAIL")
    note = generate_verifier_note(ctx, verification)
    assert "FAIL" in note
    assert "failed" in note.lower()


def test_generate_verifier_note_rejects_invalid_verification(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    verification = verification_payload()
    verification["verdict"] = "MAYBE"
    with pytest.raises(ValidationError):
        generate_verifier_note(ctx, verification)


def test_generate_verifier_note_falls_back_on_replay_miss(tmp_path: Path) -> None:
    events = FakeEvents()
    ctx = make_ctx(
        tmp_path,
        config=mini_config(cache_dir=tmp_path / "empty_cache"),
        llm=None,
        llm_mode="replay",
        events=events,
    )
    note = generate_verifier_note(ctx, verification_payload())
    assert "Independent verification note" in note
    assert "PASS" in note
    assert AS_OF_STR in note
    kinds = [record["kind"] for record in events.records]
    assert "agent_fallback" in kinds


def test_verification_payload_validates_against_schema() -> None:
    assert validate_verification_dict(verification_payload()) == []
