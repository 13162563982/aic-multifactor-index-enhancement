"""Tests for JSON schemas and the domain/citation validators."""

from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal

import pytest

from aicwf.agents.validate import (
    build_allowed_numbers,
    check_cited_numbers,
    check_markdown_contract,
    load_schema,
    numbers_from,
    schema_path,
    synthesize_json,
    validate_approval_payload,
    validate_factor_flags,
    validate_hypothesis,
    validate_payload,
    validate_verification_dict,
    validate_version_payload,
)
from test_agents_support import (
    AS_OF,
    DEFAULT_WEIGHTS,
    factor_report_payload,
    hypothesis_payload,
    mini_config,
    verification_payload,
)

CONFIG = mini_config()


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def test_all_schemas_load_and_are_valid() -> None:
    for name in (
        "hypothesis",
        "factor_flags",
        "verification",
        "approval",
        "version",
    ):
        schema = load_schema(name)
        assert isinstance(schema, dict)
        assert schema.get("$id")
        assert schema_path(name).exists()


def test_schema_synthesis_round_trips_every_schema() -> None:
    for name in ("hypothesis", "factor_flags", "verification", "approval", "version"):
        schema = load_schema(name)
        payload = synthesize_json(schema)
        assert validate_payload(payload, schema) == [], name


# ---------------------------------------------------------------------------
# Hypothesis schema + domain
# ---------------------------------------------------------------------------


def test_hypothesis_valid_payload() -> None:
    payload = hypothesis_payload()
    assert validate_payload(payload, "hypothesis") == []
    assert validate_hypothesis(payload, config=CONFIG, as_of=AS_OF) == []


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda payload: payload.update(thesis="short"), "thesis"),
        (lambda payload: payload.update(rebalance="weekly"), "rebalance"),
        (lambda payload: payload.update(extra_field=1), "extra_field"),
        (lambda payload: payload.update(information_cutoff="31-12-2024"), "information_cutoff"),
        (
            lambda payload: payload["constraints"].update({"max_weight": 0.5}),
            "max_weight",
        ),
        (lambda payload: payload.update(factor_weights={}), "factor_weights"),
    ],
)
def test_hypothesis_schema_rejections(mutate, expected: str) -> None:
    payload = hypothesis_payload()
    mutate(payload)
    errors = validate_payload(payload, "hypothesis")
    assert errors, f"expected a schema error containing {expected}"
    assert any(expected in error for error in errors)


def test_hypothesis_domain_rejects_unknown_factor_ids() -> None:
    payload = hypothesis_payload()
    payload["factor_weights"] = {"not_a_factor": 1.0}
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("unknown factor ids" in error for error in errors)


def test_hypothesis_domain_tolerates_presentation_rounding() -> None:
    payload = hypothesis_payload()
    # Sums to 1.000001: normal 6-decimal presentation rounding.
    payload["factor_weights"] = {
        "value_ep": 0.333334,
        "quality_roe": 0.333333,
        "momentum_12_1": 0.166667,
        "low_vol_60d": 0.166667,
    }
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert not any("normalised" in error for error in errors)


def test_hypothesis_domain_rejects_unnormalised_weights() -> None:
    payload = hypothesis_payload()
    payload["factor_weights"] = dict(DEFAULT_WEIGHTS)
    payload["factor_weights"]["value_ep"] = 0.90
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("normalised" in error for error in errors)


def test_hypothesis_domain_rejects_negative_weights() -> None:
    payload = hypothesis_payload()
    payload["factor_weights"] = {
        "value_ep": 1.2,
        "quality_roe": -0.2,
        "momentum_12_1": 0.0,
        "low_vol_60d": 0.0,
    }
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any(">= 0" in error for error in errors)


def test_hypothesis_domain_rejects_cutoff_mismatch() -> None:
    payload = hypothesis_payload()
    payload["information_cutoff"] = "2025-01-01"
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("information_cutoff" in error for error in errors)


def test_hypothesis_domain_rejects_constraint_envelope_breach() -> None:
    payload = hypothesis_payload()
    payload["constraints"]["te_target"] = 0.11  # schema-valid but above strategy 0.05
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("envelope" in error for error in errors)


def test_hypothesis_domain_requires_few_falsifiers() -> None:
    payload = hypothesis_payload()
    payload["falsification"] = ["only one condition that is testable"]
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("falsification" in error for error in errors)


def test_hypothesis_domain_requires_three_factors() -> None:
    payload = hypothesis_payload()
    payload["factor_weights"] = {"value_ep": 0.6, "quality_roe": 0.4}
    errors = validate_hypothesis(payload, config=CONFIG, as_of=AS_OF)
    assert any("between 3 and 8" in error for error in errors)


# ---------------------------------------------------------------------------
# Factor flags
# ---------------------------------------------------------------------------


def valid_flags_payload() -> dict:
    return {
        "assessment_markdown": (
            "Information cutoff: 2024-12-31. Value EP RankIC is 0.041 with t-statistic "
            "3.1, while momentum RankIC is 0.019 with t-statistic 1.6, so momentum is "
            "the weaker factor."
        ),
        "summary": "Value and quality are healthy; momentum is marginal.",
        "flags": [
            {
                "id": "momentum_ic_marginal",
                "severity": "warn",
                "message": "Momentum RankIC t-statistic is 1.6, below the 2.0 reference.",
            },
            {
                "id": "value_coverage_ok",
                "severity": "info",
                "message": "Value coverage is stable across the sample.",
            },
        ],
    }


def test_factor_flags_valid_payload() -> None:
    assert validate_factor_flags(valid_flags_payload()) == []


def test_factor_flags_rejects_warn_without_number() -> None:
    payload = valid_flags_payload()
    payload["flags"][0]["message"] = "Momentum looks weak and should be watched carefully."
    errors = validate_factor_flags(payload)
    assert any("cite the diagnostic number" in error for error in errors)


def test_factor_flags_rejects_duplicate_ids() -> None:
    payload = valid_flags_payload()
    payload["flags"][1]["id"] = payload["flags"][0]["id"]
    errors = validate_factor_flags(payload)
    assert any("duplicate flag id" in error for error in errors)


def test_factor_flags_rejects_bad_severity_and_id() -> None:
    payload = valid_flags_payload()
    payload["flags"][0]["severity"] = "critical"
    payload["flags"][1]["id"] = "Bad ID!"
    errors = validate_factor_flags(payload)
    assert any("severity" in error for error in errors)
    assert any("id" in error for error in errors)


# ---------------------------------------------------------------------------
# Verification / approval / version
# ---------------------------------------------------------------------------


def test_verification_valid_and_contradiction_rejected() -> None:
    payload = verification_payload()
    assert validate_verification_dict(payload) == []
    payload["checks"].append(
        {"id": "ledger_reconcile", "status": "fail", "detail": "mismatch"}
    )
    errors = validate_verification_dict(payload)
    assert any("contradicts failed checks" in error for error in errors)


def test_verification_rejects_bad_hash_and_verdict() -> None:
    payload = verification_payload()
    payload["artifact_hashes"]["ledger.parquet"] = "not-a-hash"
    payload["verdict"] = "MAYBE"
    errors = validate_verification_dict(payload)
    assert any("artifact_hashes" in error for error in errors)
    assert any("verdict" in error for error in errors)


def test_approval_valid_and_rejections() -> None:
    payload = {
        "run_id": "run-1",
        "action": "publish_report",
        "approver": "Zhengjia Zhu",
        "decided_at_utc": "2026-11-12T10:00:00+00:00",
        "scope": ["10_report/research_note.md"],
        "artifact_hashes": {"10_report/research_note.md": "a" * 64},
        "decision": "approve",
        "note": "Reviewed the verified note.",
    }
    assert validate_approval_payload(payload) == []
    payload["action"] = "delete_ledger"
    payload["decision"] = "maybe"
    errors = validate_approval_payload(payload)
    assert any("action" in error for error in errors)
    assert any("decision" in error for error in errors)


def test_version_valid_and_rejections() -> None:
    payload = {
        "version": "v1.1.0",
        "git_commit": "abc1234",
        "config_hash": "f" * 64,
        "prompt_hashes": {"prompts/researcher.md": "a" * 64},
        "snapshot_hash": "e" * 64,
        "metrics": {"annualized_excess_return": 0.0234},
        "verification_verdict": "PASS",
        "approval": {"decision": "approve"},
        "decision": "accept",
        "previous_version": "v1.0.0",
    }
    assert validate_version_payload(payload) == []
    payload["version"] = "1.1"
    payload["decision"] = "maybe"
    errors = validate_version_payload(payload)
    assert any("version" in error for error in errors)
    assert any("decision" in error for error in errors)


# ---------------------------------------------------------------------------
# Number and citation guards
# ---------------------------------------------------------------------------


def test_numbers_from_ignores_booleans_and_collects_nested() -> None:
    values = numbers_from({"a": 1, "b": [0.5, {"c": 12.25}], "d": True, "e": "not a number"})
    assert values == {Decimal("1"), Decimal("0.5"), Decimal("12.25")}


def test_numbers_from_includes_numbers_embedded_in_prose() -> None:
    values = numbers_from(
        {
            "warnings": ["neutralised z coverage 58% for turnover_20d"],
            "detail": "eligible rows: 28,000",
        }
    )
    assert Decimal("58") in values
    assert Decimal("0.58") in values
    assert Decimal("28000") in values


def test_check_cited_numbers_allows_prose_numbers_and_thousands_separators() -> None:
    allowed = build_allowed_numbers(
        {"warnings": ["neutralised z coverage 58%"], "detail": "eligible rows: 28,000"}
    )
    unverified, verified = check_cited_numbers(
        "Warning: coverage is 58% and there were 28,000 eligible rows.", allowed
    )
    assert unverified == []
    assert "58%" in verified
    assert any(token.replace(",", "") == "28000" for token in verified)


def test_check_cited_numbers_allows_rounding_and_exempts_years() -> None:
    allowed = {Decimal("0.1234"), Decimal("0.05"), Decimal("123.4567")}
    unverified, verified = check_cited_numbers(
        "In 2018-2020 the TE was 0.05, the ratio 12.34% and the level 123.46; 3 checks passed.",
        allowed,
    )
    assert unverified == []
    assert "0.05" in verified
    assert "12.34%" in verified
    assert "123.46" in verified


def test_check_cited_numbers_flags_inventions() -> None:
    allowed = {Decimal("0.05")}
    unverified, _ = check_cited_numbers("An invented 0.9876 and 42.5 appear here.", allowed)
    assert "0.9876" in unverified
    assert "42.5" in unverified


def test_check_markdown_contract_reports_all_problems() -> None:
    allowed = build_allowed_numbers({"metrics": {"annualized_excess_return": 0.0234}})
    errors = check_markdown_contract(
        "too short 0.5",
        allowed_numbers=allowed,
        min_cited=2,
        as_of=AS_OF,
        min_chars=100,
        required_phrases=("Executive summary",),
        label="note",
    )
    assert any("too short" in error for error in errors)
    assert any("information cutoff" in error for error in errors)
    assert any("verified artifact numbers" in error for error in errors)
    assert any("required section" in error for error in errors)


def test_check_markdown_contract_accepts_valid_note() -> None:
    allowed = build_allowed_numbers({"metrics": {"annualized_excess_return": 0.0234, "te": 0.049}})
    text = (
        "## Executive summary\n\n"
        "Information cutoff: 2024-12-31. Verified annualized excess return is 0.0234 "
        "with tracking error 0.049. This note is at least one hundred characters long, "
        "which lets the contract checker pass on length as well as on numbers."
    )
    assert (
        check_markdown_contract(
            text,
            allowed_numbers=allowed,
            min_cited=2,
            as_of=AS_OF,
            min_chars=100,
            required_phrases=("Executive summary",),
        )
        == []
    )


def test_factor_report_numbers_are_citable() -> None:
    report = factor_report_payload()
    allowed = build_allowed_numbers(report)
    unverified, verified = check_cited_numbers(
        "Value EP RankIC is 0.041 with t-statistic 3.1.", allowed
    )
    assert unverified == []
    assert len(verified) == 2
