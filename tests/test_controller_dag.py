"""Tests for workflow DAG parsing, validation and deterministic ordering."""

from __future__ import annotations

import pytest

from aicwf.contracts import STAGES, ConfigError
from aicwf.controller import handler_map, load_workflow, stage_order, topological_order
from aicwf.controller.dag import StageSpec
from test_controller_helpers import make_config, stage_spec, workflow_section


def _config(stages):
    return make_config(stages)


def test_valid_workflow_order_is_deterministic() -> None:
    stages = [
        stage_spec("data_prep", needs=["intake"]),
        stage_spec("intake"),
        stage_spec("hypothesis", needs=["intake"]),
    ]
    workflow = load_workflow(_config(stages))
    # Declaration order breaks ties between independent stages (intake index 1,
    # data_prep index 0, hypothesis index 2 -> intake, data_prep, hypothesis).
    assert workflow.stage_names == ("intake", "data_prep", "hypothesis")
    assert stage_order(_config(stages)) == ("intake", "data_prep", "hypothesis")
    assert [spec.name for spec in topological_order(list(workflow.stages.values()))] == list(
        workflow.stage_names
    )
    # Parsing is deterministic across calls.
    assert load_workflow(_config(stages)).stage_names == workflow.stage_names


def test_workflow_graph_helpers() -> None:
    workflow = load_workflow(
        _config(
            [
                stage_spec("intake"),
                stage_spec("data_prep", needs=["intake"]),
                stage_spec("factor_build", needs=["intake", "data_prep"]),
                stage_spec("backtest", needs=["factor_build"]),
                stage_spec("approval", needs=["backtest"], human_gate=True, approval_action="publish_report"),
            ]
        )
    )
    assert len(workflow) == 5
    assert any(spec.name == "intake" for spec in workflow)
    assert "intake" in workflow and "nope" not in workflow
    assert workflow.stage("intake").handler == "intake"
    assert workflow.needs("factor_build") == ("intake", "data_prep")
    assert workflow.children("intake") == ("data_prep", "factor_build")
    assert workflow.descendants("intake") == (
        "data_prep",
        "factor_build",
        "backtest",
        "approval",
    )
    assert workflow.ancestors("backtest") == ("intake", "data_prep", "factor_build")
    assert workflow.downstream_closure(["data_prep", "backtest"], include_self=False) == (
        "factor_build",
        "backtest",
        "approval",
    )
    assert [spec.name for spec in workflow.human_gates()] == ["approval"]
    assert workflow.stage("approval").human_gate is True
    assert workflow.stage("approval").approval_action == "publish_report"
    assert handler_map(workflow) == {
        spec.name: spec.name for spec in workflow.order
    }
    with pytest.raises(ConfigError):
        workflow.stage("missing-stage")


def test_workflow_fingerprint_changes_with_stage_spec() -> None:
    base = _config([stage_spec("intake"), stage_spec("data_prep", needs=["intake"])])
    changed_timeout = _config(
        [stage_spec("intake", timeout_s=99.0), stage_spec("data_prep", needs=["intake"])]
    )
    changed_handler = _config(
        [stage_spec("intake", handler="other"), stage_spec("data_prep", needs=["intake"])]
    )
    assert load_workflow(base).fingerprint() == load_workflow(base).fingerprint()
    assert load_workflow(base).fingerprint() != load_workflow(changed_timeout).fingerprint()
    assert load_workflow(base).fingerprint() != load_workflow(changed_handler).fingerprint()
    assert load_workflow(base).as_dict()["stages"][0]["name"] == "intake"


def test_bare_and_nested_workflow_mappings_accepted() -> None:
    stages = [stage_spec("intake"), stage_spec("data_prep", needs=["intake"])]
    bare = workflow_section(stages)
    nested = {"workflow": bare}
    assert load_workflow(bare).stage_names == ("intake", "data_prep")
    assert load_workflow(nested).stage_names == ("intake", "data_prep")
    # util.paths.load_config_bundle keeps the YAML root key nested one level deeper.
    double_nested = {"workflow": {"workflow": bare}}
    assert load_workflow(double_nested).stage_names == ("intake", "data_prep")
    assert all(spec.name in STAGES for spec in load_workflow(bare).order)


def _expect_config_error(config, match: str | None = None) -> None:
    with pytest.raises(ConfigError, match=match):
        load_workflow(config)


def test_invalid_graphs_raise_config_error() -> None:
    # duplicate stage names
    _expect_config_error(
        _config([stage_spec("intake"), stage_spec("intake")]), "duplicate stage name"
    )
    # unknown dependency
    _expect_config_error(
        _config([stage_spec("intake"), stage_spec("data_prep", needs=["missing"])]),
        "unknown stage",
    )
    # unknown stage name (must stay within aicwf.contracts.STAGES)
    _expect_config_error(_config([stage_spec("not_a_real_stage")]), "not a known stage")
    # self dependency
    _expect_config_error(_config([stage_spec("data_prep", needs=["data_prep"])]), "itself")
    # cycle
    _expect_config_error(
        _config(
            [
                stage_spec("factor_build", needs=["backtest"]),
                stage_spec("backtest", needs=["factor_build"]),
            ]
        ),
        "cycle",
    )
    # empty stage list
    _expect_config_error({"workflow": {"name": "x", "version": "1", "stages": []}}, "non-empty")
    # missing workflow section
    _expect_config_error({"strategy": {}}, "missing the 'workflow' section")
    # non-mapping stages
    _expect_config_error({"workflow": {"stages": "nope"}}, "non-empty list")


def test_invalid_stage_fields_raise_config_error() -> None:
    _expect_config_error(_config([stage_spec("intake", retries=-1)]), "retries")
    _expect_config_error(_config([stage_spec("intake", retries=True)]), "retries")
    _expect_config_error(_config([stage_spec("intake", retries="1")]), "retries")
    _expect_config_error(_config([stage_spec("intake", timeout_s=0)]), "timeout_s")
    _expect_config_error(_config([stage_spec("intake", timeout_s=-5)]), "timeout_s")
    _expect_config_error(_config([stage_spec("intake", timeout_s="60")]), "timeout_s")
    _expect_config_error(
        {"workflow": {"stages": [{"name": "intake", "handler": "intake", "needs": "hypothesis"}]}},
        "needs",
    )
    _expect_config_error(
        {"workflow": {"stages": [{"name": "intake", "handler": ""}]}}, "handler"
    )
    _expect_config_error(
        {"workflow": {"stages": [{"name": "Intake"}]}}, "must match"
    )
    _expect_config_error(
        {"workflow": {"stages": [{"name": "intake", "handler": "intake", "human_gate": "yes"}]}},
        "human_gate",
    )
    _expect_config_error(
        {
            "workflow": {
                "stages": [{"name": "intake", "handler": "intake", "agent": "prompt.md"}],
            }
        },
        "agent",
    )
    _expect_config_error(
        {
            "workflow": {
                "stages": [{"name": "intake", "handler": "intake", "agent": {"prompt": 3}}],
            }
        },
        "agent.prompt",
    )
    _expect_config_error(
        {"workflow": {"execution": {"mode": "parallel"}, "stages": [{"name": "intake"}]}},
        "mode",
    )
    _expect_config_error(
        {
            "workflow": {
                "execution": {"stop_on_first_failure": False},
                "stages": [{"name": "intake"}],
            }
        },
        "stop_on_first_failure",
    )


def test_stage_spec_helpers() -> None:
    spec = StageSpec(
        name="backtest",
        handler="backtest",
        needs=("factor_build",),
        timeout_s=120.0,
        retries=2,
        produces=("metrics.json",),
        human_gate=False,
    )
    assert spec.max_attempts_hint == 3
    payload = spec.as_dict()
    assert payload["name"] == "backtest"
    assert payload["needs"] == ["factor_build"]
    assert payload["produces"] == ["metrics.json"]
