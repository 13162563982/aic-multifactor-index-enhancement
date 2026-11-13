"""Shared helpers for the agent test modules.

This file contains **no test functions**; it only builds deterministic mini artifacts so
the stage tests run fully offline against hand-built ``RunContext`` objects.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from aicwf.contracts import LLMResponse, RunContext, StageOutput
from aicwf.util.jsonio import read_json, write_json, write_parquet_atomic, write_text

AS_OF = dt.date(2024, 12, 31)
AS_OF_STR = "2024-12-31"

FACTOR_IDS = ("value_ep", "quality_roe", "momentum_12_1", "low_vol_60d")
FACTOR_CATALOG = {
    "value_ep": {"direction": 1.0, "category": "value", "description": "trailing EPS / price"},
    "quality_roe": {"direction": 1.0, "category": "quality", "description": "EPS / BPS"},
    "momentum_12_1": {
        "direction": 1.0,
        "category": "momentum",
        "description": "12-1 month return",
    },
    "low_vol_60d": {
        "direction": -1.0,
        "category": "risk",
        "description": "60 day realized volatility",
    },
}
DEFAULT_WEIGHTS = {
    "value_ep": 0.35,
    "quality_roe": 0.30,
    "momentum_12_1": 0.25,
    "low_vol_60d": 0.10,
}

WORKFLOW_AGENTS = [
    {
        "name": "hypothesis",
        "agent": {
            "role": "researcher",
            "prompt": "prompts/researcher.md",
            "skill": "skills/hypothesis_spec/SKILL.md",
            "schema": "aicwf/schemas/hypothesis.schema.json",
            "max_tokens": 1800,
        },
    },
    {
        "name": "factor_review",
        "agent": {
            "role": "factor_analyst",
            "prompt": "prompts/factor_analyst.md",
            "skill": "skills/factor_diagnostics/SKILL.md",
            "schema": "aicwf/schemas/factor_flags.schema.json",
            "max_tokens": 1400,
        },
    },
    {
        "name": "robustness_review",
        "agent": {
            "role": "risk_critic",
            "prompt": "prompts/risk_critic.md",
            "skill": "skills/risk_review/SKILL.md",
            "schema": None,
            "max_tokens": 1600,
        },
    },
    {
        "name": "verification",
        "agent": {
            "role": "verifier",
            "prompt": "prompts/verifier.md",
            "skill": "skills/pit_and_cutoff/SKILL.md",
            "schema": "aicwf/schemas/verification.schema.json",
            "max_tokens": 1200,
        },
    },
    {
        "name": "report",
        "agent": {
            "role": "report_writer",
            "prompt": "prompts/report_writer.md",
            "skill": "skills/report_template/SKILL.md",
            "schema": None,
            "max_tokens": 2200,
        },
    },
]


def mini_config(cache_dir: Path | str | None = None) -> dict[str, Any]:
    model = {
        "default": {
            "backend": "ollama",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1600,
            "num_ctx": 8192,
            "timeout_s": 5,
            "think": False,
            "format_json": True,
            "keep_alive": "10m",
            "retry": {"attempts": 3, "backoff_seconds": [0.0, 0.0, 0.0]},
        },
        "replay": {
            "cache_dir": str(cache_dir) if cache_dir is not None else "evidence/llm_cache",
            "strict": True,
            "record": False,
        },
        "openai_compat": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": "AICWF_TEST_API_KEY",
            "model": "",
            "timeout_s": 5,
        },
        "prompts": {"dir": "prompts", "skill_dir": "skills"},
        "agent_loop": {
            "max_tool_rounds": 3,
            "max_retries_invalid_json": 2,
            "repair_temperature": 0.0,
            "max_read_chars": 12000,
        },
    }
    strategy = {
        "universe": {"min_price_coverage": 0.90},
        "data": {"snapshot": "data/snapshots/does-not-exist"},
        "factors": {
            "min_cross_section": 100,
            "catalog": FACTOR_CATALOG,
            "composite": {"default_weights": DEFAULT_WEIGHTS},
        },
        "backtest": {
            "rebalance": "monthly",
            "constraints": {
                "max_weight": 0.03,
                "max_active_weight": 0.015,
                "te_target": 0.05,
                "turnover_limit": 0.30,
            },
        },
    }
    return {
        "model": model,
        "strategy": strategy,
        "workflow": {"stages": WORKFLOW_AGENTS},
    }


def make_ctx(
    tmp_path: Path,
    *,
    llm: Any = None,
    as_of: dt.date = AS_OF,
    config: dict[str, Any] | None = None,
    llm_mode: str = "null",
    events: Any = None,
    run_id: str = "test-run",
) -> RunContext:
    run_dir = tmp_path / "outputs" / "runs" / run_id
    return RunContext(
        run_id=run_id,
        as_of=as_of,
        run_dir=run_dir,
        config=config if config is not None else mini_config(cache_dir=tmp_path / "cache"),
        llm=llm,
        llm_mode=llm_mode,
        events=events,
    )


# ---------------------------------------------------------------------------
# Deterministic mini artifacts
# ---------------------------------------------------------------------------


def hypothesis_payload(as_of: dt.date = AS_OF) -> dict[str, Any]:
    return {
        "thesis": (
            "Cheap, profitable and momentum-leading CSI 300 members should earn positive "
            "excess return versus the benchmark because value and quality premia are "
            "compensated in the A-share cross-section and institutional flows reinforce "
            "medium-term momentum."
        ),
        "factor_weights": dict(DEFAULT_WEIGHTS),
        "rebalance": "monthly",
        "constraints": {
            "max_weight": 0.03,
            "max_active_weight": 0.015,
            "te_target": 0.05,
            "turnover_limit": 0.30,
        },
        "falsification": [
            "Reject if the full-sample RankIC t-statistic is below 2.0.",
            "Reject if annualised excess return is negative under 3x costs.",
        ],
        "information_cutoff": as_of.isoformat(),
        "source_limits": [
            "frozen CSI 300 snapshot only; no network access during the run",
            "fundamentals are usable only after their announcement date",
        ],
    }


def factor_report_payload() -> dict[str, Any]:
    return {
        "coverage": {factor: 0.96 for factor in FACTOR_IDS},
        "rank_ic": {"value_ep": 0.041, "quality_roe": 0.028, "momentum_12_1": 0.019, "low_vol_60d": 0.012},
        "ic_t_stat": {"value_ep": 3.1, "quality_roe": 2.4, "momentum_12_1": 1.6, "low_vol_60d": 1.2},
        "autocorrelation": {"value_ep": 0.62, "quality_roe": 0.55, "momentum_12_1": 0.31, "low_vol_60d": 0.44},
        "correlation_matrix": {
            "value_ep": {"value_ep": 1.0, "quality_roe": 0.21, "momentum_12_1": -0.12, "low_vol_60d": -0.08},
            "quality_roe": {"value_ep": 0.21, "quality_roe": 1.0, "momentum_12_1": 0.09, "low_vol_60d": 0.15},
            "momentum_12_1": {"value_ep": -0.12, "quality_roe": 0.09, "momentum_12_1": 1.0, "low_vol_60d": 0.33},
            "low_vol_60d": {"value_ep": -0.08, "quality_roe": 0.15, "momentum_12_1": 0.33, "low_vol_60d": 1.0},
        },
        "composite_weights": dict(DEFAULT_WEIGHTS),
        "warnings": ["winsorized 34 cross-sectional outliers in 2019-03"],
    }


def metrics_payload() -> dict[str, Any]:
    return {
        "metrics": {
            "annualized_return": 0.0812,
            "annualized_excess_return": 0.0234,
            "tracking_error": 0.049,
            "information_ratio": 0.478,
            "max_drawdown": -0.184,
            "annualized_turnover": 2.8,
        },
        "benchmark_metrics": {
            "annualized_return": 0.0576,
            "max_drawdown": -0.221,
        },
        "by_year": [
            {"year": 2019, "strategy": 0.211, "benchmark": 0.183},
            {"year": 2020, "strategy": 0.142, "benchmark": 0.121},
        ],
        "constraints": {
            "max_weight": 0.03,
            "max_active_weight": 0.015,
            "te_target": 0.05,
            "turnover_limit": 0.30,
        },
        "config_hash": "a" * 64,
    }


def robustness_payload() -> dict[str, Any]:
    return {
        "subperiods": [
            {"start": "2018-01-02", "end": "2020-12-31", "excess_return": 0.0312, "information_ratio": 0.61},
            {"start": "2021-01-04", "end": "2023-12-29", "excess_return": 0.0148, "information_ratio": 0.31},
            {"start": "2024-01-02", "end": "2024-12-31", "excess_return": 0.0196, "information_ratio": 0.42},
        ],
        "cost_sensitivity": [
            {"multiplier": 0.0, "annualized_excess_return": 0.0301},
            {"multiplier": 1.0, "annualized_excess_return": 0.0234},
            {"multiplier": 3.0, "annualized_excess_return": 0.0091},
        ],
        "leave_one_out": [
            {"factor": "momentum_12_1", "annualized_excess_return": 0.0121},
            {"factor": "value_ep", "annualized_excess_return": 0.0135},
        ],
        "parameter_grid": [
            {"te_target": 0.03, "annualized_excess_return": 0.0161},
            {"te_target": 0.05, "annualized_excess_return": 0.0234},
            {"te_target": 0.07, "annualized_excess_return": 0.0282},
        ],
        "placebo": {"repetitions": 20, "strategy_percentile": 95.0, "p_value": 0.05},
        "bias_stress": {"drop_top_k": 30, "annualized_excess_return": 0.0112},
        "deflated_sharpe": 1.12,
        "summary": {"pass": 4, "concerns": ["cost sensitivity", "placebo tail"]},
    }


def robustness_checks_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": "subperiod_consistency",
            "status": "PASS",
            "value": 3,
            "threshold": 2,
            "detail": "all three subperiods have positive excess return",
        },
        {
            "name": "cost_3x",
            "status": "FAIL",
            "value": 0.0091,
            "threshold": 0.01,
            "detail": "annualised excess return at 3x costs is below 1%",
        },
        {
            "name": "placebo",
            "status": "PASS",
            "value": 0.05,
            "threshold": 0.10,
            "detail": "strategy beats 19 of 20 random-score runs",
        },
    ]


def risk_report_payload() -> dict[str, Any]:
    return {
        "ex_ante_te": 0.048,
        "industry_active": {"银行": 0.0075, "电子": -0.0061, "医药": 0.0044},
        "factor_active": {"value": 0.0121, "momentum": 0.0084},
        "turnover": 0.274,
        "n_names": 95,
        "optimizer": {
            "status": "ok",
            "iterations": 210,
            "te_lambda": 0.031,
            "fallback_used": False,
        },
        "constraints": {"te_target": 0.05, "max_active_weight": 0.015},
    }


def verification_payload(as_of: dt.date = AS_OF, verdict: str = "PASS") -> dict[str, Any]:
    """Mirror the real verifier's artifact shape (lowercase check statuses)."""

    metrics = metrics_payload()
    pass_checks = [
        {"id": "artifact_schema", "status": "pass", "detail": "all artifacts validate", "warnings": []},
        {"id": "artifact_hashes", "status": "pass", "detail": "declared hashes match files", "warnings": []},
        {"id": "ledger_reconcile", "status": "pass", "detail": "cash and shares reconcile"},
        {"id": "weights_bounds", "status": "pass", "detail": "weights within bounds"},
        {"id": "pit_fundamentals", "status": "pass", "detail": "notice_date <= decision date"},
    ]
    fail_checks = [
        {"id": "artifact_schema", "status": "pass", "detail": "all artifacts validate", "warnings": []},
        {
            "id": "ledger_reconcile",
            "status": "fail",
            "detail": "cash discrepancy of 1200 CNY",
            "discrepancies": [{"check": "ledger_reconcile", "detail": "cash mismatch"}],
        },
    ]
    checks = pass_checks if verdict == "PASS" else fail_checks
    return {
        "verdict": verdict,
        "run_id": "test-run",
        "as_of": as_of.isoformat(),
        "checks": checks,
        "recomputed": {
            "metrics": dict(metrics["metrics"]),
            "benchmark_metrics": dict(metrics["benchmark_metrics"]),
            "weights_sum": 1.0,
            "max_abs_active_weight": 0.0148,
        },
        "discrepancies": []
        if verdict == "PASS"
        else [{"check": "ledger_reconcile", "detail": "cash discrepancy of 1200 CNY"}],
        "artifact_hashes": {
            "ledger.parquet": "b" * 64,
            "holdings.parquet": "c" * 64,
            "metrics.json": "d" * 64,
        },
        "tolerances": {
            "metrics_relative": 1e-06,
            "weights_absolute": 1e-06,
            "cash_absolute": 0.01,
        },
        "snapshot": {"snapshot_hash": "e" * 64, "as_of": as_of.isoformat(), "path": "tests/fixtures/mini_snapshot"},
        "verifier": {
            "name": "independent-verifier",
            "version": "1.0.0",
            "mode": "deterministic_recompute",
            "llm_mode": "null",
            "checks_total": len(checks),
            "checks_failed": sum(1 for check in checks if check.get("status") != "pass"),
        },
    }


def equity_curve_csv() -> str:
    lines = ["date,nav,benchmark_nav,active_nav,daily_return,benchmark_return,excess_return,drawdown"]
    nav = 1.0
    benchmark = 1.0
    active = 1.0
    for month in range(12):
        nav *= 1.006
        benchmark *= 1.004
        active = nav / benchmark
        drawdown = -0.02 - 0.005 * (month % 3)
        lines.append(
            f"2024-{month + 1:02d}-28,{nav:.6f},{benchmark:.6f},{active:.6f},0.006,0.004,"
            f"{active - 1:.6f},{drawdown:.6f}"
        )
    return "\n".join(lines) + "\n"


def target_weights_frame() -> Any:
    import pandas as pd

    rows = []
    industries = ["银行", "电子", "医药"]
    for index, code in enumerate(["600519.SH", "000001.SZ", "300750.SZ"] * 3):
        weight = 0.01
        benchmark_weight = 0.008
        rows.append(
            {
                "date": "2024-12-31",
                "code": code,
                "weight": weight,
                "benchmark_weight": benchmark_weight,
                "active_weight": weight - benchmark_weight,
                "industry": industries[index % len(industries)],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Artifact registration helpers
# ---------------------------------------------------------------------------


def _write_artifact(ctx: RunContext, stage: str, filename: str, payload: Any) -> str:
    path = ctx.artifact_path(stage, filename)
    if isinstance(payload, str):
        write_text(path, payload)
    else:
        write_json(path, payload)
    return str(path)


def register_stage(ctx: RunContext, stage: str, artifacts: dict[str, str], summary: dict[str, Any]) -> None:
    ctx.register(stage, StageOutput(artifacts=dict(artifacts), summary=dict(summary)))


def populate_upstream(ctx: RunContext, *, verdict: str = "PASS") -> None:
    """Create and register the mini upstream stage outputs used by the stage tests."""

    as_of = ctx.as_of if isinstance(ctx.as_of, dt.date) else AS_OF
    run_request = _write_artifact(
        ctx,
        "intake",
        "run_request.json",
        {
            "run_id": ctx.run_id,
            "as_of": as_of.isoformat(),
            "snapshot": "tests/fixtures/mini_snapshot",
            "snapshot_hash": "e" * 64,
            "config_hash": "f" * 64,
            "llm_mode": ctx.llm_mode,
            "seed": 0,
            "workflow_version": "1.0.0",
        },
    )
    register_stage(ctx, "intake", {"run_request": run_request}, {"as_of": as_of.isoformat()})

    data_quality = _write_artifact(
        ctx,
        "data_prep",
        "data_quality.json",
        {
            "snapshot_hash": "e" * 64,
            "rows": 3000,
            "universe_size": 4,
            "coverage": 0.98,
            "checks": [{"id": "price_coverage", "status": "PASS", "detail": "98% coverage"}],
            "warnings": [],
        },
    )
    pit_audit = _write_artifact(
        ctx,
        "data_prep",
        "pit_audit.json",
        {"as_of": as_of.isoformat(), "violations": [], "max_panel_date": as_of.isoformat()},
    )
    register_stage(
        ctx,
        "data_prep",
        {"data_quality": data_quality, "pit_audit": pit_audit},
        {"rows": 3000, "universe_size": 4, "max_date": as_of.isoformat(), "coverage": 0.98, "warnings": []},
    )

    hypothesis = hypothesis_payload(as_of)
    hypothesis_path = _write_artifact(ctx, "hypothesis", "hypothesis.json", hypothesis)
    rationale_path = _write_artifact(
        ctx, "hypothesis", "rationale.md", "# Hypothesis rationale\n\nInformation cutoff: %s\n" % as_of
    )
    register_stage(
        ctx,
        "hypothesis",
        {"hypothesis": hypothesis_path, "rationale": rationale_path},
        {
            "thesis": hypothesis["thesis"],
            "factor_weights": hypothesis["factor_weights"],
            "rebalance": hypothesis["rebalance"],
            "constraints": hypothesis["constraints"],
            "information_cutoff": as_of.isoformat(),
        },
    )

    factor_report = factor_report_payload()
    factor_report_path = _write_artifact(ctx, "factor_build", "factor_report.json", factor_report)
    register_stage(
        ctx,
        "factor_build",
        {"factor_report": factor_report_path},
        {
            "factor_weights": dict(DEFAULT_WEIGHTS),
            "coverage": 0.96,
            "ic_mean": 0.025,
            "eligible_rows": 240,
        },
    )

    metrics = metrics_payload()
    metrics_path = _write_artifact(ctx, "backtest", "metrics.json", metrics)
    equity_curve_path = _write_artifact(ctx, "backtest", "equity_curve.csv", equity_curve_csv())
    register_stage(
        ctx,
        "backtest",
        {"metrics": metrics_path, "equity_curve": equity_curve_path},
        {
            "annualized_excess_return": metrics["metrics"]["annualized_excess_return"],
            "tracking_error": metrics["metrics"]["tracking_error"],
            "information_ratio": metrics["metrics"]["information_ratio"],
            "max_drawdown": metrics["metrics"]["max_drawdown"],
            "rebalance_count": 11,
        },
    )

    robustness = robustness_payload()
    checks = robustness_checks_payload()
    robustness_path = _write_artifact(ctx, "robustness", "robustness.json", robustness)
    checks_path = _write_artifact(ctx, "robustness", "robustness_checks.json", checks)
    register_stage(
        ctx,
        "robustness",
        {"robustness": robustness_path, "robustness_checks": checks_path},
        {
            "passed_checks": sum(1 for check in checks if check["status"] == "PASS"),
            "failed_checks": sum(1 for check in checks if check["status"] != "PASS"),
            "concerns": robustness["summary"]["concerns"],
        },
    )

    portfolio = risk_report_payload()
    risk_report_path = _write_artifact(ctx, "portfolio", "risk_report.json", portfolio)
    weights_path = _write_artifact(ctx, "portfolio", "target_weights.parquet", "")
    write_parquet_atomic(weights_path, target_weights_frame())
    register_stage(
        ctx,
        "portfolio",
        {"risk_report": risk_report_path, "target_weights": weights_path},
        {
            "date": as_of.isoformat(),
            "n_names": portfolio["n_names"],
            "ex_ante_te": portfolio["ex_ante_te"],
            "turnover": portfolio["turnover"],
            "fallback_used": False,
        },
    )

    verification = verification_payload(as_of, verdict=verdict)
    verification_path = _write_artifact(ctx, "verification", "verification.json", verification)
    register_stage(
        ctx,
        "verification",
        {"verification": verification_path},
        {
            "verdict": verification["verdict"],
            "checks_total": len(verification["checks"]),
            "checks_failed": sum(1 for check in verification["checks"] if check["status"] == "FAIL"),
        },
    )


# ---------------------------------------------------------------------------
# Scripted LLM used by runtime/stage tests
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Deterministic LLM stub that returns queued raw replies and records calls."""

    def __init__(self, responses: list[Any] | None = None, *, default: Any = None) -> None:
        self.model = "scripted-model"
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Any = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "format_json": format_json,
            }
        )
        if self._responses:
            item = self._responses.pop(0)
        elif self._default is not None:
            item = self._default
        else:
            item = "{}"
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(
            text=text,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            cached=False,
            latency_ms=1,
        )


class FakeEvents:
    """Minimal event sink matching ``RunContext.event`` expectations."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def emit(self, kind: str, *, level: str = "info", run_id: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {"kind": kind, "level": level, "run_id": run_id, "payload": payload or {}}
        self.records.append(record)
        return record


# ---------------------------------------------------------------------------
# Markdown helpers for markdown-output tests
# ---------------------------------------------------------------------------


def valid_risk_markdown(as_of: dt.date = AS_OF) -> str:
    return (
        "# Robustness critique\n\n"
        f"Information cutoff: {as_of.isoformat()}.\n\n"
        "## Verdict on robustness\n\n"
        "The evidence is fragile: the 3x cost case reports annualized excess return of "
        "0.0091, below the 0.01 threshold, and the placebo p-value is 0.05. "
        "Hypothesis (speculation, not a finding): cost sensitivity may dominate in the "
        "next regime.\n\n"
        "## Failed or weak checks\n\n"
        "- cost_3x: 0.0091 against a 0.01 threshold.\n"
        "- deflated_sharpe: 1.12 leaves limited margin after multiple trials.\n\n"
        "## Threats to validity\n\n"
        "Subperiod excess returns 0.0312 and 0.0148 suggest regime dependence. "
        "Tracking error of 0.049 is close to the 0.05 target.\n\n"
        "## What would change our mind\n\n"
        "An out-of-sample cost-3x excess return above the 0.01 robustness threshold with "
        "a placebo p-value at or below 0.05.\n"
    )


def valid_report_markdown(as_of: dt.date = AS_OF) -> str:
    return (
        "# CSI 300 multi-factor index enhancement: verification-ready evidence\n\n"
        f"Information cutoff: {as_of.isoformat()}.\n\n"
        "## Executive summary\n\n"
        "The frozen backtest reports annualized excess return of 0.0234 with tracking "
        "error 0.049 and information ratio 0.478. Independent verification verdict: PASS. "
        "This is research documentation, not investment advice.\n\n"
        "## Hypothesis\n\n"
        "A tilt toward cheap, profitable and momentum-leading CSI 300 members with monthly "
        "rebalancing and a 0.05 tracking-error target.\n\n"
        "## Data and point-in-time discipline\n\n"
        "The frozen snapshot is used; fundamentals are available only from notice_date. "
        "The information cutoff is %s.\n\n" % as_of.isoformat()
        + "## Factor construction\n\n"
        "RankIC is positive for value_ep at 0.041 and quality_roe at 0.028.\n\n"
        "## Backtest results\n\n"
        "Annualized return 0.0812 versus benchmark 0.0576; max drawdown -0.184. The "
        "companion table is metrics_table.md.\n\n"
        "## Robustness\n\n"
        "Three subperiods are positive, but the 3x cost case reports 0.0091, below its "
        "robustness threshold, and deflated_sharpe is 1.12.\n\n"
        "## Portfolio construction and risk\n\n"
        "Ex-ante tracking error 0.048 with 95 names and turnover 0.274.\n\n"
        "## Independent verification\n\n"
        "Verdict PASS: every required check passed, no discrepancies were reported, and "
        "artifact hashes were recomputed.\n\n"
        "## Limitations and information cutoff\n\n"
        f"Information cutoff {as_of.isoformat()}. The universe is current CSI 300 "
        "membership (survivorship caveat), costs are modelled, and past performance is not "
        "indicative. This is research documentation, not investment advice.\n"
    )
