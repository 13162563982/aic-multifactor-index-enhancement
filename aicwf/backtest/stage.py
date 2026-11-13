"""Stage entrypoints: monthly backtest (``run_backtest``) and robustness suite.

Both handlers read their inputs exclusively through the :class:`RunContext`:

* ``data_prep``: ``panel`` and ``benchmark`` (daily, point-in-time up to as_of),
* ``factor_build``: ``factor_panel`` (composite scores + neutralised z-scores),
* ``hypothesis``: ``hypothesis`` (constraint proposals, clamped to config bounds),
* the frozen snapshot (``ctx.snapshot_dir``): ``universe`` for benchmark weights
  and board/ST metadata, ``prices`` for the exchange price-limit returns.

The backtest writes ``ledger.parquet``, ``holdings.parquet``, ``metrics.json`` and
``equity_curve.csv``; the robustness stage re-runs the same engine under the
perturbations described in :mod:`aicwf.backtest.robustness` and writes
``robustness.json`` + ``robustness_checks.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.backtest import metrics as metric_model
from aicwf.backtest import robustness as robustness_mod
from aicwf.backtest.engine import BacktestModel
from aicwf.contracts import (
    DataQualityError,
    RunContext,
    StageOutput,
    ValidationError,
)
from aicwf.factors import common
from aicwf.factors import library as factor_library
from aicwf.util.jsonio import read_json, write_text


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _universe_fallback(panel: pd.DataFrame) -> pd.DataFrame:
    """Minimal universe table when the snapshot universe is unavailable."""

    frame = panel.copy()
    frame["date"] = frame["date"].astype(str).str[:10]
    last = frame.sort_values("date").groupby("code", sort=True).last().reset_index()
    industry = last["industry"] if "industry" in last.columns else "UNKNOWN"
    return pd.DataFrame(
        {
            "code": last["code"].astype(str),
            "name": last["code"].astype(str),
            "industry": industry,
            "weight": 1.0 / max(len(last), 1),
            "is_st": False,
            "board": "main",
        }
    )


def load_backtest_inputs(ctx: RunContext) -> dict[str, Any]:
    """Load and validate every input the backtest / robustness stages share."""

    cfg = common.load_strategy_config(ctx)
    panel = common.load_panel(ctx)
    benchmark = common.load_benchmark(ctx)
    factor_path = common.find_upstream_artifact(
        ctx, "factor_build", ("factor_panel", "factor_panel.parquet"), filename="factor_panel.parquet"
    )
    factor_panel = pd.read_parquet(factor_path)
    missing = [column for column in ("date", "code", "composite_score", "eligible") if column not in factor_panel.columns]
    if missing:
        raise ValidationError(f"factor_panel.parquet is missing required columns {missing}")
    factor_panel = factor_panel.copy()
    factor_panel["date"] = factor_panel["date"].astype(str).str[:10]
    factor_panel["code"] = factor_panel["code"].astype(str)
    if factor_panel.empty:
        raise DataQualityError("factor_panel.parquet is empty; cannot run a backtest")

    universe = common.load_snapshot_table(ctx, "universe", columns=("code",))
    if universe is None:
        universe = _universe_fallback(panel)
    snapshot_prices = common.load_snapshot_table(ctx, "prices", columns=("date", "code", "pct_chg"))
    hypothesis = common.load_hypothesis(ctx)
    constraints, constraint_info = common.clamp_hypothesis_constraints(hypothesis, cfg)
    return {
        "cfg": cfg,
        "panel": panel,
        "benchmark": benchmark,
        "factor_panel": factor_panel,
        "universe": universe,
        "snapshot_prices": snapshot_prices,
        "hypothesis": hypothesis,
        "constraints": constraints,
        "constraint_info": constraint_info,
        "as_of": str(ctx.as_of)[:10],
    }


def _build_model(inputs: Mapping[str, Any], ctx: RunContext) -> BacktestModel:
    try:
        return BacktestModel(
            inputs["panel"],
            inputs["benchmark"],
            inputs["factor_panel"],
            inputs["universe"],
            inputs["cfg"],
            as_of=inputs["as_of"],
            snapshot_prices=inputs["snapshot_prices"],
        )
    except (ValidationError, DataQualityError):
        raise
    except Exception as exc:  # noqa: BLE001 - convert into a stage-level data error
        raise DataQualityError(f"cannot construct backtest model: {type(exc).__name__}: {exc}") from exc


def _run_baseline(inputs: Mapping[str, Any], model: BacktestModel):
    try:
        return model.run(constraints=inputs["constraints"])
    except ValueError as exc:
        raise DataQualityError(f"backtest produced no usable rebalances: {exc}") from exc


# ---------------------------------------------------------------------------
# Metric / constraint serialisation
# ---------------------------------------------------------------------------


def _constraint_payload(
    constraints: Mapping[str, Any], info: Mapping[str, Any], strategy: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "max_weight",
        "max_active_weight",
        "industry_active_limit",
        "turnover_limit",
        "te_target",
        "cash_buffer",
        "min_names",
        "benchmark_weight_cap",
        "weight_margin",
    ):
        if key in constraints:
            payload[key] = constraints[key]
    payload["bounds"] = info.get("bounds") or {}
    payload["sources"] = info.get("source") or {}
    payload["clamped"] = info.get("clamped") or {}
    payload["clamping_notes"] = list(info.get("notes") or [])
    payload["benchmark_weight_note"] = (
        "benchmark_weight = universe free-float proxy capped at benchmark_weight_cap and"
        " renormalised; raw CSI 300 weights above the name cap make a hard |active| <= "
        "max_active_weight constraint infeasible for megacaps."
    )
    return payload


def _write_equity_curve(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, float_format="%.10f", lineterminator="\n")
    write_text(path, text)


# ---------------------------------------------------------------------------
# Backtest stage
# ---------------------------------------------------------------------------


def run_backtest(ctx: RunContext) -> StageOutput:
    inputs = load_backtest_inputs(ctx)
    model = _build_model(inputs, ctx)
    result = _run_baseline(inputs, model)
    capital = float(
        common.section(inputs["cfg"], "backtest", "initial_capital", default=1.0e8) or 1.0e8
    )
    if not len(result.ledger):
        raise DataQualityError("backtest produced an empty ledger; refusing to publish metrics")

    ledger_path = ctx.artifact_path("backtest", "ledger.parquet")
    holdings_path = ctx.artifact_path("backtest", "holdings.parquet")
    equity_path = ctx.artifact_path("backtest", "equity_curve.csv")
    metrics_path = ctx.artifact_path("backtest", "metrics.json")
    common.write_frame_artifact(ledger_path, result.ledger)
    common.write_frame_artifact(holdings_path, result.holdings)
    _write_equity_curve(equity_path, result.equity_curve)

    metrics = dict(result.metrics)
    metrics["initial_capital"] = float(capital)
    metrics["rebalance_count"] = int(len(result.rebalances))
    metrics["turnover_average"] = float(
        np.mean([item["turnover"] for item in result.rebalances]) if result.rebalances else 0.0
    )
    metrics["turnover_annualized"] = float(metrics["turnover_average"] * 12.0)
    metrics["n_holdings_average"] = float(
        np.mean([item["n_names"] for item in result.rebalances]) if result.rebalances else 0.0
    )
    metrics["te_estimate_average"] = float(
        np.mean([item["te_estimate"] for item in result.rebalances]) if result.rebalances else 0.0
    )
    metrics_doc = {
        "initial_capital": float(capital),
        "metrics": metrics,
        "benchmark_metrics": dict(result.benchmark_metrics),
        "by_year": {str(year): dict(values) for year, values in result.by_year.items()},
        "constraints": _constraint_payload(inputs["constraints"], inputs["constraint_info"], inputs["cfg"]),
        "config_hash": common.config_bundle_hash(ctx),
        "warnings": {str(key): int(value) for key, value in result.warnings.items()},
        "rebalances": [
            {
                "decision_date": item["decision_date"],
                "execution_date": item["execution_date"],
                "signal_date": item["signal_date"],
                "turnover": round(float(item["turnover"]), 10),
                "n_names": int(item["n_names"]),
                "max_weight": round(float(item["max_weight"]), 10),
                "max_active": round(float(item["max_active"]), 10),
                "te_estimate": round(float(item["te_estimate"]), 10),
                "industry_active_max": round(float(item["industry_active_max"]), 10),
            }
            for item in result.rebalances
        ],
        "benchmark_weight_policy": "capped_free_float_proxy",
    }
    common.write_json_artifact(metrics_path, metrics_doc)

    summary = {
        "annualized_excess_return": result.metrics.get("annualized_excess_return"),
        "tracking_error": result.metrics.get("tracking_error"),
        "information_ratio": result.metrics.get("information_ratio"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "rebalance_count": len(result.rebalances),
        "initial_capital": float(capital),
        "ledger_rows": int(len(result.ledger)),
    }
    artifacts = {
        "ledger": common.rel_artifact(ledger_path),
        "holdings": common.rel_artifact(holdings_path),
        "metrics": common.rel_artifact(metrics_path),
        "equity_curve": common.rel_artifact(equity_path),
        "ledger.parquet": common.rel_artifact(ledger_path),
        "holdings.parquet": common.rel_artifact(holdings_path),
        "metrics.json": common.rel_artifact(metrics_path),
        "equity_curve.csv": common.rel_artifact(equity_path),
    }
    ctx.event(
        "backtest_completed",
        rebalances=len(result.rebalances),
        trades=int(len(result.ledger)),
        annualized_excess_return=summary["annualized_excess_return"],
    )
    return StageOutput(artifacts=artifacts, summary=summary)


# ---------------------------------------------------------------------------
# Robustness stage
# ---------------------------------------------------------------------------


def _composite_weights_from_inputs(inputs: Mapping[str, Any], ctx: RunContext) -> dict[str, float]:
    """Effective composite weights actually applied by the factor stage."""

    try:
        report_path = common.find_upstream_artifact(
            ctx, "factor_build", ("factor_report", "factor_report.json"), filename="factor_report.json"
        )
        report = read_json(report_path)
        weights = ((report.get("composite_weights") or {}).get("effective_signed")) or {}
        resolved = {str(key): float(value) for key, value in dict(weights).items() if abs(float(value)) > 0}
        if resolved:
            return resolved
    except Exception:  # noqa: BLE001 - fall back to deriving from the hypothesis below
        pass
    catalog = factor_library.factor_catalog(inputs["cfg"])
    hypothesis_weights = dict((inputs["hypothesis"] or {}).get("factor_weights") or {})
    if not hypothesis_weights:
        hypothesis_weights = dict(
            common.section(inputs["cfg"], "factors", "composite", "default_weights", default={}) or {}
        )
    total = sum(abs(float(value)) for value in hypothesis_weights.values()) or 1.0
    return {
        str(factor_id): float(value) / total * float(catalog[factor_id]["direction"])
        for factor_id, value in hypothesis_weights.items()
        if factor_id in catalog and abs(float(value)) > 0
    }


def run_robustness(ctx: RunContext) -> StageOutput:
    inputs = load_backtest_inputs(ctx)
    model = _build_model(inputs, ctx)
    baseline = _run_baseline(inputs, model)
    composite_weights = _composite_weights_from_inputs(inputs, ctx)

    robustness_cfg = dict(inputs["cfg"].get("robustness") or {})
    document, checks = robustness_mod.run_robustness_suite(
        model=model,
        baseline=baseline,
        composite_weights=composite_weights,
        cfg=inputs["cfg"],
        seed=int(getattr(ctx, "seed", 0) or 0),
        constraints=inputs["constraints"],
        hypothesis=inputs["hypothesis"],
    )
    document["composite_weights"] = {str(key): float(value) for key, value in composite_weights.items()}
    document["constraints"] = {
        key: inputs["constraints"].get(key)
        for key in ("max_weight", "max_active_weight", "te_target", "turnover_limit")
    }

    # Cross-check the recomputed baseline against the published backtest metrics.
    try:
        upstream_metrics_path = common.find_upstream_artifact(
            ctx, "backtest", ("metrics", "metrics.json"), filename="metrics.json"
        )
        upstream = read_json(upstream_metrics_path)
        declared = (upstream.get("metrics") or {}) if isinstance(upstream, dict) else {}
        for key in ("annualized_excess_return", "information_ratio", "tracking_error"):
            left = declared.get(key)
            right = baseline.metrics.get(key)
            if left is None or right is None:
                continue
            try:
                if abs(float(left) - float(right)) > 1e-6 * max(1.0, abs(float(right))):
                    document.setdefault("warnings", []).append(
                        f"recomputed baseline {key}={float(right):.10g} differs from published backtest"
                        f" metrics {float(left):.10g}"
                    )
            except (TypeError, ValueError):
                continue
    except Exception:  # noqa: BLE001 - the comparison is advisory
        pass

    robustness_path = ctx.artifact_path("robustness", "robustness.json")
    checks_path = ctx.artifact_path("robustness", "robustness_checks.json")
    common.write_json_artifact(robustness_path, document)
    common.write_json_artifact(checks_path, checks)

    passed = [check["name"] for check in checks if check.get("status") == "pass"]
    failed = [check["name"] for check in checks if check.get("status") != "pass"]
    summary = {
        "passed_checks": len(passed),
        "failed_checks": len(failed),
        "concerns": list(document.get("summary", {}).get("concerns") or []),
        "checks": [check["name"] for check in checks],
    }
    artifacts = {
        "robustness": common.rel_artifact(robustness_path),
        "robustness_checks": common.rel_artifact(checks_path),
        "robustness.json": common.rel_artifact(robustness_path),
        "robustness_checks.json": common.rel_artifact(checks_path),
    }
    ctx.event("robustness_completed", passed=len(passed), failed=len(failed))
    return StageOutput(artifacts=artifacts, summary=summary)


#: Controller registry mapping (the Lead wires these in ``aicwf/pipeline.py``).
HANDLERS = {"backtest": run_backtest, "robustness": run_robustness}


__all__ = ["HANDLERS", "load_backtest_inputs", "run_backtest", "run_robustness"]
