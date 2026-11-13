"""Stage entrypoint: build the monthly point-in-time factor panel.

Reads (through the run context, never through global state):

* ``data_prep``: ``panel`` (daily, data-prep schema) — required,
* ``hypothesis``: ``hypothesis`` (factor weights) — optional outside the full
  workflow, defaults to the catalog weights in ``config/strategy.yaml``,
* snapshot directory (``ctx.snapshot_dir`` or ``strategy.data.snapshot``):
  ``fundamentals.parquet`` / ``universe.parquet`` — optional; when unavailable
  the fundamental factors are reported as uncovered instead of failing.

Writes:

* ``factor_panel.parquet`` — ``date, code`` + ``<id>_raw``/``<id>_z`` for every
  catalog factor + ``composite_score`` + ``eligible``,
* ``factor_report.json`` — coverage, RankIC, IC t-stats, autocorrelation,
  correlation matrix, composite weights, warnings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.contracts import (
    DataQualityError,
    RunContext,
    StageOutput,
    ValidationError,
    parse_date,
)
from aicwf.factors import common
from aicwf.factors import diagnostics as factor_diagnostics
from aicwf.factors import library as factor_library
from aicwf.factors import neutralize as factor_neutralize
from aicwf.factors import preprocess as factor_preprocess


def resolve_composite_weights(
    hypothesis: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    fallback_to_catalog: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Resolve hypothesis factor weights into normalised effective weights.

    * unknown factor ids raise :class:`ValidationError` (never silently dropped),
    * weights are normalised to sum of absolute value 1,
    * the catalog ``direction`` sign is applied so the effective weight is what
      the composite actually multiplies the neutralised z-score with.
    """

    source = "config:factors.composite.default_weights"
    raw: dict[str, float] = {}
    hypothesis = hypothesis or {}
    hypothesis_weights = hypothesis.get("factor_weights")
    if isinstance(hypothesis_weights, Mapping) and hypothesis_weights:
        source = "hypothesis.factor_weights"
        raw = {}
        for factor_id, value in hypothesis_weights.items():
            if factor_id not in catalog:
                raise ValidationError(
                    f"hypothesis.factor_weights contains unknown factor id {factor_id!r};"
                    f" supported ids are {sorted(catalog)}"
                )
            raw[str(factor_id)] = float(value)
    elif fallback_to_catalog:
        default_weights = common.section(cfg, "factors", "composite", "default_weights") or {}
        raw = {str(key): float(value) for key, value in dict(default_weights).items()}
        unknown = sorted(set(raw) - set(catalog))
        if unknown:
            raise ValidationError(f"config factors.composite.default_weights has unknown ids {unknown}")

    weights = {factor_id: float(raw.get(factor_id, 0.0)) for factor_id in catalog}
    total_abs = float(sum(abs(value) for value in weights.values()))
    if not np.isfinite(total_abs) or total_abs <= 0.0:
        raise ValidationError("composite factor weights are all zero or non-finite")
    normalised = {factor_id: value / total_abs for factor_id, value in weights.items() if value != 0.0}
    effective = {
        factor_id: normalised[factor_id] * float(catalog[factor_id]["direction"])
        for factor_id in normalised
    }
    info = {
        "source": source,
        "raw": raw,
        "normalised": normalised,
        "effective_signed": effective,
        "directions": {factor_id: float(meta["direction"]) for factor_id, meta in catalog.items()},
    }
    return normalised, effective, info


def _flatten_matrix(matrix: pd.DataFrame, formation_dates: list[str], codes: list[str]) -> np.ndarray:
    return matrix.reindex(index=formation_dates, columns=codes).to_numpy(dtype=float).ravel()


def _factor_panel_frame(
    data: factor_library.MonthlyFactorData,
    neutral_scores: Mapping[str, pd.DataFrame],
    composite_score: pd.DataFrame,
) -> pd.DataFrame:
    dates = data.formation_dates
    codes = data.codes
    base = data.long_frame()
    keep = base["close"].notna() & np.isfinite(base["close"].to_numpy(dtype=float))
    frame = base.loc[keep, ["date", "code", "eligible"]].copy()
    for factor_id in factor_library.FACTOR_IDS:
        frame[f"{factor_id}_raw"] = base.loc[keep, factor_id].to_numpy(dtype=float)
        frame[f"{factor_id}_z"] = _flatten_matrix(neutral_scores[factor_id], dates, codes)[
            keep.to_numpy(dtype=bool)
        ]
    frame["composite_score"] = _flatten_matrix(composite_score, dates, codes)[keep.to_numpy(dtype=bool)]
    frame["eligible"] = frame["eligible"].astype(bool)
    ordered = ["date", "code"]
    for factor_id in factor_library.FACTOR_IDS:
        ordered.extend([f"{factor_id}_raw", f"{factor_id}_z"])
    ordered.extend(["composite_score", "eligible"])
    return frame.loc[:, ordered].reset_index(drop=True)


def run_factor_build(ctx: RunContext) -> StageOutput:
    cfg = common.load_strategy_config(ctx)
    catalog = factor_library.factor_catalog(cfg)
    as_of = parse_date(ctx.as_of)

    panel = common.load_panel(ctx)
    fundamentals = common.load_snapshot_table(
        ctx,
        "fundamentals",
        columns=("code", "report_date", "notice_date", "eps", "bps"),
    )
    universe = common.load_snapshot_table(ctx, "universe", columns=("code",))
    warnings: list[str] = []
    if fundamentals is None:
        warnings.append(
            "snapshot fundamentals unavailable: value/quality/growth factors have zero coverage"
        )
    if universe is None:
        warnings.append("snapshot universe unavailable: factor eligibility relies on the panel only")

    min_listed_days = int(common.section(cfg, "universe", "min_listed_days", default=250) or 0)
    vol_window = int(common.section(cfg, "factors", "windows", "vol_days", default=60) or 60)
    short_window = int(common.section(cfg, "factors", "windows", "short_days", default=20) or 20)

    data = factor_library.build_monthly_factor_data(
        panel,
        fundamentals=fundamentals,
        universe=universe,
        end_date=as_of,
        min_listed_days=min_listed_days,
        vol_window=vol_window,
        short_window=short_window,
    )
    warnings.extend(data.warnings)

    eligible = data.eligible
    eligible_rows = int(eligible.to_numpy(dtype=bool).sum())
    if eligible_rows == 0:
        raise DataQualityError(
            "no eligible (index member, trading, seasoned) names on the monthly factor grid;"
            " cannot build factors — check the data-prep panel and universe snapshot"
        )

    winsor_cfg = common.section(cfg, "factors", "winsorize") or {}
    lower = float(winsor_cfg.get("lower", 0.01))
    upper = float(winsor_cfg.get("upper", 0.99))
    z_scores: dict[str, pd.DataFrame] = {}
    raw_masked: dict[str, pd.DataFrame] = {}
    for factor_id in factor_library.FACTOR_IDS:
        raw = data.raw[factor_id].where(eligible)
        raw_masked[factor_id] = raw
        z_scores[factor_id] = factor_preprocess.preprocess_factor(raw, lower=lower, upper=upper)

    neutral_min_obs = int(common.section(cfg, "factors", "neutralize_min_obs", default=5) or 5)
    neutral_scores, neutral_diag = factor_neutralize.neutralise_factors(
        z_scores, data.industry, data.log_size, min_obs=neutral_min_obs
    )

    normalised, effective, weight_info = resolve_composite_weights(
        common.load_hypothesis(ctx), cfg, catalog
    )
    composite = _composite_score(effective, neutral_scores, eligible)

    composite_ic = factor_diagnostics.rank_ic_series(
        composite,
        data.forward_returns,
        eligible,
        min_names=max(3, int(common.section(cfg, "factors", "ic_min_names", default=5) or 5)),
    )
    composite_ic_summary = factor_diagnostics.summarise_ic(composite_ic)

    rank_ic_summary, rank_ic_series = factor_diagnostics.rank_ic_table(
        neutral_scores, data.forward_returns, eligible
    )
    autocorrelation = factor_diagnostics.autocorrelation_table(neutral_scores, eligible)
    correlation = factor_diagnostics.correlation_matrix(neutral_scores, eligible)
    coverage = factor_diagnostics.coverage_report(data.raw, neutral_scores, eligible)

    min_cross_section = int(common.section(cfg, "factors", "min_cross_section", default=100) or 0)
    if min_cross_section > 0:
        counts = eligible.sum(axis=1)
        thin_dates = [str(date) for date, count in counts.items() if int(count) < min_cross_section]
        if thin_dates:
            warnings.append(
                f"{len(thin_dates)} of {len(counts)} formation dates have fewer eligible names"
                f" than min_cross_section={min_cross_section}: e.g. {thin_dates[:5]}"
            )
    fallback_total = sum(int(detail.get("fallback_dates", 0)) for detail in neutral_diag.values())
    if fallback_total:
        warnings.append(
            f"{fallback_total} factor-date cross-sections were too thin for the industry/size"
            " regression and used the demean-only fallback"
        )
    for factor_id, detail in coverage["by_factor"].items():
        z_coverage = detail.get("z_coverage")
        if z_coverage is not None and z_coverage < 0.5:
            warnings.append(
                f"factor {factor_id}: neutralised z coverage is only {z_coverage:.1%} of eligible names"
            )

    factor_panel = _factor_panel_frame(data, neutral_scores, composite)
    factor_panel_path = ctx.artifact_path("factor_build", "factor_panel.parquet")
    common.write_frame_artifact(factor_panel_path, factor_panel)

    report = {
        "coverage": coverage,
        "rank_ic": {factor_id: summary["mean"] for factor_id, summary in rank_ic_summary.items()},
        "rank_ic_detail": rank_ic_summary,
        "rank_ic_series": rank_ic_series,
        "ic_t_stat": {factor_id: summary["t_stat"] for factor_id, summary in rank_ic_summary.items()},
        "autocorrelation": autocorrelation,
        "correlation_matrix": correlation,
        "composite_weights": {
            "normalised": normalised,
            "effective_signed": effective,
            "directions": weight_info["directions"],
            "source": weight_info["source"],
        },
        "composite_ic": composite_ic_summary,
        "factor_ids": list(factor_library.FACTOR_IDS),
        "information_cutoff": as_of.isoformat(),
        "formation_dates": {
            "first": data.formation_dates[0],
            "last": data.formation_dates[-1],
            "count": len(data.formation_dates),
        },
        "preprocessing": {
            "winsorize_lower": lower,
            "winsorize_upper": upper,
            "zscore": "rank",
            "neutralize": ["industry_dummies", "log_size"],
            "neutralize_min_obs": neutral_min_obs,
            "neutralize_diagnostics": neutral_diag,
        },
        "warnings": warnings,
    }
    report_path = ctx.artifact_path("factor_build", "factor_report.json")
    common.write_json_artifact(report_path, report)

    summary = {
        "factor_weights": effective,
        "coverage": coverage.get("eligible_share"),
        "ic_mean": composite_ic_summary.get("mean"),
        "eligible_rows": eligible_rows,
        "formation_dates": len(data.formation_dates),
        "warnings": len(warnings),
    }
    artifacts = {
        "factor_panel": common.rel_artifact(factor_panel_path),
        "factor_report": common.rel_artifact(report_path),
        # Filename aliases keep integration tolerant to either naming convention.
        "factor_panel.parquet": common.rel_artifact(factor_panel_path),
        "factor_report.json": common.rel_artifact(report_path),
    }
    ctx.event("factor_build_completed", factors=len(neutral_scores), eligible_rows=eligible_rows)
    return StageOutput(artifacts=artifacts, summary=summary)


def _composite_score(
    effective_weights: Mapping[str, float],
    neutral_scores: Mapping[str, pd.DataFrame],
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """Composite alpha: signed, normalised weighted sum of neutralised z-scores.

    The sum is re-standardised cross-sectionally (rank z-score) so the composite
    is on the same scale every month and can be consumed directly as a
    cross-sectional alpha signal.
    """

    rows = list(eligible.index)
    codes = list(eligible.columns)
    total = np.zeros((len(rows), len(codes)), dtype=float)
    for factor_id, weight in effective_weights.items():
        matrix = pd.DataFrame(neutral_scores[factor_id]).reindex(index=rows, columns=codes)
        filled = factor_preprocess.fill_neutral(matrix)
        total += float(weight) * filled.to_numpy(dtype=float)
    composite = pd.DataFrame(total, index=rows, columns=codes)
    mask_frame = pd.DataFrame(eligible).reindex(index=rows, columns=codes)
    mask_values = mask_frame.to_numpy()
    mask = pd.DataFrame(
        np.where(pd.isna(mask_values), False, mask_values),
        index=rows,
        columns=codes,
    ).astype(bool)
    composite = composite.where(mask)
    return factor_preprocess.rank_zscore(composite)


#: Controller registry mapping (the Lead wires these in ``aicwf/pipeline.py``).
HANDLERS = {"factor_build": run_factor_build}


__all__ = ["HANDLERS", "resolve_composite_weights", "run_factor_build"]
