"""Factor mathematics, point-in-time guarantees and factor-stage integration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_quant_support as sup  # noqa: E402

from aicwf.contracts import ValidationError  # noqa: E402
from aicwf.factors import library as factor_library  # noqa: E402
from aicwf.factors import neutralize as factor_neutralize  # noqa: E402
from aicwf.factors import preprocess as factor_preprocess  # noqa: E402
from aicwf.factors import stage as factor_stage  # noqa: E402
from aicwf.util.artifact_hash import hash_dataframe  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_winsorize_clips_cross_sectional_extremes():
    values = np.array([-1000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0])
    frame = pd.DataFrame([values], index=["2024-01-31"], columns=[f"C{index}" for index in range(len(values))])
    clipped = factor_preprocess.winsorize(frame, lower=0.1, upper=0.9)
    low = float(np.quantile(values, 0.1))
    high = float(np.quantile(values, 0.9))
    assert clipped.iloc[0, 0] == pytest.approx(low)
    assert clipped.iloc[0, -1] == pytest.approx(high)
    # Interior order is preserved.
    interior = clipped.iloc[0, 1:-1].to_numpy(dtype=float)
    assert np.all(np.diff(interior) >= 0)


def test_winsorize_preserves_missing_values():
    frame = pd.DataFrame(
        [[1.0, np.nan, 3.0, 4.0, 5.0]],
        index=["2024-01-31"],
        columns=["A", "B", "C", "D", "E"],
    )
    clipped = factor_preprocess.winsorize(frame)
    assert np.isnan(clipped.loc["2024-01-31", "B"])
    assert clipped.notna().sum().sum() == 4


def test_rank_zscore_has_zero_mean_and_unit_std():
    values = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0])
    frame = pd.DataFrame([values], index=["2024-01-31"], columns=[f"C{index}" for index in range(len(values))])
    z = factor_preprocess.rank_zscore(frame)
    row = z.iloc[0].to_numpy(dtype=float)
    assert float(np.mean(row)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(row)) == pytest.approx(1.0, abs=1e-12)
    # Rank transform is monotone in the raw value.
    order = np.argsort(values, kind="mergesort")
    assert np.all(np.diff(row[order]) > 0)


def test_rank_zscore_handles_ties_and_missing():
    values = np.array([2.0, 2.0, 5.0, np.nan, 7.0])
    frame = pd.DataFrame([values], index=["2024-01-31"], columns=list("ABCDE"))
    z = factor_preprocess.rank_zscore(frame).iloc[0]
    assert np.isnan(z["D"])
    assert z["A"] == pytest.approx(z["B"])
    assert z["C"] < z["E"]


def test_preprocess_factor_chain_is_deterministic():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.normal(size=(4, 12)),
        index=["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"],
        columns=[f"C{index}" for index in range(12)],
    )
    first = factor_preprocess.preprocess_factor(frame)
    second = factor_preprocess.preprocess_factor(frame)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Neutralisation
# ---------------------------------------------------------------------------


def test_neutralization_removes_industry_and_size_exposure():
    rng = np.random.default_rng(7)
    dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    codes = [f"C{index:02d}" for index in range(24)]
    industries = ["Banks"] * 8 + ["Tech"] * 8 + ["Consumer"] * 8
    industry = pd.DataFrame([industries] * len(dates), index=dates, columns=codes)
    size = pd.DataFrame(
        rng.normal(16.0, 1.0, size=(len(dates), len(codes))), index=dates, columns=codes
    )
    scores = pd.DataFrame(rng.normal(size=(len(dates), len(codes))), index=dates, columns=codes)
    neutral, diagnostics = factor_neutralize.neutralise_cross_section(scores, industry, size)
    for date in dates:
        row = neutral.loc[date]
        assert float(np.nanmean(row)) == pytest.approx(0.0, abs=1e-10)
        assert float(np.nanstd(row, ddof=1)) == pytest.approx(1.0, abs=1e-10)
        for group in ("Banks", "Tech", "Consumer"):
            mask = industry.loc[date] == group
            assert float(np.nanmean(row[mask])) == pytest.approx(0.0, abs=1e-10)
        correlation = float(np.corrcoef(row.to_numpy(dtype=float), size.loc[date].to_numpy(dtype=float))[0, 1])
        assert abs(correlation) < 1e-10
    assert diagnostics["fallback_dates"] == 0


def test_neutralization_thin_cross_section_uses_fallback():
    industry = pd.DataFrame([["A", "B"]], index=["2024-01-31"], columns=["C0", "C1"])
    size = pd.DataFrame([[16.0, 16.5]], index=["2024-01-31"], columns=["C0", "C1"])
    scores = pd.DataFrame([[1.0, 2.0]], index=["2024-01-31"], columns=["C0", "C1"])
    neutral, diagnostics = factor_neutralize.neutralise_cross_section(
        scores, industry, size, min_obs=10
    )
    assert diagnostics["fallback_dates"] == 1
    row = neutral.iloc[0].to_numpy(dtype=float)
    assert float(np.mean(row)) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Library / PIT
# ---------------------------------------------------------------------------


def test_factor_library_is_point_in_time_at_each_formation_date():
    market = sup.make_market(n_codes=8, n_months=30)
    panel = pd.DataFrame(market["panel"])
    full = factor_library.build_monthly_factor_data(
        panel, fundamentals=market["fundamentals"], universe=market["universe"], min_listed_days=0
    )
    cutoff = full.formation_dates[20]
    truncated_panel = panel[panel["date"] <= cutoff]
    truncated = factor_library.build_monthly_factor_data(
        truncated_panel,
        fundamentals=market["fundamentals"],
        universe=market["universe"],
        min_listed_days=0,
        end_date=cutoff,
    )
    assert truncated.formation_dates[-1] == cutoff
    for factor_id in factor_library.FACTOR_IDS:
        left = full.raw[factor_id].loc[cutoff]
        right = truncated.raw[factor_id].loc[cutoff]
        pd.testing.assert_series_equal(left, right, check_names=False)


def test_factor_library_momentum_skips_most_recent_month():
    market = sup.make_market(n_codes=6, n_months=20)
    data = factor_library.build_monthly_factor_data(
        market["panel"], fundamentals=None, universe=market["universe"], min_listed_days=0
    )
    formation = data.formation_dates[13]
    close = data.close
    previous = data.formation_dates[12]
    twelve_back = data.formation_dates[1]
    expected = close.loc[previous] / close.loc[twelve_back] - 1.0
    actual = data.raw["momentum_12_1"].loc[formation]
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-12)


def test_factor_catalog_rejects_unknown_ids():
    cfg = sup.mini_strategy_config()
    cfg["factors"]["catalog"]["not_a_factor"] = {"direction": 1.0, "category": "x", "description": "y"}
    with pytest.raises(ValidationError):
        factor_library.factor_catalog(cfg)
    with pytest.raises(ValidationError):
        factor_stage.resolve_composite_weights(
            {"factor_weights": {"unknown_factor": 0.5}}, cfg, factor_library.factor_catalog(sup.mini_strategy_config())
        )


def test_composite_weights_are_normalised_and_direction_adjusted():
    cfg = sup.mini_strategy_config()
    catalog = factor_library.factor_catalog(cfg)
    hypothesis = {"factor_weights": {"value_ep": 2.0, "low_vol_60d": 1.0}}
    normalised, effective, info = factor_stage.resolve_composite_weights(hypothesis, cfg, catalog)
    assert sum(abs(value) for value in normalised.values()) == pytest.approx(1.0)
    assert effective["value_ep"] == pytest.approx(2.0 / 3.0)
    assert effective["low_vol_60d"] == pytest.approx(-1.0 / 3.0)
    assert info["source"] == "hypothesis.factor_weights"


# ---------------------------------------------------------------------------
# Factor stage on the frozen mini fixture
# ---------------------------------------------------------------------------


def test_factor_stage_on_mini_fixture():
    pipeline = sup.cached_fixture_pipeline()
    output = pipeline["outputs"]["factor_build"]
    panel_path = sup.artifact_path(pipeline, "factor_build", "factor_panel")
    report_path = sup.artifact_path(pipeline, "factor_build", "factor_report")
    assert panel_path.exists() and report_path.exists()
    panel = pd.read_parquet(panel_path)
    expected_columns = ["date", "code"]
    for factor_id in factor_library.FACTOR_IDS:
        expected_columns.extend([f"{factor_id}_raw", f"{factor_id}_z"])
    expected_columns.extend(["composite_score", "eligible"])
    assert list(panel.columns) == expected_columns
    assert panel["eligible"].dtype == bool
    assert panel.duplicated(subset=["date", "code"]).sum() == 0
    assert panel["date"].is_monotonic_increasing or (
        panel[["date", "code"]].apply(tuple, axis=1).is_monotonic_increasing
    )
    eligible = panel[panel["eligible"]]
    # The composite score is a fresh cross-sectional z-score every month for the
    # dates where at least one weighted factor is available (the first formation
    # dates of the mini fixture only have the zero-weight liquidity factors).
    grouped = eligible.groupby("date")["composite_score"]
    means = grouped.mean()
    stds = grouped.apply(lambda values: float(values.std(ddof=0)))
    assert float(means.abs().max()) < 1e-8
    informative = eligible.groupby("date")["composite_score"].nunique() > 1
    assert bool(informative.any())
    # Rank z-scores with tied raw composites keep a variance slightly below one
    # (average-tie ranks); it must still be a proper unit-scale signal.
    assert float((stds[informative] - 1.0).abs().max()) < 0.05

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("coverage", "rank_ic", "ic_t_stat", "autocorrelation", "correlation_matrix", "composite_weights", "warnings"):
        assert key in report
    assert set(report["composite_weights"]["effective_signed"]).issubset(set(factor_library.FACTOR_IDS))
    assert report["information_cutoff"] == pipeline["ctx"].as_of.isoformat()
    summary = output.summary
    for key in ("factor_weights", "coverage", "ic_mean", "eligible_rows"):
        assert key in summary


def test_factor_stage_ignores_rows_after_as_of(tmp_path):
    as_of = "2023-12-29"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    ctx_a, market_a, _ = sup.fixture_run_context(first_root, as_of=as_of)
    ctx_b, market_b, paths_b = sup.fixture_run_context(second_root, as_of=as_of)

    # Corrupt every row after the run's as_of: a point-in-time stage must not see it.
    panel_b = market_b["panel"].copy()
    future = panel_b["date"] > as_of
    panel_b.loc[future, "close"] = panel_b.loc[future, "close"] * 3.0
    panel_b.loc[future, "ret"] = 0.5
    panel_b.loc[future, "volume"] = 0.0
    sup.write_upstream(second_root / "fixture", market_b, hypothesis=sup.make_hypothesis(market_b, information_cutoff=as_of), panel=panel_b)

    output_a = factor_stage.run_factor_build(ctx_a)
    output_b = factor_stage.run_factor_build(ctx_b)
    panel_a_frame = pd.read_parquet(output_a.artifacts["factor_panel"])
    panel_b_frame = pd.read_parquet(output_b.artifacts["factor_panel"])
    assert hash_dataframe(panel_a_frame) == hash_dataframe(panel_b_frame)
    report_a = json.loads(Path(output_a.artifacts["factor_report"]).read_text(encoding="utf-8"))
    report_b = json.loads(Path(output_b.artifacts["factor_report"]).read_text(encoding="utf-8"))
    assert report_a["coverage"] == report_b["coverage"]


def test_factor_stage_is_deterministic(tmp_path):
    root = tmp_path / "run"
    pipeline = sup.run_pipeline(root, with_robustness=False)
    first_hash = sup.content_hash(sup.artifact_path(pipeline, "factor_build", "factor_panel"))
    second = sup.run_pipeline(tmp_path / "run2", with_robustness=False)
    second_hash = sup.content_hash(sup.artifact_path(second, "factor_build", "factor_panel"))
    assert first_hash == second_hash
