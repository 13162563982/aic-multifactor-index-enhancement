"""Deterministic synthetic market data and run-context builders for quant tests.

This module is shared by the quant test files.  It is deliberately *not* a
conftest (subsystem tests must not modify ``tests/conftest.py``) and it never
touches the network: the mini market is generated from a fixed seed and mirrors
the documented data-prep / snapshot schemas, so the tests are fully offline and
can run before the real snapshot or the market-data fixture exists.

The following artifacts are produced inside ``tmp_path``:

* ``upstream/panel.parquet``, ``upstream/benchmark.parquet`` (data_prep),
* ``upstream/hypothesis.json`` (hypothesis stage),
* ``snapshot/{universe,fundamentals,benchmark}.parquet`` (frozen snapshot subset).
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(ROOT))

from aicwf.contracts import RunContext, StageOutput  # noqa: E402
from aicwf.factors.common import _deep_merge, read_strategy_file  # noqa: E402
from aicwf.util.jsonio import write_json, write_parquet_atomic  # noqa: E402

INDUSTRIES = ("Banks", "Materials", "Consumer", "Technology")
BOARDS = ("main", "main", "main", "chinext", "main", "main", "star", "main", "main", "main")


def _stock_table(n_codes: int) -> list[dict[str, Any]]:
    stocks: list[dict[str, Any]] = []
    for index in range(n_codes):
        industry = INDUSTRIES[index % len(INDUSTRIES)]
        board = BOARDS[index % len(BOARDS)]
        stocks.append(
            {
                "code": f"{600100 + index}.SH" if index % 2 == 0 else f"{index + 1:06d}.SZ",
                "name": f"Stock{index:02d}",
                "industry": industry,
                "board": board,
            }
        )
    return stocks


def trading_calendar(end: str, months: int) -> list[str]:
    end_date = pd.Timestamp(end)
    start_date = (end_date - pd.DateOffset(months=months)).normalize()
    return [day.strftime("%Y-%m-%d") for day in pd.bdate_range(start_date, end_date)]


def make_market(
    *,
    n_codes: int = 8,
    n_months: int = 36,
    end: str = "2024-12-31",
    seed: int = 20260101,
    adv_scale: float = 1.0,
) -> dict[str, Any]:
    """Generate a deterministic synthetic CSI-300-like market."""

    rng = np.random.default_rng(seed)
    dates = trading_calendar(end, n_months)
    stocks = _stock_table(n_codes)
    codes = [stock["code"] for stock in stocks]
    n_days = len(dates)

    # Per-stock expected return / volatility with heterogeneous characteristics.
    qualities = rng.normal(0.0, 1.0, size=n_codes)
    drift = 0.00025 + 0.00035 * qualities
    volatility = 0.012 + 0.002 * rng.random(n_codes)
    daily_returns = rng.normal(drift, volatility, size=(n_days, n_codes))
    # Small persistent momentum so factor diagnostics are not pure noise.
    momentum_component = np.zeros((n_days, n_codes))
    for index in range(1, n_days):
        momentum_component[index] = 0.004 * daily_returns[index - 1] / np.maximum(volatility, 1e-6)
    daily_returns = np.clip(daily_returns + momentum_component, -0.15, 0.15)
    close = 8.0 * np.cumprod(1.0 + daily_returns, axis=0)
    turnover = np.abs(rng.normal(0.8, 0.25, size=(n_days, n_codes))) + 0.05
    volume = (np.abs(rng.normal(2.0e6, 4.0e5, size=(n_days, n_codes))) + 1.0e5) * float(adv_scale)
    amount = volume * close

    rows: list[pd.DataFrame] = []
    weights = np.abs(rng.normal(1.0, 0.35, size=n_codes))
    weights = weights / weights.sum()
    for index, stock in enumerate(stocks):
        frame = pd.DataFrame(
            {
                "date": dates,
                "code": stock["code"],
                "close": close[:, index],
                "ret": daily_returns[:, index],
                "volume": volume[:, index],
                "amount": amount[:, index],
                "turnover": turnover[:, index],
                "suspended": False,
                "in_universe": True,
                "industry": stock["industry"],
                "log_size": np.log(amount[:, index]),
                "listed_days": np.arange(1, n_days + 1, dtype=float) + 400.0,
            }
        )
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)

    benchmark_returns = daily_returns @ weights
    benchmark_close = 1000.0 * np.cumprod(1.0 + benchmark_returns)
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": benchmark_close,
            "ret": benchmark_returns,
        }
    )

    universe = pd.DataFrame(
        [
            {
                "code": stock["code"],
                "name": stock["name"],
                "industry": stock["industry"],
                "weight": float(weight),
                "free_cap": float(weight) * 1.0e9,
                "total_cap": float(weight) * 2.0e9,
                "listed_date": "2015-01-05",
                "is_st": False,
                "board": stock["board"],
            }
            for stock, weight in zip(stocks, weights)
        ]
    )

    # Quarterly fundamentals: announce 30 days after the report period, which
    # makes the newest report unavailable for one month (point-in-time lag).
    fundamentals: list[dict[str, Any]] = []
    first_report = (pd.Timestamp(dates[0]) - pd.DateOffset(months=18)).normalize()
    report_dates = pd.date_range(
        first_report + pd.offsets.QuarterEnd(0), pd.Timestamp(end) + pd.offsets.QuarterEnd(0), freq="QE"
    )
    for index, stock in enumerate(stocks):
        quality = qualities[index]
        for report_date in report_dates:
            notice_date = report_date + pd.Timedelta(days=30)
            eps = 0.8 + 0.25 * quality + 0.05 * rng.normal()
            bps = 6.0 + 0.8 * quality + 0.2 * rng.normal()
            revenue = 2.0e9 * (1.0 + 0.1 * quality)
            net_profit = revenue * (0.04 + 0.01 * quality)
            fundamentals.append(
                {
                    "code": stock["code"],
                    "report_date": report_date.strftime("%Y-%m-%d"),
                    "notice_date": notice_date.strftime("%Y-%m-%d"),
                    "eps": eps,
                    "bps": bps,
                    "roe": 0.10 + 0.02 * quality + 0.005 * float(rng.normal()),
                    "gross_margin": 0.25 + 0.03 * quality + 0.01 * float(rng.normal()),
                    "revenue": revenue,
                    "revenue_yoy": 0.08 + 0.03 * quality + 0.01 * float(rng.normal()),
                    "net_profit": net_profit,
                    "net_profit_yoy": 0.10 + 0.04 * quality + 0.01 * float(rng.normal()),
                    "cfo_per_share": eps * (0.9 + 0.15 * quality) + 0.02 + 0.01 * float(rng.normal()),
                    "notice_date_estimated": False,
                }
            )
    fundamentals_frame = pd.DataFrame(fundamentals).sort_values(["code", "report_date"]).reset_index(drop=True)

    return {
        "panel": panel,
        "benchmark": benchmark,
        "universe": universe,
        "fundamentals": fundamentals_frame,
        "codes": codes,
        "trading_dates": dates,
        "weights": weights,
    }


def make_hypothesis(
    market: Mapping[str, Any],
    *,
    constraints: Mapping[str, Any] | None = None,
    factor_weights: Mapping[str, float] | None = None,
    information_cutoff: str | None = None,
) -> dict[str, Any]:
    cutoff = information_cutoff or market["trading_dates"][-1]
    return {
        "thesis": (
            "Cheap profitable CSI 300 names with positive 12-1 momentum should outperform the"
            " benchmark once industry and size exposures are neutralised, because valuation and"
            " quality premia are compensated while low-beta and reversal effects damp drawdowns."
        ),
        "factor_weights": dict(
            factor_weights
            or {
                "value_ep": 0.25,
                "quality_roe": 0.20,
                "momentum_12_1": 0.20,
                "reversal_1m": 0.10,
                "low_vol_60d": 0.15,
                "growth_revenue": 0.10,
            }
        ),
        "rebalance": "monthly",
        "constraints": dict(
            constraints
            or {
                "max_weight": 0.30,
                "max_active_weight": 0.25,
                "te_target": 0.05,
                "turnover_limit": 0.60,
            }
        ),
        "falsification": [
            "The realised information ratio is below 0.3 over the frozen backtest window.",
            "More than half of the annualised excess return comes from three rebalances.",
        ],
        "information_cutoff": cutoff,
        "source_limits": [
            "Frozen CSI 300 snapshot only; no live data access during the run.",
            "Fundamentals are point-in-time by notice_date, not by report period.",
        ],
    }


MINI_OVERRIDES: dict[str, Any] = {
    "universe": {"min_listed_days": 0},
    "factors": {
        "min_cross_section": 3,
        "neutralize_min_obs": 3,
        "ic_min_names": 3,
    },
    "backtest": {
        "initial_capital": 100000000.0,
        "lot_size": 100,
        "constraints": {
            "max_weight": 0.30,
            "max_active_weight": 0.25,
            "industry_active_limit": 0.60,
            "turnover_limit": 0.60,
            "te_target": 0.05,
            "cash_buffer": 0.0,
            "min_names": 3,
            "benchmark_weight_cap": 0.40,
        },
        "optimizer": {"lambda_bisection_steps": 16, "max_iter": 120},
    },
    "portfolio": {
        "constraints_bounds": {
            "max_weight": [0.005, 0.90],
            "max_active_weight": [0.001, 0.90],
            "te_target": [0.01, 0.12],
            "turnover_limit": [0.05, 1.00],
        },
        "defaults": {
            "max_weight": 0.30,
            "max_active_weight": 0.25,
            "te_target": 0.05,
            "turnover_limit": 0.60,
        },
        "benchmark_weight_cap": 0.40,
        # The synthetic fixture's traded amounts are tiny relative to a 100M
        # portfolio; the participation cap is relaxed for tests so it never
        # binds (it is exercised by a dedicated unit test with explicit amounts).
        "participation_cap": 100.0,
        "optimizer": {"lambda_bisection_steps": 16, "max_iter": 120, "tol": 1.0e-8},
        "covariance": {"lookback_days": 60, "min_obs": 20},
        "fallback": {"enabled": True, "min_names": 2},
    },
    "robustness": {
        "subperiods": [
            ["2023-03-01", "2023-12-29"],
            ["2024-01-02", "2024-12-31"],
        ],
        "placebo_repetitions": 4,
        "parameter_grid": {"te_target": [0.03, 0.05], "rebalance": ["monthly", "quarterly"]},
        "bias_stress_drop_top_k": 2,
        "dsr_trials": 8,
        "max_loo_factors": 6,
        "thresholds": {
            "subperiod_pass_share": 0.33,
            "cost_3x_excess_min": -1.0,
            "loo_min_information_ratio": -10.0,
            "grid_positive_share": 0.0,
            "placebo_p_value_max": 1.0,
            "bias_stress_excess_min": -1.0,
            "dsr_min": 0.0,
        },
    },
}


def mini_strategy_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = _deep_merge(read_strategy_file(), MINI_OVERRIDES)
    config = _deep_merge(config, dict(overrides or {}))
    # Keep the backtest window inside the synthetic calendar: start after enough
    # history for the 12-1 momentum and its one-month signal lag.
    return config


def mini_backtest_window(market: Mapping[str, Any], *, skip_months: int = 15) -> tuple[str, str]:
    dates = market["trading_dates"]
    months = sorted({date[:7] for date in dates})
    if len(months) <= skip_months + 2:
        skip_months = max(1, len(months) - 3)
    start_month = months[skip_months]
    start = min(date for date in dates if date.startswith(start_month))
    end = dates[-1]
    return start, end


def write_upstream(
    root: Path,
    market: Mapping[str, Any],
    *,
    hypothesis: Mapping[str, Any] | None = None,
    panel: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Path]:
    """Write data-prep/hypothesis/snapshot artifacts and return their paths."""

    root = Path(root)
    upstream = root / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    panel_path = upstream / "panel.parquet"
    benchmark_path = upstream / "benchmark.parquet"
    hypothesis_path = upstream / "hypothesis.json"
    write_parquet_atomic(panel_path, panel if panel is not None else market["panel"], index=False)
    write_parquet_atomic(
        benchmark_path, benchmark if benchmark is not None else market["benchmark"], index=False
    )
    write_json(
        hypothesis_path,
        dict(hypothesis or make_hypothesis(market)),
        indent=2,
    )
    paths = {
        "upstream_dir": upstream,
        "panel": panel_path,
        "benchmark": benchmark_path,
        "hypothesis": hypothesis_path,
    }
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(snapshot_dir / "universe.parquet", market["universe"], index=False)
        write_parquet_atomic(snapshot_dir / "fundamentals.parquet", market["fundamentals"], index=False)
        write_parquet_atomic(snapshot_dir / "benchmark.parquet", market["benchmark"], index=False)
        paths["snapshot_dir"] = snapshot_dir
    return paths


def make_run_context(
    tmp_path: Path,
    *,
    market: Mapping[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    hypothesis: Mapping[str, Any] | None = None,
    with_snapshot: bool = True,
    as_of: str | None = None,
    run_id: str = "mini-quant-run",
    seed: int = 20260101,
) -> tuple[RunContext, dict[str, Any], dict[str, Path]]:
    """Build a hand-crafted RunContext with tmp upstream artifacts registered."""

    tmp_path = Path(tmp_path)
    market = market or make_market()
    config = mini_strategy_config(config_overrides)
    window_start, window_end = mini_backtest_window(market)
    explicit = dict((config_overrides or {}).get("backtest") or {})
    config = _deep_merge(
        config,
        {
            "backtest": {
                "start_date": explicit.get("start_date", window_start),
                "end_date": explicit.get("end_date", window_end),
            }
        },
    )
    cutoff = as_of or window_end
    if hypothesis is None:
        hypothesis = make_hypothesis(market, information_cutoff=cutoff)
    snapshot_dir = tmp_path / "snapshot" if with_snapshot else None
    paths = write_upstream(tmp_path, market, hypothesis=hypothesis, snapshot_dir=snapshot_dir)

    ctx = RunContext(
        run_id=run_id,
        as_of=pd.Timestamp(cutoff).date(),
        run_dir=tmp_path / "run",
        config={"strategy": config},
        llm_mode="null",
        seed=seed,
        snapshot_dir=snapshot_dir,
    )
    ctx.register(
        "data_prep",
        StageOutput(
            artifacts={
                "panel": str(paths["panel"]),
                "panel.parquet": str(paths["panel"]),
                "benchmark": str(paths["benchmark"]),
                "benchmark.parquet": str(paths["benchmark"]),
            },
            summary={"rows": int(len(market["panel"])), "universe_size": len(market["codes"])},
        ),
    )
    ctx.register(
        "hypothesis",
        StageOutput(
            artifacts={"hypothesis": str(paths["hypothesis"]), "hypothesis.json": str(paths["hypothesis"])},
            summary={"factor_weights": dict(hypothesis["factor_weights"])},
        ),
    )
    return ctx, market, paths


def run_stage_chain(
    ctx: RunContext,
    *,
    run_backtest_stage: bool = True,
    run_robustness_stage: bool = False,
):
    """Run factor build -> backtest -> (robustness) -> portfolio on a mini context."""

    from aicwf.backtest.stage import run_backtest, run_robustness
    from aicwf.factors.stage import run_factor_build
    from aicwf.portfolio.stage import run_portfolio

    outputs: dict[str, StageOutput] = {}
    outputs["factor_build"] = run_factor_build(ctx)
    ctx.register("factor_build", outputs["factor_build"])
    if run_backtest_stage:
        outputs["backtest"] = run_backtest(ctx)
        ctx.register("backtest", outputs["backtest"])
    if run_robustness_stage and run_backtest_stage:
        outputs["robustness"] = run_robustness(ctx)
        ctx.register("robustness", outputs["robustness"])
    if run_backtest_stage:
        outputs["portfolio"] = run_portfolio(ctx)
        ctx.register("portfolio", outputs["portfolio"])
    return outputs


def read_artifact(ctx: RunContext, stage: str, logical: str) -> pd.DataFrame | dict[str, Any]:
    """Read a quoted artifact from a stage output (tests / debugging helper)."""

    path = Path(ctx.upstream(stage).artifacts[logical])
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Frozen mini-fixture support (tests/fixtures/mini_snapshot)
# ---------------------------------------------------------------------------

MINI_FIXTURE = ROOT / "tests" / "fixtures" / "mini_snapshot"


def mini_fixture_available() -> bool:
    path = MINI_FIXTURE
    return (path / "prices.parquet").exists() and (path / "universe.parquet").exists()


def market_from_snapshot(snapshot_dir: Path | str) -> dict[str, Any]:
    """Build a data-prep-shaped market dict from a frozen snapshot directory.

    The frozen snapshot stores daily bars only for days a stock actually traded
    (four bars are deliberately missing in the mini fixture).  The data-prep panel
    schema wants one row per (trading day, index member) with a *suspended* flag
    and a carried-forward close, so this adapter mirrors that documented
    behaviour; it is only used by tests that exercise the quant stages before the
    real ``run_data_prep`` stage has been wired, never by production code.
    """

    root = Path(snapshot_dir)
    prices = pd.read_parquet(root / "prices.parquet")
    universe = pd.read_parquet(root / "universe.parquet")
    benchmark = pd.read_parquet(root / "benchmark.parquet")
    fundamentals = pd.read_parquet(root / "fundamentals.parquet")
    calendar = (
        pd.read_parquet(root / "calendar.parquet")
        if (root / "calendar.parquet").exists()
        else pd.DataFrame({"date": sorted(prices["date"].unique())})
    )
    prices = prices.copy()
    prices["date"] = prices["date"].astype(str).str[:10]
    prices["code"] = prices["code"].astype(str)
    universe = universe.copy()
    universe["code"] = universe["code"].astype(str)
    dates = [str(value)[:10] for value in calendar["date"].astype(str).tolist()]
    codes = universe["code"].tolist()
    industry_map = dict(zip(universe["code"], universe["industry"].astype(str)))
    listed_map = dict(zip(universe["code"], universe["listed_date"].astype(str))) if "listed_date" in universe else {}

    index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    frame = pd.DataFrame(index=index).reset_index()
    merged = frame.merge(
        prices[["date", "code", "close", "volume", "amount", "turnover", "pct_chg"]],
        on=["date", "code"],
        how="left",
    )
    close_pivot = prices.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    close_pivot = close_pivot.reindex(index=dates, columns=codes).ffill()
    frame["date"] = merged["date"]
    frame["code"] = merged["code"]
    frame["close"] = merged["close"].to_numpy(dtype=float)
    carried = close_pivot.to_numpy(dtype=float).ravel()
    frame["close"] = np.where(np.isfinite(frame["close"].to_numpy(dtype=float)), frame["close"], carried)
    raw_pct = merged["pct_chg"].to_numpy(dtype=float)
    median_pct = float(np.nanmedian(np.abs(raw_pct))) if np.any(np.isfinite(raw_pct)) else 0.0
    if np.isfinite(median_pct) and median_pct > 0.2:  # snapshot pct_chg stored in percent
        raw_pct = raw_pct / 100.0
    frame["ret"] = np.where(np.isfinite(raw_pct), raw_pct, 0.0)
    frame["volume"] = merged["volume"].fillna(0.0).to_numpy(dtype=float)
    frame["amount"] = merged["amount"].fillna(0.0).to_numpy(dtype=float)
    frame["turnover"] = merged["turnover"].fillna(0.0).to_numpy(dtype=float)
    frame["suspended"] = merged["close"].isna().to_numpy()
    frame["in_universe"] = frame["close"].notna()
    frame["industry"] = frame["code"].map(industry_map)
    frame["log_size"] = np.log(frame["amount"].to_numpy(dtype=float) + 1.0)
    date_position = {day: position for position, day in enumerate(dates)}
    listed_position = {
        code: next((position for position, day in enumerate(dates) if day >= listed), 0)
        for code, listed in listed_map.items()
    }
    frame["listed_days"] = (
        frame["date"].map(date_position).astype(float) - frame["code"].map(listed_position).fillna(0.0) + 1.0
    ).clip(lower=1.0)
    frame.loc[frame["date"].map(lambda day: day < "2010-01-01"), "listed_days"] = 400.0
    panel = frame.loc[
        :,
        [
            "date",
            "code",
            "close",
            "ret",
            "volume",
            "amount",
            "turnover",
            "suspended",
            "in_universe",
            "industry",
            "log_size",
            "listed_days",
        ],
    ].sort_values(["date", "code"]).reset_index(drop=True)
    benchmark = benchmark.copy()
    benchmark["date"] = benchmark["date"].astype(str).str[:10]
    benchmark["code"] = "000300.SH"
    return {
        "panel": panel,
        "benchmark": benchmark[["date", "close", "ret"]],
        "universe": universe,
        "fundamentals": fundamentals,
        "codes": codes,
        "trading_dates": dates,
        "weights": universe["weight"].to_numpy(dtype=float),
    }


def fixture_run_context(
    tmp_path: Path,
    *,
    fixture_dir: Path | str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    hypothesis_overrides: Mapping[str, Any] | None = None,
    as_of: str = "2024-12-31",
    run_id: str = "mini-fixture-run",
) -> tuple[RunContext, dict[str, Any], dict[str, Path]]:
    """Run context backed by the committed mini snapshot fixture."""

    snapshot = Path(fixture_dir) if fixture_dir is not None else MINI_FIXTURE
    market = market_from_snapshot(snapshot)
    start, _ = mini_backtest_window(market, skip_months=13)
    window_overrides = _deep_merge(
        {"backtest": {"start_date": start, "end_date": as_of}},
        dict(config_overrides or {}),
    )
    tmp_path = Path(tmp_path)
    config = mini_strategy_config(window_overrides)
    hypothesis = make_hypothesis(market, information_cutoff=as_of) if hypothesis_overrides is None else None
    if hypothesis_overrides is not None:
        base = make_hypothesis(market, information_cutoff=as_of)
        hypothesis = _deep_merge(base, dict(hypothesis_overrides))
    paths = write_upstream(tmp_path, market, hypothesis=hypothesis, snapshot_dir=None)
    ctx = RunContext(
        run_id=run_id,
        as_of=pd.Timestamp(as_of).date(),
        run_dir=tmp_path / "run",
        config={"strategy": config},
        llm_mode="null",
        seed=20260101,
        snapshot_dir=snapshot,
    )
    ctx.register(
        "data_prep",
        StageOutput(
            artifacts={
                "panel": str(paths["panel"]),
                "panel.parquet": str(paths["panel"]),
                "benchmark": str(paths["benchmark"]),
                "benchmark.parquet": str(paths["benchmark"]),
            },
            summary={"rows": int(len(market["panel"]))},
        ),
    )
    ctx.register(
        "hypothesis",
        StageOutput(
            artifacts={"hypothesis": str(paths["hypothesis"]), "hypothesis.json": str(paths["hypothesis"])},
            summary={"factor_weights": dict(hypothesis["factor_weights"])},
        ),
    )
    return ctx, market, paths


def run_pipeline(root: Path | str, *, with_robustness: bool = True) -> dict[str, Any]:
    """Run factor_build -> backtest -> (robustness) -> portfolio on the fixture."""

    from aicwf.backtest.stage import run_backtest, run_robustness
    from aicwf.factors.stage import run_factor_build
    from aicwf.portfolio.stage import run_portfolio

    root = Path(root)
    ctx, market, paths = fixture_run_context(root / "fixture")
    outputs: dict[str, Any] = {}
    outputs["factor_build"] = run_factor_build(ctx)
    ctx.register("factor_build", outputs["factor_build"])
    outputs["backtest"] = run_backtest(ctx)
    ctx.register("backtest", outputs["backtest"])
    if with_robustness:
        outputs["robustness"] = run_robustness(ctx)
        ctx.register("robustness", outputs["robustness"])
    outputs["portfolio"] = run_portfolio(ctx)
    ctx.register("portfolio", outputs["portfolio"])
    return {"ctx": ctx, "market": market, "paths": paths, "outputs": outputs, "root": root}


_PIPELINE_CACHE: dict[bool, dict[str, Any]] = {}


def cached_fixture_pipeline(*, with_robustness: bool = True) -> dict[str, Any]:
    """Session-level cache: the fixture pipeline is deterministic, run it once."""

    key = bool(with_robustness)
    if key not in _PIPELINE_CACHE:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="quant_fixture_pipeline_"))
        _PIPELINE_CACHE[key] = run_pipeline(root, with_robustness=with_robustness)
    return _PIPELINE_CACHE[key]


def artifact_path(pipeline: Mapping[str, Any], stage: str, logical: str) -> Path:
    """Project-artifact path from a cached pipeline result (absolute in tests)."""

    return Path(pipeline["outputs"][stage].artifacts[logical])


def content_hash(path: Path | str) -> str:
    from aicwf.util.artifact_hash import hash_artifact

    return hash_artifact(Path(path))


__all__ = [
    "INDUSTRIES",
    "MINI_FIXTURE",
    "MINI_OVERRIDES",
    "artifact_path",
    "cached_fixture_pipeline",
    "content_hash",
    "fixture_run_context",
    "make_hypothesis",
    "make_market",
    "make_run_context",
    "market_from_snapshot",
    "mini_backtest_window",
    "mini_fixture_available",
    "mini_strategy_config",
    "read_artifact",
    "run_pipeline",
    "run_stage_chain",
    "trading_calendar",
    "write_upstream",
]
