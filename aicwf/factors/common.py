"""Shared helpers for the quant subsystems (factors, backtest, portfolio).

These helpers are deliberately small and dependency-light so that the three
subsystems owned by the quant module can share one implementation of the
cross-cutting conventions:

* how the strategy configuration is obtained from a :class:`RunContext`,
* how upstream artifacts are located (tolerant to the exact logical key),
* how tabular artifacts are written deterministically (sorted rows, stable
  float formatting, no wall-clock values),
* generic numeric helpers (clamping, box/sum projection, lot rounding).

All *quantitative* definitions live in the subsystem modules themselves
(``library``, ``preprocess``, ``neutralize``, ``engine``, ``projection`` ...).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.contracts import (
    PROJECT_ROOT,
    RunContext,
    ValidationError,
    WorkflowError,
    parse_date,
    relpath,
)
from aicwf.util.hashing import hash_json
from aicwf.util.jsonio import read_json, read_parquet, read_yaml, write_parquet_atomic

STRATEGY_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy.yaml"

PANEL_REQUIRED_COLUMNS = ("date", "code", "close")
PANEL_COLUMNS = (
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
)

BENCHMARK_COLUMNS = ("date", "close", "ret")

FLOAT_DIGITS = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""

    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = _deep_merge(value, {}) if isinstance(value, Mapping) else value
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_strategy_file(path: Path | str = STRATEGY_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return dict(read_yaml(path) or {})


def load_strategy_config(ctx: RunContext) -> dict[str, Any]:
    """Return the strategy configuration used by the quant stages.

    ``RunContext.config`` may be the whole merged workflow config (with a
    ``strategy`` section), the strategy section itself, or empty.  Missing
    sections are filled from ``config/strategy.yaml`` so a stage always sees a
    complete configuration.  Values explicitly supplied through ``ctx.config``
    always win, which is how tests inject mini-fixture overrides.
    """

    raw = getattr(ctx, "config", None) or {}
    if not isinstance(raw, Mapping):
        raise ValidationError("RunContext.config must be a mapping")
    if isinstance(raw.get("strategy"), Mapping):
        override = dict(raw["strategy"])
    elif any(key in raw for key in ("factors", "backtest", "portfolio", "universe", "robustness")):
        override = dict(raw)
    else:
        override = {}
    return _deep_merge(read_strategy_file(), override)


def section(cfg: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    """Nested ``dict.get`` helper: ``section(cfg, "backtest", "costs")``."""

    node: Any = cfg
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def strategy_config_hash(cfg: Mapping[str, Any]) -> str:
    return hash_json(cfg)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def to_iso_dates(values: Iterable[Any]) -> list[str]:
    return [parse_date(value).isoformat() for value in values]


def normalise_date_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame


def month_end_dates(dates: Iterable[Any], *, start: Any = None, end: Any = None) -> list[str]:
    """Last trading day of every calendar month covered by ``dates``.

    ``dates`` must be the trading calendar actually present in the data so the
    result never contains a non-trading day.
    """

    index = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(list(dates))).unique()))
    if start is not None:
        index = index[index >= pd.Timestamp(parse_date(start))]
    if end is not None:
        index = index[index <= pd.Timestamp(parse_date(end))]
    if len(index) == 0:
        return []
    frame = pd.DataFrame({"date": index})
    frame["month"] = frame["date"].dt.to_period("M")
    last = frame.groupby("month", sort=True)["date"].max()
    return [d.date().isoformat() for d in last]


def previous_month_end(value: Any) -> str:
    """ISO date of the last calendar day of the month *before* ``value``."""

    ts = pd.Timestamp(parse_date(value))
    return (ts.replace(day=1) - pd.Timedelta(days=1)).date().isoformat()


def latest_on_or_before(sorted_dates: Sequence[str], cutoff: Any) -> str | None:
    """Largest ISO date in ``sorted_dates`` that is ``<= cutoff``."""

    if not sorted_dates:
        return None
    cutoff_iso = parse_date(cutoff).isoformat()
    lo, hi = 0, len(sorted_dates) - 1
    answer: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= cutoff_iso:
            answer = sorted_dates[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return answer


# ---------------------------------------------------------------------------
# Upstream artifacts
# ---------------------------------------------------------------------------


def _artifact_candidates(path: Path) -> list[Path]:
    """Resolve a declared artifact path relative to the project root if needed."""

    if path.is_absolute():
        return [path]
    return [PROJECT_ROOT / path, Path.cwd() / path, path]


def find_upstream_artifact(
    ctx: RunContext,
    stage: str,
    candidates: Sequence[str],
    *,
    filename: str | None = None,
) -> Path:
    """Locate an upstream artifact declared by ``stage``.

    The workflow documents logical artifact names but different subsystems may
    use the file name (``panel`` vs ``panel.parquet``) as the logical key, so
    the lookup is tolerant and only fails when nothing matches.
    """

    try:
        output = ctx.upstream(stage)
    except WorkflowError as exc:
        raise ValidationError(
            f"stage {stage!r} has not produced output; cannot read its artifacts"
        ) from exc
    artifacts = getattr(output, "artifacts", {}) or {}
    attempts: list[tuple[str, Path]] = []
    for key in candidates:
        value = artifacts.get(key)
        if value:
            attempts.append((key, Path(value)))
    if filename:
        for key, value in artifacts.items():
            path = Path(value)
            if path.name == filename or path.stem == filename:
                attempts.append((key, path))
    if not attempts:
        raise ValidationError(
            f"stage {stage!r} did not declare any of {list(candidates)} (filename={filename!r});"
            f" declared keys: {sorted(artifacts)}"
        )
    for _, path in attempts:
        for resolved in _artifact_candidates(path):
            if resolved.exists():
                return resolved
    raise ValidationError(
        f"declared artifact for stage {stage!r} missing on disk: {[str(p) for _, p in attempts]}"
    )


def upstream_summary(ctx: RunContext, stage: str) -> dict[str, Any]:
    try:
        return dict(ctx.upstream_summary(stage))
    except WorkflowError:
        return {}


def load_panel(ctx: RunContext, *, required: bool = True) -> pd.DataFrame:
    path = find_upstream_artifact(ctx, "data_prep", ("panel", "panel.parquet"), filename="panel.parquet")
    frame = read_parquet(path)
    return normalise_panel(frame, required=required)


def normalise_panel(frame: pd.DataFrame, *, required: bool = True) -> pd.DataFrame:
    """Validate/normalise a data-prep panel to the documented schema."""

    frame = pd.DataFrame(frame).copy()
    missing = [column for column in PANEL_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationError(f"panel is missing required columns {missing}")
    if len(frame) == 0:
        raise ValidationError("panel is empty")
    frame = normalise_date_column(frame, "date")
    frame["code"] = frame["code"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    frame = frame[frame["code"].notna()]
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValidationError("panel has no valid close prices")

    frame["ret"] = _numeric(frame.get("ret"), default=np.nan)
    derived_ret = frame.groupby("code", sort=False)["close"].pct_change()
    frame["ret"] = frame["ret"].where(frame["ret"].notna(), derived_ret)
    frame["ret"] = frame["ret"].fillna(0.0)

    for column in ("volume", "amount", "turnover"):
        frame[column] = _numeric(frame.get(column), default=0.0).fillna(0.0)
    frame["suspended"] = _as_bool(frame.get("suspended"), default=False)
    frame["in_universe"] = _as_bool(frame.get("in_universe"), default=True)
    frame["industry"] = (
        frame["industry"].astype(str).fillna("UNKNOWN").replace({"nan": "UNKNOWN", "None": "UNKNOWN"})
        if "industry" in frame.columns
        else "UNKNOWN"
    )
    frame["log_size"] = _numeric(frame.get("log_size"), default=np.nan)
    if "log_size" not in frame.columns or frame["log_size"].isna().all():
        frame["log_size"] = np.log(frame["close"].where(frame["close"] > 0))
    else:
        frame["log_size"] = frame["log_size"].fillna(np.log(frame["close"].where(frame["close"] > 0)))
    frame["listed_days"] = _numeric(frame.get("listed_days"), default=np.nan)
    if "listed_days" not in frame.columns:
        frame["listed_days"] = 1_000_000.0
    else:
        frame["listed_days"] = frame["listed_days"].fillna(1_000_000.0)

    for column in ("volume", "amount", "turnover", "log_size", "listed_days", "ret"):
        frame[column] = frame[column].astype(float)
    return frame.loc[:, [column for column in PANEL_COLUMNS if column in frame.columns]]


def _numeric(series: Any, *, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series([default], dtype=float)
    series = pd.Series(series)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _as_bool(series: Any, *, default: bool = False) -> pd.Series:
    """Coerce common truthy representations (1/0, "true"/"false", bool)."""

    if series is None:
        return pd.Series([default], dtype=bool)
    series = pd.Series(series)
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(float(default)).astype(float) != 0.0
    text = series.astype(str).str.strip().str.lower()
    truthy = text.isin({"true", "1", "1.0", "yes", "y", "t"})
    return truthy.fillna(default).astype(bool)


def load_benchmark(ctx: RunContext) -> pd.DataFrame:
    path = find_upstream_artifact(
        ctx, "data_prep", ("benchmark", "benchmark.parquet"), filename="benchmark.parquet"
    )
    frame = pd.read_parquet(path)
    frame = pd.DataFrame(frame).copy()
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValidationError("benchmark artifact must contain date and close columns")
    frame = normalise_date_column(frame, "date")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    if len(frame) < 2:
        raise ValidationError("benchmark artifact has fewer than two usable rows")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if "ret" in frame.columns:
        frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    else:
        frame["ret"] = np.nan
    derived = frame["close"].pct_change()
    frame["ret"] = frame["ret"].where(frame["ret"].notna(), derived)
    frame["ret"] = frame["ret"].fillna(0.0)
    return frame.loc[:, ["date", "close", "ret"]].astype({"close": float, "ret": float})


def load_hypothesis(ctx: RunContext, *, required: bool = False) -> dict[str, Any]:
    """Read the hypothesis artifact produced by the agent-driven stage."""

    try:
        path = find_upstream_artifact(
            ctx, "hypothesis", ("hypothesis", "hypothesis.json"), filename="hypothesis.json"
        )
    except ValidationError:
        if required:
            raise
        return {}
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValidationError("hypothesis.json must contain a JSON object")
    return dict(payload)


def snapshot_root(ctx: RunContext) -> Path | None:
    """Directory of the frozen data snapshot, when available."""

    direct = getattr(ctx, "snapshot_dir", None)
    if direct:
        path = Path(direct)
        if path.exists():
            return path
    cfg = load_strategy_config(ctx)
    raw = section(cfg, "data", "snapshot")
    if isinstance(raw, str) and raw:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
    return None


def load_snapshot_table(ctx: RunContext, name: str, *, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
    """Read one snapshot parquet (e.g. ``universe``, ``fundamentals``) if present."""

    root = snapshot_root(ctx)
    if root is None:
        return None
    path = root / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame = pd.DataFrame(frame).copy()
    if columns is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            return None
    return frame


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


def jsonable(value: Any, *, digits: int = FLOAT_DIGITS) -> Any:
    """Recursively convert a payload into strict, deterministic JSON values.

    ``NaN``/``Inf`` become ``None`` (JSON has no such literals), numpy scalars
    become Python scalars, dates become ISO strings and floats are rounded to a
    fixed number of decimals so small platform-dependent differences do not
    leak into artifacts.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        if isinstance(value, pd.Timestamp) and pd.isna(value):
            return None
        return parse_date(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item, digits=digits) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item, digits=digits) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item, digits=digits) for item in value.tolist()]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item(), digits=digits)
        except Exception:  # pragma: no cover - exotic scalars
            return str(value)
    return str(value)


def write_json_artifact(path: Path | str, payload: Any) -> Path:
    """Write strict, deterministic JSON (no NaN/Inf, stable float rounding)."""

    from aicwf.util.jsonio import write_json

    return write_json(Path(path), jsonable(payload), indent=2, sort_keys=False)


def write_frame_artifact(
    path: Path | str,
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] = ("date", "code"),
) -> Path:
    """Write a tabular artifact sorted deterministically by ``sort_by``."""

    frame = pd.DataFrame(frame).copy()
    present = [column for column in sort_by if column in frame.columns]
    if present:
        frame = frame.sort_values(present, kind="mergesort").reset_index(drop=True)
    float_columns = [column for column in frame.columns if pd.api.types.is_float_dtype(frame[column])]
    for column in float_columns:
        frame[column] = frame[column].astype(float).round(FLOAT_DIGITS)
    return write_parquet_atomic(Path(path), frame, index=False)


def round_floats(frame: pd.DataFrame, digits: int = FLOAT_DIGITS) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype(float).round(digits)
    return frame


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def clamp(value: float, lower: float, upper: float) -> float:
    if math.isnan(float(value)):
        return float(lower)
    return float(min(max(float(value), float(lower)), float(upper)))


def safe_div(numerator: Any, denominator: Any, *, default: float = np.nan) -> Any:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, default, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=out, where=den != 0)
    return out if out.shape else float(out)


def lot_floor(shares: float, lot: int = 100) -> int:
    """Round a share count down to an A-share board lot (100 shares)."""

    if lot <= 1:
        return int(max(0, math.floor(shares)))
    return int(max(0, math.floor(shares / lot) * lot))


def project_box_sum(
    vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float = 1.0,
    *,
    tolerance: float = 1e-13,
    max_iter: int = 200,
) -> np.ndarray:
    """Exact Euclidean projection onto ``{lb <= w <= ub, sum(w) = total}``.

    It is a one-dimensional dual problem: the projection is
    ``w_i = clip(v_i - mu, lb_i, ub_i)`` for a scalar ``mu``; because the sum is
    monotone decreasing in ``mu`` we solve for it by bisection.  When ``total``
    is not attainable inside the box, the target is projected onto the feasible
    sum interval ``[sum(lb), sum(ub)]`` first (there is no exact solution then).
    """

    vector = np.asarray(vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if vector.size == 0:
        return vector.copy()
    if np.any(upper < lower):
        raise ValueError("upper bounds must be >= lower bounds")
    total = float(np.clip(total, float(lower.sum()), float(upper.sum())))

    def clipped(mu: float) -> np.ndarray:
        return np.clip(vector - mu, lower, upper)

    # sum(mu=-inf) = sum(upper); sum(mu=+inf) = sum(lower).
    lo = float(np.min(vector - upper)) - 1.0
    hi = float(np.max(vector - lower)) + 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(clipped(mid).sum()) > total:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            break
    weights = clipped(0.5 * (lo + hi))
    # Reconcile any residual rounding error so the identity sum(w) == total
    # holds to machine precision (the residual is spread over interior names).
    residual = total - float(weights.sum())
    if abs(residual) > 1e-15:
        weights = _add_residual(weights, residual, lower, upper)
    return weights


def _add_residual(
    weights: np.ndarray, residual: float, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    weights = weights.copy()
    if residual > 0:
        room = upper - weights
        capacity = room.sum()
        if capacity > 0:
            weights = np.minimum(weights + room * (residual / capacity), upper)
    else:
        room = weights - lower
        capacity = room.sum()
        if capacity > 0:
            weights = np.maximum(weights + room * (residual / capacity), lower)
    return weights


def project_capped_benchmark(weights: pd.Series, cap: float) -> pd.Series:
    """Cap a benchmark weight vector at ``cap`` and renormalise to one.

    The capped vector is the exact Euclidean projection of the raw weights onto
    ``{0 <= w <= cap, sum(w) = 1}``.  It is used as the *benchmark proxy* for
    per-name active bounds because raw CSI 300 weights exceed a 3% name cap for
    the largest constituents, which makes a hard ``|w - wb| <= max_active``
    constraint infeasible for those names.
    """

    weights = pd.Series(weights, dtype=float).fillna(0.0)
    positive = weights[weights > 0]
    if positive.empty:
        if len(weights) == 0:
            return weights
        return pd.Series(np.full(len(weights), 1.0 / len(weights)), index=weights.index)
    cap = float(cap)
    if len(weights) * cap < 1.0 - 1e-12:
        cap = 1.0 / len(weights) + 1e-12
    projected = project_box_sum(
        weights.to_numpy(),
        np.zeros(len(weights)),
        np.full(len(weights), cap),
        1.0,
    )
    return pd.Series(projected, index=weights.index)


def normalise_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def rel_artifact(path: Path | str) -> str:
    """Project-relative POSIX path for a StageOutput artifact mapping."""

    return relpath(Path(path))


def stable_sort_frame(frame: pd.DataFrame, keys: Sequence[str] = ("date", "code")) -> pd.DataFrame:
    present = [key for key in keys if key in frame.columns]
    if not present:
        return frame.reset_index(drop=True)
    return frame.sort_values(present, kind="mergesort").reset_index(drop=True)


def reindex_bool(
    frame: pd.DataFrame,
    *,
    index: Iterable[Any] | None = None,
    columns: Iterable[Any] | None = None,
    default: bool = False,
) -> pd.DataFrame:
    """Reindex a boolean matrix without the pandas object-downcast warning.

    Missing (date, code) combinations are filled with ``default``: ``False`` for
    "not in the index / not tradable", ``True`` for "suspended" when the panel is
    missing the row entirely.
    """

    out = frame.reindex(index=index, columns=columns)
    values = out.to_numpy()
    if values.dtype == bool:
        return out.astype(bool)
    mask = pd.isna(values)
    filled = np.where(mask, bool(default), values)
    return pd.DataFrame(filled.astype(bool), index=out.index, columns=out.columns)


def format_bytes(payload: Any) -> str:  # pragma: no cover - debugging helper
    return json.dumps(jsonable(payload), indent=2)


# ---------------------------------------------------------------------------
# Constraints and configuration hashes
# ---------------------------------------------------------------------------

CONSTRAINT_KEYS: tuple[str, ...] = ("max_weight", "max_active_weight", "te_target", "turnover_limit")

#: Fallback bounds used when ``config/strategy.yaml`` has no explicit bounds
#: (the shipped configuration defines them in ``portfolio.constraints_bounds``).
DEFAULT_CONSTRAINT_BOUNDS: dict[str, tuple[float, float]] = {
    "max_weight": (0.005, 0.03),
    "max_active_weight": (0.001, 0.015),
    "te_target": (0.01, 0.12),
    "turnover_limit": (0.05, 0.60),
}


def config_bundle_hash(ctx: RunContext) -> str:
    """Hash of the configuration sections the run request binds (same as intake).

    ``intake`` hashes ``{workflow, strategy, gates, budgets}`` from the merged
    configuration bundle; metrics.json must carry the same value so verification
    can prove the run is bound to its configuration.  When a stage is exercised
    with a bare strategy config (tests), a deterministic hash of that config is
    used instead.
    """

    raw = getattr(ctx, "config", None) or {}
    if not isinstance(raw, Mapping):
        return hash_json({"config": str(raw)})
    if any(name in raw for name in ("workflow", "gates", "budgets")):
        payload = {name: raw.get(name) for name in ("workflow", "strategy", "gates", "budgets")}
    elif "strategy" in raw:
        payload = {"strategy": raw.get("strategy")}
    else:
        payload = {"strategy": dict(raw)}
    return hash_json(payload)


def clamp_hypothesis_constraints(
    hypothesis: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clamp hypothesis-proposed constraint values to configuration bounds.

    The hypothesis agent proposes ``max_weight``, ``max_active_weight``,
    ``te_target`` and ``turnover_limit``.  Those choices are accepted only inside
    the bounds published in ``config/strategy.yaml`` under
    ``portfolio.constraints_bounds``; values are clamped (never rejected at this
    stage — the agent runtime validates the schema) and the applied values,
    their source and every clamping decision are recorded in the returned info
    so ``risk_report.json`` / ``metrics.json`` can document them.
    """

    hypothesis = dict(hypothesis or {})
    proposed = dict(hypothesis.get("constraints") or {})
    portfolio_cfg = dict(cfg.get("portfolio") or {})
    backtest_cfg = dict(cfg.get("backtest") or {})
    backtest_constraints = dict(backtest_cfg.get("constraints") or {})
    bounds_cfg = dict(portfolio_cfg.get("constraints_bounds") or {})
    defaults_cfg = dict(portfolio_cfg.get("defaults") or {})
    bounds: dict[str, list[float]] = {}
    values: dict[str, Any] = {}
    info: dict[str, Any] = {
        "source": {},
        "requested": {},
        "applied": {},
        "bounds": {},
        "clamped": {},
        "notes": [],
    }
    for key in CONSTRAINT_KEYS:
        default_low, default_high = DEFAULT_CONSTRAINT_BOUNDS[key]
        configured = bounds_cfg.get(key)
        if isinstance(configured, (list, tuple)) and len(configured) == 2:
            low, high = float(configured[0]), float(configured[1])
        else:
            low, high = default_low, default_high
        bounds[key] = [low, high]
        fallback = defaults_cfg.get(key, backtest_constraints.get(key, (low + high) / 2.0))
        fallback_value = clamp(normalise_float(fallback, default=(low + high) / 2.0), low, high)
        if key in proposed:
            requested = clamp(normalise_float(proposed[key], default=fallback_value), -1e12, 1e12)
            source = "hypothesis"
        else:
            requested = fallback_value
            source = "config_default"
        applied = clamp(requested, low, high)
        values[key] = applied
        info["source"][key] = source
        info["requested"][key] = requested
        info["applied"][key] = applied
        if abs(applied - requested) > 1e-15:
            info["clamped"][key] = {"requested": requested, "applied": applied, "bounds": [low, high]}
            info["notes"].append(
                f"{key}: proposed {requested:.6g} clamped to {applied:.6g} within config bounds [{low:g}, {high:g}]"
            )
    for key in ("cash_buffer", "min_names"):
        values[key] = normalise_float(backtest_constraints.get(key, 0.0), default=0.0)
    values["min_names"] = int(values["min_names"])
    values["industry_active_limit"] = normalise_float(
        backtest_constraints.get("industry_active_limit", 1.0), default=1.0
    )
    values["benchmark_weight_cap"] = min(
        normalise_float(
            backtest_constraints.get("benchmark_weight_cap", values["max_weight"]),
            default=values["max_weight"],
        ),
        float(values["max_weight"]),
    )
    values["weight_margin"] = normalise_float(backtest_constraints.get("weight_margin", 0.985), default=0.985)
    values["weight_margin"] = clamp(values["weight_margin"], 0.5, 1.0)
    info["bounds"] = bounds
    return values, info


__all__ = [
    "CONSTRAINT_KEYS",
    "BENCHMARK_COLUMNS",
    "FLOAT_DIGITS",
    "PANEL_COLUMNS",
    "PANEL_REQUIRED_COLUMNS",
    "STRATEGY_CONFIG_PATH",
    "clamp",
    "clamp_hypothesis_constraints",
    "config_bundle_hash",
    "find_upstream_artifact",
    "format_bytes",
    "jsonable",
    "latest_on_or_before",
    "load_benchmark",
    "load_hypothesis",
    "load_panel",
    "load_snapshot_table",
    "load_strategy_config",
    "lot_floor",
    "month_end_dates",
    "normalise_date_column",
    "normalise_float",
    "normalise_panel",
    "previous_month_end",
    "project_box_sum",
    "project_capped_benchmark",
    "read_strategy_file",
    "reindex_bool",
    "rel_artifact",
    "round_floats",
    "safe_div",
    "section",
    "snapshot_root",
    "stable_sort_frame",
    "strategy_config_hash",
    "to_iso_dates",
    "upstream_summary",
    "write_frame_artifact",
    "write_json_artifact",
]
