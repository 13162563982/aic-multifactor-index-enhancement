"""Monthly rebalance index-enhancement backtest engine.

Trading conventions (see also ``docs/artifact_contract.md``):

* Decision dates are the **last trading day of each month**.  The signal used on
  a decision date is the factor score formed at the end of the *previous* month
  (``signal_lag_months = 1``): a month-``t`` signal is traded in month ``t+1``.
* Trades execute at the **T+1 close** (the next trading day after the decision
  date, ``signal_lag_days = 1``) at that day's frozen adjusted close.
* The first rebalance builds the portfolio from cash; later rebalances start
  from the current holdings, so the strategy is a genuine *enhancement* of the
  benchmark rather than a stand-alone stock-picking sleeve.
* Execution respects A-share market rules: 100-share board lots, no trades of
  suspended names, no buys at limit-up, no sells at limit-down, no buys of ST
  names, and a participation cap on the traded amount.
* Costs come from ``config/strategy.yaml`` (commission, stamp duty on sells,
  slippage, square-root market impact) and are paid in cash.
* The NAV path is built on the **entire frozen trading calendar up to
  ``as_of``** (capital is in cash before the first rebalance), exactly matching
  how the independent verifier rebuilds the portfolio from the ledger, so the
  declared metrics can be reproduced from the artifacts alone.  The equity
  curve, holdings, ledger and metrics therefore reconcile with the verifier's
  independent reconstruction.

Portfolio construction at each rebalance
----------------------------------------
Let ``wb`` be the benchmark proxy (universe free-float weights capped at the name
cap and renormalised), ``alpha`` the neutralised composite score and ``sigma``
the trailing 60-day annualised volatility.  The active tilt is scaled so a
diagonal estimate of ex-ante tracking error hits ``te_target``::

    a_i = clip(k * alpha_i, -max_active, +max_active)
    k   = TE_target / sqrt( sum_i (alpha_i * sigma_i)^2 )
    w_raw = wb + a

``w_raw`` is then projected onto the feasible set

    { 0 <= w_i <= max_weight, |w_i - wb_i| <= max_active,
      sum_i w_i = 1, sum_{i in g} w_i = sum_{i in g} wb_i for every industry g }

with :func:`aicwf.portfolio.projection.project_industry_neutral_box` (exact
per-industry dual bisection).  The projection bounds include a minimum holding
weight ``lot_size * price_i * (1 + limit_i + 0.02) / initial_capital`` so every
benchmark constituent can actually be bought in board lots; this keeps the
realised holdings industry-neutral as well.  Finally the target is blended
towards the previous feasible portfolio to respect the one-way turnover cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aicwf.backtest import costs as cost_model
from aicwf.backtest import metrics as metric_model
from aicwf.factors import common
from aicwf.portfolio.projection import project_industry_neutral_box

DEFAULT_BOARD_LIMITS: dict[str, float] = {
    "main": 0.10,
    "chinext": 0.20,
    "star": 0.20,
    "bse": 0.30,
    "st": 0.05,
    "limit_tolerance": 0.005,
}

GROWTH_BOARDS = {"chinext", "gem", "star", "kcb", "sse_star", "szse_gem", "300", "688", "cyb"}
BSE_BOARDS = {"bse", "bj", "beijing", "北交所", "北证"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    holdings: pd.DataFrame
    equity_curve: pd.DataFrame
    nav: pd.Series
    benchmark_nav: pd.Series
    metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any]
    by_year: dict[str, dict[str, Any]]
    constraints: dict[str, Any]
    warnings: dict[str, int]
    rebalances: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _board_limit(board: Any, is_st: bool, board_limits: Mapping[str, float]) -> float:
    if is_st:
        return float(board_limits.get("st", 0.05))
    text = str(board or "").strip().lower()
    if text in GROWTH_BOARDS or text in {"创业板", "科创板"}:
        return float(board_limits.get("chinext", 0.20))
    if text in BSE_BOARDS:
        return float(board_limits.get("bse", 0.30))
    return float(board_limits.get("main", 0.10))


def _lot_floor(shares: float, lot: int) -> int:
    return common.lot_floor(shares, lot)


def _safe_price(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) and number > 0 else float("nan")


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(index="date", columns="code", values=column, aggfunc="last")


def _as_wide(matrix: pd.DataFrame, dates: list[str], codes: list[str], fill: float = np.nan) -> pd.DataFrame:
    return matrix.reindex(index=dates, columns=codes).astype(float).fillna(fill)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestModel:
    """Reusable, pre-computed backtest model (robustness runs many variants)."""

    def __init__(
        self,
        panel: pd.DataFrame,
        benchmark: pd.DataFrame,
        factor_panel: pd.DataFrame,
        universe: pd.DataFrame | None,
        strategy: Mapping[str, Any],
        *,
        as_of: str | None = None,
        snapshot_prices: pd.DataFrame | None = None,
    ) -> None:
        self.strategy = dict(strategy)
        frame = pd.DataFrame(panel).copy()
        frame["date"] = frame["date"].astype(str).str[:10]
        frame["code"] = frame["code"].astype(str)
        end_iso = str(self._strategy_value("backtest", "end_date", default=frame["date"].max()))[:10]
        if as_of:
            end_iso = min(end_iso, str(as_of)[:10])
        dates = sorted(day for day in frame["date"].unique().tolist() if day <= end_iso)
        if not dates:
            raise ValueError("no panel trading dates inside the run window")
        self.dates = dates
        self.date_index = {day: index for index, day in enumerate(dates)}
        self.codes = sorted(frame["code"].unique().tolist())
        self.code_index = {code: index for index, code in enumerate(self.codes)}
        frame = frame[frame["date"].isin(set(dates))]
        self.close = _as_wide(_pivot(frame, "close"), dates, self.codes)
        self.valuation_close = self.close.ffill().bfill()
        self.ret = _as_wide(_pivot(frame, "ret"), dates, self.codes, fill=0.0)
        self.amount = _as_wide(_pivot(frame, "amount"), dates, self.codes, fill=0.0)
        self.suspended = common.reindex_bool(
            _pivot(frame.assign(_s=frame["suspended"].astype(bool)), "_s"),
            index=dates,
            columns=self.codes,
            default=True,
        )
        self.in_universe = common.reindex_bool(
            _pivot(frame.assign(_u=frame["in_universe"].astype(bool)), "_u"),
            index=dates,
            columns=self.codes,
            default=False,
        )
        industries = frame.pivot_table(index="date", columns="code", values="industry", aggfunc="last")
        self.industry_matrix = industries.reindex(index=dates, columns=self.codes).ffill().fillna("UNKNOWN")
        self.listed_days = (
            _pivot(frame, "listed_days").reindex(index=dates, columns=self.codes).astype(float).ffill()
        )

        universe_frame = pd.DataFrame(universe) if universe is not None else pd.DataFrame({"code": self.codes})
        universe_frame["code"] = universe_frame["code"].astype(str)
        universe_frame = universe_frame.drop_duplicates(subset=["code"], keep="last").set_index("code")
        if "weight" in universe_frame.columns:
            weights = pd.to_numeric(universe_frame["weight"], errors="coerce").fillna(0.0)
        else:
            weights = pd.Series(1.0, index=universe_frame.index, dtype=float)
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=universe_frame.index, dtype=float)
        self.universe_weights = weights
        self.is_st = (
            universe_frame["is_st"].fillna(False).astype(bool)
            if "is_st" in universe_frame.columns
            else pd.Series(False, index=universe_frame.index)
        )
        self.board = (
            universe_frame["board"].astype(str)
            if "board" in universe_frame.columns
            else pd.Series("main", index=universe_frame.index)
        )
        self.board_limits = dict(DEFAULT_BOARD_LIMITS)
        self.board_limits.update(dict(self._strategy_value("backtest", "board_limits", default={}) or {}))

        # Factor score matrices, restricted to the model calendar (dates <= end_iso).
        factor_frame = pd.DataFrame(factor_panel).copy()
        factor_frame["date"] = factor_frame["date"].astype(str).str[:10]
        factor_frame["code"] = factor_frame["code"].astype(str)
        factor_frame = factor_frame[factor_frame["date"] <= end_iso]
        self.factor_dates = sorted(factor_frame["date"].unique().tolist())
        self.composite = _pivot(factor_frame, "composite_score").reindex(index=self.factor_dates, columns=self.codes)
        self.eligible = common.reindex_bool(
            _pivot(factor_frame.assign(_e=factor_frame["eligible"].astype(bool)), "_e"),
            index=self.factor_dates,
            columns=self.codes,
            default=False,
        )
        self.factor_zscores: dict[str, pd.DataFrame] = {}
        for column in factor_frame.columns:
            if column.endswith("_z"):
                factor_id = column[: -len("_z")]
                matrix = _pivot(factor_frame, column).reindex(index=self.factor_dates, columns=self.codes)
                self.factor_zscores[factor_id] = matrix

        # Benchmark NAV (full frozen history, normalised at its first date).
        self.benchmark_nav_full = metric_model.benchmark_nav_from_frame(benchmark)

        # Price-limit returns: prefer frozen snapshot pct_chg, fall back to panel ret.
        self.limit_ret = self.ret
        if snapshot_prices is not None and len(snapshot_prices):
            price_frame = pd.DataFrame(snapshot_prices)
            if {"date", "code", "pct_chg"}.issubset(price_frame.columns):
                price_frame = price_frame.copy()
                price_frame["date"] = price_frame["date"].astype(str).str[:10]
                price_frame["code"] = price_frame["code"].astype(str)
                pivot = _pivot(price_frame, "pct_chg").reindex(index=dates, columns=self.codes)
                median = float(np.nanmedian(np.abs(pivot.to_numpy(dtype=float))))
                if np.isfinite(median) and median > 0.2:  # pct_chg stored in percent
                    pivot = pivot / 100.0
                self.limit_ret = pivot.fillna(self.ret)

    # -- configuration helpers ----------------------------------------------
    def _strategy_value(self, *path: str, default: Any = None) -> Any:
        return common.section(self.strategy, *path, default=default)

    def constraints(self) -> dict[str, Any]:
        values = dict(self._strategy_value("backtest", "constraints", default={}) or {})
        values.setdefault("max_weight", 1.0)
        values.setdefault("max_active_weight", 1.0)
        values.setdefault("industry_active_limit", 1.0)
        values.setdefault("turnover_limit", 1.0)
        values.setdefault("te_target", 0.05)
        values.setdefault("cash_buffer", 0.0)
        values.setdefault("min_names", 0)
        values.setdefault("benchmark_weight_cap", values["max_weight"])
        return values

    # -- decision grid -------------------------------------------------------
    def signal_date_for(self, decision_date: str, *, lag_months: int = 1) -> str | None:
        cutoff = decision_date
        if lag_months > 0:
            cutoff = common.previous_month_end(decision_date)
            for _ in range(max(0, int(lag_months) - 1)):
                cutoff = common.previous_month_end(cutoff)
        return common.latest_on_or_before(self.factor_dates, cutoff)

    def decision_grid(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        rebalance: str = "monthly",
        lag_months: int = 1,
    ) -> list[dict[str, str]]:
        start_iso = str(start or self._strategy_value("backtest", "start_date", default=self.dates[0]))[:10]
        end_iso = str(end or self._strategy_value("backtest", "end_date", default=self.dates[-1]))[:10]
        window = [day for day in self.dates if start_iso <= day <= end_iso]
        month_last: dict[str, str] = {}
        for day in window:
            month_last[day[:7]] = day
        months = sorted(month_last)
        if str(rebalance).lower() == "quarterly" and months:
            months = months[::3]
        grid: list[dict[str, str]] = []
        for month in months:
            decision = month_last[month]
            signal = self.signal_date_for(decision, lag_months=lag_months)
            if signal is None:
                continue
            index = self.date_index[decision]
            if index + 1 >= len(self.dates):
                continue
            grid.append(
                {
                    "decision_date": decision,
                    "execution_date": self.dates[index + 1],
                    "signal_date": signal,
                }
            )
        return grid

    # -- benchmark proxy -----------------------------------------------------
    def benchmark_proxy(self, *, cap: float) -> pd.Series:
        raw = self.universe_weights.copy()
        raw = raw[raw > 0]
        if raw.empty:
            raw = pd.Series(1.0, index=self.codes, dtype=float)
        capped_cap = float(cap)
        if len(raw) * capped_cap < 1.0:
            capped_cap = 1.0 / len(raw) + 1e-12
        projected = common.project_box_sum(
            raw.to_numpy(dtype=float),
            np.zeros(len(raw)),
            np.full(len(raw), capped_cap),
            1.0,
        )
        return pd.Series(projected, index=raw.index)

    # -- universe screening --------------------------------------------------
    def _screen(
        self,
        decision_date: str,
        signal_date: str,
        *,
        exclude_codes: set[str] | None = None,
    ) -> list[str]:
        excluded = set(exclude_codes or ())
        min_listed = float(self._strategy_value("universe", "min_listed_days", default=0) or 0)
        codes: list[str] = []
        for code in self.codes:
            if code in excluded:
                continue
            if code not in self.universe_weights.index or not (float(self.universe_weights.get(code, 0.0)) > 0):
                continue
            if code in self.is_st.index and bool(self.is_st.get(code, False)):
                continue
            if not bool(self.in_universe.at[decision_date, code]):
                continue
            if not np.isfinite(_safe_price(self.close.at[decision_date, code])):
                continue
            if signal_date not in self.eligible.index or not bool(self.eligible.at[signal_date, code]):
                continue
            listed = float(self.listed_days.at[decision_date, code])
            if np.isfinite(listed) and listed < min_listed:
                continue
            codes.append(code)
        return sorted(codes)

    # -- target construction -------------------------------------------------
    def target_weights(
        self,
        *,
        decision_date: str,
        signal_date: str,
        current_weights: pd.Series | None = None,
        scores: pd.DataFrame | None = None,
        constraints: Mapping[str, Any] | None = None,
        te_target: float | None = None,
        exclude_codes: set[str] | None = None,
        initial_capital: float | None = None,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Build the constrained target portfolio for one rebalance.

        Delegates the constrained construction to
        :func:`aicwf.portfolio.construction.construct_index_enhancement_weights`:
        untradable names are fixed at their current holding level, names that left
        the benchmark are forced to exit, and every industry total is set to the
        benchmark proxy's industry total (exact industry neutrality).
        """

        values = dict(constraints or self.constraints())
        excluded = set(exclude_codes or ())
        weight_margin = float(values.get("weight_margin", 0.985))
        decision_prices = self.valuation_close.loc[decision_date]
        member_codes = [
            code
            for code in self.codes
            if code not in excluded
            and code in self.universe_weights.index
            and float(self.universe_weights.get(code, 0.0)) > 0
            and np.isfinite(_safe_price(decision_prices.get(code)))
        ]
        supplied_current = (
            pd.Series(dtype=float) if current_weights is None else pd.Series(current_weights, dtype=float)
        )
        codes = sorted(set(member_codes) | set(supplied_current[supplied_current > 0].index.tolist()))
        if not codes:
            raise ValueError(f"no investable names on {decision_date} (signal {signal_date})")

        benchmark_cap = min(
            float(values.get("benchmark_weight_cap", values.get("max_weight", 1.0))),
            float(values.get("max_weight", 1.0)),
        ) * weight_margin
        wb_static = self.benchmark_proxy(cap=benchmark_cap).reindex(codes).fillna(0.0).astype(float)
        if float(wb_static.sum()) > 0:
            wb_static = wb_static / float(wb_static.sum())

        if scores is not None:
            alpha = scores.reindex(index=[signal_date], columns=self.codes).iloc[0]
        else:
            alpha = self.composite.loc[signal_date]
        alpha = alpha.reindex(codes).astype(float).fillna(0.0)

        decision_position = self.date_index[decision_date]
        window = self.dates[max(0, decision_position - 60) : decision_position + 1]
        sigma = (self.ret.loc[window, codes].std(ddof=1) * np.sqrt(252.0)).reindex(codes).fillna(0.0)

        industry = self.industry_matrix.loc[decision_date].reindex(codes).fillna("UNKNOWN").astype(str)
        in_universe = self.in_universe.loc[decision_date].reindex(codes).fillna(False).astype(bool)
        suspended = self.suspended.loc[decision_date].reindex(codes).fillna(True).astype(bool)
        raw_prices = self.close.loc[decision_date].reindex(codes)
        listed = self.listed_days.loc[decision_date].reindex(codes).astype(float)
        min_listed = float(self._strategy_value("universe", "min_listed_days", default=0) or 0)
        eligible = (
            self.eligible.reindex(index=[signal_date], columns=codes).iloc[0].fillna(False).astype(bool)
        )
        listed_values = np.nan_to_num(listed.to_numpy(dtype=float), nan=1e6)
        tradable = (
            in_universe.to_numpy(dtype=bool)
            & (~suspended.to_numpy(dtype=bool))
            & np.isfinite(raw_prices.to_numpy(dtype=float))
            & (listed_values >= min_listed)
            & eligible.to_numpy(dtype=bool)
            & np.array([code not in excluded for code in codes], dtype=bool)
        )

        capital = float(
            initial_capital
            if initial_capital is not None
            else self._strategy_value("backtest", "initial_capital", default=1.0e8)
        )
        lot = int(self._strategy_value("backtest", "lot_size", default=100) or 100)
        capital_floor = max(0.5 * capital, 1.0)
        limit_frac = np.array(
            [
                _board_limit(
                    self.board.get(code, "main"),
                    bool(self.is_st.get(code, False)),
                    self.board_limits,
                )
                for code in codes
            ],
            dtype=float,
        )
        prices = decision_prices.reindex(codes).astype(float)
        min_holding = pd.Series(
            lot * prices.to_numpy(dtype=float) * (1.0 + limit_frac + 0.02) / capital_floor,
            index=codes,
        )
        min_holding = min_holding.where(np.isfinite(min_holding), 0.0)

        if current_weights is None:
            current = wb_static.copy()
            first_rebalance = True
        else:
            current = supplied_current.reindex(codes).fillna(0.0).astype(float)
            first_rebalance = False

        from aicwf.portfolio.construction import construct_index_enhancement_weights

        weights, info = construct_index_enhancement_weights(
            alpha=alpha,
            benchmark=wb_static,
            industry=industry,
            prices=prices,
            sigma=sigma,
            current_weights=current,
            tradable=pd.Series(tradable, index=codes),
            constraints=values,
            te_target=te_target,
            min_holding=min_holding,
            lot=lot,
            capital=capital,
        )
        info["first_rebalance"] = first_rebalance
        info["n_names"] = int((weights > 0).sum())
        return weights.astype(float), info

    # -- execution -----------------------------------------------------------
    def _execute_rebalance(
        self,
        *,
        execution_date: str,
        target_weights: pd.Series,
        cash: float,
        positions: dict[str, int],
        benchmark_weights: Mapping[str, float],
        upper_bounds: Mapping[str, float],
        costs: Mapping[str, float],
        lot: int,
        cash_buffer: float,
        min_trade_notional: float,
        participation_cap: float,
        warnings: dict[str, int],
        ledger_rows: list[dict[str, Any]],
        holdings_rows: list[dict[str, Any]],
    ) -> tuple[float, dict[str, int], dict[str, Any]]:
        execution_index = self.date_index[execution_date]
        price_row = self.close.loc[execution_date]
        valuation_row = self.valuation_close.loc[execution_date]
        amount_row = self.amount.loc[execution_date]
        suspended_row = self.suspended.loc[execution_date]
        limit_row = self.limit_ret.loc[execution_date]
        tolerance = float(self.board_limits.get("limit_tolerance", 0.005))

        nav_pre = float(cash)
        for code, shares in positions.items():
            price = _safe_price(valuation_row.get(code))
            if np.isfinite(price):
                nav_pre += shares * price

        codes = sorted(set(target_weights.index.tolist()) | set(positions))
        target_shares: dict[str, int] = {}
        for code in codes:
            current = int(positions.get(code, 0))
            weight = float(target_weights.get(code, 0.0))
            price = _safe_price(price_row.get(code))
            is_suspended = bool(suspended_row.get(code, True))
            if weight <= 0.0:
                # Not in the target portfolio: exit the position when tradable.
                target_shares[code] = 0
                continue
            if is_suspended or not np.isfinite(price):
                target_shares[code] = current
                continue
            notional = weight * nav_pre * (1.0 - cash_buffer)
            desired = _lot_floor(notional / price, lot)
            if desired == 0 and current == 0:
                forced_weight = lot * price / nav_pre if nav_pre > 0 else float("inf")
                if forced_weight <= float(upper_bounds.get(code, weight)) + 1e-12:
                    desired = lot  # establish at least one board lot when within bounds
            target_shares[code] = desired

        def sellable(code: str) -> tuple[bool, str]:
            price = _safe_price(price_row.get(code))
            if not np.isfinite(price):
                warnings["missing_price"] = warnings.get("missing_price", 0) + 1
                return False, "missing_price"
            if bool(suspended_row.get(code, True)):
                warnings["suspended_skips"] = warnings.get("suspended_skips", 0) + 1
                return False, "suspended"
            limit = _board_limit(self.board.get(code, "main"), bool(self.is_st.get(code, False)), self.board_limits)
            daily = float(limit_row.get(code, 0.0))
            if np.isfinite(daily) and daily <= -(limit - tolerance):
                warnings["limit_down_blocked"] = warnings.get("limit_down_blocked", 0) + 1
                return False, "limit_down"
            return True, "ok"

        def buyable(code: str) -> tuple[bool, str]:
            price = _safe_price(price_row.get(code))
            if not np.isfinite(price):
                warnings["missing_price"] = warnings.get("missing_price", 0) + 1
                return False, "missing_price"
            if bool(suspended_row.get(code, True)):
                warnings["suspended_skips"] = warnings.get("suspended_skips", 0) + 1
                return False, "suspended"
            limit = _board_limit(self.board.get(code, "main"), bool(self.is_st.get(code, False)), self.board_limits)
            daily = float(limit_row.get(code, 0.0))
            if np.isfinite(daily) and daily >= (limit - tolerance):
                warnings["limit_up_blocked"] = warnings.get("limit_up_blocked", 0) + 1
                return False, "limit_up"
            return True, "ok"

        # Sells first (frees cash and never sells shares acquired today).
        for code in codes:
            current = int(positions.get(code, 0))
            desired = int(target_shares.get(code, current))
            if desired >= current:
                continue
            delta = current - desired
            price = _safe_price(price_row.get(code))
            if not np.isfinite(price):
                warnings["missing_price"] = warnings.get("missing_price", 0) + 1
                continue
            allowed, reason = sellable(code)
            if not allowed:
                continue
            notional = delta * price
            if desired > 0 and notional < min_trade_notional:
                warnings["small_trade_skips"] = warnings.get("small_trade_skips", 0) + 1
                continue
            participation = notional / float(amount_row.get(code, 0.0)) if float(amount_row.get(code, 0.0)) > 0 else 0.0
            if participation_cap > 0 and participation > participation_cap:
                allowed_shares = _lot_floor(participation_cap * float(amount_row.get(code, 0.0)) / price, lot)
                delta = min(delta, allowed_shares)
                if delta <= 0:
                    warnings["participation_capped"] = warnings.get("participation_capped", 0) + 1
                    continue
                notional = delta * price
            cost = cost_model.trade_cost(notional, "sell", costs, adv=float(amount_row.get(code, 0.0)))
            cash += cost["cash_flow"]
            positions[code] = current - delta
            if positions[code] == 0:
                positions.pop(code, None)
            ledger_rows.append(
                {
                    "date": execution_date,
                    "code": code,
                    "side": "sell",
                    "shares": int(delta),
                    "price": float(price),
                    "gross_amount": float(notional),
                    "commission": float(cost["commission"]),
                    "stamp_duty": float(cost["stamp_duty"]),
                    "slippage": float(cost["slippage"]),
                    "total_cost": float(cost["total_cost"]),
                    "cash_flow": float(cost["cash_flow"]),
                    "reason": "rebalance_sell" if desired > 0 else "exit_position",
                }
            )

        # Buys, limited by remaining cash.
        fee_buffer = max(
            0.002,
            (
                float(costs.get("commission_bps", 0.0))
                + float(costs.get("slippage_bps", 0.0))
                + float(costs.get("stamp_duty_bps", 0.0))
            )
            / 1e4
            + 0.001,
        )
        for code in codes:
            current = int(positions.get(code, 0))
            desired = int(target_shares.get(code, current))
            if desired <= current:
                continue
            delta = desired - current
            price = _safe_price(price_row.get(code))
            if not np.isfinite(price):
                warnings["missing_price"] = warnings.get("missing_price", 0) + 1
                continue
            allowed, reason = buyable(code)
            if not allowed:
                continue
            notional = delta * price
            if current > 0 and notional < min_trade_notional:
                warnings["small_trade_skips"] = warnings.get("small_trade_skips", 0) + 1
                continue
            participation = (
                notional / float(amount_row.get(code, 0.0)) if float(amount_row.get(code, 0.0)) > 0 else 0.0
            )
            if participation_cap > 0 and participation > participation_cap:
                allowed_shares = _lot_floor(participation_cap * float(amount_row.get(code, 0.0)) / price, lot)
                delta = min(delta, allowed_shares)
                if delta <= 0:
                    warnings["participation_capped"] = warnings.get("participation_capped", 0) + 1
                    continue
                notional = delta * price
            affordable = _lot_floor(cash / (price * (1.0 + fee_buffer)), lot)
            if affordable < delta:
                delta = affordable
                warnings["insufficient_cash"] = warnings.get("insufficient_cash", 0) + 1
            if delta <= 0:
                continue
            notional = delta * price
            cost = cost_model.trade_cost(notional, "buy", costs, adv=float(amount_row.get(code, 0.0)))
            cash += cost["cash_flow"]
            positions[code] = current + delta
            ledger_rows.append(
                {
                    "date": execution_date,
                    "code": code,
                    "side": "buy",
                    "shares": int(delta),
                    "price": float(price),
                    "gross_amount": float(notional),
                    "commission": float(cost["commission"]),
                    "stamp_duty": float(cost["stamp_duty"]),
                    "slippage": float(cost["slippage"]),
                    "total_cost": float(cost["total_cost"]),
                    "cash_flow": float(cost["cash_flow"]),
                    "reason": "rebalance_buy",
                }
            )

        if cash < 0.0:
            warnings["negative_cash_repair"] = warnings.get("negative_cash_repair", 0) + 1
            cash = self._repair_negative_cash(cash, positions, ledger_rows)

        # Enforce the published weight / active-weight caps on tradable holdings
        # that drifted since the previous rebalance (small trades are skipped and
        # prices move); names bought today cannot be sold again (T+1).
        bought_today = {
            str(row["code"])
            for row in ledger_rows
            if row["side"] == "buy" and str(row["date"]) == execution_date
        }
        cash = self._enforce_holding_bounds(
            execution_date=execution_date,
            cash=cash,
            positions=positions,
            benchmark_weights=benchmark_weights,
            bought_today=bought_today,
            costs=costs,
            lot=lot,
            warnings=warnings,
            ledger_rows=ledger_rows,
        )

        # Holdings snapshot after trades (date = execution date).
        nav_post = float(cash)
        for code, shares in positions.items():
            price = _safe_price(valuation_row.get(code))
            if np.isfinite(price):
                nav_post += shares * price
        for code in sorted(positions):
            shares = int(positions[code])
            if shares <= 0:
                continue
            price = _safe_price(valuation_row.get(code))
            if not np.isfinite(price):
                continue
            industry = str(self.industry_matrix.at[execution_date, code])
            weight = shares * price / nav_post if nav_post > 0 else 0.0
            benchmark_weight = float(benchmark_weights.get(code, 0.0))
            holdings_rows.append(
                {
                    "date": execution_date,
                    "code": code,
                    "shares": int(shares),
                    "price": float(price),
                    "weight": float(weight),
                    "benchmark_weight": float(benchmark_weight),
                    "active_weight": float(weight - benchmark_weight),
                    "industry": industry,
                }
            )
        return cash, positions, {"nav": nav_post}

    @staticmethod
    def _repair_negative_cash(
        cash: float, positions: dict[str, int], ledger_rows: list[dict[str, Any]]
    ) -> float:
        """Undo the most recent buys until cash is non-negative (defensive only)."""

        index = len(ledger_rows) - 1
        while cash < -1e-6 and index >= 0:
            row = ledger_rows[index]
            if row["side"] != "buy":
                index -= 1
                continue
            code = str(row["code"])
            shares = int(row["shares"])
            positions[code] = int(positions.get(code, 0)) - shares
            if positions[code] <= 0:
                positions.pop(code, None)
            cash -= float(row["cash_flow"])
            ledger_rows.pop(index)
            index -= 1
        return cash

    def _enforce_holding_bounds(
        self,
        *,
        execution_date: str,
        cash: float,
        positions: dict[str, int],
        benchmark_weights: Mapping[str, float],
        bought_today: set[str],
        costs: Mapping[str, float],
        lot: int,
        warnings: dict[str, int],
        ledger_rows: list[dict[str, Any]],
    ) -> float:
        """Trim / top-up tradable holdings whose weights drifted outside the caps.

        The verifier checks the weight and active-weight bounds on the holdings
        artifact, which is dated at each execution date.  Between rebalances a
        position's weight drifts with relative prices and skipped small trades, so
        before writing the holdings snapshot we bring every *tradable* holding
        back inside the published bounds.  Shares bought on this execution date
        cannot be sold again (T+1).
        """

        raw_constraints = dict(self._strategy_value("backtest", "constraints", default={}) or {})
        raw_max_weight = float(raw_constraints.get("max_weight", 1.0))
        raw_max_active = float(raw_constraints.get("max_active_weight", 1.0))
        margin = 1e-6
        price_row = self.close.loc[execution_date]
        valuation_row = self.valuation_close.loc[execution_date]
        suspended_row = self.suspended.loc[execution_date]
        limit_row = self.limit_ret.loc[execution_date]
        tolerance = float(self.board_limits.get("limit_tolerance", 0.005))
        fee_buffer = max(
            0.004,
            (
                float(costs.get("commission_bps", 0.0))
                + float(costs.get("slippage_bps", 0.0))
                + float(costs.get("stamp_duty_bps", 0.0))
            )
            / 1e4,
        )
        for _ in range(3):
            nav = float(cash)
            for code, shares in positions.items():
                price = _safe_price(valuation_row.get(code))
                if np.isfinite(price):
                    nav += shares * price
            if nav <= 0:
                break
            changed = False
            for code in sorted(positions):
                shares = int(positions[code])
                price = _safe_price(price_row.get(code))
                valuation = _safe_price(valuation_row.get(code))
                if not np.isfinite(price) or not np.isfinite(valuation):
                    continue
                if bool(suspended_row.get(code, True)):
                    continue
                weight = shares * valuation / nav
                benchmark_weight = float(benchmark_weights.get(code, 0.0))
                upper = min(raw_max_weight, benchmark_weight + raw_max_active) - margin
                lower = max(0.0, benchmark_weight - raw_max_active) + margin
                limit = _board_limit(
                    self.board.get(code, "main"), bool(self.is_st.get(code, False)), self.board_limits
                )
                daily_return = float(limit_row.get(code, 0.0))
                if weight > upper and code not in bought_today:
                    if np.isfinite(daily_return) and daily_return <= -(limit - tolerance):
                        warnings["limit_down_blocked"] = warnings.get("limit_down_blocked", 0) + 1
                        continue
                    excess_shares = int(np.ceil((weight - upper) * nav / price / lot) * lot)
                    quantity = min(shares, excess_shares)
                    if quantity <= 0:
                        continue
                    notional = quantity * price
                    cost = cost_model.trade_cost(
                        notional, "sell", costs, adv=float(self.amount.at[execution_date, code])
                    )
                    cash += cost["cash_flow"]
                    positions[code] = shares - quantity
                    if positions[code] <= 0:
                        positions.pop(code, None)
                    ledger_rows.append(
                        {
                            "date": execution_date,
                            "code": code,
                            "side": "sell",
                            "shares": int(quantity),
                            "price": float(price),
                            "gross_amount": float(notional),
                            "commission": float(cost["commission"]),
                            "stamp_duty": float(cost["stamp_duty"]),
                            "slippage": float(cost["slippage"]),
                            "total_cost": float(cost["total_cost"]),
                            "cash_flow": float(cost["cash_flow"]),
                            "reason": "bounds_repair_trim",
                        }
                    )
                    warnings["bounds_repair_sells"] = warnings.get("bounds_repair_sells", 0) + 1
                    changed = True
                    break
                if weight < lower and benchmark_weight > 0 and cash > 0:
                    deficit_shares = int(np.floor((lower - weight) * nav / price / lot) * lot)
                    affordable = int(
                        np.floor(cash / (price * (1.0 + fee_buffer)) / lot) * lot
                    )
                    quantity = min(deficit_shares, affordable)
                    if quantity <= 0:
                        continue
                    notional = quantity * price
                    cost = cost_model.trade_cost(
                        notional, "buy", costs, adv=float(self.amount.at[execution_date, code])
                    )
                    cash += cost["cash_flow"]
                    positions[code] = shares + quantity
                    bought_today.add(code)
                    ledger_rows.append(
                        {
                            "date": execution_date,
                            "code": code,
                            "side": "buy",
                            "shares": int(quantity),
                            "price": float(price),
                            "gross_amount": float(notional),
                            "commission": float(cost["commission"]),
                            "stamp_duty": float(cost["stamp_duty"]),
                            "slippage": float(cost["slippage"]),
                            "total_cost": float(cost["total_cost"]),
                            "cash_flow": float(cost["cash_flow"]),
                            "reason": "bounds_repair_topup",
                        }
                    )
                    warnings["bounds_repair_buys"] = warnings.get("bounds_repair_buys", 0) + 1
                    changed = True
                    break
            if not changed:
                break
        return cash

    # -- main entry point ----------------------------------------------------
    def run(
        self,
        *,
        scores: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
        rebalance: str | None = None,
        cost_multiplier: float = 1.0,
        te_target: float | None = None,
        constraints: Mapping[str, Any] | None = None,
        exclude_codes: Sequence[str] = (),
        initial_capital: float | None = None,
        drop_rebalances: int = 0,
    ) -> BacktestResult:
        values = dict(constraints or self.constraints())
        cfg_rebalance = str(rebalance or self._strategy_value("backtest", "rebalance", default="monthly"))
        lag_months = int(self._strategy_value("backtest", "signal_lag_months", default=1) or 0)
        grid = self.decision_grid(start=start, end=end, rebalance=cfg_rebalance, lag_months=lag_months)
        if drop_rebalances:
            grid = grid[int(drop_rebalances) :]
        if not grid:
            raise ValueError("no rebalance dates available for the requested backtest window")

        capital = float(
            initial_capital
            if initial_capital is not None
            else self._strategy_value("backtest", "initial_capital", default=1.0e8)
        )
        lot = int(self._strategy_value("backtest", "lot_size", default=100) or 100)
        costs = cost_model.scaled_costs(self._strategy_value("backtest", "costs", default={}), cost_multiplier)
        min_trade_notional = float(costs.get("min_trade_notional", 0.0) or 0.0)
        cash_buffer = float(values.get("cash_buffer", 0.0) or 0.0)
        participation_cap = float(self._strategy_value("portfolio", "participation_cap", default=0.10) or 0.0)
        excluded = set(str(code) for code in exclude_codes)

        cash = capital
        positions: dict[str, int] = {}
        ledger_rows: list[dict[str, Any]] = []
        holdings_rows: list[dict[str, Any]] = []
        state_records: list[tuple[int, float, dict[str, int]]] = []
        warnings: dict[str, int] = {"n_rebalances": 0, "n_trades": 0}
        rebalances: list[dict[str, Any]] = []

        for item in grid:
            decision = item["decision_date"]
            execution = item["execution_date"]
            signal = item["signal_date"]
            nav_decision = float(cash)
            values_at_decision: dict[str, float] = {}
            for code, shares in positions.items():
                price = _safe_price(self.valuation_close.at[decision, code])
                if np.isfinite(price):
                    nav_decision += shares * price
                    values_at_decision[code] = shares * price
            current_weights = (
                pd.Series(
                    {
                        code: value / nav_decision
                        for code, value in values_at_decision.items()
                        if nav_decision > 0
                    },
                    dtype=float,
                )
                if nav_decision > 0
                else pd.Series(dtype=float)
            )
            weights, info = self.target_weights(
                decision_date=decision,
                signal_date=signal,
                current_weights=current_weights if state_records else None,
                scores=scores,
                constraints=values,
                te_target=te_target,
                exclude_codes=excluded,
                initial_capital=capital,
            )
            cash, positions, execution_info = self._execute_rebalance(
                execution_date=execution,
                target_weights=weights,
                cash=cash,
                positions=positions,
                benchmark_weights=info["benchmark_weights"],
                upper_bounds=info.get("upper_bounds", {}),
                costs=costs,
                lot=lot,
                cash_buffer=cash_buffer,
                min_trade_notional=min_trade_notional,
                participation_cap=participation_cap,
                warnings=warnings,
                ledger_rows=ledger_rows,
                holdings_rows=holdings_rows,
            )
            warnings["n_rebalances"] = warnings.get("n_rebalances", 0) + 1
            state_records.append((self.date_index[execution], cash, dict(positions)))
            rebalances.append(
                {
                    "decision_date": decision,
                    "execution_date": execution,
                    "signal_date": signal,
                    "turnover": float(info.get("turnover", 0.0)),
                    "n_names": int(info.get("n_held", 0)),
                    "te_target": float(info.get("te_target", 0.0)),
                    "te_estimate": float(info.get("te_estimate_diagonal", 0.0)),
                    "max_weight": float(info.get("max_weight", 0.0)),
                    "max_active": float(info.get("max_active", 0.0)),
                    "industry_active_max": float(info.get("industry_active_max", 0.0)),
                    "projection_status": str(info.get("projection_status", "")),
                }
            )

        warnings["n_trades"] = len(ledger_rows)
        ledger = pd.DataFrame(
            ledger_rows,
            columns=[
                "date",
                "code",
                "side",
                "shares",
                "price",
                "gross_amount",
                "commission",
                "stamp_duty",
                "slippage",
                "total_cost",
                "cash_flow",
                "reason",
            ],
        )
        holdings = pd.DataFrame(
            holdings_rows,
            columns=[
                "date",
                "code",
                "shares",
                "price",
                "weight",
                "benchmark_weight",
                "active_weight",
                "industry",
            ],
        )
        nav, benchmark_nav = self._build_nav_path(state_records, capital)
        equity_curve = self._build_equity_curve(nav, benchmark_nav)
        metrics = metric_model.compute_performance_metrics(nav, self.benchmark_nav_full)
        benchmark_metrics = metric_model.compute_benchmark_metrics(self.benchmark_nav_full)
        by_year = metric_model.by_year_table(nav, benchmark_nav)
        return BacktestResult(
            ledger=ledger,
            holdings=holdings,
            equity_curve=equity_curve,
            nav=nav,
            benchmark_nav=benchmark_nav,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            by_year=by_year,
            constraints=dict(values),
            warnings={key: int(value) for key, value in warnings.items()},
            rebalances=rebalances,
        )

    # -- NAV / equity curve --------------------------------------------------
    def _build_nav_path(
        self, state_records: Sequence[tuple[int, float, dict[str, int]]], capital: float
    ) -> tuple[pd.Series, pd.Series]:
        n_dates = len(self.dates)
        n_codes = len(self.codes)
        positions_matrix = np.zeros((n_dates, n_codes), dtype=float)
        cash_series = np.full(n_dates, float(capital), dtype=float)
        cursor = 0
        current_positions = np.zeros(n_codes, dtype=float)
        current_cash = float(capital)
        for index in range(n_dates):
            while cursor < len(state_records) and state_records[cursor][0] <= index:
                _, current_cash, state_positions = state_records[cursor]
                current_positions = np.zeros(n_codes, dtype=float)
                for code, shares in state_positions.items():
                    code_index = self.code_index.get(code)
                    if code_index is not None:
                        current_positions[code_index] = float(shares)
                cursor += 1
            positions_matrix[index] = current_positions
            cash_series[index] = current_cash
        valuation = self.valuation_close.to_numpy(dtype=float)
        valuation = np.where(np.isfinite(valuation), valuation, 0.0)
        holdings_value = positions_matrix * valuation
        nav_raw = cash_series + holdings_value.sum(axis=1)
        nav = pd.Series(nav_raw / float(capital), index=pd.Index(self.dates, name="date"), dtype=float)
        benchmark = self.benchmark_nav_full.reindex(self.dates)
        benchmark = benchmark.ffill()
        return nav, benchmark

    def _build_equity_curve(self, nav: pd.Series, benchmark_nav: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame({"date": nav.index.tolist(), "nav": nav.to_numpy(dtype=float)})
        frame["benchmark_nav"] = benchmark_nav.reindex(nav.index).to_numpy(dtype=float)
        frame["active_nav"] = frame["nav"] / frame["benchmark_nav"]
        frame["daily_return"] = frame["nav"].pct_change()
        frame["benchmark_return"] = frame["benchmark_nav"].pct_change()
        frame["excess_return"] = frame["daily_return"] - frame["benchmark_return"]
        running_max = np.maximum.accumulate(frame["nav"].to_numpy(dtype=float))
        frame["drawdown"] = frame["nav"].to_numpy(dtype=float) / running_max - 1.0
        return frame.loc[
            :,
            [
                "date",
                "nav",
                "benchmark_nav",
                "active_nav",
                "daily_return",
                "benchmark_return",
                "excess_return",
                "drawdown",
            ],
        ]


__all__ = ["BacktestModel", "BacktestResult", "DEFAULT_BOARD_LIMITS"]
