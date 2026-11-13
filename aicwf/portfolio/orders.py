"""Order generation: target weights -> board-lot share orders.

Orders are computed from the constrained target weights of the latest decision
date and the shares currently held (from the backtest's final holdings).  Share
quantities are floored to the A-share board lot and capped by a participation
limit (``portfolio.participation_cap``) against the decision day's traded amount.

``participation_pct`` is stored as a **fraction** of that day's traded amount
(e.g. ``0.05`` means 5% of ADV); this convention is documented in
``risk_report.json`` under ``participation``.  No order is emitted for zero
share changes, and a partial adjustment smaller than ``min_trade_notional`` is
skipped (a full exit is always executed).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from aicwf.factors import common

ORDER_COLUMNS = (
    "date",
    "code",
    "side",
    "shares",
    "weight_change",
    "notional",
    "participation_pct",
)


def build_orders(
    *,
    date: str,
    target_weights: pd.Series,
    current_shares: Mapping[str, Any],
    nav: float,
    prices: pd.Series,
    amount: pd.Series | None = None,
    lot_size: int = 100,
    participation_cap: float = 0.10,
    min_trade_notional: float = 0.0,
) -> pd.DataFrame:
    """Build the order list to move from current shares to target weights."""

    nav = float(nav)
    rows: list[dict[str, Any]] = []
    if nav <= 0:
        return pd.DataFrame(columns=list(ORDER_COLUMNS))
    lot = int(lot_size) if int(lot_size) > 1 else 1
    codes = sorted(set(target_weights.index.tolist()) | {str(code) for code in current_shares})
    for code in codes:
        price = float(prices.get(code, float("nan")))
        if not np.isfinite(price) or price <= 0:
            continue
        weight = float(target_weights.get(code, 0.0))
        target_shares = common.lot_floor(weight * nav / price, lot)
        current = int(current_shares.get(code, 0) or 0)
        delta = int(target_shares) - current
        if delta == 0:
            continue
        notional = abs(delta) * price
        if (
            min_trade_notional > 0
            and current > 0
            and target_shares > 0
            and notional < float(min_trade_notional)
        ):
            continue
        adv = float(amount.get(code, 0.0)) if amount is not None else 0.0
        participation = (notional / adv) if adv and np.isfinite(adv) and adv > 0 else 0.0
        if participation_cap > 0 and adv > 0 and participation > float(participation_cap):
            allowed = common.lot_floor(float(participation_cap) * adv / price, lot)
            if allowed <= 0:
                continue
            delta = int(np.sign(delta)) * int(allowed)
            if delta == 0:
                continue
            notional = abs(delta) * price
            participation = notional / adv
        side = "buy" if delta > 0 else "sell"
        rows.append(
            {
                "date": str(date)[:10],
                "code": str(code),
                "side": side,
                "shares": int(abs(delta)),
                "weight_change": float(delta * price / nav),
                "notional": float(notional),
                "participation_pct": float(participation),
            }
        )
    frame = pd.DataFrame(rows, columns=list(ORDER_COLUMNS))
    if len(frame):
        frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return frame


__all__ = ["ORDER_COLUMNS", "build_orders"]
