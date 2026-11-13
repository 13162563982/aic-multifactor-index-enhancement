"""Transaction-cost model for the monthly rebalance backtest.

Cost components (all rates are one-way percentages of the traded notional, bps
in configuration):

* ``commission``   — ``commission_bps`` on both buys and sells;
* ``stamp_duty``   — ``stamp_duty_bps`` on sells only (A-share 2023 rule);
* ``slippage``     — ``slippage_bps`` on both sides;
* ``impact``       — square-root market-impact model
  ``impact_bps = impact_coef * 100 * sqrt(notional / ADV)`` where ``ADV`` is the
  execution day's traded amount.  The impact cost is **folded into the ledger's
  ``slippage`` column** so the contract's ledger identity
  ``total_cost = commission + stamp_duty + slippage`` holds exactly.

The engine executes at the next day's close and pays these costs explicitly in
cash; the verifier reconciles the same identities from the written ledger.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

DEFAULT_COSTS: dict[str, float] = {
    "commission_bps": 2.5,
    "stamp_duty_bps": 5.0,
    "slippage_bps": 5.0,
    "impact_coef": 0.10,
    "min_trade_notional": 10000.0,
}


def resolve_costs(costs: Mapping[str, Any] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_COSTS)
    for key, value in dict(costs or {}).items():
        if key in resolved and value is not None:
            try:
                resolved[key] = float(value)
            except (TypeError, ValueError):
                continue
    return resolved


def scaled_costs(costs: Mapping[str, Any] | None, multiplier: float) -> dict[str, float]:
    """Scale every *rate* by ``multiplier`` (``0`` = frictionless sensitivity run)."""

    resolved = resolve_costs(costs)
    factor = float(multiplier)
    for key in ("commission_bps", "stamp_duty_bps", "slippage_bps", "impact_coef"):
        resolved[key] = float(resolved[key]) * factor
    return resolved


def impact_bps(notional: float, adv: float | None, impact_coef: float) -> float:
    """Square-root market impact in basis points."""

    if adv is None:
        return 0.0
    notional = abs(float(notional))
    adv = float(adv)
    if not math.isfinite(adv) or adv <= 0.0 or not math.isfinite(notional) or notional <= 0.0:
        return 0.0
    participation = notional / adv
    return float(impact_coef) * 100.0 * math.sqrt(participation)


def trade_cost(
    notional: float,
    side: str,
    costs: Mapping[str, Any] | None,
    *,
    adv: float | None = None,
) -> dict[str, float]:
    """Return the cost breakdown for one executed trade.

    ``notional`` is the absolute traded value (shares * price).  The returned
    dictionary contains the exact ledger columns (``slippage`` already includes
    the impact component) plus the raw ``impact`` number for reporting.
    """

    resolved = resolve_costs(costs)
    notional = abs(float(notional))
    side = str(side).lower()
    if side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    commission = notional * resolved["commission_bps"] / 1e4
    stamp_duty = notional * resolved["stamp_duty_bps"] / 1e4 if side == "sell" else 0.0
    slippage_component = notional * resolved["slippage_bps"] / 1e4
    impact_component = notional * impact_bps(notional, adv, resolved["impact_coef"]) / 1e4
    slippage = slippage_component + impact_component
    total = commission + stamp_duty + slippage
    cash_flow = -(notional + total) if side == "buy" else notional - total
    return {
        "commission": float(commission),
        "stamp_duty": float(stamp_duty),
        "slippage": float(slippage),
        "impact": float(impact_component),
        "total_cost": float(total),
        "cash_flow": float(cash_flow),
    }


__all__ = ["DEFAULT_COSTS", "impact_bps", "resolve_costs", "scaled_costs", "trade_cost"]
