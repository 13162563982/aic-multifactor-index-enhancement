# Hypothesis rationale (deterministic demo)

Variant: **rejected** - Candidate v1.2.0: weekly rebalance (turnover regression).

The short-horizon reversal effect is motivated by transitory liquidity demand. The signal is lagged to the decision date, execution happens at the next trading day's close, and the active tilt is scaled so that the portfolio stays inside the index-enhancement bounds. Industry active weights are limited by the demo config; the universe is the frozen mini fixture with six synthetic names.

Information cutoff: **2024-12-31** (no observation after the cutoff is used for the signal).

Signal lookback: 15 trading days; rebalance: weekly; amplitude: 0.03.
