# Hypothesis rationale

Information cutoff: 2026-11-12. The hypothesis uses only the frozen CSI 300 snapshot available on or before this date; no live data or external sources are used.

## Thesis

We posit that a tilt toward high-quality earnings (Quality) and low-risk characteristics (Risk) compensates for the benchmark's exposure to speculative momentum and liquidity-driven noise. Specifically, we expect excess return from firms with strong operating cash flow relative to EPS and low realized volatility, as these factors capture fundamental stability and downside protection absent in the CSI 300 index. The mechanism relies on the risk premium of avoiding idiosyncratic crash risk and the behavioral bias against 'boring' high-quality stocks during market stress.

## Factor tilts

| factor | weight | category | direction | description |
|---|---|---|---|---|
| quality_cashflow | 0.25 | quality | 1 | operating cash flow per share / EPS |
| low_vol_60d | 0.25 | risk | -1 | 60 day realized volatility |
| value_ep | 0.25 | value | 1 | trailing EPS / price |
| momentum_12_1 | 0.25 | momentum | 1 | 12-1 month return |

## Rebalance and risk constraints

- Rebalance: monthly
- Max single weight: 0.03
- Max active weight: 0.015
- Tracking-error target: 0.03
- Turnover limit: 0.3

## Falsification criteria

1. The full-sample RankIC t-statistic for the composite factor must exceed 2.0; failure indicates no statistical edge.
2. Excess return after applying 3x transaction costs must remain positive over the backtest horizon; otherwise, the alpha is not economically viable.
3. Sign stability: The individual factor ICs must maintain the same sign (positive for long factors, negative for short factors) in at least 70% of rolling 12-month subperiods.
4. Deflated Sharpe Ratio must exceed 1.5; if lower, returns are likely driven by common risk factors rather than idiosyncratic skill.

## Information and source limits

- Research relies exclusively on the frozen snapshot provided (as_of: 2026-11-12).
- No network access or live data feeds are permitted during this stage.
- Fundamental data is treated as point-in-time available only after its announcement date.
- Universe consists of current CSI 300 constituents with survivorship bias caveats noted in the manifest.
- Factor calculations must strictly adhere to the direction signs defined in the catalog.
- No external news or post-cutoff events may influence the hypothesis construction.

Deterministic later stages (factor build, backtest, robustness, verification) test this contract; any claim of performance is intentionally absent here.
