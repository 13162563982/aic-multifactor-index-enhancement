# Skill: risk review (devil's advocate)

## Purpose

Attack the evidence for a strategy before a human does. Used by the risk-critic agent to
produce a quantified, structured critique of the backtest, robustness suite and portfolio
construction so the approver sees the weakest points first.

## When to use

- After `05_backtest` and `06_robustness` artifacts exist and before `11_approval`.
- On monitoring cycles, when a strategy's realised tracking error or drawdown breaches
  its alert thresholds and the robustness of the original evidence must be re-read.

## Inputs

- `metrics.json`: strategy/benchmark metrics, by-year table, constraints.
- `robustness.json` and `robustness_checks.json`: subperiods, cost sensitivity, factor
  leave-one-out, parameter grid, placebo, bias stress, deflated Sharpe, check statuses.
- `risk_report.json`: ex-ante tracking error, industry/factor active exposure, turnover,
  number of names, optimizer status and fallback flag.

## Procedure

1. **Start from failures.** List every check whose status is not PASS with its measured
   value and threshold. These are the highest-value findings.
2. **Regime test.** Compare subperiod excess returns; name any subperiod with negative or
   flat excess return. State whether the full-sample result depends on one regime.
3. **Cost test.** Compare 1x and 3x cost results; note the degradation and any turnover
   increase. A strategy that only works at zero cost is a false positive.
4. **Concentration test.** Use factor leave-one-out: if removing one factor destroys the
   edge, the strategy is a single-factor bet. Also check if a small number of names or
   months dominates performance where the evidence allows.
5. **Parameter sensitivity.** Check the TE-target / rebalance grid for sign changes; a
   sign flip across adjacent parameters means the reported point is not robust.
6. **Randomness test.** Placebo score runs must be clearly worse than the strategy;
   deflated Sharpe must remain acceptable after counting trials.
7. **Bias stress.** If dropping the top-k full-period winners erases the edge, flag
   survivorship/bias dependence explicitly.
8. **Construction risk.** Ex-ante TE close to its cap, large industry/factor active
   exposure, high turnover or optimizer fallback are implementation risks to disclose.
9. **Write the critique.** Sections: Verdict on robustness; Failed or weak checks;
   Threats to validity; What would change our mind. Cite at least three concrete numbers
   with the artifact they come from; label speculation as hypothesis.

## Checks

- Every cited number matches an input artifact and is quoted at the artifact's precision.
- Failed/WARN checks are named first; no cherry-picking of passes.
- Cutoff date stated; no claim about live or post-cutoff performance.
- No trade recommendation, price target or external citation.

## Pitfalls

- **Summary instead of critique.** Restating the marketing metrics is not a review; the
  note must say what is weak.
- **Unfalsifiable worries.** "Risks remain" is noise; say which measurement would
  invalidate the strategy.
- **Numbers without provenance.** Each number must be traceable to `metrics`,
  `robustness_checks` or `risk_report`.
- **Ignoring the optimizer.** Fallback weights or a binding industry constraint can mean
  the realised portfolio is not the intended one.
