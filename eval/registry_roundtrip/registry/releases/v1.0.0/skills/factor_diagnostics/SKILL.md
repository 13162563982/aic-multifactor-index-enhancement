# Skill: factor diagnostics

## Purpose

Read a deterministic factor-report artifact critically and produce traceable advisory
flags. Used by the factor analyst agent after the factor build and before the backtest is
trusted.

## When to use

- After `03_factors/factor_report.json` has been produced for a new snapshot, hypothesis
  or preprocessing change.
- When a human asks whether the factor set is healthy enough to proceed.

## Inputs

- `factor_report.json`: `coverage`, `rank_ic`, `ic_t_stat`, `autocorrelation`,
  `correlation_matrix`, `composite_weights`, `warnings`.
- The hypothesis actually used: factor weights, rebalance, cutoff.
- Data-quality and PIT-audit digests when available.
- Configured thresholds: minimum cross-section size, minimum price coverage.

## Procedure

1. **Coverage first.** For each factor, confirm cross-sectional coverage is adequate in
   every period, not only on average. Thin coverage early in the sample invalidates IC
   comparisons across periods.
2. **Sign and significance.** Compare each factor's RankIC mean with the sign implied by
   its catalog direction. Read the IC t-statistic: a mean IC with |t| below roughly 2 is
   not distinguishable from noise. Treat marginal IC as a risk, not a result.
3. **Stability.** Inspect IC autocorrelation and per-year IC. A factor whose IC is
   carried by one or two months is a hidden event risk, not a premium.
4. **Crowding.** From the correlation matrix, identify factor pairs above ~0.6; the
   composite is less diversified than the weights suggest. Say which pair and how
   correlated.
5. **Composite sanity.** Check that the composite weights match the hypothesis
   normalised weights and that neutralisation (industry, log size) was applied.
6. **Diagnostics from the build.** Promote every construction warning into an explicit
   finding; do not explain warnings away.
7. **Write flags.** One flag per finding, stable snake_case id, severity chosen
   conservatively: `info` for observations, `warn` for material concerns, `block` only
   when you can name the measurement that should stop the pipeline. `warn`/`block`
   messages must contain the diagnostic number.
8. **Write the assessment.** Narrative for a human: what was measured, what is healthy,
   what is not, and what it implies for downstream stages.

## Checks

- Every flagged claim maps to a number present in `factor_report.json`.
- `warn`/`block` messages contain the number; ids are unique and stable.
- No factor modified; no re-computation; flags are advisory only.
- The assessment states the information cutoff.

## Pitfalls

- **Averaging over regimes.** Full-sample IC can hide a sign flip in the last two years;
  use by-year or subperiod IC when supplied.
- **Correlation as causation.** A high composite correlation with the benchmark can be
  intentional hedging; flag it as a diagnostic, not automatically as a defect.
- **Alarming language without numbers.** Generic warnings are useless; every flag must
  point to a measurement.
- **Peeking.** Do not use information after the cutoff to explain a diagnostic.
