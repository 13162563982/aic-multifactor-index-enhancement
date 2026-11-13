# Skill: hypothesis specification

## Purpose

Turn economic priors into a falsifiable, implementable multi-factor tilt for CSI 300
index enhancement. This skill is used by the researcher agent before any calculation; its
output is the research contract that the factor build, backtest, robustness and
verification stages test.

## When to use

- At the start of a research run, or when a human re-opens the hypothesis after a failed
  verification / change-control decision.
- Whenever a candidate change to factor weights must be expressed as a new testable
  hypothesis rather than an ad-hoc tweak.

## Inputs

- Frozen snapshot manifest: `as_of`, snapshot hash, row counts, date ranges.
- `config/strategy.yaml` factor catalog: `id`, `category`, `direction`, `description`.
- Default composite weights and the risk-constraint envelope from the strategy config.
- Information cutoff (`as_of`) and stage budget.

## Procedure

1. **Name the mechanism.** Write one sentence per factor family explaining why it should
   be compensated in the A-share cross-section (risk premium, behavioural bias,
   institutional constraints, slow information diffusion). No mechanism, no factor.
2. **Choose 3-8 factors** from the catalog, covering at least two categories. Prefer
   economically distinct factors; check the correlation matrix from prior runs when
   choosing between near-duplicates.
3. **Set weights.** Start from the default composite weights, then tilt. Weights are
   non-negative; direction lives in the catalog. Normalise to exactly 1.0
   (use the `normalize_weights` check to verify).
4. **Set the risk envelope.** Choose `max_weight`, `max_active_weight`, `te_target`,
   `turnover_limit` at or inside the configured limits. Tighter is legitimate; looser is
   rejected.
5. **Pick rebalance frequency.** Monthly unless the hypothesis is explicitly
   slower-moving; quarterly raises turnover per rebalance but lowers trading frequency.
6. **Write falsification conditions.** 2-4 conditions that deterministic code can test:
   full-sample RankIC t-statistic, excess return after 3x costs, subperiod sign
   consistency, deflated Sharpe ratio, factor leave-one-out dependence, drawdown limit.
7. **Set the cutoff.** `information_cutoff` equals the run `as_of` exactly. List the
   source limits explicitly (frozen snapshot, no network, PIT fundamentals only).
8. **Self-check** with the `factor_catalog`, `constraint_envelope` and
   `normalize_weights` checks before submitting.

## Checks

- Every factor id exists in `config/strategy.yaml`; weights sum to 1.0; 3-8 factors.
- Constraints inside the strategy envelope and schema bounds.
- `information_cutoff == as_of`; no dates after the cutoff anywhere.
- Thesis explains *why* the tilt is compensated and states expected direction.
- Falsification conditions are testable by later deterministic stages.

## Pitfalls

- **Kitchen-sink weighting.** 13 factors at ~7% each is not a hypothesis; it is a
  benchmark clone with extra turnover.
- **Sign confusion.** Do not encode reversal/low-vol as negative weights; the catalog
  direction already flips the score. Negative weights fail validation.
- **Copying the default.** Submitting the default weights unchanged is acceptable but
  should still state a mechanism and falsifiers.
- **Forecasts as facts.** A hypothesis is a conditional claim, not a predicted return.
- **Cutoff drift.** Using "today" instead of `as_of` silently leaks future information
  into a time-sensitive workflow.
