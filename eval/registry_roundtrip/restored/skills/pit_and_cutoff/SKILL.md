# Skill: point-in-time discipline and information cutoff

## Purpose

Prevent future-information leakage in a time-sensitive research workflow. Every agent
role, and every deterministic stage, must use only information available on or before the
run's information cutoff (`as_of`). This skill is the shared checklist for prompts,
artifacts, verification and monitoring.

## When to use

- Any stage that touches prices, fundamentals, universe membership or benchmark data.
- Any agent prompt that asks a model to describe, cite or reason about a run.
- When writing or reviewing the verifier note, the research note or monitor reports.

## Inputs

- Run request: `as_of`, snapshot directory and `snapshot_hash`.
- `02_data/pit_audit.json`: `as_of`, `violations`, `max_panel_date`,
  `fundamentals_available_as_of`, checks.
- `fundamentals.parquet`: `report_date`, `notice_date`, `notice_date_estimated`.
- The stage prompt's stated cutoff.

## Procedure

1. **Fix the cutoff once.** Take `as_of` from the run request / `RunContext`; never let a
   stage infer "today". Store it in every artifact summary and in every agent prompt.
2. **Filter prices.** Drop every row with `date > as_of`; the panel's maximum date must be
   `<= as_of`. A verification check asserts this.
3. **Use announcement dates, not report dates, for fundamentals.** A fundamental value is
   available only from `notice_date`. If `notice_date` is missing and the row is flagged
   `notice_date_estimated`, use the estimate (typically `report_date + 90 days`) and carry
   the flag into the PIT audit.
4. **Signal lag.** Signals computed at a decision date use data up to and including that
   date; execution uses the next available close. Never use the execution day's return in
   the signal.
5. **Prose cutoff.** Every generated note states the cutoff date explicitly and does not
   mention observed events after it. Agents are told: later information is unavailable.
6. **Future perturbation test.** Perturb a future price/fundamental value after `as_of`
   and re-run; artifacts and metrics must be byte-stable (or content-hash stable). Treat
   any change as leakage.
7. **Survivorship caveat.** The universe is current CSI 300 membership; say so in source
   limits and limitations. Do not claim point-in-time index membership.

## Checks

- `max_panel_date <= as_of`; `pit_audit.violations == []`.
- Fundamental coverage computed from rows whose `notice_date <= as_of`.
- Agent markdown mentions `as_of`; appositive numbers match artifacts.
- Future-perturbation determinism test passes.
- No wall-clock timestamps inside numeric artifacts.

## Pitfalls

- **Report date leakage.** Using `report_date` alone makes Q4 results available before
  their announcement; the PIT audit must catch it.
- **Universe leakage.** Reconstructing history from current constituents overstates
  performance; disclose it.
- **Rebalance look-ahead.** Forming weights on the month-end close and filling at the same
  close is look-ahead; use the next close.
- **Cutoff in prose only.** A prompt that says "cutoff" while the artifact was built from
  a later snapshot is still a leak: enforce at the data stage.
