# Skill: research note template

## Purpose

Produce a short, decision-ready research note for a completed and independently verified
CSI 300 index-enhancement run. The note is an approval-package artifact: a human signs
off on it, so every sentence must be traceable to verified run artifacts.

## When to use

- After `09_verification/verification.json` exists with verdict `PASS` (a FAIL run must
  not be presented as publishable).
- When a monitor or change-control cycle needs an updated note for a re-run.

## Inputs

- Verified numbers: `verification.json` (`recomputed`, `checks`, `discrepancies`),
  `metrics.json`, robustness and risk-report digests.
- The hypothesis: thesis, factor weights, rebalance, constraints, cutoff.
- The deterministic `metrics_table.md` companion table.
- The information cutoff `as_of`.

## Procedure

1. **Open with the verdict.** Executive summary: what was tested, the headline verified
   metrics, the verification verdict, and the main caveat. No promotional language.
2. **Hypothesis.** Restate the thesis, factor weights, rebalance and risk envelope
   exactly as submitted.
3. **Data and point-in-time discipline.** Frozen snapshot, cutoff, PIT fundamentals
   (announcement dates), survivorship caveat, data-quality summary.
4. **Factor construction.** Coverage, IC summary, neutralisation, composite. Two or three
   sentences with the reported numbers; do not re-derive.
5. **Backtest results.** Reference `metrics_table.md`; cite annualised excess return,
   tracking error, information ratio, max drawdown, turnover - all verified values.
6. **Robustness.** Summarise passed/failed checks, cost sensitivity, subperiod
   consistency, deflated Sharpe; point to the critic's critique.
7. **Portfolio construction and risk.** Ex-ante tracking error, industry/factor active
   exposure, number of names, optimizer status.
8. **Independent verification.** Verdict, which independent checks ran, any
   discrepancies, artifact hashes. Do not soften the result.
9. **Limitations and information cutoff.** State the cutoff, the known limitations
   (survivorship, no transaction-cost microstructure beyond the model, frozen snapshot,
   past performance not indicative), and the disclaimer that this is research
   documentation, not investment advice.

## Checks

- At least five numbers, each present in verified artifacts and attributed.
- Cutoff date stated in the note.
- Verification verdict and discrepancies reproduced verbatim.
- No forecasts, price targets, "proven" claims or external references.
- No wall-clock timestamps.

## Pitfalls

- **Marketing tone.** "Strong", "robust", "outstanding" without the number and its check
  is noise; replace with the metric and its verification status.
- **Rounding drift.** Reproducing a metric to a different precision than the artifact
  makes reconciliation fail; reuse the supplied precision.
- **Silent caveats.** If a check is WARN/SKIP or a robustness test failed, the note must
  say so where a reviewer will see it, not in a footnote.
- **New claims.** Any claim not supported by the verified evidence is a hallucination,
  even if it sounds plausible.
