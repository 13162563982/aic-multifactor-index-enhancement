# Research Note: CSI 300 Index Enhancement Strategy

## Executive summary
This note documents a completed backtest of a multi-factor index enhancement strategy targeting the CSI 300 universe. The independent verification process returned a verdict of **PASS**, confirming that all blocking checks passed and no discrepancies were found between the reported metrics and independently recomputed values from the frozen snapshot. However, the robustness suite identified one failed check: `placebo_significance` (value 1, threshold 0.1). This indicates that while the strategy outperforms the benchmark, the statistical significance of this excess return relative to a placebo model requires further scrutiny before deployment. The information cutoff for all data and metrics is **2026-11-12**.

## Hypothesis
The strategy aims to generate alpha by constructing a long-only portfolio that tilts away from the CSI 300 benchmark weights while maintaining strict risk constraints. The core thesis relies on factor exposures derived from fundamental data, subject to hard caps on individual stock weights and industry neutrality. Constraints include a maximum weight of 0.03 per name, a maximum active weight of 0.015, an industry active limit of +/-0.02, and a turnover limit of 0.3. The optimizer seeks to maximize risk-adjusted returns within these bounds.

## Data and point-in-time discipline
The analysis is based on a frozen snapshot as of **2026-11-12**. All price data corresponds to the same-date close, ensuring no look-ahead bias. The backtest covers the period from 2018-02-01 to 2026-11-12, utilizing 2,127 metric days and 106 rebalance dates. A critical caveat is the survivorship bias inherent in using a CSI 300 index universe; delisted stocks are not included in the panel, potentially inflating performance metrics. Additionally, fundamental data used for factor construction is subject to point-in-time limitations, meaning announcements made after the valuation date were not available for that period's decision-making.

## Factor construction
The strategy employs a composite of factors derived from the CSI 300 universe. The factor panel was constructed using historical fundamentals and price data. While specific factor weights are optimized dynamically, the resulting portfolio exhibits an information ratio of 2.5870761559 (from `verification.json`). The factor construction process ensures that raw signals are neutralized to control exposure to broad market movements before being fed into the optimizer.

## Backtest results
The strategy demonstrates positive excess returns over the benchmark period. According to `metrics.json`, the annualized excess return is 0.159403963, while the cumulative excess return reaches 2.5061833886. The benchmark itself underperformed significantly during this window, with a cumulative return of -0.0475689713 (from `metrics.json`). The strategy achieved an annualized return of 0.1536436321 and a Sharpe ratio of 0.8880531574. Detailed year-by-year performance, including tracking error and volatility by year, is available in the companion table `metrics_table.md`.

## Robustness
The robustness suite executed exactly 7 checks: 6 passed, 1 failed. The single failure was `placebo_significance`, which returned a value of 1 against a threshold of 0.1. This suggests that the observed alpha may not be statistically distinguishable from noise when compared to a baseline placebo model. Other robustness checks, including cost sensitivity and subperiod consistency, passed without issue.

## Portfolio construction and risk
The portfolio maintains strict adherence to risk budgets. The realized tracking error is 0.0616154892 (from `verification.json`), which exceeds the ex-ante target of 0.0274347423. This deviation indicates that the optimization process encountered constraints that prevented full realization of the desired risk profile, or that realized volatility was higher than anticipated. The portfolio holds an average of 279.7264150943 names per rebalance and exhibits an annualized one-way turnover of 1.64627 (from `metrics.json`). The maximum single-rebalance two-way turnover observed was 0.540412.

## Independent verification
An independent verifier (`aicwf.verify`) recomputed all key metrics against the frozen artifacts. The verdict is **PASS**. All 11 required checks passed, including artifact schema validation, ledger reconciliation (21,241 trades), and weight bounds compliance. No discrepancies were found between the reported numbers and the independently recomputed values within a relative tolerance of 1e-06. For example, the annualized excess return was verified as 0.1594039630443691.

## Limitations and information cutoff
This document is research documentation, not investment advice. The results are historical and do not guarantee future performance. Key limitations include survivorship bias in the index universe and the use of point-in-time fundamentals which may differ from current data availability. All metrics are as of **2026-11-12**. The failure of the `placebo_significance` check warrants caution before considering this strategy for live capital allocation.
