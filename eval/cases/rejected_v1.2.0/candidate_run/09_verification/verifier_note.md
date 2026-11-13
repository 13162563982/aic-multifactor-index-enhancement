# Independent verification note - PASS

- **Run:** `demo-rejected-2024-12-31` as of **2024-12-31**
- **Verifier:** independent-verifier v1.0.0 (deterministic_recompute, llm_mode=replay)
- **Independence:** aicwf.verify (independent of aicwf.backtest/aicwf.portfolio)
- **Checks:** 11 total, 0 failed

## Checks

| id | status | detail |
|---|---|---|
| artifact_schema | pass | 18/18 contract artifacts present, parseable and well-formed |
| artifact_hashes | pass | 18 artifact content hashes computed and match declared hashes |
| ledger_reconcile | pass | 889 ledger rows reconciled across 153 holdings dates; cash and weights tie out |
| metrics_recompute | pass | 7 headline metrics and 4 benchmark metrics reproduced independently within relative tolerance 1e-06 (e.g. annualized_excess_return=0.2216019623401777, tracking_error=0.2191396191011285) |
| weights_bounds | pass | all weights <= 0.4, \|active\| <= 0.1, no leverage and >= 4 names at every rebalance |
| industry_neutrality | pass | industry active weights within +/-0.1 at every rebalance (max \|industry active\| 0.060009 on 2022-11-04 in 银行) |
| forbidden_securities | pass | 889 trades checked against ST/suspended/limit rules (6 universe names, 889 price-limit observations); 1 warning(s) |
| t1_tradability | pass | 889 trades respect T+1 settlement and the frozen trading calendar (781 days) |
| pit_prices | pass | all artifact dates <= as_of 2024-12-31; every ledger/holdings price equals the frozen same-date close |
| pit_fundamentals | pass | pit_audit clean (violations=[], max_panel_date=2024-12-31) and no fundamental row available after as_of 2024-12-31 was usable; 1 warning(s) |
| determinism_summary | pass | numeric artifacts contain no wall-clock state; two independent rebuilds are byte-identical; run_request and metrics bind the same config_hash |

## Independently recomputed headline metrics

| metric | recomputed |
|---|---|
| annualized_return | 0.19994710867847965 |
| annualized_volatility | 0.1215476073238854 |
| sharpe | 1.6450106512231435 |
| max_drawdown | -0.09868661149877178 |
| annualized_excess_return | 0.2216019623401777 |
| tracking_error | 0.2191396191011285 |
| information_ratio | 1.0112364128821127 |

## Benchmark

- `annualized_return` = -0.021676444581363308
- `annualized_volatility` = 0.17935887228890954
- `sharpe` = -0.12085515650682226
- `max_drawdown` = -0.34488935261550924

## Discrepancies

_None: every reported headline number was reproduced independently within tolerance._

## Declaration

The verifier recomputed positions, cash, the daily NAV path, benchmark returns and all headline metrics from `ledger.parquet`, the frozen prices and `benchmark.parquet` without importing `aicwf.backtest`, `aicwf.portfolio` or `aicwf.factors`. It did not modify any upstream artifact. The frozen snapshot file hashes, the declared artifact content hashes, weight/active bounds, industry neutrality, forbidden-security rules, T+1 settlement, and point-in-time availability were checked independently.
