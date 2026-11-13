# Monthly monitoring - monitor-2026-11-12

- **As of:** 2026-11-12
- **Verdict:** ALERT
- **Latest research run:** `final2-2026-11-12` (previous: `final2-2026-11-12`)
- **Trigger:** scheduled (new_data=False)
- **Snapshot hash:** `8ab22713074fcfef...`

## Alerts

| id | severity | message |
|---|---|---|
| tracking_error_alert | warn | realized tracking error 0.0867 exceeds the alert threshold 0.0800 |

## Realized performance since the last rebalance

- Window: 2026-11-02 -> 2026-11-12 (8 trading days, method=mark_to_market)
- Portfolio return: -0.0030
- Benchmark return: 0.0067
- Cumulative excess (tracking difference): -0.0097
- Realized tracking error (annualized): 0.0867
- Realized active drawdown: -0.0162

## Factor RankIC drift

- Method: spearman(composite_score_t, forward_return_t+20d)
- Recent mean RankIC: 0.0679
- Baseline mean RankIC: 0.0346
- Decay: -0.0333
- Observations: 121

## Exposures

- Latest holdings date: 2026-11-02
- Names held: 300
- Max |name active weight|: 0.0080
- Max |industry active weight|: 0.0006 (limit 0.0200)

## Data freshness and coverage

- Latest frozen price date: 2026-11-12
- Trading days stale: 0 (limit 3)
- Price coverage: 1.000
- Factor coverage: 1.000

## Thresholds

- `data_staleness_days` = 3.0
- `excess_drawdown_alert` = 0.1
- `ic_decay_absolute_alert` = 0.05
- `ic_decay_alert` = -0.05
- `industry_active_limit` = 0.02
- `min_coverage` = 0.9
- `te_alert` = 0.08
- `te_stop_level` = 0.16
- `te_stop_multiple` = 2.0

This note was generated deterministically from the frozen artifacts; no model output is used for numbers.
