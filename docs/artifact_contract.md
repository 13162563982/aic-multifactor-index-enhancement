# Artifact contract

This file is the interface between the workflow subsystems. Paths are relative to the
project root. Every stage writes into `outputs/runs/<run_id>/<NN_stage>/` and declares
its outputs in a `StageOutput.artifacts` mapping (logical name -> relative path).

Determinism rules:

* Dates are stored as ISO `YYYY-MM-DD` strings in parquet/JSON artifacts.
* Tabular artifacts are sorted by `(date, code)` unless stated otherwise.
* No wall-clock timestamps inside numeric artifacts. Run metadata (fetch time,
  creation time) lives only in `manifest.json`, `run_request.json` and event logs.
* Deterministic content hashes are computed with `aicwf.util.artifact_hash`, never
  from raw parquet bytes (parquet embeds library metadata).

## 1. Frozen data snapshot

`data/snapshots/<as_of>/` (committed; produced by `scripts/build_snapshot.py`):

| File | Columns / keys | Notes |
|---|---|---|
| `prices.parquet` | `date, code, open, high, low, close, volume, amount, turnover, pct_chg, board` | qfq-adjusted daily bars, `code` = `600519.SH` / `000001.SZ`, sorted by date, code |
| `fundamentals.parquet` | `code, report_date, notice_date, eps, bps, roe, gross_margin, revenue, revenue_yoy, net_profit, net_profit_yoy, cfo_per_share, notice_date_estimated` | one row per report period; `notice_date` = announcement date used for point-in-time availability, estimated rows flagged true |
| `universe.parquet` | `code, name, industry, weight, free_cap, total_cap, listed_date, is_st, board` | current CSI 300 members; `weight` = normalised free-float cap proxy |
| `benchmark.parquet` | `date, close, ret` | CSI 300 index (000300.SH) |
| `calendar.parquet` | `date` | trading days covered by the snapshot |
| `manifest.json` | `schema_version, as_of, created_at_utc, source_urls, row_counts, date_ranges, files, snapshot_hash, notes` | `snapshot_hash` = canonical hash of all file hashes |

`tests/fixtures/mini_snapshot/` mirrors the same layout with 6 synthetic stocks and 36
months of data, fully offline and deterministic.

## 2. Run artifacts

| Stage dir | Artifact | Format | Required content |
|---|---|---|---|
| `00_intake` | `run_request.json` | JSON | `run_id, as_of, snapshot, snapshot_hash, config_hash, llm_mode, seed, budget, workflow_version` |
| `01_hypothesis` | `hypothesis.json` | JSON | validated by `aicwf/schemas/hypothesis.schema.json`: `thesis, factor_weights, rebalance, constraints, falsification, information_cutoff, source_limits` |
| | `rationale.md` | Markdown | narrative rationale, must mention cutoff |
| `02_data` | `panel.parquet` | Parquet | `date, code, close, ret, volume, amount, turnover, suspended, in_universe, industry, log_size, listed_days` |
| | `benchmark.parquet` | Parquet | `date, close, ret` |
| | `data_quality.json` | JSON | `snapshot_hash, rows, universe_size, coverage, checks:[{id,status,detail}], warnings` |
| | `pit_audit.json` | JSON | `as_of, violations:[], max_panel_date, fundamentals_available_as_of, checks` |
| `03_factors` | `factor_panel.parquet` | Parquet | `date, code` + `<factor_id>_raw`, `<factor_id>_z` per catalog factor + `composite_score, eligible` |
| | `factor_report.json` | JSON | `coverage, rank_ic, ic_t_stat, autocorrelation, correlation_matrix, composite_weights, warnings` |
| `04_factor_review` | `assessment.md`, `flags.json` | Markdown / JSON | `flags:[{id,severity,message}]` severity in `info/warn/block`; advisory only |
| `05_backtest` | `ledger.parquet` | Parquet | `date, code, side, shares, price, gross_amount, commission, stamp_duty, slippage, total_cost, cash_flow, reason` |
| | `holdings.parquet` | Parquet | `date, code, shares, price, weight, benchmark_weight, active_weight, industry` |
| | `metrics.json` | JSON | `metrics, benchmark_metrics, by_year, constraints, config_hash` (numbers only, no timestamps) |
| | `equity_curve.csv` | CSV | `date, nav, benchmark_nav, active_nav, daily_return, benchmark_return, excess_return, drawdown` |
| `06_robustness` | `robustness.json` | JSON | `subperiods, cost_sensitivity, leave_one_out, parameter_grid, placebo, bias_stress, deflated_sharpe, summary:{pass,concerns}` |
| | `robustness_checks.json` | JSON | list of `{name,status,value,threshold,detail}` |
| `07_robustness_review` | `critique.md` | Markdown | devil's-advocate review; must reference concrete robustness numbers |
| `08_portfolio` | `target_weights.parquet` | Parquet | `date, code, weight, benchmark_weight, active_weight, industry` |
| | `orders.csv` | CSV | `date, code, side, shares, weight_change, notional, participation_pct` |
| | `risk_report.json` | JSON | `ex_ante_te, industry_active, factor_active, turnover, n_names, optimizer:{status,iterations,te_lambda,fallback_used}, constraints` |
| `09_verification` | `verification.json` | JSON | `verdict (PASS/FAIL), checks:[{id,status,detail}], recomputed, discrepancies, artifact_hashes, verifier` |
| | `verifier_note.md` | Markdown | independent narrative; cannot modify upstream files |
| `10_report` | `research_note.md`, `metrics_table.md` | Markdown | report cites only verified numbers |
| | `charts/*.png` | PNG | `cumulative_excess.png, drawdown.png, factor_ic.png, exposures.png` |
| `11_approval` | `approval.json` | JSON | `run_id, action, approver, decided_at_utc, scope, artifact_hashes, decision, note` |
| `12_registry` | `version_entry.json` | JSON | `version, git_commit, config_hash, prompt_hashes, snapshot_hash, metrics, verification_verdict, approval, decision, previous_version` |

Every run also writes `events.jsonl` (controller), `state_snapshot.json` (exported state)
and `run_summary.json` under the run directory root.

## 3. Stage output summary conventions

`StageOutput.summary` carries small digests that downstream stages read without loading
parquet:

* `hypothesis`: `{thesis, factor_weights, rebalance, constraints, information_cutoff}`
* `data_prep`: `{rows, universe_size, max_date, coverage, warnings}`
* `factor_build`: `{factor_weights, coverage, ic_mean, eligible_rows}`
* `backtest`: `{annualized_excess_return, tracking_error, information_ratio, max_drawdown, rebalance_count}`
* `robustness`: `{passed_checks, failed_checks, concerns}`
* `portfolio`: `{date, n_names, ex_ante_te, turnover, fallback_used}`
* `verification`: `{verdict, checks_total, checks_failed}`
* `report`: `{paths:[...]}`
* `approval`: `{decision, approver}`
* `registry`: `{version, decision}`

## 4. Monitor artifacts

`outputs/monitor/<monitor_id>/`:

| File | Content |
|---|---|
| `monitor_request.json` | `monitor_id, as_of, previous_run_id, snapshot_hash, trigger` |
| `performance_update.json` | realized returns since last rebalance, realized TE, tracking difference |
| `drift_report.json` | factor RankIC drift, exposure drift, data freshness, coverage |
| `monitoring.json` | `alerts:[{id,severity,message}], thresholds, verdict` |
| `monitoring_report.md` | human-readable monitoring note |

## 5. Registry

`registry/versions.json` is an append-only list of version entries (one per accepted
release) plus rejected candidates. `registry/releases/<version>/` stores the approved
config, prompts and skills as a rollback bundle. Rollback restores the bundle, re-runs
the smoke evaluation and appends a `rollback` event.
