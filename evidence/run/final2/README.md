# Evidence for the final verified run

Run: `final2-2026-11-12` · snapshot `8ab22713…` (information cutoff 2026-11-12) ·
status `completed`.

These files are copies of the small, reviewable artifacts of the run (large parquet files stay in
`outputs/runs/final2-2026-11-12/` and are not duplicated here):

| Directory / file | Content |
|---|---|
| `run_summary.json` | final state, stage table, budget, approvals |
| `events.jsonl` | controller event log (stage starts, skips, agent stats, gate and promotion events) |
| `state_snapshot.json` | exported SQLite state at run end |
| `approval_request.json` | the approval request written when the run reached the human gate |
| `00_intake/run_request.json` | pinned config, snapshot, prompt hashes, information cutoff |
| `05_backtest/metrics.json`, `equity_curve.csv` | backtest results (2018-02-01 → 2026-11-12) |
| `06_robustness/robustness.json`, `robustness_checks.json` | robustness suite: 6/7 pass, placebo fails |
| `08_portfolio/risk_report.json`, `orders.csv` | target construction, ex-ante TE, orders |
| `09_verification/verification.json`, `verifier_note.md` | independent recomputation, 11/11 checks PASS |
| `10_report/research_note.md`, `metrics_table.md`, `charts/` | committee-ready note and deterministic charts |
| `11_approval/approval.json` | human decision bound to artifact hashes |
| `12_registry/version_entry.json` | immutable registry entry promoted as v1.0.0 |

The curated example copy for readers who only want the outputs is in `examples/demo_run/`.
