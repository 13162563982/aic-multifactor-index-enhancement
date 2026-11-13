# Operations runbook

This runbook is for the analyst or reviewer who operates the workflow. It covers setup, the
monthly procedure, recovery, change control, scheduling and troubleshooting. The design rationale
is in `design_note.md`; the artifact schemas are in `artifact_contract.md`.

## 1. Setup

```powershell
python -m pip install -r requirements.txt      # or: python -m pip install -e .
python -m aicwf doctor                         # environment, config, snapshot, model
```

The repository already contains the frozen snapshot (`data/snapshots/2026-11-12/`) and the
replay cache (`evidence/llm_cache/`), so the submitted results can be reproduced offline. To use
the real local model, start Ollama (`ollama serve`) and make sure `qwen3.5:9b` is available.

Optional environment overrides (used by tests and isolated runs):

| Variable | Default | Purpose |
|---|---|---|
| `AICWF_PROJECT_ROOT` | repository root | relocate the project |
| `AICWF_CONFIG_DIR` | `config/` | use an alternative config bundle |
| `AICWF_STATE_DB` | `state/runs.db` | isolate the run database |
| `AICWF_RUNS_DIR` | `outputs/runs/` | relocate run artifacts |

The test suite uses these to run the entire workflow on the deterministic mini snapshot without
touching the real state.

## 2. Command reference

| Command | Purpose | Example |
|---|---|---|
| `aicwf doctor` | check environment, config, snapshot, state, model | `python -m aicwf doctor` |
| `aicwf snapshot` | rebuild the frozen snapshot from public endpoints | `python -m aicwf snapshot --as-of 2026-11-12` |
| `aicwf run` | start a research run | `python -m aicwf run --as-of 2026-11-12 --llm replay --run-id demo-2026-11-12` |
| `aicwf resume` | resume an interrupted, paused or approved run | `python -m aicwf resume demo-2026-11-12` |
| `aicwf status` | inspect one run or list recent runs | `python -m aicwf status demo-2026-11-12 --events 10` |
| `aicwf approve` | record a human decision (and resume) | `python -m aicwf approve demo-2026-11-12 --action publish_report --note "reviewed"` |
| `aicwf state-export` | export the SQLite state to JSON | `python -m aicwf state-export --run-id demo-2026-11-12` |
| `aicwf fault-demo` | prove crash recovery end to end | `python -m aicwf fault-demo --as-of 2026-11-12` |
| `aicwf monitor` | run the monthly monitoring workflow | `python -m aicwf monitor --as-of 2026-11-12 --llm replay` |
| `aicwf eval` | evaluate a candidate change against a baseline | `python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0` |
| `aicwf rollback` | roll the registry back to a version | `python -m aicwf rollback --to v1.0.0` |

Exit codes: `0` ok · `1` failed · `2` usage · `3` awaiting approval · `4` blocked · `5` budget
exhausted · `6` not implemented · `42` simulated crash (fault injection).

## 3. Monthly procedure

1. **Refresh data.** Build the snapshot for the latest trading day. It is a deliberate,
   separately verified action; the run itself never touches the network.
   `python -m aicwf snapshot --as-of <latest-trading-day>`
2. **Run the research pipeline.**
   `python -m aicwf run --as-of <latest-trading-day> --llm replay --run-id run-<date>`
   The runner stops at the first human gate and prints the run id and status.
3. **Review the package.** Read `09_verification/verifier_note.md`,
   `10_report/research_note.md`, `08_portfolio/risk_report.json` and the charts. Check that the
   verification verdict is `PASS` and that no warning was ignored.
4. **Approve or reject.** One run-wide approval satisfies both the publication and promotion
   gates:
   `python -m aicwf approve run-<date> --action publish_report --scope run --note "..."`
   Rejecting records a rejection and leaves the run blocked.
5. **Confirm the registry entry.** `registry/versions.json` gains an immutable entry with the
   config/prompt/data hashes, metrics and verdict.
6. **Schedule monitoring.** The monthly monitor compares the approved portfolio with the
   benchmark, reports realised tracking error and factor drift, and raises alerts. Two
   consecutive months beyond twice the tracking-error target stop new rebalances until a human
   reviews the product.

## 4. Recovery procedures

**Interrupted run (crash, power loss, closed laptop).**
`python -m aicwf resume <run_id>` re-reads the state database, skips stages whose input hashes are
unchanged (`skipped_completed` events), re-runs the interrupted stage and continues. Artifact
hashes are unchanged. `evidence/resume/` contains a recorded demonstration.

**Budget exhausted.** The run is paused with status `budget_exhausted`. Raise the relevant limit in
`config/budgets.yaml` (or run with a larger budget), then `python -m aicwf resume <run_id>`.
Completed stages are not repeated.

**Blocked / verification failure.** The run is blocked and `escalation.json` names the failed
check and the stage to revise. Inspect `09_verification/verification.json`; fix the cause; then
`python -m aicwf resume <run_id> --force`. Revisions are limited to two; after that the run must
be restarted with a new hypothesis or config.

**Approval rejected.** The decision is recorded and the run stays blocked. Nothing is published or
promoted. Re-running the hypothesis stage or changing the portfolio configuration creates a new
run.

## 5. Change control

A change is any edit to `config/`, `prompts/`, `skills/` or code that can alter results.

1. **Freeze the baseline.** `eval/baseline_v1.0.0.json` records the snapshot hash, config and
   prompt hashes and the approved metrics. Refresh it only through the documented helper, never by
   hand.
2. **Evaluate the candidate.**
   `python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0`
   Hard gates (leakage, verification, deterministic replay, budget) must pass. Soft tolerances
   (excess return, information ratio, tracking error, drawdown, turnover) are read from
   `config/gates.yaml`.
3. **Read the decision record.** Accepted and rejected candidates both get an entry with the
   metric-by-metric delta, the deciding rule and the rationale. Rejected candidates stay on
   record so the same idea is not retried silently.
4. **Promote.** Promotion writes a release bundle under `registry/releases/<version>/` and
   requires the approval recorded in step 4 of the monthly procedure.
5. **Roll back if needed.** `python -m aicwf rollback --to <previous-version>` restores the
   bundle, re-runs the smoke evaluation and appends a rollback event with before/after metrics.
   History is append-only.

## 6. Scheduling

*Windows Task Scheduler* — `scripts/scheduled_monitor.ps1` registers a monthly task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/scheduled_monitor.ps1 -Install
```

*cron / macOS / Linux*:

```
0 18 1 * *  cd /path/to/project && python -m aicwf monitor --as-of $(date +\%F)
```

The domain trading calendar, not the scheduler, decides the last trading day. New data triggers a
monitoring run when the snapshot hash changes; on a rebalance month the analyst starts the full
research run from the monitoring report's link. If the harness supports scheduled sessions
(e.g. DSH schedules), point the schedule at the same command.

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `snapshot not found` / `as_of mismatch` | snapshot for that date was never built, or the config end date moved | `python -m aicwf snapshot --as-of <date>`; check `data/snapshots/<date>/manifest.json` |
| `snapshot file hash mismatch` | snapshot edited after build | rebuild the snapshot; never edit snapshot files in place |
| eastmoney timeouts / disconnects | public endpoint throttling | the builder retries and rotates hosts; if it still fails, wait and rerun. The frozen snapshot is unaffected |
| `ReplayCacheMiss` | prompt or inputs changed since the cache was recorded | run once with `--llm record` (real model) or `--llm null` for a structural smoke run, then replay |
| `Ollama ... not reachable` | local server not running | start `ollama serve`, or use `--llm replay` |
| `approval required` / exit 3 | human gate reached | review the package and run `aicwf approve` |
| `blocked` / exit 4 | verification failed after revisions | inspect `escalation.json`, fix, `aicwf resume --force` |
| `budget exhausted` / exit 5 | wall clock, LLM calls or tokens exhausted | raise the limit in `config/budgets.yaml`, resume |
| CI/import error while a teammate edits modules | transient partial implementation | rerun `python -m pytest -q`; check `git status` |

## 8. Regenerating the submitted evidence

```powershell
python -m pytest -q
python -m aicwf fault-demo --as-of 2026-11-12
python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0
python -m aicwf rollback --to v1.0.0
python -m aicwf monitor --as-of 2026-11-12 --llm replay
python scripts/package_submission.py
```

The commands write to `evidence/` and `exports/`; `package_submission.py` rebuilds and verifies the
submission ZIP, including the complete `.git/` history.
