# Evidence index

This file maps every claim in the README and design note to a concrete file that a reviewer can
open without running anything. Commands that regenerate the evidence are listed at the bottom.

| # | Claim | Evidence file(s) | What to look for |
|---|---|---|---|
| E1 | The full 13-stage workflow runs end to end on the frozen snapshot and stops at the human gate | `evidence/run/run_summary.json`, `evidence/run/events.jsonl`, `evidence/run/state_export.json` | `status: awaiting_approval`, all stages `completed`, `snapshot_hash`, `config_hash` |
| E2 | A human approval is required before publication/promotion and is bound to artifact hashes | `evidence/run/approval_request.json`, `evidence/run/11_approval/approval.json` | gate blocks first; approval records approver, scope, action and `artifact_hashes` |
| E3 | The independent verifier recomputes the result and the run passes | `evidence/run/09_verification/verification.json`, `09_verification/verifier_note.md` | `verdict: PASS`, check-by-check details, recomputed metrics |
| E4 | The real local model was used, with token accounting | `evidence/run/llm_usage.json`, `evidence/llm_cache/*.json` | model id, calls, tokens, cache entries for each agent stage |
| E5 | An interrupted run resumes without repeating completed work | `evidence/resume/fault_run.log`, `evidence/resume/resume_run.log`, `evidence/resume/stage_status.json`, `evidence/resume/artifact_hashes.json` | skipped stages, `crash_after`/`crash_before`, identical hashes, no duplicated ledger rows |
| E6 | Independent verification catches tampering | `tests/test_verify_checks.py::test_tampered_ledger_fails`, `tests/test_verify_stage.py` | assertions that a modified ledger/PIT violation produces `FAIL` |
| E7 | A proposed change is evaluated against a frozen baseline and accepted | `eval/cases/accepted_v1.1.0/decision.json`, `eval/baseline_v1.0.0.json`, `evidence/eval/accepted.log` | hard gates pass, metric deltas within tolerance, `decision: accept` |
| E8 | A proposed change is evaluated against a frozen baseline and rejected | `eval/cases/rejected_v1.2.0/decision.json`, `evidence/eval/rejected.log` | the deciding rule and the failing metric |
| E9 | An accepted version can be rolled back with an audit record | `registry/versions.json`, `evidence/rollback/rollback.json`, `evidence/rollback/smoke_eval.json` | append-only entries, `previous_version`, before/after metrics |
| E10 | Monthly monitoring detects drift and enforces thresholds | `evidence/monitor/run1/monitoring.json`, `evidence/monitor/run2/monitoring.json`, `monitoring_report.md` | realised TE, RankIC drift, freshness, alerts |
| E11 | Future information cannot leak into a run | `tests/test_data_pit.py`, `tests/test_quant_leakage.py`, `evidence/leakage_test.txt` | future-perturbation test output, max date assertions |
| E12 | The replay cache makes results reproducible offline | `evidence/llm_cache/`, `evidence/reproducibility/hash_comparison.json` | two runs, identical content hashes |
| E13 | The ZIP contains the complete `.git` history and is clean | `evidence/package_check.txt` | `git rev-list --count`, `git status --porcelain`, extracted import check |
| E14 | The test suite is green offline | `evidence/test_run.txt` | pytest summary, skipped network/slow tests |

## Regenerating the evidence

```powershell
# Environment and data
python -m aicwf doctor

# Main run (offline replay; the live-model run is recorded in evidence/llm_cache)
python -m aicwf run --as-of 2026-11-12 --llm replay --run-id evidence-2026-11-12
python -m aicwf status evidence-2026-11-12 --json          # awaiting_approval
python -m aicwf approve evidence-2026-11-12 --action publish_report --scope run --note "Reviewed verification package"

# Recovery demonstration
python -m aicwf fault-demo --as-of 2026-11-12 --run-id resume-demo

# Change control
python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0
python -m aicwf rollback --to v1.0.0

# Monitoring
python -m aicwf monitor --as-of 2026-11-12 --llm replay

# Tests and packaging
python -m pytest -q | Tee-Object evidence/test_run.txt
python scripts/package_submission.py | Tee-Object evidence/package_check.txt
```

## Notes on what is deliberately *not* claimed

* The backtest is not a live-track record and the universe is current index membership, so
  survivorship bias remains; the bias-stress result is reported next to the headline result.
* The local 9B model writes advisory prose only; its output is schema-checked and human-reviewed.
* Nothing in this repository places orders or connects to a broker.
