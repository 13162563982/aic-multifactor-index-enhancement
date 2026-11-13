# Evidence index

This file maps every claim in the README and design note to a concrete file a reviewer can open
without running anything. The authoritative run is `final2-2026-11-12` against the frozen
snapshot `8ab22713074fcfef89f9e1fd25116536c53e3a8f385adbadcf4e41c154c125c8`
(information cutoff 2026-11-12).

| # | Claim | Evidence file(s) | What to look for |
|---|---|---|---|
| E1 | The full 13-stage workflow runs end to end and produces a verified package | `evidence/run/final2/run_summary.json`, `events.jsonl`, `state_snapshot.json` | `status: completed`, every stage `completed`, snapshot/config hashes |
| E2 | A human approval is required and bound to artifact hashes | `evidence/run/final2/11_approval/approval.json`, `evidence/run/final2/approval_request.json` | approver, action, scope, `artifact_hashes`, verification PASS |
| E3 | An independent verifier recomputes the result and the run passes | `evidence/run/final2/09_verification/verification.json`, `verifier_note.md` | verdict PASS, 11/11 checks, discrepancies empty, recomputed metrics |
| E4 | The real local model was used, with token/call accounting | `outputs/runs/final2-2026-11-12/events.jsonl` (`agent_stage_completed`), `evidence/llm_cache/` | Qwen3.5 9B calls, prompt/completion tokens, cache entries |
| E5 | An interrupted run resumes without repeating completed work | `evidence/resume/fault_demo.log`, `resume_run.log`, `stage_status.json`, `attempts.json`, `artifact_hashes.json` | crash_after at data_prep, data_prep attempt 2, other stages attempt 1, identical artifact hashes |
| E6 | Independent verification catches tampering and violations | `tests/test_verify_*.py`, `tests/test_data_pit.py`, `tests/test_quant_*.py` | tampered ledger/weights/PIT/snapshot cases produce FAIL |
| E7 | A proposed change is evaluated against a frozen baseline and accepted | `eval/baseline_v1.0.0.json`, `eval/cases/accepted_v1.1.0/decision.json`, `evidence/eval/` | hard gates pass, metric deltas within tolerance, decision accept |
| E8 | A proposed change can be rejected | `eval/cases/rejected_v1.2.0/decision.json` | rejection rule and failing metric (fixture-backed case; real-data candidate B blocked by reviewer-validator context, documented) |
| E9 | An accepted version can be rolled back with an audit record | `registry/versions.json`, `registry/releases/`, `evidence/rollback/` | append-only history v1.0.0 → v1.1.0 → rollback, before/after metrics |
| E10 | Monthly monitoring detects drift and enforces thresholds | `evidence/monitor/run1/`, `run2/`, `outputs/monitor/monitor-2026-11-12/` | realised TE, RankIC drift, freshness, verdict alert |
| E11 | Future information cannot leak into a run | `evidence/leakage_test.txt`, `tests/test_data_pit.py`, `tests/test_quant_factors.py` | future-perturbation and PIT assertions |
| E12 | Replay mode is deterministic | `evidence/reproducibility/`, `evidence/llm_cache/` | two replay runs, identical artifact hashes and pure cache hits (scope documented) |
| E13 | The ZIP contains the complete `.git` history and is clean | `evidence/package_check.txt` | commit count, `git status --porcelain`, extracted import check |
| E14 | The offline test suite is green | `evidence/test_run.txt` | pytest summary |
| E15 | No credentials appear in the submitted files or the commit history | `evidence/secret_scan.txt` | working-tree findings 0, all Git objects scanned (including unreachable), verdict CLEAN; reproduce with `python scripts/scan_secrets.py` |

## Commands that regenerate the evidence

```powershell
python -m aicwf run --as-of 2026-11-12 --llm replay --run-id final2-2026-11-12
python -m aicwf approve final2-2026-11-12 --action publish_report --scope run --approver "Zhengjia Zhu"
python -m aicwf fault-demo --as-of 2026-11-12 --run-id resume-demo-2 --llm null
python -m aicwf monitor --as-of 2026-11-12 --llm replay
python -m aicwf.eval --run outputs/runs/cand-a-2026-11-12 --baseline eval/baseline_v1.0.0.json
python -m aicwf.registry rollback --to v1.0.0
python -m pytest -q | Tee-Object evidence/test_run.txt
python scripts/package_submission.py | Tee-Object evidence/package_check.txt
```

## Honest scope notes

* The backtest universe is the current CSI 300 membership; survivorship and reconstitution bias
  remain. The robustness suite quantifies part of the bias but does not remove it.
* The placebo significance check fails (p = 1.0000 over 20 repetitions) and the deflated Sharpe
  ratio is only marginally above its threshold; the research note discloses both.
* Realised tracking error (6.16%) is roughly twice the 3.00% ex-ante target; monitoring treats a
  persistent gap as a stop condition.
* Candidate B's real-data run could not complete because the factor-review citation guard rejects
  rounded nested numbers; the rejection evidence is therefore the deterministic fixture case, and
  this is stated rather than hidden.
* The approval in the evidence was recorded through the CLI by the project author acting as the
  investment analyst (the human approval gate is exercised; it is a demonstration, not an
  investment committee decision).
