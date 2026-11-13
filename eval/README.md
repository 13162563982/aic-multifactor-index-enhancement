# Eval harness and change-control evidence

This directory holds the frozen baseline, candidate-change cases and the
generated decision records used for the accept/reject/rollback evidence.

## Files

| Path | Meaning |
|---|---|
| `baseline_v1.0.0.json` | Frozen baseline: snapshot hash, config hash, prompt hashes, headline metrics, hard gates and the soft change-control tolerances (from `config/gates.yaml`). |
| `runs/baseline_v1.0.0/run/` | The reference run the baseline was frozen from (fixture-backed until the real integration run replaces it). |
| `cases/accepted_v1.1.0/` | Candidate prompt override + config override + candidate run + `decision.json` (**accept**). |
| `cases/rejected_v1.2.0/` | Candidate prompt override + config override + candidate run + `decision.json` (**reject**). |
| `registry_roundtrip/` | Append-only promote -> promote -> rollback transcript with frozen bundles and restored config, plus before/after metrics. |
| `evidence_summary.json` | Machine-readable summary of every generated artifact. |
| `make_fixture_evidence.py` | Regenerates all of the above deterministically, offline. |

## Decision rule

`aicwf/eval/runner.py` evaluates a candidate run in two stages:

1. **Hard gates** (`change_control.hard_gates`): `leakage_tests_pass`,
   `verification_passes`, `deterministic_replay`, `budget_within_limit`.
   Any failure rejects the candidate regardless of its metrics.
2. **Soft metric tolerances** (`change_control.metric_acceptance`): annualized
   excess return degradation, information-ratio degradation, absolute tracking
   error band, max-drawdown degradation and turnover increase. A candidate is
   accepted only when it is inside every tolerance.

## Fixture evidence vs real evidence

`eval/baseline_v1.0.0.json` is now the **real-data baseline**, frozen from the
promoted run `outputs/runs/final-2026-11-12` (frozen snapshot
`8ab22713…`, `data_scope: real_csi300`, `evidence_mode: real`, prompt hashes
pinned, verification PASS). The earlier fixture-backed placeholder is preserved
as `eval/baseline_v1.0.0.fixture.json`; the fixture case directories under
`eval/cases/` (and their `decision.json`) remain as offline mechanism examples and
are labelled `evidence_mode: fixture`.

Regenerate the real baseline from any accepted integrated run:

```powershell
python -m aicwf.eval --baseline-refresh --run outputs/runs/<run_id> `
    --data-scope real_csi300 --evidence-mode real --prompt-dir prompts
```

Real accepted/rejected change-control evidence requires candidate runs from the
pipeline with the case prompt/config overrides; then:

```powershell
python -m aicwf.eval --run outputs/runs/<candidate_run_id> --case-dir eval/cases/accepted_v1.1.0
python -m aicwf.eval --run outputs/runs/<candidate_run_id> --case-dir eval/cases/rejected_v1.2.0
```

The fixture-only sections below describe the original placeholder evidence.

The committed baseline/decision files under `eval/cases/` were originally
**fixture-backed**: they are built from the deterministic mini snapshot
(`tests/fixtures/mini_snapshot`, 6 synthetic names, 36 months) so the whole
mechanism runs offline in CI. They are labelled `"evidence_mode": "fixture"` and
`"data_scope": "mini_fixture"`. The fixture generator and evidence script remain
available:

```powershell
# 3. or refresh everything fixture-side after changing the generators
python eval/make_fixture_evidence.py
```

Registry evidence:

```powershell
python -m aicwf.registry versions
python -m aicwf.registry verify
python -m aicwf.registry promote --run-dir outputs/runs/<accepted_run_id> --version v1.1.0
python -m aicwf.registry rollback --to v1.0.0
```

## Real change-control evidence (final submission)

The final real-data evidence chain is:

| Step | Evidence | Verdict |
|---|---|---|
| Frozen real baseline | `eval/baseline_v1.0.0.json` (snapshot 8ab22713..., 2018-02-01..2026-11-12; ann. excess 0.15940, TE 0.06162, IR 2.587) | baseline |
| Baseline release | `registry/versions.json` #1 `v1.0.0` (run `final2-2026-11-12`, verification PASS) | ACCEPT |
| Candidate v1.1.0 | `eval/cases/real_accepted_v1.1.0/` and `evidence/eval/real_accepted_v1.1.0_decision.json` (report-writer disclosure prompt override; 4/4 hard gates, 5/5 change-control metrics within tolerance) | ACCEPT |
| Candidate v1.1.0 release | `registry/versions.json` #2 `v1.1.0` | promoted |
| Deterministic replay proof | `evidence/eval/real_accepted_v1.1.0_replay_evidence.json` (13/13 numeric artifacts reproduced; scope documented) | PASS |
| Rollback | `evidence/rollback/rollback_v1.1.0_to_v1.0.0.json` + `evidence/rollback/registry_history.json` (bundle verified + restored, smoke evaluation pass, before/after metrics) | ROLLBACK OK |
| Rejected candidate v1.2.0 | `eval/cases/rejected_v1.2.0/decision.json` (fixture) + `evidence/eval/real_rejected_v1.2.0_note.json` | REJECT |

The real rejected candidate run (`cand-b-2026-11-12`, 5x execution-cost assumption) could not
complete within the submission deadline because its live-Qwen `factor_review` Markdown tripped the
reviewer numeric-citation context gap (a number rendered as `0.0076` vs the artifact `0.00759...`);
after the agent-runtime guard widening the run progressed through factor_review/backtest but did not
reach completion in time. This is a tooling/validator limitation, **not** a metric result, so the
committed fixture rejection record is used to document the reject path (excess-return degradation and
turnover breach beyond the change-control tolerances).
