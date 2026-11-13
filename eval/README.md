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

The committed baseline/decision files are **fixture-backed**: they are built from
the deterministic mini snapshot (`tests/fixtures/mini_snapshot`, 6 synthetic
names, 36 months) so the whole mechanism runs offline in CI. They are labelled
`"evidence_mode": "fixture"` and `"data_scope": "mini_fixture"`. After the Lead
integrates the real snapshot and the quant stack, regenerate the real evidence:

```powershell
# 1. freeze the real baseline from the accepted real integration run
python -m aicwf.eval --baseline-refresh --run outputs/runs/<real_run_id> `
    --data-scope real_csi300 --evidence-mode real

# 2. run the pipeline with a candidate's prompt/config override, then evaluate it
python -m aicwf.eval --run outputs/runs/<candidate_run_id> `
    --case-dir eval/cases/accepted_v1.1.0

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
