# Real-data change-control evidence (v1.0.0 -> v1.1.0 -> rollback)

| file | content |
|---|---|
| `real_accepted_v1.1.0_decision.json` | aicwf.eval decision record: candidate v1.1.0 vs frozen real baseline -> **ACCEPT** (4/4 hard gates pass, 5/5 change-control metrics within tolerance). |
| `real_accepted_v1.1.0_replay_evidence.json` | deterministic numeric-stage replay proof (13/13 artifacts content-identical; scope data_prep->factor_build->backtest->portfolio; robustness/agent markdown excluded and documented). |
| `baseline_v1.0.0_summary.json` | the frozen real baseline the candidate was compared against (snapshot 8ab22713..., 2018-02-01..2026-11-12). |
| `real_rejected_v1.2.0_note.json` | why the real rejected candidate run could not complete in the deadline (reviewer-citation validator gap) and the committed fixture rejection record used instead. |

Reproduce the accepted decision:

```powershell
python -m aicwf.eval --run outputs/runs/cand-a-2026-11-12 --baseline eval/baseline_v1.0.0.json
```
