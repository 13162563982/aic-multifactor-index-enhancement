# Crash / resume evidence

This directory is produced by the fault-injection demonstration on the real frozen snapshot.

What happened:

1. `python -m aicwf fault-demo --as-of 2026-11-12 --run-id resume-demo-2 --llm replay`
   created run `resume-demo-2` and deliberately aborted the process with `os._exit(42)` after the
   `data_prep` stage had written its artifacts but before the controller committed the checkpoint.
   This is the `crash_after` case in which the artifact exists but the step is not marked complete.
2. The demo then resumed the run. The state database shows `data_prep` at attempt 2 and every
   other completed stage at attempt 1; the resume skipped the completed stages
   (`skipped_completed`) and re-ran only the interrupted stage.
3. The re-run produced byte-identical artifacts: `artifact_hashes.json` compares the resumed run's
   deterministic artifacts with the final unrestricted run (`final2-2026-11-12`) and every hash
   is identical.
4. The LLM stages that had already completed were replayed from the cache, not called again; this
   is visible in `events.jsonl` (`agent_stage_completed` with `cached_calls == calls`).

Files:

| File | Content |
|---|---|
| `fault_demo.log` | fault-injection console log (crash then resume) |
| `resume_run.log` | completion of the resumed run (report stage had to be re-recorded with the live model after the cache prune) |
| `attempts.json` | controller attempt table: the crash is recorded and is not charged as a normal retry |
| `stage_status.json` | stage status, attempt, disposition and input/output hashes |
| `artifact_hashes.json` | resumed vs uninterrupted content hashes for nine deterministic artifacts |
| `events.jsonl` | full event log for the run |
| `run_summary.json` | final state export of the run |
