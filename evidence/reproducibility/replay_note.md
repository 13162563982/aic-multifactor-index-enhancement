# Strict replay verification (deterministic, offline)

This note documents the replay evidence for the agreed `v1.0.0` research run. It is
generated from artifacts on disk; the authoritative machine-readable hash comparison is
`evidence/reproducibility/replay_artifact_hashes.json`.

## 1. What is being verified

| item | value |
|---|---|
| Recorded run | `final2-2026-11-12` (approved; registry promoted `v1.0.0`) |
| Information cutoff | `2026-11-12` |
| Frozen snapshot | `data/snapshots/2026-11-12`, `snapshot_hash = 8ab22713074fcfef89f9e1fd25116536c53e3a8f385adbadcf4e41c154c125c8` |
| LLM backend | local Ollama `qwen3.5:9b`, `format=json`, `think=false`, `num_ctx=16384` |
| Replay mode | `--llm replay` (strict sha256 cache; a cache miss raises `ReplayCacheMiss`, never calls a model) |
| Replay runs | `replay-final-a`, `replay-final-b` (fresh run ids, isolated state DBs) |
| Cache | `evidence/llm_cache/`, 5 entries, 156.1 KB |

The pinned prompt/skill/schema content hashes are recorded in
`outputs/runs/final2-2026-11-12/00_intake/run_request.json` (`prompt_hashes`); prompts,
skills and JSON schemas were frozen before the recorded run.

## 2. Cache contents

The cache key is `sha256(canonical_json({cache_version, model, system, user, schema,
params:{temperature, max_tokens, format_json}}))`. Run ids, timestamps and backend modes
are deliberately excluded: prompts are deterministic functions of the prompt/skill files,
config, frozen snapshot and cutoff. Run-scoped metadata (`run_id`, `llm_mode`, and the
artifact-hash map that contains the hash of `run_request.json`) is stripped from the
verification artifact before it is put into a prompt, so replay under a fresh run id
hits the same entry.

| cache file (sha256 prefix) | agent role | recorded at (UTC) | prompt+request chars | completion chars | recorded usage (prompt/completion tokens) | recorded latency |
|---|---|---|---|---|---|---|
| `81c4baa6d660…` | researcher (hypothesis) | 2026-11-13T03:38:54 | 20,497 | 2,118 | 6,148 / 549 | 15.3 s |
| `773f083ee23a…` | factor analyst (factor review) | 2026-11-13T03:40:01 | 26,686 | 4,823 | 9,542 / 1,333 | 35.1 s |
| `8d077602e86e…` | risk critic (robustness review) | 2026-11-13T03:57:48 | 30,573 | 2,745 | 11,000 / 599 | 21.6 s |
| `11146796ae5d…` | verifier (verifier note) | 2026-11-13T04:24:59 | 16,840 | 2,497 | — | — |
| `645085c31858…` | report writer (research note) | 2026-11-13T04:24:59 | 31,862 | 5,169 | — | — |

Honest provenance note: the first three entries are genuine live recordings from the
Qwen model during the evidence runs (latency/token usage above). The verifier-note and
report entries were re-filled after the prompt was made run-id invariant, with the
response texts being exactly the approved `final2` note texts; therefore their
`usage/latency` metadata is `0` (the fill was not a model call) even though the texts
themselves were originally produced by the model and approved. All five entries replay
deterministically; none of them calls a model.

## 3. Replay procedure

```powershell
# fresh strict-replay runs, isolated controller state so the recorded run is untouched
$env:AICWF_STATE_DB = "$env:TEMP\aicwf-replay-a.db"
python -m aicwf run --as-of 2026-11-12 --llm replay --run-id replay-final-a
$env:AICWF_STATE_DB = "$env:TEMP\aicwf-replay-b.db"
python -m aicwf run --as-of 2026-11-12 --llm replay --run-id replay-final-b
```

Both runs executed the full pipeline and stopped at the human approval gate
(`status = awaiting_approval`), as expected; no `approval.json`/`version_entry.json` is
created because no human approved during the replay.

## 4. Results

### 4.1 Every LLM call was a cache hit

For both replay runs, every agent stage reports `cached_calls == calls` (one call each,
zero live model calls):

| agent stage | calls | cached | attempts | result |
|---|---|---|---|---|
| hypothesis (researcher) | 1 | 1 | 1 | cache hit |
| factor_review (factor analyst) | 1 | 1 | 1 | cache hit |
| robustness_review (risk critic) | 1 | 1 | 1 | cache hit |
| verification (verifier note) | 1 | 1 | 1 | cache hit |
| report (report writer) | 1 | 1 | 1 | cache hit |

Run budget used by each replay: 5 LLM calls, 43,261 recorded tokens, ~200 s wall time
(limits: 40 calls / 200,000 tokens / 900 s). The report stage used the agent-produced
note (`report_mode = "agent"`); the deterministic fallback was not needed.

### 4.2 Artifact content hashes

42 artifacts are declared by the recorded run; the replay runs declare 40 (they stop
before approval/registry). Comparing by `(stage, logical_name)` content hash:

* **38 artifacts are byte-identical across the recorded run and both replay runs**,
  including every data, factor, backtest, robustness and portfolio artifact, the
  hypothesis and its rationale, factor-review flags/assessment, robustness critique,
  `metrics_table.md`, all four chart PNGs, and — importantly — both agent-written notes
  (`09_verification/verifier_note.md` and `10_report/research_note.md`).
* **4 declared artifacts differ, all run-scoped metadata or gate artifacts:**
  * `intake/run_request.json` — embeds the run id / llm mode;
  * `verification/verification.json` — embeds `run_id`, `llm_mode`, and the artifact-hash
    map containing the hash of `run_request.json`;
  * `approval/approval.json`, `registry/version_entry.json` — exist only in the recorded
    run; the replay runs correctly stop at the approval gate.

Between the two replay runs, the only differing artifacts are exactly
`intake/run_request.json` and `verification/verification.json`; all other 38 artifacts
are byte-identical.

Key hashes (canonical content hash, `aicwf.util.artifact_hash`):

| artifact | recorded run | replay A | replay B |
|---|---|---|---|
| `10_report/research_note.md` | `8e6b8e0e58c947a08e8f7364…` | same | same |
| `09_verification/verifier_note.md` | `6a15bacdd9fb49683504cb07…` | same | same |
| `05_backtest/metrics.json` | `097d2be2cd42ba6609b54ca4…` | same | same |
| `06_robustness/robustness_checks.json` | `e655e06898dedc430b807887…` | same | same |
| `08_portfolio/target_weights.parquet` | `b60b81a669e264760904eba1…` | same | same |

(The JSON evidence file lists the full 42×3 hash table and the boolean equality flags.)

## 5. Caveats and limits of this evidence

* Strict replay proves the agent pipeline is deterministic **given the same snapshot,
  config, prompt/skill/schema files and cache**; it does not prove that regenerating the
  text with the model would reproduce the wording (sampling is not bit-reproducible).
* Wall-clock durations and per-call token accounting of cached calls differ from the
  recorded run; only content hashes are claimed identical.
* PNG chart bytes are compared above and matched on this machine/Matplotlib version;
  their numeric inputs are identical.
* The two re-filled cache entries are disclosed in §2; a reviewer can inspect
  `request`/`response` text in each cache entry and compare it with the recorded run's
  artifacts, which are identical.
* The replay runs intentionally do not re-create approval/registry artifacts: those are
  human decisions by design.
