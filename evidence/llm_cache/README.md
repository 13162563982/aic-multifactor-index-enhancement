# Replay LLM cache

This directory is the committed cache that makes the demo run reproducible and offline.

## What is stored

One JSON file per completion request, named `<sha256>.json`. Each entry contains:

- `cache_version` – schema version of the cache entry (bump invalidates old entries);
- `key` – the sha256 cache key (must equal the file name);
- `recorded_at_utc` – when the real model response was recorded (audit only);
- `backend`, `model` – the backend and exact model that produced the response;
- `request` – `system`, `user`, full `schema`, `temperature`, `max_tokens`, `format_json`;
- `response` – `text`, `model`, token `usage` and `latency_ms`.

The cache key is:

```
sha256(canonical_json({
  cache_version,
  model,
  system,
  user,
  schema,
  params: {temperature, max_tokens, format_json}
}))
```

Run ids, wall-clock timestamps and event metadata are deliberately **not** part of the
key. Agent prompts are deterministic functions of the prompt/skill files, the frozen
config, the frozen snapshot and the information cutoff (`as_of`), so a run recorded under
one run id replays exactly under another.

## Modes

- `--llm replay` (the demo/evaluation mode): strict replay. A cache hit returns the
  recorded response with `cached=true`; a cache miss raises `ReplayCacheMiss`. Replay
  **never** calls a live model and never silently degrades – that is a hard determinism
  requirement for the evaluation and the submitted demo.
- `--llm ollama` with `AICWF_LLM_RECORD=1` (or `make_llm(mode="record")`): recording mode.
  Every new prompt calls the local Qwen model and the response is written atomically
  (temporary file + `os.replace`) before it is returned.

## How this cache was produced

The committed evidence cache contains exactly the 5 responses used by the approved
`final2-2026-11-12` run (5 entries, 156.1 KB). One recorded pass of the complete real
pipeline on the frozen 300-name CSI 300 snapshot (`as_of = 2026-11-12`) with the local
Ollama model `qwen3.5:9b`:

```powershell
$env:AICWF_LLM_RECORD = "1"
python -m aicwf run --as-of 2026-11-12 --llm ollama --run-id final2-2026-11-12
Remove-Item Env:AICWF_LLM_RECORD
```

The subsequent demo/evaluation runs use `--llm replay` and touch no network or model.
See `evidence/reproducibility/replay_note.md` for the two-run strict replay evidence and
`evidence/reproducibility/replay_artifact_hashes.json` for the content-hash comparison.

| role | entry sha256 prefix | recorded by |
|---|---|---|
| researcher (hypothesis) | `81c4baa6d660…` | live Qwen call |
| factor analyst (factor review) | `773f083ee23a…` | live Qwen call |
| risk critic (robustness review) | `8d077602e86e…` | live Qwen call |
| verifier (verifier note) | `11146796ae5d…` | filled with the approved final2 note text after prompt de-metadataing |
| report writer (research note) | `645085c31858…` | filled with the approved final2 note text after prompt de-metadataing |

The last two entries are disclosed in the replay note: the response texts are exactly the
approved final2 artifacts, but their `usage`/`latency` metadata is zero because the fill
itself was not a model call. The three live entries carry the real Qwen latency and token
usage.

## Integrity rules

- Cache entries are never edited by hand. To change a prompt, a skill or a schema, edit
  the source file and re-record the cache; the old entries simply become unreferenced.
- Do not delete individual entries: a partial cache causes strict replay misses for the
  affected stage, which is intentional (fail loudly rather than silently call a model).
- The cache contains no credentials and no private data; only prompts built from the
  frozen public snapshot and the model's responses.
