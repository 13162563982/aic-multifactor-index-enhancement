"""LLM backends for the AICWF agent stages.

Four backends implement the :class:`aicwf.contracts.LLMClient` protocol:

``OllamaLLM``
    Local Ollama server (``POST /api/generate``), JSON format mode, bounded retries on
    5xx/timeouts, token usage from ``prompt_eval_count`` / ``eval_count``. This is the
    default backend and needs no API key.
``OpenAICompatLLM``
    Any OpenAI-compatible ``/chat/completions`` endpoint. The API key is read from an
    environment variable and never logged, stored or committed.
``ReplayLLM``
    Strict sha256 cache under ``evidence/llm_cache/``. Replay mode never touches the
    network: an unseen prompt raises :class:`ReplayCacheMiss`. Record mode calls a
    wrapped live backend on a miss and stores the response atomically, which is how the
    committed demo cache is produced.
``NullLLM``
    Deterministic offline backend used by tests and by ``--llm-mode null`` dry runs. It
    synthesises schema-valid payloads and markdown that only cites numbers already
    present in the prompt, so the no-hallucination guards stay meaningful.

The cache key is ``sha256(cache_version + model + system + user + schema + sampling
parameters)``. Run ids, timestamps and event payloads are deliberately *not* part of
the key: prompts are deterministic functions of the frozen snapshot and config, so a
recorded demo can be replayed by a later run with a different ``run_id``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicwf.agents.validate import synthesize_json
from aicwf.contracts import (
    ConfigError,
    LLMResponse,
    NonRetryableError,
    PROJECT_ROOT,
    RetryableError,
    WorkflowError,
)
from aicwf.util.hashing import canonical_json, hash_json
from aicwf.util.jsonio import read_json, write_json

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = "evidence/llm_cache"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 180.0

_TRUTHY = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReplayCacheMiss(NonRetryableError):
    """A strict replay was asked for a prompt that was never recorded.

    This is never silently resolved by calling a model: deterministic replay is a hard
    requirement for the evaluation and demo runs.
    """


# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------


def model_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the ``model.yaml`` mapping from a full config bundle or a section."""

    cfg = dict(config or {})
    nested = cfg.get("model")
    if isinstance(nested, Mapping):
        return dict(nested)
    return cfg


def _agent_loop_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(model_section(config).get("agent_loop") or {})


def resolve_cache_dir(
    cache_dir: Path | str | None, config: Mapping[str, Any] | None = None
) -> Path:
    """Resolve the replay cache directory (relative paths are project-root relative)."""

    model_cfg = model_section(config)
    raw: Path | str | None = cache_dir
    if raw is None:
        raw = (model_cfg.get("replay") or {}).get("cache_dir") or DEFAULT_CACHE_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def cache_key(
    *,
    model: str,
    system: str,
    user: str,
    schema: Mapping[str, Any] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1600,
    format_json: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable cache key for one completion request.

    Rounding is only applied to the temperature so that ``0.1`` and ``0.10000000002``
    from config/CLI parsing hash identically.
    """

    params: dict[str, Any] = {
        "temperature": round(float(temperature), 6),
        "max_tokens": int(max_tokens),
        "format_json": bool(format_json),
    }
    if extra:
        params.update(dict(extra))
    return hash_json(
        {
            "cache_version": CACHE_VERSION,
            "model": str(model),
            "system": str(system),
            "user": str(user),
            "schema": dict(schema) if schema is not None else None,
            "params": params,
        }
    )


# ---------------------------------------------------------------------------
# Replay cache
# ---------------------------------------------------------------------------


class ReplayCache:
    """Directory of one-JSON-file-per-entry responses, written atomically."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.dir = resolve_cache_dir(cache_dir)

    def entry_path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.entry_path(key)
        if not path.exists():
            return None
        try:
            entry = read_json(path)
        except (OSError, ValueError) as exc:  # corrupt cache entries must not be silent
            raise NonRetryableError(f"corrupt replay cache entry {path}: {exc}") from exc
        if not isinstance(entry, dict) or entry.get("key") != key:
            raise NonRetryableError(f"replay cache entry {path} has an inconsistent key")
        if int(entry.get("cache_version", 0)) != CACHE_VERSION:
            raise NonRetryableError(
                f"replay cache entry {path} uses cache_version {entry.get('cache_version')!r}; "
                f"expected {CACHE_VERSION} (re-record the cache)"
            )
        return entry

    def put(self, key: str, entry: dict[str, Any]) -> Path:
        path = self.entry_path(key)
        write_json(path, entry, indent=2, sort_keys=True)
        return path

    def keys(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(path.stem for path in self.dir.glob("*.json"))

    def __len__(self) -> int:
        return len(self.keys())


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _RetryingBackend:
    """Shared retry policy for HTTP LLM backends.

    Subclasses provide ``model``, ``timeout_s``, ``attempts``, ``backoff_seconds``,
    ``transport`` (a callable ``(url, payload, timeout) -> response``) and ``sleep``.
    """

    model: str = ""
    backend_name: str = "http"
    timeout_s: float = DEFAULT_TIMEOUT_S
    attempts: int = 3
    backoff_seconds: tuple[float, ...] = (0.5, 1.0, 2.0)

    def __init__(
        self,
        *,
        transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
        | None = None,
        sleep: Callable[[float], None] = time.sleep,
        recorder: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.transport = transport or self._default_transport
        self.sleep = sleep
        self.recorder = recorder
        self.calls = 0
        self.retries = 0
        self._last_latency_ms = 0

    # -- transport -----------------------------------------------------
    @staticmethod
    def _default_transport(
        url: str,
        payload: dict[str, Any],
        timeout: float,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        import requests  # local import: tests that never call the network do not need it

        return requests.post(url, json=payload, headers=dict(headers or {}), timeout=timeout)

    def _sleep_before_retry(self, attempt_index: int) -> None:
        if not self.backoff_seconds:
            return
        self.sleep(self.backoff_seconds[min(attempt_index, len(self.backoff_seconds) - 1)])

    def _emit(self, record: dict[str, Any]) -> None:
        if self.recorder is None:
            return
        try:  # a recorder must never break an LLM call
            self.recorder(record)
        except Exception:  # noqa: BLE001 - deliberately ignored
            pass

    # -- request -------------------------------------------------------
    def _post_once(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Mapping[str, str] | None,
    ) -> tuple[dict[str, Any], int]:
        """One HTTP attempt: returns ``(decoded_body, latency_ms)`` or raises."""

        started = time.perf_counter()
        try:
            response = self.transport(url, payload, self.timeout_s, headers)
        except Exception as exc:  # noqa: BLE001 - timeouts, connection resets, DNS
            raise RetryableError(
                f"{self.backend_name} transport error after {self.timeout_s}s: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429 or status >= 500:
            raise RetryableError(
                f"{self.backend_name} returned HTTP {status}: "
                f"{str(getattr(response, 'text', ''))[:200]}"
            )
        if status >= 400:
            detail = str(getattr(response, "text", ""))[:400]
            raise NonRetryableError(
                f"{self.backend_name} request rejected with HTTP {status}: {detail}"
            )
        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - malformed body is a server fault
            raise RetryableError(f"{self.backend_name} returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise NonRetryableError(
                f"{self.backend_name} returned a non-object JSON body: {type(data).__name__}"
            )
        return data, latency_ms

    def _request_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        parse: Callable[[dict[str, Any]], Any] | None = None,
    ) -> Any:
        """POST JSON with bounded retries.

        HTTP-level failures *and* backend-level parse failures (empty completion, missing
        fields) raise :class:`RetryableError` and are retried up to ``attempts`` times.
        ``parse`` converts the decoded body into the backend-specific result.
        """

        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                data, latency_ms = self._post_once(url, payload, headers)
                result = parse(data) if parse is not None else data
            except NonRetryableError:
                raise
            except RetryableError as exc:
                last_error = exc
                self.retries += 1
                if attempt + 1 < self.attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break
            self._last_latency_ms = latency_ms
            self.calls += 1
            return result
        if isinstance(last_error, RetryableError):
            raise last_error
        raise RetryableError(
            f"{self.backend_name} request failed after {self.attempts} attempts: {last_error}"
        ) from last_error


class OllamaLLM(_RetryingBackend):
    """Local Ollama ``/api/generate`` backend."""

    backend_name = "ollama"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
        | None = None,
        sleep: Callable[[float], None] = time.sleep,
        recorder: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(transport=transport, sleep=sleep, recorder=recorder)
        model_cfg = model_section(config)
        default = dict(model_cfg.get("default") or {})
        self.model = str(
            model or os.environ.get("AICWF_LLM_MODEL") or default.get("model") or DEFAULT_MODEL
        )
        self.base_url = str(
            base_url or default.get("base_url") or DEFAULT_OLLAMA_URL
        ).rstrip("/")
        self.temperature = float(default.get("temperature", 0.1))
        self.top_p = float(default.get("top_p", 0.9))
        self.num_ctx = int(default.get("num_ctx", 8192))
        self.think = bool(default.get("think", False))
        self.format_json_default = bool(default.get("format_json", True))
        self.keep_alive = default.get("keep_alive", "10m")
        self.timeout_s = float(default.get("timeout_s", DEFAULT_TIMEOUT_S))
        retry = dict(default.get("retry") or {})
        self.attempts = int(retry.get("attempts", default.get("attempts", 3)))
        backoff = retry.get("backoff_seconds") or default.get("backoff_seconds") or [0.5, 1.0, 2.0]
        self.backoff_seconds = tuple(float(x) for x in backoff)

    # -- LLMClient -----------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(temperature),
                "top_p": self.top_p,
                "num_predict": int(max_tokens),
                "num_ctx": self.num_ctx,
            },
        }
        # ``format`` = "json" is the documented happy path for Qwen on Ollama. A full
        # JSON-schema ``format`` is not portable across Ollama versions, and the runtime
        # validates the payload locally anyway, so it is not requested.
        if format_json:
            payload["format"] = "json"

        def parse(data: dict[str, Any]) -> dict[str, Any]:
            error = data.get("error")
            if error:
                raise NonRetryableError(f"ollama error for model {self.model!r}: {error}")
            text = data.get("response")
            if not isinstance(text, str) or not text.strip():
                raise RetryableError(
                    f"ollama returned an empty response for model {self.model!r} "
                    f"(done={data.get('done')!r})"
                )
            prompt_tokens = int(data.get("prompt_eval_count") or 0)
            completion_tokens = int(data.get("eval_count") or 0)
            return {
                "text": text,
                "model": str(data.get("model") or self.model),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        parsed = self._request_json(
            f"{self.base_url}/api/generate", payload, parse=parse
        )
        latency_ms = self._last_latency_ms
        response = LLMResponse(
            text=parsed["text"],
            model=parsed["model"],
            usage=parsed["usage"],
            latency_ms=latency_ms,
        )
        self._emit(
            {
                "kind": "llm_call",
                "backend": self.backend_name,
                "model": response.model,
                "cached": False,
                "latency_ms": latency_ms,
                "usage": response.usage,
            }
        )
        return response


class OpenAICompatLLM(_RetryingBackend):
    """OpenAI-compatible ``/chat/completions`` backend (key from the environment)."""

    backend_name = "openai_compat"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
        | None = None,
        sleep: Callable[[float], None] = time.sleep,
        recorder: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(transport=transport, sleep=sleep, recorder=recorder)
        model_cfg = model_section(config)
        compat = dict(model_cfg.get("openai_compat") or {})
        default = dict(model_cfg.get("default") or {})
        self.model = str(
            model
            or os.environ.get("AICWF_LLM_MODEL")
            or compat.get("model")
            or default.get("model")
            or ""
        )
        if not self.model:
            raise ConfigError("openai_compat backend requires a model name")
        self.base_url = str(
            base_url or os.environ.get("AICWF_LLM_BASE_URL") or compat.get("base_url") or ""
        ).rstrip("/")
        if not self.base_url:
            raise ConfigError(
                "openai_compat backend requires base_url (config or AICWF_LLM_BASE_URL)"
            )
        key_env = str(compat.get("api_key_env") or "AICWF_LLM_API_KEY")
        self._api_key = api_key if api_key is not None else os.environ.get(key_env, "")
        self._key_env = key_env
        if not self._api_key:
            raise ConfigError(
                f"openai_compat backend requires an API key in environment variable "
                f"{key_env} (never commit it)"
            )
        self.timeout_s = float(compat.get("timeout_s", 120.0))
        retry = dict(compat.get("retry") or {})
        self.attempts = int(retry.get("attempts", 3))
        self.backoff_seconds = tuple(
            float(x) for x in (retry.get("backoff_seconds") or [1.0, 2.0, 4.0])
        )
        self.temperature = float(compat.get("temperature", 0.1))
        self.use_response_format = bool(compat.get("use_response_format", False))

    def __repr__(self) -> str:  # never expose the key
        return f"OpenAICompatLLM(model={self.model!r}, base_url={self.base_url!r})"

    # -- LLMClient -----------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if format_json and self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        def parse(data: dict[str, Any]) -> dict[str, Any]:
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise NonRetryableError(
                    "openai_compat response missing choices[0].message.content: "
                    f"{str(data)[:300]}"
                ) from exc
            if not isinstance(text, str) or not text.strip():
                raise RetryableError("openai_compat returned an empty completion")
            usage_raw = data.get("usage") or {}
            prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
            completion_tokens = int(usage_raw.get("completion_tokens") or 0)
            return {
                "text": text,
                "model": str(data.get("model") or self.model),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(
                        usage_raw.get("total_tokens") or prompt_tokens + completion_tokens
                    ),
                },
            }

        parsed = self._request_json(
            f"{self.base_url}/chat/completions", payload, headers=headers, parse=parse
        )
        latency_ms = self._last_latency_ms
        response = LLMResponse(
            text=parsed["text"],
            model=parsed["model"],
            usage=parsed["usage"],
            latency_ms=latency_ms,
        )
        self._emit(
            {
                "kind": "llm_call",
                "backend": self.backend_name,
                "model": response.model,
                "cached": False,
                "latency_ms": latency_ms,
                "usage": response.usage,
            }
        )
        return response


class ReplayLLM:
    """Deterministic cache backend.

    ``mode="strict"`` serves only recorded responses and raises
    :class:`ReplayCacheMiss` otherwise. ``mode="record"`` requires a wrapped ``inner``
    live backend: a cache miss calls it once and persists the response atomically.
    """

    def __init__(
        self,
        *,
        mode: str = "strict",
        cache_dir: Path | str | None = None,
        inner: Any | None = None,
        model: str = "replay",
        recorder: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"strict", "record"}:
            raise ConfigError(f"ReplayLLM mode must be 'strict' or 'record', got {mode!r}")
        if mode == "record" and inner is None:
            raise ConfigError("ReplayLLM record mode requires a live inner backend")
        self.mode = mode
        self.model = str(model)
        self.inner = inner
        self.cache = ReplayCache(cache_dir)
        self.recorder = recorder

    def __repr__(self) -> str:
        return f"ReplayLLM(mode={self.mode!r}, dir={str(self.cache.dir)!r})"

    def _emit(self, record: dict[str, Any]) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder(record)
        except Exception:  # noqa: BLE001
            pass

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse:
        key = cache_key(
            model=self.model,
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
        )
        entry = self.cache.get(key)
        if entry is not None:
            payload = entry.get("response") or {}
            response = LLMResponse(
                text=str(payload.get("text", "")),
                model=str(payload.get("model") or entry.get("model") or self.model),
                usage={str(k): int(v) for k, v in (payload.get("usage") or {}).items()},
                cached=True,
                latency_ms=int(payload.get("latency_ms") or 0),
            )
            self._emit(
                {
                    "kind": "llm_call",
                    "backend": "replay",
                    "model": response.model,
                    "cached": True,
                    "cache_key": key,
                    "latency_ms": 0,
                    "usage": response.usage,
                }
            )
            return response
        if self.mode == "strict" or self.inner is None:
            raise ReplayCacheMiss(
                f"replay cache miss for model {self.model!r} (key {key}); "
                f"cache dir: {self.cache.dir}. Deterministic replay never calls a live "
                f"model - re-record the cache with AICWF_LLM_RECORD=1 and --llm-mode ollama."
            )
        started = time.perf_counter()
        response = self.inner.complete(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
        )
        latency_ms = response.latency_ms or int((time.perf_counter() - started) * 1000)
        entry = {
            "cache_version": CACHE_VERSION,
            "key": key,
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "backend": getattr(self.inner, "backend_name", type(self.inner).__name__),
            "model": response.model,
            "request": {
                "system": system,
                "user": user,
                "schema": dict(schema) if schema is not None else None,
                "temperature": round(float(temperature), 6),
                "max_tokens": int(max_tokens),
                "format_json": bool(format_json),
            },
            "response": {
                "text": response.text,
                "model": response.model,
                "usage": dict(response.usage or {}),
                "latency_ms": latency_ms,
            },
        }
        path = self.cache.put(key, entry)
        self._emit(
            {
                "kind": "llm_recorded",
                "backend": entry["backend"],
                "model": response.model,
                "cache_key": key,
                "cache_path": str(path),
                "latency_ms": latency_ms,
                "usage": dict(response.usage or {}),
            }
        )
        return LLMResponse(
            text=response.text,
            model=response.model,
            usage=dict(response.usage or {}),
            cached=False,
            latency_ms=latency_ms,
        )


class NullLLM:
    """Deterministic offline backend for tests and dry runs.

    * ``responses`` is an optional script: strings (or JSON-serialisable objects) are
      returned in order before any synthesis.
    * With a JSON schema it synthesises a bounded, schema-valid payload. The built-in
      defaults for the hypothesis and factor-flag schemas use real factor ids, describe
      the configured constraints and echo the information cutoff found in the prompt.
    * Without a schema (markdown roles) it emits deterministic prose that cites up to
      three numbers already present in the prompt, so number-integrity guards can run
      offline exactly as they do for a live model.
    """

    model = "null-model"
    backend_name = "null"

    def __init__(
        self,
        responses: Sequence[str | Mapping[str, Any]] | None = None,
        *,
        markdown: str | None = None,
        model: str = "null-model",
    ) -> None:
        self._queue = list(responses or [])
        self._markdown = markdown
        self.model = model
        self.calls = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        format_json: bool = True,
    ) -> LLMResponse:
        self.calls += 1
        if self._queue:
            item = self._queue.pop(0)
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        elif schema is not None:
            payload = _canned_for_schema(schema, user)
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        else:
            text = self._markdown or _canned_markdown(user)
        return LLMResponse(
            text=text,
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cached=False,
            latency_ms=0,
        )


# ---------------------------------------------------------------------------
# NullLLM canned payloads
# ---------------------------------------------------------------------------


def _find_cutoff(text: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else "2024-12-31"


def _schema_identity(schema: Mapping[str, Any]) -> str:
    return " ".join(
        str(schema.get(key) or "") for key in ("$id", "title", "description")
    ).lower()


def _canned_for_schema(schema: Mapping[str, Any], user: str) -> Any:
    identity = _schema_identity(schema)
    cutoff = _find_cutoff(user)
    if "hypothesis" in identity:
        return {
            "thesis": (
                "A diversified tilt toward cheap, profitable and momentum-leading CSI 300 "
                "members should earn positive excess return versus the benchmark after "
                "costs, because those characteristics have historically been compensated "
                "in the Chinese A-share cross-section and the frozen snapshot covers a "
                "full valuation cycle."
            ),
            "factor_weights": {"value_ep": 0.40, "quality_roe": 0.35, "momentum_12_1": 0.25},
            "rebalance": "monthly",
            "constraints": {
                "max_weight": 0.03,
                "max_active_weight": 0.015,
                "te_target": 0.05,
                "turnover_limit": 0.30,
            },
            "falsification": [
                "Reject the thesis if the full-sample RankIC t-statistic is below 2.0.",
                "Reject the thesis if annualised excess return is negative after 3x costs.",
            ],
            "information_cutoff": cutoff,
            "source_limits": [
                "frozen CSI 300 snapshot only; no network access during the run",
                "fundamentals are usable only after their announcement date",
            ],
        }
    if "factor_flag" in identity or "factor_flags" in identity:
        return {
            "assessment_markdown": (
                f"Information cutoff: {cutoff}. Null-mode factor review: coverage and "
                "information content are read from the provided factor report excerpt, "
                "and the flag list records the diagnostics a live analyst would raise. "
                "The assessment only cites numbers supplied in the prompt and never "
                "reaches outside the frozen snapshot. No blocking issue is asserted by "
                "this placeholder review; severity info is used deliberately because a "
                "live model must justify warn/block findings with a quoted diagnostic "
                "number."
            ),
            "summary": "Factors show usable coverage; momentum stability is the main risk.",
            "flags": [
                {
                    "id": "coverage_acceptable",
                    "severity": "info",
                    "message": "Cross-sectional coverage is reported as acceptable in the "
                    "supplied factor report excerpt.",
                },
                {
                    "id": "momentum_ic_instability",
                    "severity": "info",
                    "message": "Momentum RankIC stability should be checked against the "
                    "reported IC t-statistic before increasing its weight.",
                },
            ],
        }
    if "verification" in identity:
        return {
            "verdict": "PASS",
            "checks": [
                {"id": "artifact_schema", "status": "PASS", "detail": "null-mode placeholder"},
                {"id": "artifact_hashes", "status": "PASS", "detail": "null-mode placeholder"},
            ],
            "recomputed": {},
            "discrepancies": [],
            "artifact_hashes": {},
            "verifier": {"role": "verifier", "model": "null-model"},
        }
    if "approval" in identity:
        return {
            "run_id": "null-run",
            "action": "publish_report",
            "approver": "null-approver",
            "decided_at_utc": "2024-12-31T00:00:00+00:00",
            "scope": ["research_note.md"],
            "artifact_hashes": {},
            "decision": "approve",
            "note": "Null-mode approval; not a real human decision.",
        }
    if "version" in identity:
        return {
            "version": "v0.0.0-null",
            "git_commit": "",
            "config_hash": "",
            "prompt_hashes": {},
            "snapshot_hash": "",
            "metrics": {},
            "verification_verdict": "PASS",
            "approval": {"decision": "approve"},
            "decision": "accept",
            "previous_version": None,
        }
    return synthesize_json(schema)


def _canned_markdown(user: str) -> str:
    """Deterministic markdown that cites only numbers already present in the prompt."""

    cutoff = _find_cutoff(user)
    numbers: list[str] = []
    # Strip ISO dates so their day/month components are never quoted as measurements.
    scan_text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", user)
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?%?", scan_text):
        token = match.group(0)
        # Only quote values that read as measurements, not years/counts in prose.
        if _is_measurement_token(token) and token not in numbers:
            numbers.append(token)
        if len(numbers) >= 6:
            break
    cited = ", ".join(numbers) if numbers else "the supplied values"
    verdict_match = re.search(r'"verdict":\s*"(PASS|FAIL)"', user)
    verdict = verdict_match.group(1) if verdict_match else "PASS"
    verdict_sentence = (
        "The independently recomputed artifacts match the produced artifacts within the "
        "configured tolerances."
        if verdict == "PASS"
        else "At least one independent check failed; the run must be revised or escalated "
        "before approval."
    )
    return (
        "## Executive summary\n\n"
        f"Information cutoff: {cutoff}. Verification verdict: {verdict}. {verdict_sentence} "
        "This note is a deterministic null-mode placeholder produced without any model "
        "inference. It exists so that offline tests and dry runs exercise the full agent "
        "workflow while keeping the replay cache deterministic.\n\n"
        "## Independent verification\n\n"
        f"The supplied context reports the following measurement values: {cited}. A live "
        "model would discuss what these numbers imply, name the artifact that produced "
        "each value, and flag what remains uncertain. No number is introduced here that "
        "was not present in the prompt.\n\n"
        "## Verdict on robustness\n\n"
        "Speculation (not a finding): the placeholder cannot judge whether the evidence is "
        "robust; it only confirms that the offline agent plumbing, schema validation and "
        "citation guards are wired correctly. The companion table is metrics_table.md.\n\n"
        "## Checks performed\n\n"
        "Null mode performs no checks of its own; the deterministic stages produce and "
        "verify all numbers.\n\n"
        "## Discrepancies\n\n"
        "None reported by this placeholder. This is research documentation, not investment "
        "advice.\n\n"
        "## What would change our mind\n\n"
        "A real answer requires the deterministic artifacts supplied to a live model.\n\n"
        "## Limitations and information cutoff\n\n"
        f"The information cutoff is {cutoff}. This paragraph is intentionally conservative: "
        "the placeholder does not interpret the evidence, does not forecast future returns "
        "and does not recommend a trade. It only confirms that the offline agent plumbing, "
        "schema validation and citation guards are wired correctly end to end.\n"
    )


def _is_measurement_token(token: str) -> bool:
    """True for decimal/percentage values, not years or small integer counts."""

    if "." in token or token.endswith("%"):
        return True
    try:
        value = int(token)
    except ValueError:
        return False
    if 1900 <= value <= 2100 or abs(value) <= 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _cache_model_name(model_cfg: Mapping[str, Any]) -> str:
    default = dict(model_cfg.get("default") or {})
    compat = dict(model_cfg.get("openai_compat") or {})
    model = default.get("model") or compat.get("model") or DEFAULT_MODEL
    return str(os.environ.get("AICWF_LLM_MODEL") or model)


def _build_live_backend(
    model_cfg: Mapping[str, Any],
    *,
    backend: str,
    player: Callable[[dict[str, Any]], None] | None,
    transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
    | None,
    sleep: Callable[[float], None] | None,
) -> Any:
    kwargs: dict[str, Any] = {"recorder": player}
    if transport is not None:
        kwargs["transport"] = transport
    if sleep is not None:
        kwargs["sleep"] = sleep
    if backend == "ollama":
        return OllamaLLM(model_cfg, **kwargs)
    if backend == "openai_compat":
        return OpenAICompatLLM(model_cfg, **kwargs)
    if backend == "null":
        return NullLLM()
    raise ConfigError(f"unknown live LLM backend {backend!r}")


def make_llm(
    mode: str = "null",
    config: Mapping[str, Any] | None = None,
    cache_dir: Path | str | None = None,
    run_id: str | None = None,
    recorder: Callable[[dict[str, Any]], None] | None = None,
    *,
    transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
    | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Build the LLM backend for one run.

    Parameters
    ----------
    mode:
        ``"ollama"`` (default local model), ``"openai_compat"``, ``"null"``,
        ``"replay"`` (strict cache) or ``"record"`` (dev-only: live model + cache
        recording of every new prompt).
    config:
        Full config bundle (``{"model": ..., ...}``) or the ``model.yaml`` mapping.
    cache_dir:
        Overrides the replay cache location; defaults to
        ``config.model.replay.cache_dir`` (``evidence/llm_cache``).
    run_id:
        Run identifier used only for diagnostics/recording; it is deliberately not part
        of the cache key.
    recorder:
        Optional callback receiving one small dict per call/record event, used to emit
        ``events.jsonl`` entries without coupling the LLM layer to the controller.

    Notes
    -----
    * ``mode="ollama"`` / ``"openai_compat"`` are transparently wrapped with recording
      when ``config.model.replay.record`` is true or ``AICWF_LLM_RECORD=1`` is set. This
      is how the committed demo cache is produced (run once with recording enabled,
      then demo with ``--llm-mode replay``).
    * No backend ever writes a credential anywhere.
    """

    mode_norm = str(mode or "null").strip().lower()
    aliases = {
        "strict": "replay",
        "cache": "replay",
        "replay_strict": "replay",
        "openai": "openai_compat",
        "compat": "openai_compat",
        "none": "null",
        "offline": "null",
    }
    mode_norm = aliases.get(mode_norm, mode_norm)
    valid = {"ollama", "openai_compat", "null", "replay", "record"}
    if mode_norm not in valid:
        raise ConfigError(
            f"unknown llm_mode {mode!r}; expected one of {sorted(valid)}"
        )

    model_cfg = model_section(config)
    cache_path = resolve_cache_dir(cache_dir, model_cfg)
    replay_cfg = dict(model_cfg.get("replay") or {})
    record_enabled = bool(replay_cfg.get("record", False)) or _env_truthy("AICWF_LLM_RECORD")
    cache_model = _cache_model_name(model_cfg)

    def replay_player(record: dict[str, Any]) -> None:
        if recorder is not None:
            try:
                recorder(record)
            except Exception:  # noqa: BLE001
                pass

    if mode_norm == "replay":
        return ReplayLLM(
            mode="strict",
            cache_dir=cache_path,
            model=cache_model,
            recorder=replay_player,
        )
    if mode_norm == "record":
        live = _build_live_backend(
            model_cfg,
            backend=str(dict(model_cfg.get("default") or {}).get("backend") or "ollama"),
            player=replay_player,
            transport=transport,
            sleep=sleep,
        )
        if isinstance(live, NullLLM):
            return live
        return ReplayLLM(
            mode="record",
            cache_dir=cache_path,
            inner=live,
            model=cache_model,
            recorder=replay_player,
        )
    if mode_norm == "null":
        return NullLLM()

    backend = str(dict(model_cfg.get("default") or {}).get("backend") or "ollama")
    if mode_norm == "openai_compat":
        backend = "openai_compat"
    live = _build_live_backend(
        model_cfg,
        backend=backend,
        player=replay_player,
        transport=transport,
        sleep=sleep,
    )
    if record_enabled and not isinstance(live, NullLLM):
        return ReplayLLM(
            mode="record",
            cache_dir=cache_path,
            inner=live,
            model=cache_model,
            recorder=replay_player,
        )
    return live


def smoke_request(
    *,
    config: Mapping[str, Any] | None = None,
    transport: Callable[[str, dict[str, Any], float, Mapping[str, str] | None], Any]
    | None = None,
    prompt: str = "Reply with the JSON object {\"ok\": true}.",
) -> LLMResponse:
    """One tiny request used by the manual Ollama smoke test and doctor command."""

    llm = OllamaLLM(config, transport=transport)
    return llm.complete(
        system="You are a connectivity probe. Return one JSON object and nothing else.",
        user=prompt,
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        temperature=0.0,
        max_tokens=64,
        format_json=True,
    )
