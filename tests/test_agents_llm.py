"""Tests for the LLM backends (all offline: fake transports only, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicwf.agents.llm import (
    NullLLM,
    OllamaLLM,
    OpenAICompatLLM,
    ReplayCache,
    ReplayCacheMiss,
    ReplayLLM,
    cache_key,
    make_llm,
    smoke_request,
)
from aicwf.agents.validate import validate_payload
from aicwf.contracts import (
    ConfigError,
    LLMResponse,
    NonRetryableError,
    RetryableError,
)
from test_agents_support import AS_OF_STR, make_ctx, mini_config


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class FakeTransport:
    """Records every request and serves a scripted list of responses/exceptions."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, payload, timeout, headers=None):
        self.calls.append(
            {"url": url, "payload": payload, "timeout": timeout, "headers": dict(headers or {})}
        )
        item = self.responses.pop(0) if self.responses else FakeResponse(500, {"error": "exhausted"})
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    """Tiny live backend used as the inner client of a recording ReplayLLM."""

    model = "fake-live-model"
    backend_name = "fake"

    def __init__(self, text: str = '{"ok": true}') -> None:
        self.text = text
        self.calls = 0

    def complete(self, *, system, user, schema=None, temperature=0.1, max_tokens=1600, format_json=True):
        self.calls += 1
        return LLMResponse(
            text=self.text,
            model=self.model,
            usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            cached=False,
            latency_ms=7,
        )


# ---------------------------------------------------------------------------
# Cache key and replay cache
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_and_sensitive() -> None:
    base = dict(model="m", system="sys", user="usr", schema={"type": "object"}, temperature=0.1, max_tokens=100)
    first = cache_key(**base)
    assert first == cache_key(**base)
    # Key order inside the schema must not matter (canonical JSON).
    reordered = dict(base, schema={"type": "object"}, user="usr")
    assert cache_key(**reordered) == first
    assert cache_key(**dict(base, temperature=0.0)) != first
    assert cache_key(**dict(base, max_tokens=200)) != first
    assert cache_key(**dict(base, user="other")) != first
    assert cache_key(**dict(base, model="m2")) != first
    assert cache_key(**dict(base, schema=None)) != first


def test_replay_record_then_strict_replay_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    inner = FakeLLM('{"answer": 42}')
    recorder = ReplayLLM(
        mode="record", cache_dir=cache_dir, inner=inner, model="fake-live-model"
    )
    kwargs = dict(
        system="you are a test",
        user="give me the answer",
        schema={"type": "object", "required": ["answer"]},
        temperature=0.1,
        max_tokens=64,
        format_json=True,
    )
    recorded = recorder.complete(**kwargs)
    assert recorded.cached is False
    assert recorded.text == '{"answer": 42}'
    assert inner.calls == 1
    entries = list(cache_dir.glob("*.json"))
    assert len(entries) == 1

    replayer = ReplayLLM(mode="strict", cache_dir=cache_dir, model="fake-live-model")
    replayed = replayer.complete(**kwargs)
    assert replayed.cached is True
    assert replayed.text == recorded.text
    assert replayed.usage["prompt_tokens"] == 11
    assert replayed.latency_ms == 7
    assert inner.calls == 1  # replay never calls the wrapped backend
    entry = json.loads(entries[0].read_text(encoding="utf-8"))
    assert entry["response"]["text"] == '{"answer": 42}'
    assert entry["cache_version"] == 1


def test_replay_strict_cache_miss_raises(tmp_path: Path) -> None:
    replayer = ReplayLLM(mode="strict", cache_dir=tmp_path / "empty", model="m")
    with pytest.raises(ReplayCacheMiss) as info:
        replayer.complete(system="s", user="u", temperature=0.0, max_tokens=10)
    assert "replay cache miss" in str(info.value)
    assert "AICWF_LLM_RECORD" in str(info.value)


def test_replay_corrupt_entry_is_a_hard_error(tmp_path: Path) -> None:
    key = "ab" * 32
    path = tmp_path / "cache" / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    cache = ReplayCache(tmp_path / "cache")
    with pytest.raises(NonRetryableError):
        cache.get(key)


def test_replay_cache_flat_file_layout(tmp_path: Path) -> None:
    cache = ReplayCache(tmp_path / "cache")
    key = "cd" * 32
    cache.put(key, {"cache_version": 1, "key": key, "response": {"text": "x"}})
    assert cache.keys() == [key]
    assert cache.entry_path(key) == (tmp_path / "cache" / f"{key}.json")
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_replay_record_requires_inner_backend(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ReplayLLM(mode="record", cache_dir=tmp_path / "c")
    with pytest.raises(ConfigError):
        ReplayLLM(mode="bogus", cache_dir=tmp_path / "c")


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def test_ollama_request_shape_and_usage_parsing() -> None:
    transport = FakeTransport(
        [
            FakeResponse(
                200,
                {
                    "response": '{"ok": true}',
                    "model": "qwen3.5:9b",
                    "prompt_eval_count": 123,
                    "eval_count": 45,
                },
            )
        ]
    )
    llm = OllamaLLM(mini_config(), transport=transport, sleep=lambda _s: None)
    response = llm.complete(
        system="system prompt", user="user prompt", temperature=0.0, max_tokens=64
    )
    assert response.text == '{"ok": true}'
    assert response.usage == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    assert response.cached is False
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:9/api/generate"
    payload = call["payload"]
    assert payload["model"] == "test-model"
    assert payload["system"] == "system prompt"
    assert payload["prompt"] == "user prompt"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 64
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_ctx"] == 8192


def test_ollama_retries_on_5xx_then_succeeds() -> None:
    sleeps: list[float] = []
    transport = FakeTransport(
        [
            FakeResponse(500, {"error": "boom"}),
            FakeResponse(200, {"response": '{"ok": true}', "prompt_eval_count": 1, "eval_count": 2}),
        ]
    )
    llm = OllamaLLM(mini_config(), transport=transport, sleep=sleeps.append)
    response = llm.complete(system="s", user="u")
    assert response.text == '{"ok": true}'
    assert len(transport.calls) == 2
    assert llm.retries == 1
    assert sleeps == [0.0]


def test_ollama_retries_on_timeout_exception() -> None:
    sleeps: list[float] = []
    transport = FakeTransport(
        [
            TimeoutError("read timed out"),
            FakeResponse(200, {"response": '{"ok": true}'}),
        ]
    )
    llm = OllamaLLM(mini_config(), transport=transport, sleep=sleeps.append)
    assert llm.complete(system="s", user="u").text == '{"ok": true}'
    assert len(transport.calls) == 2


def test_ollama_exhausted_retries_raises_retryable() -> None:
    transport = FakeTransport([FakeResponse(500, text="down") for _ in range(3)])
    llm = OllamaLLM(mini_config(), transport=transport, sleep=lambda _s: None)
    with pytest.raises(RetryableError):
        llm.complete(system="s", user="u")
    assert len(transport.calls) == 3


def test_ollama_4xx_is_non_retryable() -> None:
    transport = FakeTransport([FakeResponse(404, text="model not found")])
    llm = OllamaLLM(mini_config(), transport=transport, sleep=lambda _s: None)
    with pytest.raises(NonRetryableError):
        llm.complete(system="s", user="u")
    assert len(transport.calls) == 1


def test_ollama_empty_response_is_retried_then_fails() -> None:
    transport = FakeTransport(
        [FakeResponse(200, {"response": "", "done": True, "model": "test-model"}) for _ in range(3)]
    )
    llm = OllamaLLM(mini_config(), transport=transport, sleep=lambda _s: None)
    with pytest.raises(RetryableError):
        llm.complete(system="s", user="u")
    assert len(transport.calls) == 3


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


def test_openai_compat_reads_key_from_env_and_hides_it(monkeypatch) -> None:
    monkeypatch.setenv("AICWF_TEST_API_KEY", "sk-secret-value")
    monkeypatch.setenv("AICWF_LLM_MODEL", "compat-model")
    transport = FakeTransport(
        [
            FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                    "model": "compat-model",
                },
            )
        ]
    )
    llm = OpenAICompatLLM(mini_config(), transport=transport, sleep=lambda _s: None)
    response = llm.complete(system="s", user="u", format_json=True)
    assert response.text == '{"ok": true}'
    assert response.usage["total_tokens"] == 13
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-secret-value"
    assert "sk-secret-value" not in repr(llm)
    assert call["payload"]["messages"][0] == {"role": "system", "content": "s"}


def test_openai_compat_missing_key_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("AICWF_TEST_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenAICompatLLM(mini_config())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_llm_null_mode(tmp_path: Path) -> None:
    llm = make_llm("null", mini_config(cache_dir=tmp_path / "cache"))
    assert isinstance(llm, NullLLM)
    response = llm.complete(system="s", user="u")
    assert response.text


def test_make_llm_replay_mode_is_strict(tmp_path: Path) -> None:
    config = mini_config(cache_dir=tmp_path / "cache")
    llm = make_llm("replay", config)
    assert isinstance(llm, ReplayLLM)
    assert llm.mode == "strict"
    assert llm.model == "test-model"
    assert llm.cache.dir == (tmp_path / "cache").resolve()


def test_make_llm_ollama_mode_and_recording_wrapper(tmp_path: Path, monkeypatch) -> None:
    config = mini_config(cache_dir=tmp_path / "cache")
    transport = FakeTransport([FakeResponse(200, {"response": '{"ok": true}'})])
    llm = make_llm("ollama", config, transport=transport, sleep=lambda _s: None)
    assert isinstance(llm, OllamaLLM)
    monkeypatch.setenv("AICWF_LLM_RECORD", "1")
    wrapped = make_llm("ollama", config, transport=transport, sleep=lambda _s: None)
    assert isinstance(wrapped, ReplayLLM)
    assert wrapped.mode == "record"


def test_make_llm_record_mode_records_and_replays(tmp_path: Path) -> None:
    config = mini_config(cache_dir=tmp_path / "cache")
    transport = FakeTransport(
        [FakeResponse(200, {"response": '{"ok": true}', "prompt_eval_count": 3, "eval_count": 2})]
    )
    recorder = make_llm("record", config, transport=transport, sleep=lambda _s: None)
    recorded = recorder.complete(system="s", user="u", temperature=0.0, max_tokens=20)
    assert recorded.cached is False
    replayer = make_llm("replay", config)
    replayed = replayer.complete(system="s", user="u", temperature=0.0, max_tokens=20)
    assert replayed.cached is True
    assert replayed.text == '{"ok": true}'
    assert len(transport.calls) == 1


def test_make_llm_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        make_llm("magic", mini_config(cache_dir=tmp_path / "cache"))


def test_make_llm_resolves_answer_from_full_bundle(tmp_path: Path) -> None:
    full_bundle = {"model": mini_config(cache_dir=tmp_path / "c")["model"], "strategy": {}}
    llm = make_llm("replay", full_bundle, cache_dir=tmp_path / "c2")
    assert llm.model == "test-model"


def test_smoke_request_uses_fake_transport() -> None:
    transport = FakeTransport([FakeResponse(200, {"response": '{"ok": true}'})])
    response = smoke_request(config=mini_config(), transport=transport)
    assert response.text == '{"ok": true}'
    assert transport.calls[0]["payload"]["format"] == "json"


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------


def test_null_llm_is_deterministic_and_schema_valid() -> None:
    from aicwf.agents.validate import load_schema

    schema = load_schema("hypothesis")
    llm = NullLLM()
    prompt = f"Information cutoff (as_of): {AS_OF_STR}\nFactor catalog: value_ep, quality_roe, momentum_12_1"
    first = llm.complete(system="s", user=prompt, schema=schema)
    second = llm.complete(system="s", user=prompt, schema=schema)
    assert first.text == second.text
    payload = json.loads(first.text)
    assert validate_payload(payload, schema) == []
    assert payload["information_cutoff"] == AS_OF_STR


def test_null_llm_markdown_cites_prompt_numbers_and_cutoff() -> None:
    llm = NullLLM()
    user = f"Information cutoff: {AS_OF_STR}\nmetrics: 0.0812, 0.0234, 0.049, 0.478, 0.184, 2.8"
    text = llm.complete(system="s", user=user).text
    assert AS_OF_STR in text
    for token in ("0.0812", "0.0234", "0.049"):
        assert token in text
    assert "PASS" in text  # default verdict when none supplied


def test_null_llm_respects_scripted_responses() -> None:
    llm = NullLLM(['{"a": 1}', {"b": 2}], model="scripted-null")
    assert json.loads(llm.complete(system="s", user="u").text) == {"a": 1}
    assert json.loads(llm.complete(system="s", user="u").text) == {"b": 2}


def test_null_llm_in_stage_context(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, llm=NullLLM())
    from aicwf.agents.roles import resolve_llm

    assert resolve_llm(ctx) is ctx.llm
