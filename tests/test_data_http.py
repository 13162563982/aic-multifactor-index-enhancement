"""Offline tests for the retry / backoff / host-rotation HTTP layer."""

from __future__ import annotations

import json

import pytest

from aicwf.contracts import RetryableError
from aicwf.data.http import HttpClient, TransportError


class FakeTransport:
    """Scripted transport: each ``get`` pops the next response or raises it."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def get(self, url: str, *, params=None, timeout: float = 20.0):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self.script:
            raise AssertionError("FakeTransport script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(script: list, **overrides):
    sleeps: list[float] = []
    transport = FakeTransport(script)
    kwargs = dict(
        transport=transport,
        attempts=3,
        backoff_seconds=(0.5, 1.0),
        jitter_fraction=0.0,
        polite_delay_s=0.0,
        timeout_s=5.0,
        sleep=sleeps.append,
    )
    kwargs.update(overrides)
    return HttpClient(**kwargs), transport, sleeps


def test_success_first_attempt():
    client, transport, sleeps = make_client([(200, json.dumps({"ok": True}))])
    assert client.get_json("https://example.test/x") == {"ok": True}
    assert len(transport.calls) == 1
    assert client.last_attempts == 1
    assert sleeps == []


def test_retries_transport_error_then_succeeds():
    client, transport, sleeps = make_client(
        [TransportError("connection reset"), (200, json.dumps({"ok": 1}))]
    )
    assert client.get_json("https://example.test/x") == {"ok": 1}
    assert len(transport.calls) == 2
    assert client.last_attempts == 2
    assert sleeps == [0.5]  # first backoff value, jitter disabled


def test_retries_http_500_then_succeeds():
    client, transport, sleeps = make_client([(500, "server error"), (200, '{"ok": 2}')])
    assert client.get_json("https://example.test/x") == {"ok": 2}
    assert sleeps == [0.5]


def test_retries_invalid_json_then_succeeds():
    client, transport, sleeps = make_client([(200, "<html>nope</html>"), (200, '{"ok": 3}')])
    assert client.get_json("https://example.test/x") == {"ok": 3}


def test_validator_rejects_empty_payload_and_retries():
    script = [
        (200, '{"rc": 0, "data": null}'),
        (200, '{"rc": 0, "data": {"klines": ["2024-01-02,1,2,3,4,5,6"]}}'),
    ]
    client, transport, _ = make_client(script)
    payload = client.get_json(
        "https://example.test/kline",
        validator=lambda body: bool((body.get("data") or {}).get("klines")),
    )
    assert payload["data"]["klines"]
    assert len(transport.calls) == 2


def test_exhausted_retries_raise_retryable_error():
    client, transport, sleeps = make_client(
        [TransportError("a"), (500, "b"), TransportError("c")],
        attempts=3,
    )
    with pytest.raises(RetryableError, match="failed after 3 attempt"):
        client.get_json("https://example.test/x", describe="kline 1.600519")
    assert len(transport.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_host_rotation_cycles_urls_on_retries():
    urls = ["https://host-a.test/api", "https://host-b.test/api", "https://host-c.test/api"]
    client, transport, _ = make_client([TransportError("x"), TransportError("y"), (200, '{"ok": 1}')])
    assert client.get_json(urls) == {"ok": 1}
    assert [call["url"] for call in transport.calls] == urls


def test_polite_delay_applied_per_request():
    client, transport, sleeps = make_client([(200, '{"ok": 1}')], polite_delay_s=0.25)
    client.get_json("https://example.test/x")
    assert sleeps == [0.25]
    assert client.total_polite_sleep == pytest.approx(0.25)


def test_jitter_bounds():
    sleeps: list[float] = []
    transport = FakeTransport([TransportError("x"), (200, '{"ok": 1}')])
    client = HttpClient(
        transport=transport,
        attempts=2,
        backoff_seconds=(1.0,),
        jitter_fraction=0.5,
        polite_delay_s=0.0,
        sleep=sleeps.append,
    )
    client.get_json("https://example.test/x")
    assert len(sleeps) == 1
    assert 0.5 <= sleeps[0] <= 1.5


def test_params_and_timeout_forwarded_to_transport():
    client, transport, _ = make_client([(200, '{"ok": 1}')])
    client.get_json("https://example.test/x", params={"pn": 2, "pz": 100})
    assert transport.calls[0]["params"] == {"pn": 2, "pz": 100}
    assert transport.calls[0]["timeout"] == 5.0


def test_empty_url_sequence_rejected():
    client, _, _ = make_client([])
    with pytest.raises(ValueError):
        client.get_json([])
