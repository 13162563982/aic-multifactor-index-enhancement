"""HTTP transport with retries, jitter, host rotation and polite throttling.

The public Eastmoney endpoints are flaky (frequent ``RemoteDisconnected``
errors and occasional empty payloads). This module wraps a tiny transport
abstraction so the retry/rotation policy is testable offline with a fake
transport, and so the snapshot builder can record what happened.

The transport interface is deliberately minimal::

    transport.get(url, params={...}, timeout=20.0) -> (status_code, text)

``RequestsTransport`` implements it with ``requests.Session`` (keep-alive);
tests inject a scripted fake.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from aicwf.contracts import RetryableError


class TransportError(RuntimeError):
    """A network-level failure (connection reset, timeout, DNS, TLS)."""


class InvalidResponseError(RuntimeError):
    """The response arrived but is not usable (bad status, bad JSON, empty)."""


_IPV4_ONLY: bool = False


def force_ipv4_resolution(enable: bool = True) -> None:
    """Prefer IPv4 addresses for all later socket resolutions (process-wide).

    Diagnosed 2026-11-13: the Eastmoney public hosts advertise IPv6 addresses
    whose paths reset/black-hole most connections from this environment, while
    the IPv4 path is fast and stable (8/8 requests succeeded with keep-alive vs
    1/6 over IPv6). ``requests``/urllib3 prefer the first ``getaddrinfo`` result,
    so ``socket.getaddrinfo`` is constrained to ``AF_INET`` once, inside the
    data-fetching process only (snapshot builder CLI / live network tests).
    """

    global _IPV4_ONLY
    if not enable or _IPV4_ONLY:
        return
    import socket

    original = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only  # type: ignore[assignment]
    _IPV4_ONLY = True


@runtime_checkable
class HttpTransport(Protocol):
    """Minimal transport used by :class:`HttpClient`."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> tuple[int, str]: ...


class RequestsTransport:
    """``requests``-based transport with UTF-8/GB18030 tolerant decoding."""

    def __init__(
        self,
        headers: Mapping[str, str] | None = None,
        session: Any = None,
        *,
        ipv4_only: bool = True,
    ) -> None:
        import requests  # local import keeps the module importable without requests

        if ipv4_only:
            force_ipv4_resolution(True)
        self._requests = requests
        self.session = session or requests.Session()
        if headers:
            self.session.headers.update(dict(headers))
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> tuple[int, str]:
        try:
            response = self.session.get(url, params=dict(params or {}), timeout=timeout)
        except self._requests.RequestException as exc:  # connection reset/timeout/TLS
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        raw = response.content
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gb18030", errors="replace")
        return int(response.status_code), text


def _default_rng(seed: int = 20261112) -> random.Random:
    return random.Random(seed)


@dataclass
class HttpClient:
    """JSON GET client with bounded retries, backoff, jitter and host rotation.

    ``urls`` may be a single URL or a sequence; attempt ``i`` uses
    ``urls[i % len(urls)]``, which implements round-robin host rotation while
    retrying. A response is retried when the transport raises, the HTTP status is
    not 200, the body is not valid JSON, or ``validator(payload)`` returns false
    (used to reject the "HTTP 200 but empty data" behaviour of some hosts).
    """

    transport: HttpTransport
    attempts: int = 4
    backoff_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    jitter_fraction: float = 0.3
    timeout_s: float = 20.0
    polite_delay_s: float = 0.15
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=_default_rng)
    last_attempts: int = 0
    total_polite_sleep: float = 0.0
    total_backoff_sleep: float = 0.0

    # -- internals -----------------------------------------------------
    def _jittered(self, delay: float) -> float:
        if delay <= 0:
            return 0.0
        spread = abs(float(self.jitter_fraction))
        return max(0.0, delay * (1.0 + self.rng.uniform(-spread, spread)))

    def _backoff(self, attempt: int, override: Sequence[float] | None = None) -> float:
        backoffs = tuple(override) if override else tuple(self.backoff_seconds)
        if not backoffs:
            return 0.0
        base = float(backoffs[min(attempt, len(backoffs) - 1)])
        return self._jittered(base)

    def _polite(self) -> None:
        if self.polite_delay_s > 0:
            self.sleep(self.polite_delay_s)
            self.total_polite_sleep += float(self.polite_delay_s)

    # -- public --------------------------------------------------------
    def get_json(
        self,
        url: str | Sequence[str],
        *,
        params: Mapping[str, Any] | None = None,
        validator: Callable[[Any], bool] | None = None,
        describe: str = "",
        attempts: int | None = None,
        backoff_seconds: Sequence[float] | None = None,
    ) -> Any:
        """Fetch and decode JSON, retrying transient failures.

        Raises :class:`aicwf.contracts.RetryableError` when every attempt fails.
        """

        urls = [url] if isinstance(url, str) else list(url)
        if not urls:
            raise ValueError("get_json requires at least one URL")
        max_attempts = int(attempts if attempts is not None else self.attempts)
        label = describe or urls[0]
        last_error = "no attempt made"
        self.last_attempts = 0
        for attempt in range(max_attempts):
            self.last_attempts = attempt + 1
            target = urls[attempt % len(urls)]
            self._polite()
            try:
                status, text = self.transport.get(target, params=params, timeout=self.timeout_s)
            except TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts - 1:
                    delay = self._backoff(attempt, backoff_seconds)
                    self.sleep(delay)
                    self.total_backoff_sleep += delay
                continue
            if status != 200:
                last_error = f"HTTP {status}"
                if attempt < max_attempts - 1:
                    delay = self._backoff(attempt, backoff_seconds)
                    self.sleep(delay)
                    self.total_backoff_sleep += delay
                continue
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"invalid JSON: {exc}"
                if attempt < max_attempts - 1:
                    delay = self._backoff(attempt, backoff_seconds)
                    self.sleep(delay)
                    self.total_backoff_sleep += delay
                continue
            if validator is not None:
                try:
                    accepted = bool(validator(payload))
                except Exception as exc:  # noqa: BLE001 - validator must not crash the retry loop
                    accepted = False
                    last_error = f"validator raised {type(exc).__name__}: {exc}"
                else:
                    last_error = "payload rejected by validator"
                if not accepted:
                    if attempt < max_attempts - 1:
                        delay = self._backoff(attempt, backoff_seconds)
                        self.sleep(delay)
                        self.total_backoff_sleep += delay
                    continue
            return payload
        raise RetryableError(f"{label} failed after {max_attempts} attempt(s): {last_error}")
