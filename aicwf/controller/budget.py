"""Run budget ledger and the budget-guarded LLM client.

Every expensive controller action is charged against a :class:`BudgetLedger`:

* wall-clock time since the run (or resume) started,
* number of LLM calls,
* prompt + completion tokens,
* priced LLM cost (only meaningful for paid remote backends; the local Ollama
  default has zero prices in ``config/budgets.yaml``).

The ledger is a plain in-memory object; the runner persists its snapshot into the
``budgets`` table after every stage and whenever a run stops. :class:`GuardedLLM`
wraps the raw backend from ``aicwf/pipeline.py`` so that stages cannot spend
beyond the configured limits even if they ignore the ledger directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from aicwf.contracts import BudgetExceeded, BudgetLimits, LLMClient, LLMResponse
from aicwf.util.logging import get_logger

__all__ = [
    "BUDGET_METRICS",
    "BudgetLedger",
    "BudgetLimits",
    "BudgetExceeded",
    "GuardedLLM",
]

BUDGET_METRICS: tuple[str, ...] = ("wall_seconds", "llm_calls", "tokens_total", "cost_usd")

_LOGGER = get_logger("aicwf.controller.budget")
_ESTIMATED_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough, deterministic token estimate used only for pre-call budget checks."""

    return max(1, len(text) // _ESTIMATED_CHARS_PER_TOKEN)


@dataclass
class BudgetLedger:
    """Tracks usage against :class:`aicwf.contracts.BudgetLimits`.

    Parameters
    ----------
    limits:
        Hard limits for the run.
    price_per_1k_input_usd / price_per_1k_output_usd:
        Optional token prices; zero means "free local model" and the cost metric
        stays at ``0.0``.
    clock:
        Monotonic clock, injectable in tests.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    price_per_1k_input_usd: float = 0.0
    price_per_1k_output_usd: float = 0.0
    clock: Callable[[], float] = field(default=time.monotonic, compare=False, repr=False)
    started_at: float | None = field(default=None, repr=False)
    used_wall_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    # -- construction --------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any] | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> "BudgetLedger":
        """Build a ledger from the ``config/budgets.yaml`` bundle."""

        cfg = dict(cfg or {})
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), Mapping) else {}
        llm_cfg = dict(llm_cfg or {})
        return cls(
            limits=BudgetLimits.from_config(cfg),
            price_per_1k_input_usd=float(llm_cfg.get("price_per_1k_input_usd", 0.0) or 0.0),
            price_per_1k_output_usd=float(llm_cfg.get("price_per_1k_output_usd", 0.0) or 0.0),
            clock=clock,
        )

    # -- timeline ------------------------------------------------------
    def start(self) -> None:
        """Start (or restart) the wall-clock stopwatch."""

        self.started_at = self.clock()

    def _ensure_started(self) -> None:
        if self.started_at is None:
            self.started_at = self.clock()

    def wall_seconds(self) -> float:
        """Total wall-clock seconds used, including time before a resume."""

        self._ensure_started()
        assert self.started_at is not None
        return self.used_wall_seconds + max(0.0, self.clock() - self.started_at)

    def restore(self, used: Mapping[str, float]) -> None:
        """Restore usage from persisted state before resuming a run."""

        self.used_wall_seconds = float(used.get("wall_seconds", 0.0) or 0.0)
        self.llm_calls = int(used.get("llm_calls", 0) or 0)
        self.prompt_tokens = int(used.get("prompt_tokens", 0) or 0)
        self.completion_tokens = int(used.get("completion_tokens", 0) or 0)
        self.cost_usd = float(used.get("cost_usd", 0.0) or 0.0)
        self.started_at = None

    # -- usage ---------------------------------------------------------
    @property
    def tokens_total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def used(self) -> dict[str, float]:
        return {
            "wall_seconds": self.wall_seconds(),
            "llm_calls": float(self.llm_calls),
            "tokens_total": float(self.tokens_total),
            "cost_usd": float(self.cost_usd),
        }

    def limit_values(self) -> dict[str, float]:
        return {
            "wall_seconds": float(self.limits.max_wall_seconds),
            "llm_calls": float(self.limits.max_llm_calls),
            "tokens_total": float(self.limits.max_tokens_total),
            "cost_usd": float(self.limits.max_cost_usd),
        }

    def remaining(self) -> dict[str, float]:
        used = self.used()
        return {metric: max(0.0, self.limit_values()[metric] - used[metric]) for metric in BUDGET_METRICS}

    def exhausted(self) -> list[str]:
        """Metrics whose usage strictly exceeds the configured limit."""

        used = self.used()
        limits = self.limit_values()
        return [metric for metric in BUDGET_METRICS if used[metric] > limits[metric] + 1e-12]

    def snapshot(self) -> dict[str, tuple[float, float]]:
        """``metric -> (limit, used)`` payload for :meth:`StateStore.upsert_budgets`."""

        used = self.used()
        limits = self.limit_values()
        return {metric: (limits[metric], used[metric]) for metric in BUDGET_METRICS}

    def to_dict(self) -> dict[str, Any]:
        used = self.used()
        limits = self.limit_values()
        return {
            "limits": limits,
            "used": used,
            "remaining": self.remaining(),
            "exhausted": self.exhausted(),
        }

    # -- enforcement ---------------------------------------------------
    def _raise(self, metric: str, *, projected: float) -> None:
        limits = self.limit_values()
        used = self.used()
        raise BudgetExceeded(
            f"budget exhausted: {metric}={projected:.4f} exceeds limit {limits[metric]:.4f} "
            f"(used {used[metric]:.4f})"
        )

    def ensure_capacity(self, *, calls: int = 0, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Raise :class:`BudgetExceeded` if the next action would exceed a limit.

        Prospective checks use ``>`` (not ``>=``): using exactly the budget is
        allowed, the *next* unit is not.
        """

        used = self.used()
        limits = self.limit_values()
        if calls and used["llm_calls"] + calls > limits["llm_calls"]:
            self._raise("llm_calls", projected=used["llm_calls"] + calls)
        if tokens and used["tokens_total"] + tokens > limits["tokens_total"]:
            self._raise("tokens_total", projected=used["tokens_total"] + tokens)
        if cost_usd > 0.0 and used["cost_usd"] + cost_usd > limits["cost_usd"]:
            self._raise("cost_usd", projected=used["cost_usd"] + cost_usd)
        if used["wall_seconds"] > limits["wall_seconds"]:
            self._raise("wall_seconds", projected=used["wall_seconds"])

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if any metric is already over its limit."""

        reasons = self.exhausted()
        if reasons:
            metric = reasons[0]
            self._raise(metric, projected=self.used()[metric])

    # -- accounting ----------------------------------------------------
    def estimated_cost(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000.0 * self.price_per_1k_input_usd
            + completion_tokens / 1000.0 * self.price_per_1k_output_usd
        )

    def record_llm_call(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float | None = None,
        check: bool = True,
    ) -> None:
        """Account one completed (or attempted) LLM call.

        ``calls`` are counted even for failed calls because the API was hit; token
        accounting uses the response usage when available.
        """

        self.llm_calls += 1
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        if cost_usd is None:
            cost_usd = self.estimated_cost(
                prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
            )
        self.cost_usd += max(0.0, float(cost_usd))
        if check:
            self.check()


def _extract_usage(usage: Mapping[str, Any] | None) -> tuple[int, int]:
    """Read prompt/completion token counts from common usage-dict spellings."""

    usage = dict(usage or {})
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", usage.get("prompt")))
    completion = usage.get("completion_tokens", usage.get("output_tokens", usage.get("completion")))
    try:
        prompt_i = int(prompt) if prompt is not None else 0
    except (TypeError, ValueError):
        prompt_i = 0
    try:
        completion_i = int(completion) if completion is not None else 0
    except (TypeError, ValueError):
        completion_i = 0
    return max(0, prompt_i), max(0, completion_i)


class GuardedLLM:
    """``LLMClient`` decorator that charges every call to a :class:`BudgetLedger`.

    The wrapper is placed on ``RunContext.llm`` by the runner, so agent stages
    never see the raw backend. It preserves the backend's ``model`` attribute and
    the exact :class:`aicwf.contracts.LLMClient` call signature.
    """

    def __init__(
        self,
        client: LLMClient,
        ledger: BudgetLedger,
        *,
        run_id: str = "",
        stage: str = "",
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.run_id = run_id
        self.stage = stage

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "")

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
        """Call the backend under a pre-call and post-call budget check."""

        estimated_prompt = _estimate_tokens(system) + _estimate_tokens(user)
        estimated_cost = self.ledger.estimated_cost(
            prompt_tokens=estimated_prompt, completion_tokens=max_tokens
        )
        self.ledger.ensure_capacity(calls=1, tokens=estimated_prompt, cost_usd=estimated_cost)
        response = self.client.complete(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
        )
        prompt_tokens, completion_tokens = _extract_usage(getattr(response, "usage", None))
        if prompt_tokens <= 0:
            prompt_tokens = estimated_prompt
        if completion_tokens <= 0:
            completion_tokens = _estimate_tokens(getattr(response, "text", "") or "")
        actual_cost = self.ledger.estimated_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        self.ledger.record_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=actual_cost,
            check=True,
        )
        _LOGGER.debug(
            "llm call run=%s stage=%s model=%s prompt=%d completion=%d cost=%.6f",
            self.run_id,
            self.stage,
            self.model,
            prompt_tokens,
            completion_tokens,
            actual_cost,
        )
        return response
