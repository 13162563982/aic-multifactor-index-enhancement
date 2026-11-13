"""Optional real-model smoke test against the local Ollama server.

Skipped by default so the normal suite is fully offline. Run it manually with::

    $env:AICWF_OLLAMA_SMOKE = "1"
    python -m pytest tests -q -k ollama_smoke -m slow -s

The test performs exactly one tiny JSON-mode call and reports latency and token usage; it
is not part of the deterministic evaluation path.
"""

from __future__ import annotations

import json
import os

import pytest

from aicwf.agents.llm import OllamaLLM
from aicwf.contracts import PROJECT_ROOT
from aicwf.util.jsonio import read_yaml

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("AICWF_OLLAMA_SMOKE") != "1",
        reason="real-model smoke test disabled; set AICWF_OLLAMA_SMOKE=1 to run it",
    ),
]


def test_ollama_smoke_real_model() -> None:
    config = read_yaml(PROJECT_ROOT / "config" / "model.yaml")
    llm = OllamaLLM(config)
    response = llm.complete(
        system=(
            "You are a connectivity probe for a research workflow. Reply with exactly one "
            "JSON object and nothing else."
        ),
        user=(
            "Return one JSON object with exactly two fields: "
            '"ok": true and "probe": "qwen".'
        ),
        schema={
            "type": "object",
            "required": ["ok", "probe"],
            "properties": {"ok": {"type": "boolean"}, "probe": {"type": "string"}},
        },
        temperature=0.0,
        max_tokens=80,
        format_json=True,
    )
    payload = json.loads(response.text)
    assert payload.get("ok") is True
    assert isinstance(payload.get("probe"), str)
    assert response.usage.get("prompt_tokens", 0) > 0
    assert response.usage.get("completion_tokens", 0) > 0
    assert response.latency_ms >= 0
    print(
        "\nOllama smoke:"
        f" model={response.model}"
        f" latency_ms={response.latency_ms}"
        f" prompt_tokens={response.usage.get('prompt_tokens')}"
        f" completion_tokens={response.usage.get('completion_tokens')}"
    )
