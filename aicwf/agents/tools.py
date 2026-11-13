"""Harness-mediated tools available to an agent role.

The model never touches the filesystem or the network directly. It can only emit JSON
tool directives; :class:`Toolbox` executes a small whitelist of deterministic, read-only
operations plus two terminal submission directives that the runtime validates:

``read_artifact``
    Read a named resource (a bounded prompt blob prepared by the stage or a text
    artifact file) through a read-path allowlist. Binary artifacts such as parquet are
    refused with an explanation instead of being dumped into the context.
``list_artifacts``
    List the resources and deterministic checks the model may use.
``run_check``
    Run one whitelisted deterministic check registered by the stage (for example
    ``normalize_weights`` or ``factor_stats``) and return its JSON result.
``submit_json`` / ``submit_markdown``
    Terminal directives; the runtime validates their content against the output contract.

Writes are a separate, non-model API: :meth:`AgentRuntime.write_json` and
:meth:`AgentRuntime.write_markdown` only accept logical names pre-registered by the
stage, so the role can never invent an output path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicwf.contracts import PROJECT_ROOT, ValidationError
from aicwf.util.hashing import canonical_json
from aicwf.util.jsonio import write_json, write_text

READ_ONLY_TOOLS: tuple[str, ...] = ("read_artifact", "list_artifacts", "run_check")
SUBMIT_TOOLS: tuple[str, ...] = ("submit_json", "submit_markdown")
TOOL_WHITELIST: tuple[str, ...] = READ_ONLY_TOOLS + SUBMIT_TOOLS

TEXT_SUFFIXES = frozenset({".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".jsonl"})


class ToolRejected(Exception):
    """A directive was refused; the runtime returns it to the model as an observation."""


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Toolbox:
    """Resources, checks and declared output paths for one agent role."""

    def __init__(
        self,
        *,
        files: Mapping[str, Path | str] | None = None,
        blobs: Mapping[str, Any] | None = None,
        checks: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
        read_roots: Sequence[Path | str] = (),
        write_paths: Mapping[str, Path | str] | None = None,
        write_roots: Sequence[Path | str] = (),
        max_read_chars: int = 12_000,
        max_check_chars: int = 8_000,
    ) -> None:
        self.files = {str(name): _resolve(path) for name, path in (files or {}).items()}
        self.blobs = {str(name): value for name, value in (blobs or {}).items()}
        self.checks = {str(name): fn for name, fn in (checks or {}).items()}
        self.read_roots = tuple(_resolve(root) for root in read_roots)
        self.write_paths = {
            str(name): _resolve(path) for name, path in (write_paths or {}).items()
        }
        self.write_roots = tuple(_resolve(root) for root in write_roots)
        self.max_read_chars = int(max_read_chars)
        self.max_check_chars = int(max_check_chars)

    # -- introspection -------------------------------------------------
    def resource_names(self) -> list[str]:
        return sorted(set(self.blobs) | set(self.files) | set(self.checks))

    def list_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for name in sorted(self.blobs):
            resources.append(
                {
                    "name": name,
                    "kind": "check" if name in self.checks else "blob",
                    "read_only": True,
                }
            )
        for name in sorted(self.files):
            path = self.files[name]
            size = path.stat().st_size if path.exists() else 0
            resources.append(
                {"name": name, "kind": "file", "bytes": size, "read_only": True}
            )
        for name in sorted(self.checks):
            if name in self.blobs:
                continue
            resources.append(
                {"name": name, "kind": "check", "read_only": True, "usage": "run_check"}
            )
        return resources

    # -- read-only tools ----------------------------------------------
    def read_artifact(self, name: str, *, max_chars: int | None = None) -> dict[str, Any]:
        key = str(name or "").strip()
        if not key:
            raise ToolRejected("read_artifact requires a resource name")
        limit = int(max_chars or self.max_read_chars)
        if key in self.blobs:
            content = self.blobs[key]
            if isinstance(content, str):
                text = content
            else:
                text = canonical_json(content)
            return _truncate_payload(key, text, limit)
        if key in self.files:
            path = self.files[key]
            if not self._is_readable(path):
                raise ToolRejected(
                    f"read_artifact: {key!r} resolves outside the read allowlist"
                )
            if path.suffix.lower() not in TEXT_SUFFIXES:
                raise ToolRejected(
                    f"read_artifact: {key!r} is a binary {path.suffix or 'file'} artifact; "
                    "read_artifact only returns text. Use run_check for numeric digests."
                )
            if not path.exists():
                raise ToolRejected(f"read_artifact: underlying file for {key!r} is missing")
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ToolRejected(f"read_artifact: cannot read {key!r}: {exc}") from exc
            return _truncate_payload(key, text, limit)
        raise ToolRejected(
            f"read_artifact: unknown resource {key!r}; available resources: "
            f"{self.resource_names()}"
        )

    def _is_readable(self, path: Path) -> bool:
        if not self.read_roots:
            return False
        return any(path == root or path.is_relative_to(root) for root in self.read_roots)

    def run_check(self, name: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        key = str(name or "").strip()
        if key not in self.checks:
            raise ToolRejected(
                f"run_check: unknown check {key!r}; available checks: {sorted(self.checks)}"
            )
        payload = dict(args or {})
        try:
            encoded = canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise ToolRejected(f"run_check: arguments must be JSON data: {exc}") from exc
        if len(encoded) > 20_000:
            raise ToolRejected("run_check: arguments are too large; pass small lookups only")
        try:
            result = self.checks[key](payload)
        except ToolRejected:
            raise
        except KeyError as exc:
            raise ToolRejected(f"run_check {key!r}: missing argument {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - a broken check is model feedback, not a crash
            raise ToolRejected(f"run_check {key!r} failed: {exc}") from exc
        try:
            encoded_result = canonical_json(result)
        except (TypeError, ValueError) as exc:
            raise ToolRejected(f"run_check {key!r} returned non-JSON data: {exc}") from exc
        if len(encoded_result) > self.max_check_chars:
            raise ToolRejected(
                f"run_check {key!r} result is too large; request a smaller lookup"
            )
        return {"name": key, "result": result, "chars": len(encoded_result)}

    # -- declared writes (never model-controlled) ----------------------
    def _write_target(self, name: str, *, suffix: str) -> Path:
        key = str(name or "").strip()
        if key not in self.write_paths:
            raise ValidationError(
                f"write {key!r} is not permitted for this role; declared outputs: "
                f"{sorted(self.write_paths)}"
            )
        path = self.write_paths[key]
        if path.suffix.lower() != suffix:
            raise ValidationError(
                f"write {key!r} must target a {suffix} file, not {path.name}"
            )
        if self.write_roots and not any(
            path.is_relative_to(root) for root in self.write_roots
        ):
            raise ValidationError(
                f"write {key!r} escapes the permitted write roots: {path}"
            )
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self._write_target(name, suffix=".json")
        return write_json(path, payload, indent=2, sort_keys=False)

    def write_markdown(self, name: str, text: str) -> Path:
        path = self._write_target(name, suffix=".md")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"markdown output {name!r} must be a non-empty string")
        return write_text(path, text.rstrip() + "\n")


def _truncate_payload(name: str, text: str, limit: int) -> dict[str, Any]:
    truncated = len(text) > limit
    body = text[:limit]
    result: dict[str, Any] = {
        "name": name,
        "content": body,
        "chars": len(text),
        "truncated": truncated,
    }
    if truncated:
        result["note"] = (
            f"content truncated to {limit} characters; quote only the visible excerpt "
            "and say what is unknown instead of inventing numbers"
        )
    return result


def describe_tool_protocol() -> str:
    """Static description of tool directives, injected into the runtime system prompt."""

    return (
        "## Tool directives (JSON only)\n"
        "Every response must be exactly one JSON object. You may either call one tool or\n"
        "submit your final answer. Available tools:\n"
        '- {"tool": "list_artifacts"}: list readable resources and deterministic checks.\n'
        '- {"tool": "read_artifact", "name": "<resource>"}: read one bounded excerpt.\n'
        '- {"tool": "run_check", "name": "<check>", "args": {...}}: run a whitelisted\n'
        "  deterministic check (catalog lookups, weight normalisation, metric lookups).\n"
        '- {"tool": "submit_json", "content": {...}}: submit the final JSON payload.\n'
        '- {"tool": "submit_markdown", "content": "..."}: submit the final Markdown text.\n'
        "Only these five tools exist. The harness refuses anything else (for example\n"
        "write_file, http_get, python) and tells you why; you cannot write files, browse\n"
        "the web or run arbitrary code. The number of read-only tool calls is bounded, so\n"
        "read only what you need. A bare JSON object without a tool wrapper is treated as\n"
        "a submit_json payload when a JSON schema is required.\n"
    )


def describe_json_schema_contract(schema: Mapping[str, Any] | None) -> str:
    """Compact JSON-schema description appended to the runtime system prompt."""

    if schema is None:
        return ""
    try:
        encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - schemas are JSON already
        encoded = "{}"
    return (
        "## Output schema (authoritative)\n"
        "Your final submit_json content is validated against this JSON Schema. Use only\n"
        "the declared properties; additional properties are rejected.\n"
        "```json\n"
        f"{encoded}\n"
        "```\n"
    )
