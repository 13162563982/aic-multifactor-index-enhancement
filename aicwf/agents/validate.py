"""Schema loading, domain validation and citation-integrity helpers for agents.

Three layers of checks live here:

1. **JSON Schema** (``validate_payload``) wraps ``jsonschema`` so every agent payload
   and machine artifact is validated against the same schema files that the controller
   and verification stage use.
2. **Domain rules** (``validate_hypothesis``, ``validate_factor_flags``,
   ``validate_verification_dict``) encode the finance-specific invariants that JSON
   Schema cannot express: known factor ids only, weights normalised to 1.0, the
   information cutoff equal to the run's ``as_of``, hypothesis constraints inside the
   configured strategy envelope, no contradictory verification verdict, and so on.
3. **Citation integrity** (``check_markdown_contract``) is the anti-hallucination guard
   for markdown outputs: every quoted number must be a rounding of a number present in
   the supplied verified artifacts, a minimum number of citations is required, and the
   information cutoff must be stated. Years and small integer counts are exempt.

The module has no knowledge of LLM backends, so it is safe to import anywhere.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jsonschema

from aicwf.contracts import PROJECT_ROOT, ConfigError, date_str

SCHEMA_DIR = PROJECT_ROOT / "aicwf" / "schemas"

_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# "CSI 300" is the benchmark's name, not a measurement; do not treat 300 as a citation.
_BENCHMARK_NAME_RE = re.compile(r"\bCSI[\s-]?300\b", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_STRING_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def normalise_numeric_literals(text: str) -> str:
    """Remove thousands separators (28,000 -> 28000) before numeric tokenisation."""

    return _THOUSANDS_RE.sub("", str(text))


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_schema_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot load schema {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise ConfigError(f"schema {path} must be a JSON object")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:  # pragma: no cover - developer error
        raise ConfigError(f"invalid JSON schema {path}: {exc.message}") from exc
    return schema


def schema_path(name_or_path: str | Path) -> Path:
    """Resolve ``"factor_flags"``, ``"factor_flags.schema.json"`` or a project path."""

    raw = str(name_or_path)
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    candidates: list[Path] = []
    if path.suffix == ".json":
        candidates.append(PROJECT_ROOT / path)
        candidates.append(SCHEMA_DIR / path.name)
    else:
        candidates.append(SCHEMA_DIR / f"{raw}.schema.json")
        candidates.append(SCHEMA_DIR / raw)
        candidates.append(PROJECT_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ConfigError(
        f"schema {name_or_path!r} not found; looked in {[str(p) for p in candidates]}"
    )


def load_schema(name_or_path: str | Path) -> dict[str, Any]:
    """Load and validate a schema; returns a deep copy callers may keep."""

    return copy.deepcopy(_load_schema_cached(str(schema_path(name_or_path))))


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------


def validation_errors(
    payload: Any,
    schema: Mapping[str, Any] | str | Path,
    *,
    label: str = "payload",
    limit: int = 20,
) -> list[str]:
    """Return human-readable schema errors (empty list means valid)."""

    resolved = load_schema(schema) if not isinstance(schema, Mapping) else schema
    validator = jsonschema.Draft202012Validator(resolved)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: [str(part) for part in err.absolute_path],
    )
    messages: list[str] = []
    for err in errors[:limit]:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in err.absolute_path
        )
        messages.append(f"{label}{path}: {err.message}")
    if len(errors) > limit:
        messages.append(f"{label}: {len(errors) - limit} more schema errors")
    return messages


def validate_payload(
    payload: Any,
    schema: Mapping[str, Any] | str | Path | None,
    *,
    label: str = "payload",
) -> list[str]:
    """Compatibility wrapper: ``None`` schema means nothing to validate."""

    if schema is None:
        return []
    return validation_errors(payload, schema, label=label)


# ---------------------------------------------------------------------------
# Minimal schema synthesis (NullLLM and tooling)
# ---------------------------------------------------------------------------


def synthesize_json(schema: Mapping[str, Any], path: str = "$") -> Any:
    """Build one small valid instance for a schema.

    Only the constructs used by this repository's schemas are interpreted; anything
    unknown falls back to a permissive value. This is intentionally *not* a general
    schema faker: it exists so the offline backend can produce a valid placeholder.
    """

    schema = dict(schema or {})
    for combinator in ("oneOf", "anyOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options:
            return synthesize_json(options[0], path)
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]

    declared = schema.get("type")
    if isinstance(declared, list):
        declared = next((t for t in declared if t != "null"), declared[0] if declared else "string")
    if declared is None:
        if "properties" in schema or isinstance(schema.get("required"), list):
            declared = "object"
        elif "items" in schema:
            declared = "array"
        elif any(key in schema for key in ("pattern", "minLength", "format", "maxLength")):
            declared = "string"
        else:
            declared = "object"

    if declared == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        keys = list(required) or list(properties)[: max(1, int(schema.get("minProperties", 1) or 1))]
        result: dict[str, Any] = {}
        for key in keys:
            result[key] = synthesize_json(properties.get(key, {}), f"{path}.{key}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping) and not result:
            result["placeholder"] = synthesize_json(additional, f"{path}.placeholder")
        return result
    if declared == "array":
        items = schema.get("items")
        count = int(schema.get("minItems", 1) or 1)
        if items is None:
            return []
        return [synthesize_json(items, f"{path}[{idx}]") for idx in range(max(count, 1))]
    if declared == "integer":
        return _synthesize_number(schema, path, integer=True)
    if declared == "number":
        return _synthesize_number(schema, path, integer=False)
    if declared == "boolean":
        return False
    if declared == "null":
        return None
    # String
    pattern = str(schema.get("pattern") or "")
    if schema.get("format") == "date-time" or (
        "T\\d{2}" in pattern and "\\d{4}-\\d{2}-\\d{2}" in pattern
    ):
        return "2024-12-31T00:00:00+00:00"
    if schema.get("format") == "date" or "\\d{4}-\\d{2}-\\d{2}" in pattern:
        return "2024-12-31"
    if "sha256" in str(schema.get("description", "")).lower() or re.search(
        r"0-9a-f\]\{64\}", pattern
    ):
        return "0" * 64
    if pattern.startswith("^v") and "\\d" in pattern:
        return "v1.0.0"
    if pattern.startswith("^[a-z]"):
        return "placeholder"
    min_length = int(schema.get("minLength", 1) or 1)
    base = "placeholder"
    if len(base) < min_length:
        base = base + "x" * (min_length - len(base))
    return base


def _synthesize_number(schema: Mapping[str, Any], path: str, *, integer: bool) -> Any:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and maximum is not None:
        value = (float(minimum) + float(maximum)) / 2.0
    elif minimum is not None:
        value = float(minimum)
    elif maximum is not None:
        value = float(maximum)
    else:
        value = 0.0
    return int(round(value)) if integer else float(value)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def config_section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    """Return ``config[name]`` when ``config`` is a bundle, else ``config`` itself."""

    cfg = dict(config or {})
    nested = cfg.get(name)
    if isinstance(nested, Mapping):
        return dict(nested)
    return cfg


def factor_catalog(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    strategy = config_section(config, "strategy")
    catalog = ((strategy.get("factors") or {}).get("catalog")) or {}
    return {str(key): dict(value or {}) for key, value in catalog.items()}


def default_factor_weights(config: Mapping[str, Any] | None) -> dict[str, float]:
    strategy = config_section(config, "strategy")
    weights = ((strategy.get("factors") or {}).get("composite") or {}).get(
        "default_weights"
    ) or {}
    return {str(key): float(value) for key, value in weights.items()}


def strategy_constraints(config: Mapping[str, Any] | None) -> dict[str, float]:
    strategy = config_section(config, "strategy")
    raw = (strategy.get("backtest") or {}).get("constraints") or {}
    defaults = {
        "max_weight": 0.03,
        "max_active_weight": 0.015,
        "te_target": 0.05,
        "turnover_limit": 0.30,
    }
    for key, fallback in defaults.items():
        value = raw.get(key, fallback)
        try:
            defaults[key] = float(value)
        except (TypeError, ValueError):
            pass
    return defaults


# ---------------------------------------------------------------------------
# Hypothesis domain validation
# ---------------------------------------------------------------------------


def validate_hypothesis(
    payload: Any,
    *,
    config: Mapping[str, Any] | None,
    as_of: str | dt.date,
    schema: Mapping[str, Any] | str | Path | None = None,
    min_factors: int = 3,
    max_factors: int = 8,
    normalization_tolerance: float = 1e-3,
) -> list[str]:
    """Schema + domain validation for ``hypothesis.json``.

    Domain rules beyond the schema:

    * every ``factor_weights`` key is a factor id in ``config/strategy.yaml``;
    * weights are non-negative (direction is handled by the catalog) and sum to 1.0;
    * 3-8 factors are required for a diversified multi-factor hypothesis;
    * ``information_cutoff`` equals the run's ``as_of`` (leakage guard);
    * risk constraints stay inside the configured strategy envelope;
    * at least two concrete falsification conditions are given.

    ``normalization_tolerance`` accepts presentation rounding (a model that writes
    six weights of 0.227273 sums to 1.000002); the hypothesis stage then renormalises
    deterministically before writing the artifact, so the stored weights sum to 1.0.
    """

    schema_errors = validate_payload(
        payload,
        load_schema("hypothesis") if schema is None else schema,
        label="hypothesis",
    )
    if not isinstance(payload, Mapping):
        return schema_errors or ["hypothesis: payload must be a JSON object"]
    errors = list(schema_errors)

    catalog = factor_catalog(config)
    known = set(catalog)
    weights = payload.get("factor_weights")
    numeric_weights: dict[str, float] = {}
    if isinstance(weights, Mapping):
        unknown = sorted(str(key) for key in weights if str(key) not in known)
        if unknown:
            errors.append(
                "hypothesis.factor_weights: unknown factor ids "
                f"{unknown}; allowed ids are {sorted(known)}"
            )
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value < 0:
                errors.append(
                    f"hypothesis.factor_weights[{key!r}]: weight must be >= 0; the sign "
                    "belongs to the factor catalog direction"
                )
            numeric_weights[str(key)] = float(value)
        if numeric_weights:
            count = len(numeric_weights)
            if not (min_factors <= count <= max_factors):
                errors.append(
                    f"hypothesis.factor_weights: expected between {min_factors} and "
                    f"{max_factors} factor weights, got {count}"
                )
            total = sum(numeric_weights.values())
            if abs(total - 1.0) > normalization_tolerance:
                normalized = {
                    key: round(value / total, 6) for key, value in numeric_weights.items()
                } if total else {}
                errors.append(
                    "hypothesis.factor_weights: weights must be normalised to sum to "
                    f"1.0 (within {normalization_tolerance:g}), got {total:.12g}; a "
                    f"normalised version would be {normalized}"
                )

    cutoff = payload.get("information_cutoff")
    expected_cutoff = date_str(as_of) if not isinstance(as_of, str) else as_of
    if cutoff != expected_cutoff:
        errors.append(
            "hypothesis.information_cutoff: must equal the run as_of "
            f"{expected_cutoff!r}, got {cutoff!r}"
        )

    constraints = payload.get("constraints")
    envelope = strategy_constraints(config)
    if isinstance(constraints, Mapping):
        for key, limit in envelope.items():
            value = constraints.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if float(value) > limit + 1e-12:
                errors.append(
                    f"hypothesis.constraints.{key}: {value} exceeds the configured "
                    f"strategy envelope {limit}"
                )

    falsification = payload.get("falsification")
    if isinstance(falsification, list) and len(falsification) < 2:
        errors.append(
            "hypothesis.falsification: provide at least 2 concrete, testable "
            "falsification conditions"
        )

    return errors


# ---------------------------------------------------------------------------
# Factor review validation
# ---------------------------------------------------------------------------


def validate_factor_flags(
    payload: Any,
    *,
    schema: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    """Schema + domain validation for ``flags.json``.

    ``warn`` and ``block`` flags must quote a diagnostic number so that a reviewer can
    trace the flag back to the factor report instead of accepting prose.
    """

    errors = validate_payload(
        payload,
        load_schema("factor_flags") if schema is None else schema,
        label="flags",
    )
    if not isinstance(payload, Mapping):
        return errors or ["flags: payload must be a JSON object"]
    flags = payload.get("flags")
    if not isinstance(flags, list):
        return errors
    seen: set[str] = set()
    for index, flag in enumerate(flags):
        if not isinstance(flag, Mapping):
            continue
        flag_id = str(flag.get("id", ""))
        if flag_id in seen:
            errors.append(f"flags.flags[{index}].id: duplicate flag id {flag_id!r}")
        seen.add(flag_id)
        severity = flag.get("severity")
        message = str(flag.get("message", ""))
        if severity in {"warn", "block"} and not re.search(r"\d", message):
            errors.append(
                f"flags.flags[{index}].message: a {severity} flag must cite the "
                "diagnostic number it is based on"
            )
    assessment = payload.get("assessment_markdown")
    if isinstance(assessment, str) and len(assessment.strip()) < 100:
        errors.append(
            "flags.assessment_markdown: narrative assessment must be at least 100 "
            "characters so the review note is reviewable"
        )
    return errors


# ---------------------------------------------------------------------------
# Verification / approval / version validation
# ---------------------------------------------------------------------------


def validate_verification_dict(
    payload: Any,
    *,
    schema: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    errors = validate_payload(
        payload, load_schema("verification") if schema is None else schema, label="verification"
    )
    if not isinstance(payload, Mapping):
        return errors or ["verification: payload must be a JSON object"]
    verdict = payload.get("verdict")
    checks = payload.get("checks")
    if verdict == "PASS" and isinstance(checks, list):
        failed = [
            str(check.get("id"))
            for check in checks
            if isinstance(check, Mapping) and str(check.get("status", "")).lower() == "fail"
        ]
        if failed:
            errors.append(
                "verification.verdict: verdict PASS contradicts failed checks "
                f"{failed}; the verdict must be FAIL"
            )
    return errors


def validate_approval_payload(
    payload: Any,
    *,
    schema: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    return validate_payload(
        payload, load_schema("approval") if schema is None else schema, label="approval"
    )


def validate_version_payload(
    payload: Any,
    *,
    schema: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    return validate_payload(
        payload, load_schema("version") if schema is None else schema, label="version_entry"
    )


# ---------------------------------------------------------------------------
# Citation integrity / number guards
# ---------------------------------------------------------------------------


def _to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if isinstance(value, int):
            return Decimal(value)
        return Decimal(str(float(value)))
    except (ArithmeticError, ValueError):
        return None


def numbers_from(obj: Any) -> set[Decimal]:
    """Collect every number in a JSON-like object (booleans excluded).

    Numbers embedded in prose strings are included too: artifact fields such as
    ``warnings: ["neutralised z coverage 58% ..."]`` carry measurements that agents
    legitimately quote, so they must be citable (both ``58`` and ``0.58`` for "58%").
    """

    found: set[Decimal] = set()

    def add_decimal(value: Any) -> None:
        dec = _to_decimal(value)
        if dec is not None and abs(dec) < Decimal("1e15"):
            found.add(dec)

    def add_text(text: str) -> None:
        cleaned = normalise_numeric_literals(text)
        cleaned = _DATE_RE.sub(" ", cleaned)
        cleaned = _BENCHMARK_NAME_RE.sub(" ", cleaned)
        for match in _STRING_NUMBER_RE.finditer(cleaned):
            token = match.group(0)
            try:
                if token.endswith("%"):
                    number = Decimal(token[:-1])
                    if abs(number) < Decimal("1e15"):
                        found.add(number / Decimal(100))
                        found.add(number)
                else:
                    number = Decimal(token)
                    if abs(number) < Decimal("1e15"):
                        found.add(number)
            except ArithmeticError:  # pragma: no cover - regex guarantees a number
                continue

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            add_text(node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            add_decimal(node)

    walk(obj)
    return found


def build_allowed_numbers(*objects: Any) -> set[Decimal]:
    allowed: set[Decimal] = set()
    for obj in objects:
        if obj is None:
            continue
        allowed |= numbers_from(obj)
    return allowed


def _token_decimal(token: str) -> tuple[Decimal, int] | None:
    """Parse a numeric token, returning ``(value, decimal_places_of_value)``."""

    try:
        if token.endswith("%"):
            number = Decimal(token[:-1])
            value = number / Decimal(100)
            decimals = _decimal_places(token[:-1]) + 2
            return value, decimals
        return Decimal(token), _decimal_places(token)
    except ArithmeticError:
        return None


def _decimal_places(text: str) -> int:
    if "." in text:
        return len(text.split(".", 1)[1])
    return 0


def _is_exempt_token(token: str) -> bool:
    """Years and small integer counts are prose, not invented measurements."""

    if "." in token or token.endswith("%"):
        return False
    try:
        value = int(token)
    except ValueError:
        return False
    if 1900 <= value <= 2100:
        return True
    return abs(value) <= 10


def _matches_allowed(value: Decimal, decimals: int, allowed: Iterable[Decimal]) -> bool:
    """True when the quoted value is an exact or presentation-rounded artifact number."""

    if decimals > 6:
        decimals = 6
    quantum = Decimal(1).scaleb(-decimals) if decimals > 0 else Decimal(1)
    for candidate in allowed:
        if value == candidate:
            return True
        try:
            if candidate.quantize(quantum) == value:
                return True
        except ArithmeticError:  # pragma: no cover - extreme exponents
            continue
    return False


def check_cited_numbers(
    text: str,
    allowed: Iterable[Decimal] | Mapping[str, Any] | Sequence[Any],
    *,
    max_unverified: int = 10,
) -> tuple[list[str], list[str]]:
    """Return ``(unverified_tokens, verified_tokens)`` found in ``text``.

    ``allowed`` may be a set of ``Decimal`` (from :func:`build_allowed_numbers`) or a
    JSON-like object to extract numbers from.
    """

    if isinstance(allowed, Mapping) or isinstance(allowed, (list, tuple)):
        allowed_set: set[Decimal] = numbers_from(allowed)
    else:
        allowed_set = set(allowed or [])
    unverified: list[str] = []
    verified: list[str] = []
    # ISO dates (2024-12-31) and benchmark names (CSI 300) are prose, not measurements:
    # strip them before tokenising so their digits are never treated as invented numbers.
    scan_text = normalise_numeric_literals(text)
    scan_text = _DATE_RE.sub(" ", scan_text)
    scan_text = _BENCHMARK_NAME_RE.sub(" ", scan_text)
    for match in _NUMBER_TOKEN_RE.finditer(scan_text):
        token = match.group(0)
        if _is_exempt_token(token):
            continue
        parsed = _token_decimal(token)
        if parsed is None:  # pragma: no cover - regex guarantees a number
            continue
        value, decimals = parsed
        if _matches_allowed(value, decimals, allowed_set):
            verified.append(token)
        elif len(unverified) < max_unverified:
            unverified.append(token)
    return unverified, verified


def check_markdown_contract(
    text: str,
    *,
    allowed_numbers: Iterable[Decimal] | Mapping[str, Any] | Sequence[Any],
    min_cited: int,
    as_of: str | dt.date | None = None,
    min_chars: int = 0,
    required_phrases: Sequence[str] = (),
    label: str = "markdown",
) -> list[str]:
    """Validate an agent markdown output for citation integrity.

    * quotes only numbers that appear in the supplied verified artifacts (rounding to
      the quoted precision is allowed; inventions such as an unbacked 12.3% are not);
    * cites at least ``min_cited`` distinct verified numbers;
    * states the information cutoff date when ``as_of`` is given;
    * meets the minimum length and mentions the required section phrases.
    """

    errors: list[str] = []
    body = text.strip()
    if len(body) < min_chars:
        errors.append(
            f"{label}: output is too short ({len(body)} chars); at least {min_chars} "
            "characters of substantive narrative are required"
        )
    if as_of is not None:
        expected = date_str(as_of) if not isinstance(as_of, str) else as_of
        if expected not in body:
            errors.append(
                f"{label}: the information cutoff {expected} must be stated explicitly "
                "in the output"
            )
    unverified, verified = check_cited_numbers(body, allowed_numbers)
    if unverified:
        errors.append(
            f"{label}: cites numbers that are not in the supplied verified artifacts: "
            f"{unverified}; quote only artifact values"
        )
    if len(verified) < min_cited:
        errors.append(
            f"{label}: must cite at least {min_cited} verified artifact numbers, found "
            f"{len(verified)}"
        )
    lowered = body.lower()
    for phrase in required_phrases:
        if phrase.lower() not in lowered:
            errors.append(f"{label}: required section or phrase missing: {phrase!r}")
    return errors


def find_cutoff(text: str) -> str | None:
    match = _DATE_RE.search(text)
    return match.group(1) if match else None
