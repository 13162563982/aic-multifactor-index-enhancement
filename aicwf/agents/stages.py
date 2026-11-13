"""Agent-driven stage entrypoints for the AICWF workflow.

The controller calls the functions in this module from the stage registry:

* :func:`run_hypothesis`        -> stage ``hypothesis`` (01_hypothesis)
* :func:`run_factor_review`     -> stage ``factor_review`` (04_factor_review)
* :func:`run_risk_review`       -> stage ``robustness_review`` (07_robustness_review)
* :func:`run_report`            -> stage ``report`` (10_report)
* :func:`generate_verifier_note` -> helper used by the verification teammate

Each stage builds its own bounded context from upstream artifacts (never a full parquet
dump), runs one :class:`aicwf.agents.runtime.AgentRuntime`, validates the output with the
common validators and writes only the artifacts declared in ``docs/artifact_contract.md``.
All writes go through the runtime's pre-registered write allowlist.

The stages are deterministic given (snapshot, config, prompts, LLM responses). They never
write wall-clock timestamps into artifacts, never call the network and never let the model
choose a path.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicwf.agents.llm import ReplayCacheMiss
from aicwf.agents.roles import (
    agent_loop_limits,
    load_role_assets,
    resolve_llm,
)
from aicwf.agents.runtime import AgentRuntime
from aicwf.agents.tools import Toolbox
from aicwf.agents.validate import (
    build_allowed_numbers,
    check_cited_numbers,
    check_markdown_contract,
    config_section,
    default_factor_weights,
    factor_catalog,
    load_schema,
    numbers_from,
    strategy_constraints,
    validate_factor_flags,
    validate_hypothesis,
    validate_verification_dict,
)
from aicwf.contracts import (
    ConfigError,
    NonRetryableError,
    PROJECT_ROOT,
    RunContext,
    StageOutput,
    ValidationError,
    WorkflowError,
    date_str,
    parse_date,
    relpath,
)
from aicwf.util.jsonio import read_json, read_text
from aicwf.util.hashing import canonical_json

MAX_PROMPT_JSON_CHARS = 28_000
# Total JSON context budget per user prompt. The local model runs with a 16k context,
# so ~18k characters of evidence plus the system prompt leaves room for the completion.
MAX_PROMPT_CONTEXT_CHARS = 18_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _section(ctx: RunContext, name: str) -> dict[str, Any]:
    return config_section(ctx.config, name)


def _as_of(ctx: RunContext) -> dt.date:
    value = ctx.as_of
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return parse_date(str(value))


def _rel(path: Path | str) -> str:
    return relpath(path)


def _event(ctx: RunContext, kind: str, level: str = "info", **payload: Any) -> None:
    try:
        ctx.event(kind, level=level, **payload)
    except Exception:  # noqa: BLE001 - telemetry must never break a stage
        return


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _excerpt(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 15,
    max_str: int = 1200,
) -> Any:
    """Bounded, JSON-serialisable excerpt of an upstream artifact."""

    if depth > max_depth:
        return "<<nested content omitted>>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            result[str(key)] = _excerpt(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_str=max_str,
            )
        if len(items) > max_items:
            result["_omitted_keys"] = len(items) - max_items
        return result
    if isinstance(value, (list, tuple)):
        items = list(value)
        result = [
            _excerpt(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_str=max_str,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            result.append(f"... {len(items) - max_items} more items omitted")
        return result
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + "... [truncated]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _upstream_summary(ctx: RunContext, stage: str) -> dict[str, Any]:
    try:
        return dict(ctx.upstream_summary(stage))
    except WorkflowError:
        return {}


def _read_upstream_json(ctx: RunContext, stage: str, logical: str) -> Any | None:
    try:
        path = ctx.upstream_artifact(stage, logical)
    except WorkflowError:
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise WorkflowError(
            f"cannot parse upstream artifact {path} ({stage}/{logical}): {exc}"
        ) from exc


def _snapshot_manifest(ctx: RunContext) -> dict[str, Any]:
    candidates: list[Path] = []
    if ctx.snapshot_dir is not None:
        candidates.append(Path(ctx.snapshot_dir))
    strategy = _section(ctx, "strategy")
    snapshot_cfg = (strategy.get("data") or {}).get("snapshot")
    if snapshot_cfg:
        candidates.append(PROJECT_ROOT / str(snapshot_cfg))
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if manifest_path.exists():
            try:
                return _excerpt(
                    read_json(manifest_path), max_depth=3, max_items=10, max_str=400
                )
            except (OSError, ValueError) as exc:
                raise WorkflowError(f"cannot parse snapshot manifest {manifest_path}: {exc}") from exc
    return {}


def _intake_info(ctx: RunContext) -> dict[str, Any]:
    summary = _upstream_summary(ctx, "intake")
    if summary:
        return _excerpt(summary, max_depth=2, max_items=10)
    path = Path(ctx.run_dir) / "00_intake" / "run_request.json"
    if path.exists():
        try:
            return _excerpt(read_json(path), max_depth=2, max_items=10)
        except (OSError, ValueError):
            return {}
    return {}


def _toolbox(
    ctx: RunContext,
    *,
    blobs: Mapping[str, Any],
    checks: Mapping[str, Callable[[dict[str, Any]], Any]],
    write_paths: Mapping[str, Path] | None = None,
) -> Toolbox:
    limits = agent_loop_limits(ctx.config)
    read_roots: list[Path] = [Path(ctx.run_dir)]
    if ctx.snapshot_dir is not None:
        read_roots.append(Path(ctx.snapshot_dir))
    return Toolbox(
        blobs=dict(blobs),
        checks=dict(checks),
        read_roots=read_roots,
        write_paths=dict(write_paths or {}),
        write_roots=[Path(ctx.run_dir)],
        max_read_chars=int(limits["max_read_chars"]),
    )


def _make_runtime(
    ctx: RunContext,
    assets: Any,
    toolbox: Toolbox,
    *,
    as_of: dt.date,
    output_kind: str,
    validators: Sequence[Callable[[Any], list[str]]] = (),
    text_validators: Sequence[Callable[[str], list[str]]] = (),
    allowed_numbers: Any = None,
    min_cited: int = 0,
    min_chars: int = 200,
    required_phrases: Sequence[str] = (),
    max_tokens: int | None = None,
    extra_rules: str = "",
) -> AgentRuntime:
    limits = agent_loop_limits(ctx.config)
    allowed_paths: list[Path] = [Path(ctx.run_dir)]
    if ctx.snapshot_dir is not None:
        allowed_paths.append(Path(ctx.snapshot_dir))
    return AgentRuntime(
        role=assets.role,
        system_prompt=assets.prompt_text,
        skill_text=assets.skill_text,
        schema=assets.schema,
        llm=resolve_llm(ctx),
        allowed_paths=allowed_paths,
        max_tool_rounds=int(limits["max_tool_rounds"]),
        toolbox=toolbox,
        output_kind=output_kind,
        as_of=as_of,
        max_repairs=int(limits["max_retries_invalid_json"]),
        repair_temperature=float(limits["repair_temperature"]),
        max_tokens=int(max_tokens or assets.spec.max_tokens),
        validators=validators,
        text_validators=text_validators,
        allowed_numbers=allowed_numbers,
        min_cited_numbers=min_cited,
        min_chars=min_chars,
        required_phrases=required_phrases,
        extra_rules=extra_rules,
    )


def _build_user_prompt(
    *,
    title: str,
    intro: str,
    as_of: dt.date,
    sections: Sequence[tuple[str, Any]],
    instructions: str,
) -> str:
    parts: list[str] = [
        f"# {title}",
        intro.strip(),
        "",
        f"Information cutoff (as_of): {date_str(as_of)}",
        "All numbers below come from the frozen snapshot; do not use or invent anything else.",
        "",
        "# Provided context (read-only evidence)",
    ]
    used_chars = 0
    for heading, value in sections:
        encoded = _json_text(value)
        remaining = MAX_PROMPT_CONTEXT_CHARS - used_chars
        if remaining <= 200:
            parts.extend(
                [
                    f"## {heading}",
                    "... [omitted: the bounded prompt context budget is exhausted; use the "
                    "read_artifact/run_check tools if this evidence is required]",
                ]
            )
            continue
        if len(encoded) > remaining:
            encoded = encoded[:remaining] + "\n... [context truncated by harness]"
        if len(encoded) > MAX_PROMPT_JSON_CHARS:
            encoded = encoded[:MAX_PROMPT_JSON_CHARS] + "\n... [context truncated by harness]"
        used_chars += len(encoded)
        parts.extend([f"## {heading}", "```json", encoded, "```"])
    parts.extend(["", "# Instructions", instructions.strip(), ""])
    return "\n".join(parts)


def _agent_summary(runtime: AgentRuntime) -> dict[str, Any]:
    return runtime.stats_summary()


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _dig(obj: Any, path: str) -> Any:
    node = obj
    for part in str(path).split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise ValueError(f"cannot resolve path {path!r}") from None
        else:
            raise ValueError(f"cannot resolve path {path!r}")
    return node


_FACTOR_REPORT_KEYS = (
    "coverage",
    "rank_ic",
    "ic_t_stat",
    "autocorrelation",
    "correlation_matrix",
    "composite_weights",
    "composite_ic",
    "factor_ids",
    "information_cutoff",
    "formation_dates",
    "warnings",
)

# Run-scoped metadata that must never enter an LLM prompt: prompts are functions of the
# evidence, not of which run id or backend mode produced it. Without this scrub, replaying
# a recorded cache under a different run id would miss (the verification artifact embeds
# the run id and llm_mode).
_PROMPT_VOLATILE_KEYS = frozenset(
    {
        "run_id",
        "resolved_run_id",
        "llm_mode",
        "created_at_utc",
        "decided_at_utc",
        "requested_at_utc",
        "recorded_at_utc",
        "ts_utc",
        "run_dir",
        "run_path",
    }
)


def _scrub_prompt_metadata(value: Any, *, depth: int = 0) -> Any:
    """Recursively drop run-scoped metadata keys from an artifact used in a prompt."""

    if depth > 8:
        return "<<nested content omitted>>"
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_prompt_metadata(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in _PROMPT_VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_prompt_metadata(item, depth=depth + 1) for item in value]
    return value


def _factor_report_excerpt(report: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded excerpt of ``factor_report.json`` (drops the long per-date series)."""

    selected = {key: report.get(key) for key in _FACTOR_REPORT_KEYS if key in report}
    return _excerpt(selected, max_depth=3, max_items=20, max_str=600)


def _canonicalise_hypothesis_weights(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Renormalise factor weights to sum to exactly 1.0 after schema/domain validation.

    The model is allowed to write weights with normal rounding (six weights of
    0.227273 sum to 1.000002). The validator accepts that presentation rounding; this
    deterministic step then produces the canonical artifact, so the stored hypothesis
    always sums to one and downstream stages never inherit float drift. It never
    changes the *relative* tilt, only the common scale.
    """

    weights = payload.get("factor_weights")
    if not isinstance(weights, Mapping) or not weights:
        return dict(payload)
    numeric = {
        str(key): float(value)
        for key, value in weights.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if len(numeric) != len(weights) or len(numeric) == 0:
        return dict(payload)
    total = sum(numeric.values())
    if total <= 0:
        return dict(payload)
    normalized = {key: value / total for key, value in numeric.items()}
    residual = 1.0 - sum(normalized.values())
    if residual:
        key = max(normalized, key=lambda item: normalized[item])
        normalized[key] += residual
    rounded = {key: round(value, 12) for key, value in normalized.items()}
    drift = 1.0 - sum(rounded.values())
    if drift:
        key = max(rounded, key=lambda item: rounded[item])
        rounded[key] = round(rounded[key] + drift, 12)
    result = dict(payload)
    result["factor_weights"] = rounded
    return result


# ---------------------------------------------------------------------------
# Stage: hypothesis (01_hypothesis)
# ---------------------------------------------------------------------------


def run_hypothesis(ctx: RunContext) -> StageOutput:
    """Researcher agent -> validated ``hypothesis.json`` + deterministic ``rationale.md``."""

    stage = "hypothesis"
    as_of = _as_of(ctx)
    assets = load_role_assets("researcher", ctx.config)
    catalog = factor_catalog(ctx.config)
    if not catalog:
        raise ConfigError(
            "config/strategy.yaml factor catalog is empty; the researcher cannot choose "
            "factors from an empty catalog"
        )
    envelope = strategy_constraints(ctx.config)
    defaults = default_factor_weights(ctx.config)
    strategy = _section(ctx, "strategy")
    configured_rebalance = str((strategy.get("backtest") or {}).get("rebalance") or "monthly")
    manifest = _snapshot_manifest(ctx)
    intake = _intake_info(ctx)

    factor_list = [
        {
            "id": factor_id,
            "category": str(meta.get("category", "")),
            "direction": float(meta.get("direction", 1.0)),
            "description": str(meta.get("description", "")),
        }
        for factor_id, meta in sorted(catalog.items())
    ]

    def check_factor_catalog(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "factors": factor_list,
            "categories": sorted({item["category"] for item in factor_list}),
        }

    def check_constraint_envelope(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "limits": envelope,
            "rebalance_options": ["monthly", "quarterly"],
            "configured_rebalance": configured_rebalance,
            "factor_count_range": [3, 8],
        }

    def check_normalize_weights(args: dict[str, Any]) -> dict[str, Any]:
        raw = args.get("weights")
        if not isinstance(raw, Mapping):
            raise ValueError("pass weights as an object: {\"weights\": {<factor_id>: <number>}}")
        unknown = sorted(str(key) for key in raw if str(key) not in catalog)
        negative = sorted(
            str(key)
            for key, value in raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0
        )
        numeric = {
            str(key): float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        total = sum(numeric.values())
        normalized = (
            {key: round(value / total, 6) for key, value in numeric.items()} if total else {}
        )
        return {
            "sum": round(total, 12),
            "normalized": normalized,
            "count": len(numeric),
            "unknown_ids": unknown,
            "negative_ids": negative,
            "valid": not unknown and not negative and abs(total - 1.0) <= 1e-6 and 3 <= len(numeric) <= 8,
        }

    def check_cutoff_status(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "as_of": date_str(as_of),
            "snapshot_hash": manifest.get("snapshot_hash"),
            "snapshot_date_range": manifest.get("date_ranges"),
            "intake": intake,
        }

    toolbox = _toolbox(
        ctx,
        blobs={
            "factor_catalog": {"factors": factor_list},
            "strategy_constraints": {
                "envelope": envelope,
                "rebalance_options": ["monthly", "quarterly"],
                "configured_rebalance": configured_rebalance,
            },
            "default_composite_weights": defaults,
            "snapshot_manifest": manifest,
            "intake_summary": intake,
        },
        checks={
            "factor_catalog": check_factor_catalog,
            "constraint_envelope": check_constraint_envelope,
            "normalize_weights": check_normalize_weights,
            "cutoff_status": check_cutoff_status,
        },
        write_paths={
            "hypothesis": ctx.artifact_path(stage, "hypothesis.json"),
            "rationale": ctx.artifact_path(stage, "rationale.md"),
        },
    )

    user_prompt = _build_user_prompt(
        title="Stage 01_hypothesis — specify the multi-factor research hypothesis",
        intro=(
            "You are writing the research contract for this run. Use the frozen snapshot "
            "and the factor catalog below. The hypothesis must be falsifiable, "
            "implementable and consistent with the configured risk envelope."
        ),
        as_of=as_of,
        sections=[
            ("factor_catalog", {"factors": factor_list}),
            ("default_composite_weights", defaults),
            (
                "risk_constraint_envelope",
                {
                    "limits": envelope,
                    "configured_rebalance": configured_rebalance,
                    "rebalance_options": ["monthly", "quarterly"],
                },
            ),
            ("snapshot_manifest", manifest),
            ("run_intake", intake),
        ],
        instructions=(
            "Return one JSON object that validates against hypothesis.schema.json with "
            "exactly these fields: thesis, factor_weights, rebalance, constraints, "
            "falsification, information_cutoff, source_limits.\n"
            f"- factor_weights: 3-8 ids from the factor catalog, non-negative, summing to "
            "1.0 (use the normalize_weights check to verify).\n"
            "- constraints must be at or inside the supplied risk envelope.\n"
            "- falsification: 2-4 concrete testable conditions that later deterministic "
            "stages can check.\n"
            f"- information_cutoff must be exactly {date_str(as_of)!r}.\n"
            "- source_limits: state the frozen snapshot, no-network and point-in-time "
            "fundamental constraints.\n"
            "Do not claim empirical results in this stage; no backtest has been run yet."
        ),
    )

    def hypothesis_validator(payload: Any) -> list[str]:
        return validate_hypothesis(
            payload,
            config=ctx.config,
            as_of=as_of,
            schema=assets.schema,
        )

    runtime = _make_runtime(
        ctx,
        assets,
        toolbox,
        as_of=as_of,
        output_kind="json",
        validators=(hypothesis_validator,),
    )
    payload = runtime.run(user_prompt)
    raw_weights = dict(payload.get("factor_weights") or {})
    payload = _canonicalise_hypothesis_weights(payload)
    if payload["factor_weights"] != raw_weights:
        _event(
            ctx,
            "hypothesis_weights_normalised",
            stage=stage,
            raw_sum=round(sum(float(value) for value in raw_weights.values()), 12),
            information_cutoff=date_str(as_of),
        )

    rationale = _render_rationale(payload, catalog, as_of)
    hypothesis_path = runtime.write_json("hypothesis", payload)
    rationale_path = runtime.write_markdown("rationale", rationale)
    _event(
        ctx,
        "agent_stage_completed",
        stage=stage,
        role="researcher",
        factor_count=len(payload.get("factor_weights") or {}),
        information_cutoff=date_str(as_of),
        stats=_agent_summary(runtime),
    )
    return StageOutput(
        artifacts={
            "hypothesis": _rel(hypothesis_path),
            "rationale": _rel(rationale_path),
        },
        summary={
            "thesis": payload["thesis"],
            "factor_weights": payload["factor_weights"],
            "rebalance": payload["rebalance"],
            "constraints": payload["constraints"],
            "information_cutoff": payload["information_cutoff"],
            "agent": _agent_summary(runtime),
        },
    )


def _render_rationale(
    payload: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    as_of: dt.date,
) -> str:
    lines = [
        "# Hypothesis rationale",
        "",
        f"Information cutoff: {date_str(as_of)}. The hypothesis uses only the frozen CSI 300 "
        "snapshot available on or before this date; no live data or external sources are used.",
        "",
        "## Thesis",
        "",
        str(payload.get("thesis", "")).strip(),
        "",
        "## Factor tilts",
        "",
        "| factor | weight | category | direction | description |",
        "|---|---|---|---|---|",
    ]
    weights = payload.get("factor_weights") or {}
    for factor_id, weight in weights.items():
        meta = catalog.get(factor_id, {})
        lines.append(
            f"| {factor_id} | {_fmt_number(weight)} | {meta.get('category', '')} | "
            f"{_fmt_number(float(meta.get('direction', 1.0)))} | "
            f"{str(meta.get('description', '')).replace('|', '/')} |"
        )
    constraints = payload.get("constraints") or {}
    lines.extend(
        [
            "",
            "## Rebalance and risk constraints",
            "",
            f"- Rebalance: {payload.get('rebalance')}",
            f"- Max single weight: {_fmt_number(constraints.get('max_weight'))}",
            f"- Max active weight: {_fmt_number(constraints.get('max_active_weight'))}",
            f"- Tracking-error target: {_fmt_number(constraints.get('te_target'))}",
            f"- Turnover limit: {_fmt_number(constraints.get('turnover_limit'))}",
            "",
            "## Falsification criteria",
            "",
        ]
    )
    for index, item in enumerate(payload.get("falsification") or [], start=1):
        lines.append(f"{index}. {str(item).strip()}")
    lines.extend(["", "## Information and source limits", ""])
    for item in payload.get("source_limits") or []:
        lines.append(f"- {str(item).strip()}")
    lines.extend(
        [
            "",
            "Deterministic later stages (factor build, backtest, robustness, verification) "
            "test this contract; any claim of performance is intentionally absent here.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage: factor review (04_factor_review)
# ---------------------------------------------------------------------------


def run_factor_review(ctx: RunContext) -> StageOutput:
    """Factor analyst agent -> ``assessment.md`` + schema-validated ``flags.json``."""

    stage = "factor_review"
    as_of = _as_of(ctx)
    assets = load_role_assets("factor_analyst", ctx.config)
    hypothesis = _upstream_summary(ctx, "hypothesis")
    hypothesis_artifact = _read_upstream_json(ctx, "hypothesis", "hypothesis")
    if hypothesis_artifact is None and not hypothesis:
        raise WorkflowError("factor review requires the hypothesis stage output")
    factor_build = _upstream_summary(ctx, "factor_build")
    factor_report = _read_upstream_json(ctx, "factor_build", "factor_report")
    if not isinstance(factor_report, Mapping):
        raise WorkflowError(
            "factor review requires the factor_build stage artifact 'factor_report'"
        )
    report_excerpt = _factor_report_excerpt(factor_report)
    hypothesis_weights = (
        (hypothesis_artifact or {}).get("factor_weights")
        or (hypothesis or {}).get("factor_weights")
        or {}
    )
    build_summary = _excerpt(factor_build, max_depth=2, max_items=15)
    data_quality = _upstream_summary(ctx, "data_prep")
    data_quality_excerpt = _excerpt(data_quality, max_depth=2, max_items=12)
    strategy = _section(ctx, "strategy")
    thresholds = {
        "min_cross_section": int((strategy.get("factors") or {}).get("min_cross_section", 100)),
        "min_price_coverage": float((strategy.get("universe") or {}).get("min_price_coverage", 0.90)),
        "rank_ic_t_stat_reference": 2.0,
        "factor_correlation_crowding_reference": 0.6,
    }
    allowed_numbers = build_allowed_numbers(
        report_excerpt, hypothesis_weights, build_summary, data_quality_excerpt, thresholds
    )

    def _factor_section(section: str, factor_id: str) -> Any:
        payload = factor_report.get(section)
        if not isinstance(payload, Mapping):
            return None
        value = payload.get(factor_id)
        if isinstance(value, Mapping):
            return _excerpt(value, max_depth=2, max_items=8)
        return value

    def _factor_ids() -> list[str]:
        for section in ("coverage", "rank_ic", "ic_t_stat", "autocorrelation"):
            payload = factor_report.get(section)
            if isinstance(payload, Mapping) and payload:
                return sorted(str(key) for key in payload)
        return sorted(str(key) for key in (factor_report.get("composite_weights") or {}))

    def check_list_factors(_args: dict[str, Any]) -> dict[str, Any]:
        return {"factors": _factor_ids()}

    def check_factor_stats(args: dict[str, Any]) -> dict[str, Any]:
        factor_id = str(args.get("factor_id") or args.get("id") or "").strip()
        ids = _factor_ids()
        if not factor_id:
            raise ValueError(f"pass a factor_id; available factors: {ids}")
        if factor_id not in ids:
            raise ValueError(f"factor {factor_id!r} is not in the factor report; available: {ids}")
        return {
            "factor_id": factor_id,
            "coverage": _factor_section("coverage", factor_id),
            "rank_ic": _factor_section("rank_ic", factor_id),
            "ic_t_stat": _factor_section("ic_t_stat", factor_id),
            "autocorrelation": _factor_section("autocorrelation", factor_id),
        }

    def check_weight_lookup(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "hypothesis_weights": hypothesis_weights,
            "composite_weights": _excerpt(
                factor_report.get("composite_weights") or {}, max_depth=2
            ),
        }

    def check_flag_thresholds(_args: dict[str, Any]) -> dict[str, Any]:
        return {"thresholds": thresholds}

    toolbox = _toolbox(
        ctx,
        blobs={
            "factor_report": report_excerpt,
            "hypothesis_weights": hypothesis_weights,
            "factor_build_summary": build_summary,
            "data_quality_summary": data_quality_excerpt,
        },
        checks={
            "list_factors": check_list_factors,
            "factor_stats": check_factor_stats,
            "weight_lookup": check_weight_lookup,
            "flag_thresholds": check_flag_thresholds,
        },
        write_paths={
            "flags": ctx.artifact_path(stage, "flags.json"),
            "assessment": ctx.artifact_path(stage, "assessment.md"),
        },
    )

    user_prompt = _build_user_prompt(
        title="Stage 04_factor_review — diagnostic review of the factor build",
        intro=(
            "Review the deterministic factor diagnostics below and produce traceable "
            "advisory flags. Do not modify or re-analyse the factor panel."
        ),
        as_of=as_of,
        sections=[
            ("hypothesis_weights", hypothesis_weights),
            ("factor_report", report_excerpt),
            ("factor_build_summary", build_summary),
            ("data_quality_summary", data_quality_excerpt),
            ("diagnostic_reference_thresholds", thresholds),
        ],
        instructions=(
            "Return one JSON object that validates against factor_flags.schema.json: "
            "assessment_markdown (>=100 chars, narrative review for assessment.md), "
            "summary (one paragraph) and flags [{id, severity, message}] (max 20).\n"
            "- ids must be stable unique snake_case; severity in info/warn/block.\n"
            "- every warn/block message must contain the diagnostic number it is based on.\n"
            "- cite in assessment_markdown only numbers visible in this prompt, quoted as "
            "supplied; state the information cutoff.\n"
            "- flags are advisory: they never change the factor panel or the hypothesis."
        ),
    )

    def flags_validator(payload: Any) -> list[str]:
        errors = validate_factor_flags(payload, schema=assets.schema)
        assessment = payload.get("assessment_markdown") if isinstance(payload, Mapping) else None
        if isinstance(assessment, str) and assessment.strip():
            errors.extend(
                check_markdown_contract(
                    assessment,
                    allowed_numbers=allowed_numbers,
                    min_cited=0,
                    as_of=as_of,
                    min_chars=100,
                    label="factor_flags.assessment_markdown",
                )
            )
        return errors

    runtime = _make_runtime(
        ctx,
        assets,
        toolbox,
        as_of=as_of,
        output_kind="json",
        validators=(flags_validator,),
    )
    payload = runtime.run(user_prompt)

    flags = list(payload.get("flags") or [])
    assessment = _render_factor_assessment(payload, as_of)
    flags_path = runtime.write_json("flags", payload)
    assessment_path = runtime.write_markdown("assessment", assessment)
    _event(
        ctx,
        "agent_stage_completed",
        stage=stage,
        role="factor_analyst",
        n_flags=len(flags),
        information_cutoff=date_str(as_of),
        stats=_agent_summary(runtime),
    )
    return StageOutput(
        artifacts={"assessment": _rel(assessment_path), "flags": _rel(flags_path)},
        summary={
            "n_flags": len(flags),
            "n_block": sum(1 for flag in flags if flag.get("severity") == "block"),
            "n_warn": sum(1 for flag in flags if flag.get("severity") == "warn"),
            "n_info": sum(1 for flag in flags if flag.get("severity") == "info"),
            "factor_weights": hypothesis_weights,
            "information_cutoff": date_str(as_of),
            "agent": _agent_summary(runtime),
        },
    )


def _render_factor_assessment(payload: Mapping[str, Any], as_of: dt.date) -> str:
    lines = [
        "# Factor review",
        "",
        f"Information cutoff: {date_str(as_of)}. Advisory review of the deterministic factor "
        "build; no upstream artifact was modified.",
        "",
    ]
    narrative = str(payload.get("assessment_markdown") or "").strip()
    if narrative:
        lines.extend([narrative, ""])
    summary = str(payload.get("summary") or "").strip()
    if summary:
        lines.extend(["## Summary", "", summary, ""])
    lines.extend(["## Flags", ""])
    flags = list(payload.get("flags") or [])
    if not flags:
        lines.append("No flags were raised.")
    for flag in flags:
        lines.append(
            f"- **{flag.get('severity', 'info')}** `{flag.get('id', '')}`: "
            f"{str(flag.get('message', '')).strip()}"
        )
        if flag.get("recommendation"):
            lines.append(f"  - Recommendation: {str(flag['recommendation']).strip()}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage: risk review (07_robustness_review)
# ---------------------------------------------------------------------------


def run_risk_review(ctx: RunContext) -> StageOutput:
    """Risk critic agent -> adversarial ``critique.md`` grounded in robustness numbers."""

    stage = "robustness_review"
    as_of = _as_of(ctx)
    assets = load_role_assets("risk_critic", ctx.config)
    metrics = _read_upstream_json(ctx, "backtest", "metrics")
    if not isinstance(metrics, Mapping):
        raise WorkflowError("risk review requires the backtest stage artifact 'metrics'")
    robustness = _read_upstream_json(ctx, "robustness", "robustness")
    checks = _read_upstream_json(ctx, "robustness", "robustness_checks")
    if not isinstance(robustness, Mapping) and not isinstance(checks, list):
        raise WorkflowError(
            "risk review requires the robustness stage artifacts "
            "'robustness' and/or 'robustness_checks'"
        )
    risk_report: Any = None
    if "portfolio" in ctx.stage_outputs:
        risk_report = _read_upstream_json(ctx, "portfolio", "risk_report")
    backtest_summary = _excerpt(_upstream_summary(ctx, "backtest"), max_depth=2, max_items=12)
    robustness_summary = _excerpt(
        _upstream_summary(ctx, "robustness"), max_depth=2, max_items=12
    )
    metrics_excerpt = _excerpt(metrics, max_depth=3, max_items=20, max_str=400)
    robustness_excerpt = _excerpt(
        robustness if isinstance(robustness, Mapping) else {},
        max_depth=3,
        max_items=20,
        max_str=800,
    )
    checks_excerpt = _excerpt(checks if isinstance(checks, list) else [], max_depth=2, max_items=40, max_str=400)
    risk_excerpt = _excerpt(risk_report or {}, max_depth=2, max_items=15)
    allowed_numbers = build_allowed_numbers(
        metrics_excerpt,
        robustness_excerpt,
        checks_excerpt,
        risk_excerpt,
        backtest_summary,
        robustness_summary,
    )

    def check_metric_lookup(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or args.get("metric") or "").strip()
        if not name:
            raise ValueError("pass a metric name, e.g. {\"name\": \"metrics.annualized_excess_return\"}")
        return {"name": name, "value": _dig(metrics, name)}

    robustness_payload = robustness if isinstance(robustness, Mapping) else {}

    def check_list_checks(_args: dict[str, Any]) -> dict[str, Any]:
        if isinstance(checks, list):
            return {
                "checks": [
                    {
                        "name": item.get("name") or item.get("id"),
                        "status": item.get("status"),
                    }
                    for item in checks
                    if isinstance(item, Mapping)
                ]
            }
        payload = robustness_payload.get("checks")
        return {"checks": _excerpt(payload, max_depth=2, max_items=40) if payload else []}

    def check_failed_checks(_args: dict[str, Any]) -> dict[str, Any]:
        items = checks if isinstance(checks, list) else []
        failed = [
            _excerpt(item, max_depth=2)
            for item in items
            if isinstance(item, Mapping) and str(item.get("status", "")).upper() not in {"PASS", "OK"}
        ]
        summary = robustness_payload.get("summary") or {}
        return {"failed_or_warn": failed, "summary": _excerpt(summary, max_depth=2)}

    def check_robustness_lookup(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("pass a robustness field path, e.g. {\"name\": \"deflated_sharpe\"}")
        return {"name": name, "value": _dig(robustness_payload, name)}

    def check_risk_lookup(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("pass a risk_report field path, e.g. {\"name\": \"ex_ante_te\"}")
        return {"name": name, "value": _dig(risk_excerpt, name)}

    toolbox = _toolbox(
        ctx,
        blobs={
            "metrics": metrics_excerpt,
            "robustness": robustness_excerpt,
            "robustness_checks": checks_excerpt,
            "risk_report": risk_excerpt,
            "backtest_summary": backtest_summary,
            "robustness_summary": robustness_summary,
        },
        checks={
            "metric_lookup": check_metric_lookup,
            "list_checks": check_list_checks,
            "failed_checks": check_failed_checks,
            "robustness_lookup": check_robustness_lookup,
            "risk_lookup": check_risk_lookup,
        },
        write_paths={"critique": ctx.artifact_path(stage, "critique.md")},
    )

    user_prompt = _build_user_prompt(
        title="Stage 07_robustness_review — adversarial critique of the evidence",
        intro=(
            "Attack the backtest and robustness evidence. Start from failed checks, then "
            "subperiod consistency, cost sensitivity, factor dependence, parameter "
            "sensitivity, placebo/bias stress and construction risk. Do not summarise "
            "marketing metrics."
        ),
        as_of=as_of,
        sections=[
            ("backtest_metrics", metrics_excerpt),
            ("backtest_summary", backtest_summary),
            ("robustness_evidence", robustness_excerpt),
            ("robustness_checks", checks_excerpt),
            ("robustness_summary", robustness_summary),
            ("portfolio_risk_report", risk_excerpt),
        ],
        instructions=(
            "Return Markdown (submit_markdown) of at least ~500 characters with sections:\n"
            "## Verdict on robustness\n## Failed or weak checks\n## Threats to validity\n"
            "## What would change our mind\n"
            f"- state the information cutoff {date_str(as_of)} explicitly;\n"
            "- cite at least three concrete numbers from the supplied artifacts and name "
            "their source artifact;\n"
            "- label speculation as a hypothesis, never as a finding;\n"
            "- no forecasts, price targets or trading recommendations."
        ),
    )

    def critique_validator(text: str) -> list[str]:
        # The "label speculation explicitly" rule is enforced deterministically after the
        # call by _annotate_speculation(), not as a blocking validation error: a strong
        # critique can discuss hypothetical mechanisms without literally using the word
        # "hypothesis", and failing a whole run on vocabulary is not useful.
        _unverified, cited = check_cited_numbers(text, allowed_numbers)
        if len(cited) < 3:  # redundant with min_cited but gives a clearer message
            return [
                f"risk_critic.submission: only {len(cited)} verified numbers cited; name at "
                "least three concrete artifact values"
            ]
        return []

    runtime = _make_runtime(
        ctx,
        assets,
        toolbox,
        as_of=as_of,
        output_kind="markdown",
        text_validators=(critique_validator,),
        allowed_numbers=allowed_numbers,
        min_cited=int(assets.spec.min_cited_numbers),
        min_chars=int(assets.spec.min_chars),
        required_phrases=assets.spec.required_phrases,
    )
    critique = runtime.run(user_prompt)
    critique = _annotate_speculation(str(critique))
    critique_path = runtime.write_markdown("critique", _normalise_markdown(critique, as_of))
    _event(
        ctx,
        "agent_stage_completed",
        stage=stage,
        role="risk_critic",
        information_cutoff=date_str(as_of),
        stats=_agent_summary(runtime),
    )
    return StageOutput(
        artifacts={"critique": _rel(critique_path)},
        summary={
            "information_cutoff": date_str(as_of),
            "word_count": len(str(critique).split()),
            "passed_checks": (robustness_summary.get("passed_checks") if robustness_summary else None),
            "failed_checks": (robustness_summary.get("failed_checks") if robustness_summary else None),
            "agent": _agent_summary(runtime),
        },
    )


def _normalise_markdown(text: str, as_of: dt.date) -> str:
    body = text.strip()
    cutoff = date_str(as_of)
    if cutoff not in body:
        body = (
            f"> Information cutoff: {cutoff}. All numbers below are from the frozen "
            "snapshot on or before this date.\n\n" + body
        )
    return body.rstrip() + "\n"


_SPECULATIVE_LANGUAGE = (
    "may ",
    "might ",
    "could ",
    "likely",
    "possibly",
    "plausibl",
    "risk that",
    "suggests",
    "would require",
    "we would",
    "if the",
)


def _annotate_speculation(text: str) -> str:
    """Guarantee that the risk critique labels speculation as hypothesis, not finding.

    This is a deterministic post-processing step rather than a blocking validation rule:
    a good critique may discuss hypothetical mechanisms without using the literal word
    "hypothesis". If speculative language is present and the note contains no explicit
    hypothesis/speculation label, one audit sentence is appended. The model's own text is
    never rewritten.
    """

    lowered = text.lower()
    if "speculation" in lowered or "hypothes" in lowered:
        return text
    if not any(marker in lowered for marker in _SPECULATIVE_LANGUAGE):
        return text
    label = (
        "\n\n## Speculation (hypothesis, not a finding)\n\n"
        "Any mechanism, forecast or forward-looking statement above is a hypothesis to be "
        "tested, not an observed finding. Only numbers quoted from the supplied artifacts "
        "are evidence.\n"
    )
    return text.rstrip() + label


def _augment_report_disclosure(
    note: str,
    *,
    failed_checks: Sequence[Mapping[str, Any]],
    concerns: Sequence[Any],
    realized_te: Any,
    ex_ante_te: Any,
    te_target: Any,
) -> tuple[str, list[str]]:
    """Deterministically ensure the note discloses weak robustness and risk gaps.

    This is a post-processing safety net, not a blocking validation: the note is already
    validated (citations, cutoff, sections). If the model omitted a failed robustness
    check, the placebo concern or the realised-vs-target tracking-error gap, a short
    ``## Harness disclosure`` section is appended. The model's own prose is never edited,
    and the transformation is deterministic so replay is byte-identical.
    """

    lowered = note.lower()
    warnings: list[str] = []
    lines: list[str] = []
    for check in failed_checks:
        name = str(check.get("name") or check.get("id") or "check")
        if name.strip().lower() and name.strip().lower() in lowered:
            continue
        status = str(check.get("status") or "").strip().lower() or "not_passed"
        value = check.get("value")
        threshold = check.get("threshold")
        detail = str(check.get("detail") or "").strip()
        parts = [f"- `{name}`: {status}"]
        if isinstance(value, (int, float)) or isinstance(threshold, (int, float)):
            parts.append(
                f"(value {_fmt_number(value)}, threshold {_fmt_number(threshold)})"
            )
        if detail:
            parts.append(f"- {detail}")
        lines.append(" ".join(parts))
        warnings.append(f"failed_check_disclosed:{name}")

    placebo_concern = any("placebo" in str(concern).lower() for concern in concerns) or any(
        "placebo" in str(check.get("name") or check.get("id") or "").lower()
        for check in failed_checks
    )
    if placebo_concern and "placebo" not in lowered:
        lines.append(
            "- The placebo / random-score check did not pass or is a listed robustness "
            "concern; treat the excess return as not robust to a random-score null."
        )
        warnings.append("placebo_concern_disclosed")

    realised_mentioned = "tracking error" in lowered or "tracking_error" in lowered
    if (
        not realised_mentioned
        and isinstance(realized_te, (int, float))
        and isinstance(te_target, (int, float))
    ):
        ex_ante_text = _fmt_number(ex_ante_te) if isinstance(ex_ante_te, (int, float)) else "n/a"
        lines.append(
            f"- Realised tracking error {_fmt_number(realized_te)} versus ex-ante tracking "
            f"error {ex_ante_text} and target {_fmt_number(te_target)}."
        )
        warnings.append("realized_vs_target_te_disclosed")

    if not lines:
        return note, warnings
    section = (
        "\n\n## Harness disclosure (deterministic)\n\n"
        "The following items are added deterministically from the verified artifacts so "
        "the note cannot omit them:\n\n" + "\n".join(lines) + "\n"
    )
    return note.rstrip() + section, warnings


_REPORT_SECTIONS = (
    "## Executive summary",
    "## Hypothesis",
    "## Data and point-in-time discipline",
    "## Factor construction",
    "## Backtest results",
    "## Robustness",
    "## Portfolio construction and risk",
    "## Independent verification",
    "## Limitations and information cutoff",
)

_REPORT_DISCLAIMER = "This note is research documentation, not investment advice."


def _compose_deterministic_report(
    *,
    as_of: dt.date,
    verdict: str,
    verification: Mapping[str, Any],
    metrics: Mapping[str, Any],
    robustness_checks: Sequence[Mapping[str, Any]],
    robustness_summary: Mapping[str, Any],
    risk_report: Any,
    factor_report: Any,
    hypothesis: Mapping[str, Any],
    data_quality: Any,
    pit_audit: Any,
    risk_budget: Mapping[str, Any],
    metrics_table: str,
) -> str:
    """Deterministic research note composed from verified artifacts (no model call).

    Used when the report-writer agent repeatedly fails validation: the harness must still
    produce a truthful, reviewable note rather than silently dropping the stage. Every
    number here comes from an artifact; the note states its provenance and the cutoff.
    """

    cutoff = date_str(as_of)
    recomputed = verification.get("recomputed") if isinstance(verification, Mapping) else {}
    recomputed = recomputed if isinstance(recomputed, Mapping) else {}
    strategy_metrics = recomputed.get("metrics") if isinstance(recomputed.get("metrics"), Mapping) else None
    benchmark_metrics = (
        recomputed.get("benchmark_metrics")
        if isinstance(recomputed.get("benchmark_metrics"), Mapping)
        else None
    )
    if not strategy_metrics:
        strategy_metrics = metrics.get("metrics") if isinstance(metrics.get("metrics"), Mapping) else {}
    if not benchmark_metrics:
        benchmark_metrics = (
            metrics.get("benchmark_metrics")
            if isinstance(metrics.get("benchmark_metrics"), Mapping)
            else {}
        )
    strategy_metrics = strategy_metrics or {}
    benchmark_metrics = benchmark_metrics or {}

    lines: list[str] = [
        "# CSI 300 multi-factor index-enhancement research note",
        "",
        f"Information cutoff: {cutoff}.",
        "",
        "> Deterministic harness fallback: the report-writing agent failed validation, so "
        "this note was composed directly from the independently verified artifacts. It "
        "contains no model-generated claims; a human reviewer must read it before approval.",
        "",
        "## Executive summary",
        "",
        f"Independent verification verdict: **{verdict}**. "
        + (
            "All required independent checks passed; the numbers below are the "
            "independently recomputed values."
            if verdict == "PASS"
            else "The verification did not pass; this note documents the evidence without "
            "claiming publication readiness, and the run must be revised or escalated."
        ),
        "",
    ]
    headline_metrics = (
        "annualized_return",
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
        "max_drawdown",
        "annualized_turnover",
    )
    headline = [
        f"`{key}` = {_fmt_number(strategy_metrics.get(key))}"
        for key in headline_metrics
        if isinstance(strategy_metrics.get(key), (int, float))
    ]
    if headline:
        lines.append("Headline metrics: " + "; ".join(headline) + ".")
        lines.append("")

    lines.extend(["## Hypothesis", ""])
    thesis = str(hypothesis.get("thesis") or "").strip()
    if thesis:
        lines.extend([thesis, ""])
    weights = hypothesis.get("factor_weights") or {}
    if isinstance(weights, Mapping) and weights:
        lines.extend(["| factor | weight |", "|---|---|"])
        for factor_id in sorted(weights):
            lines.append(f"| {factor_id} | {_fmt_number(weights[factor_id])} |")
        lines.append("")
    constraints = hypothesis.get("constraints") or {}
    if isinstance(constraints, Mapping) and constraints:
        lines.append(
            "Rebalance: "
            f"{hypothesis.get('rebalance', 'n/a')}; constraints: "
            + ", ".join(f"{key}={_fmt_number(value)}" for key, value in sorted(constraints.items()))
            + "."
        )
        lines.append("")

    lines.extend(
        [
            "## Data and point-in-time discipline",
            "",
            f"The analysis uses the frozen CSI 300 snapshot available on or before {cutoff}. "
            "No live data, network access or post-cutoff information is used.",
            "",
        ]
    )
    quality = data_quality if isinstance(data_quality, Mapping) else {}
    if quality:
        lines.append(
            f"Data quality: universe size {_fmt_number(quality.get('universe_size'))}, "
            f"coverage {_fmt_number(quality.get('coverage'))}, warnings "
            f"{len(quality.get('warnings') or [])}."
        )
    pit = pit_audit if isinstance(pit_audit, Mapping) else {}
    if pit:
        lines.append(
            f"PIT audit: maximum panel date {str(pit.get('max_panel_date', 'n/a'))}, "
            f"violations {len(pit.get('violations') or [])}."
        )
    lines.extend(
        [
            "Fundamentals are point-in-time: a value is usable only after its announcement "
            "date; missing announcement dates were estimated and flagged in the snapshot.",
            "",
        ]
    )

    lines.extend(["## Factor construction", ""])
    factor = factor_report if isinstance(factor_report, Mapping) else {}
    rank_ic = factor.get("rank_ic") if isinstance(factor.get("rank_ic"), Mapping) else {}
    if rank_ic:
        lines.append("Factor RankIC (mean): " + ", ".join(
            f"{factor_id}={_fmt_number(value)}" for factor_id, value in sorted(rank_ic.items())
        ) + ".")
    coverage = factor.get("coverage") if isinstance(factor.get("coverage"), Mapping) else {}
    if coverage:
        values = [value for value in coverage.values() if isinstance(value, (int, float))]
        if values:
            lines.append(
                f"Coverage ranges from {_fmt_number(min(values))} to {_fmt_number(max(values))}."
            )
    lines.append("Factors are winsorised, rank-standardised and neutralised by industry and log size.")
    lines.append("")

    lines.extend(["## Backtest results", "", "The companion table `metrics_table.md` is authoritative:", ""])
    lines.extend(metrics_table.strip().splitlines()[2:] if metrics_table.startswith("#") else metrics_table.strip().splitlines())
    lines.append("")

    lines.extend(["## Robustness", ""])
    failed_for_counts = [
        check
        for check in robustness_checks
        if isinstance(check, Mapping)
        and str(check.get("status", "")).strip().lower() not in {"pass", "ok", ""}
    ]
    passed_count = sum(
        1
        for check in robustness_checks
        if isinstance(check, Mapping)
        and str(check.get("status", "")).strip().lower() in {"pass", "ok"}
    )
    failed_detail = "; ".join(
        f"{str(check.get('name') or check.get('id') or 'check')} (value "
        f"{_fmt_number(check.get('value'))}, threshold {_fmt_number(check.get('threshold'))})"
        for check in failed_for_counts
    )
    lines.append(
        f"The robustness suite has exactly {len(robustness_checks)} checks: "
        f"{passed_count} passed, {len(failed_for_counts)} failed"
        + (f" ({failed_detail})." if failed_detail else ".")
    )
    concerns = list(robustness_summary.get("concerns") or []) if isinstance(robustness_summary, Mapping) else []
    if isinstance(robustness_summary, Mapping) and robustness_summary.get("pass"):
        passed = robustness_summary.get("pass")
        if isinstance(passed, (list, tuple)) and passed:
            lines.append("Passed robustness checks: " + ", ".join(str(item) for item in passed) + ".")
        elif isinstance(passed, (int, float)):
            lines.append(f"Robustness checks passed: {_fmt_number(passed)}.")
    if concerns:
        lines.append("Robustness concerns: " + "; ".join(str(item) for item in concerns) + ".")
    if robustness_checks:
        lines.extend(
            ["", "| check | status | value | threshold | detail |", "|---|---|---|---|---|"]
        )
        for check in robustness_checks:
            lines.append(
                f"| {check.get('name') or check.get('id') or 'check'} | "
                f"{check.get('status')} | {_fmt_number(check.get('value'))} | "
                f"{_fmt_number(check.get('threshold'))} | "
                f"{str(check.get('detail') or '').replace('|', '/')} |"
            )
    else:
        lines.append("No robustness check artifact was supplied; robustness is not established.")
    lines.append("")

    lines.extend(["## Portfolio construction and risk", ""])
    risk = risk_report if isinstance(risk_report, Mapping) else {}
    turnover_text = risk_budget.get("turnover_text") if isinstance(risk_budget, Mapping) else None
    if turnover_text:
        lines.append(str(turnover_text))
    lines.append(
        f"Realised tracking error {_fmt_number(risk_budget.get('realized_tracking_error'))} "
        f"versus ex-ante tracking error {_fmt_number(risk_budget.get('ex_ante_tracking_error'))} "
        f"and target {_fmt_number(risk_budget.get('te_target'))}; turnover "
        f"{_fmt_number(risk_budget.get('turnover'))}."
    )
    if risk:
        lines.append(
            f"Portfolio report: {_fmt_number(risk.get('n_names'))} names, ex-ante TE "
            f"{_fmt_number(risk.get('ex_ante_te'))}, turnover {_fmt_number(risk.get('turnover'))}, "
            f"optimizer status {str((risk.get('optimizer') or {}).get('status', 'n/a'))} "
            f"(fallback used: {str((risk.get('optimizer') or {}).get('fallback_used', 'n/a'))})."
        )
        constraints = risk.get("constraints") if isinstance(risk.get("constraints"), Mapping) else {}
        if constraints:
            lines.append(
                "Constraints: "
                + ", ".join(f"{key}={_fmt_number(value)}" for key, value in sorted(constraints.items()))
                + "."
            )
    lines.append("")

    lines.extend(["## Independent verification", "", f"Verdict: **{verdict}**.", ""])
    checks = verification.get("checks") if isinstance(verification, Mapping) else []
    if isinstance(checks, list) and checks:
        for check in checks:
            lines.append(
                f"- {check.get('id')}: {str(check.get('status')).upper()} — "
                f"{str(check.get('detail') or '').strip()}"
            )
    discrepancies = verification.get("discrepancies") if isinstance(verification, Mapping) else []
    lines.append(
        "Discrepancies: "
        + (
            "none reported."
            if not discrepancies
            else "; ".join(str(item) for item in list(discrepancies)[:10])
        )
    )
    artifact_hashes = verification.get("artifact_hashes") if isinstance(verification, Mapping) else {}
    lines.append(
        f"Verified artifact hashes: {len(artifact_hashes or {})} sha256 digests recorded in verification.json."
    )
    lines.append("")

    lines.extend(
        [
            "## Limitations and information cutoff",
            "",
            f"Information cutoff: {cutoff}. No live or post-cutoff information is used.",
            "",
            "- Survivorship caveat: the eligible universe is the current CSI 300 membership "
            "from the frozen snapshot; historical membership is not reconstructed.",
            "- Benchmark weights are a capped, renormalised free-float proxy, not official "
            "index weights.",
            "- Some fundamentals rely on estimated announcement dates, flagged in the snapshot.",
            "- Transaction costs, impact and tradability are modelled; no live execution is "
            "claimed.",
            "- Past backtest performance is not indicative of future results.",
            "",
            _REPORT_DISCLAIMER,
            "",
        ]
    )
    return "\n".join(lines)


def _validate_deterministic_report(
    note: str,
    *,
    as_of: dt.date,
    verdict: str,
    failed_check_names: Sequence[str],
) -> list[str]:
    """Light internal check for the deterministic fallback note (not the LLM guard)."""

    errors: list[str] = []
    lowered = note.lower()
    for section in _REPORT_SECTIONS:
        if section.lower() not in lowered:
            errors.append(f"missing section {section!r}")
    if date_str(as_of) not in note:
        errors.append(f"information cutoff {date_str(as_of)} missing")
    if verdict.lower() not in lowered:
        errors.append(f"verification verdict {verdict!r} missing")
    if _REPORT_DISCLAIMER.lower() not in lowered:
        errors.append("required disclaimer sentence missing")
    for name in failed_check_names:
        if str(name).lower() not in lowered:
            errors.append(f"failed robustness check {name!r} not disclosed")
    if len(note.strip()) < 500:
        errors.append("fallback note is too short to be reviewable")
    return errors


# ---------------------------------------------------------------------------
# Stage: report (10_report)
# ---------------------------------------------------------------------------


def run_report(ctx: RunContext) -> StageOutput:
    """Report writer agent -> verified ``research_note.md`` + deterministic table/charts."""

    stage = "report"
    as_of = _as_of(ctx)
    try:
        verification_output = ctx.upstream("verification")
    except WorkflowError as exc:
        raise WorkflowError("report stage requires the verification stage output") from exc
    verification_summary = dict(verification_output.summary)
    verification = _read_upstream_json(ctx, "verification", "verification")
    if not isinstance(verification, Mapping):
        raise WorkflowError(
            "report stage requires the verification artifact 'verification'"
        )
    verdict = str(verification.get("verdict") or verification_summary.get("verdict") or "")
    if verdict != "PASS":
        raise NonRetryableError(
            "report stage cannot publish a research note: independent verification verdict "
            f"is {verdict or 'missing'!r}; a FAIL run must be revised or escalated instead"
        )
    schema_errors = validate_verification_dict(verification)
    if schema_errors:
        raise ValidationError(
            "verification artifact is invalid, refusing to cite it: "
            + "; ".join(schema_errors[:5])
        )
    metrics = _read_upstream_json(ctx, "backtest", "metrics")
    if not isinstance(metrics, Mapping):
        raise WorkflowError("report stage requires the backtest artifact 'metrics'")
    robustness = _read_upstream_json(ctx, "robustness", "robustness")
    factor_report = _read_upstream_json(ctx, "factor_build", "factor_report")
    risk_report = _read_upstream_json(ctx, "portfolio", "risk_report")
    backtest_summary = _excerpt(_upstream_summary(ctx, "backtest"), max_depth=2, max_items=12)
    robustness_summary = _excerpt(
        _upstream_summary(ctx, "robustness"), max_depth=2, max_items=12
    )
    hypothesis = _read_upstream_json(ctx, "hypothesis", "hypothesis") or {}
    metrics_excerpt = _excerpt(metrics, max_depth=3, max_items=25, max_str=400)
    verification_excerpt = _excerpt(
        _scrub_prompt_metadata(verification), max_depth=3, max_items=25, max_str=400
    )
    robustness_excerpt = _excerpt(
        robustness if isinstance(robustness, Mapping) else {}, max_depth=3, max_items=20
    )
    robustness_document = robustness if isinstance(robustness, Mapping) else {}
    robustness_checks = _read_upstream_json(ctx, "robustness", "robustness_checks")
    if not isinstance(robustness_checks, list):
        embedded = robustness_document.get("checks")
        robustness_checks = embedded if isinstance(embedded, list) else []
    failed_robustness_checks = [
        check
        for check in robustness_checks
        if isinstance(check, Mapping)
        and str(check.get("status", "")).strip().lower() not in {"pass", "ok", ""}
    ]
    failed_check_names = [
        str(check.get("name") or check.get("id") or "unnamed_check")
        for check in failed_robustness_checks
    ]
    robustness_summary_document = robustness_document.get("summary") or {}
    if not isinstance(robustness_summary_document, Mapping):
        robustness_summary_document = {}
    robustness_disclosure = {
        "summary": dict(robustness_summary_document),
        "counts": {
            "checks_total": len(robustness_checks),
            "checks_not_passed": len(failed_robustness_checks),
            "checks_passed": sum(
                1
                for check in robustness_checks
                if isinstance(check, Mapping)
                and str(check.get("status", "")).strip().lower() in {"pass", "ok"}
            ),
        },
        "failed_checks": _excerpt(failed_robustness_checks, max_depth=2, max_items=20, max_str=400),
        "concerns": _excerpt(
            robustness_summary_document.get("concerns") or [], max_depth=2, max_items=20, max_str=400
        ),
    }
    robustness_counts = robustness_disclosure["counts"]
    failed_detail_text = "; ".join(
        f"{str(check.get('name') or check.get('id') or 'check')} (value "
        f"{_fmt_number(check.get('value'))}, threshold {_fmt_number(check.get('threshold'))})"
        for check in failed_robustness_checks
    )
    robustness_counts_text = (
        f"The robustness suite has exactly {robustness_counts['checks_total']} checks: "
        f"{robustness_counts['checks_passed']} passed, {robustness_counts['checks_not_passed']} failed"
        + (f" ({failed_detail_text})." if failed_detail_text else ".")
    )
    robustness_disclosure["counts_text"] = robustness_counts_text
    robustness_disclosure["note"] = (
        "Use these exact robustness counts and check names. Do not mix independent "
        "verification checks with robustness checks and do not restate any other count."
    )
    risk_excerpt = _excerpt(risk_report or {}, max_depth=2, max_items=15)
    metrics_metrics = metrics.get("metrics") if isinstance(metrics.get("metrics"), Mapping) else {}
    realized_te = metrics_metrics.get("tracking_error")
    if not isinstance(realized_te, (int, float)) and isinstance(risk_report, Mapping):
        realized_te = risk_report.get("ex_ante_te")
    ex_ante_te = risk_report.get("ex_ante_te") if isinstance(risk_report, Mapping) else None
    te_target = None
    if isinstance(risk_report, Mapping):
        te_target = risk_report.get("te_target")
        risk_constraints = risk_report.get("constraints")
        if te_target is None and isinstance(risk_constraints, Mapping):
            te_target = risk_constraints.get("te_target")
    if te_target is None:
        hypothesis_constraints = (hypothesis or {}).get("constraints") or {}
        te_target = hypothesis_constraints.get("te_target")
    risk_budget = {
        "realized_tracking_error": realized_te,
        "ex_ante_tracking_error": ex_ante_te,
        "te_target": te_target,
        "te_utilisation": (
            round(float(ex_ante_te) / float(te_target), 6)
            if isinstance(ex_ante_te, (int, float))
            and isinstance(te_target, (int, float))
            and float(te_target) > 0
            else None
        ),
        "turnover": metrics_metrics.get("turnover_annualized", metrics_metrics.get("turnover_average")),
    }
    # Turnover is reported as the engine records it: one-way annualised (mean one-way
    # per rebalance x 12), with two-way = 2x and the maximum single-rebalance two-way
    # derived from metrics.json["rebalances"]. The note must use these definitions.
    one_way_annualized = None
    for key in ("turnover_annualized", "annualized_turnover"):
        if isinstance(metrics_metrics.get(key), (int, float)):
            one_way_annualized = float(metrics_metrics[key])
            break
    if one_way_annualized is None and isinstance(metrics_metrics.get("turnover_average"), (int, float)):
        one_way_annualized = float(metrics_metrics["turnover_average"]) * 12.0
    rebalance_rows = metrics.get("rebalances") if isinstance(metrics.get("rebalances"), list) else []
    rebalance_turnovers = [
        float(row.get("turnover"))
        for row in rebalance_rows
        if isinstance(row, Mapping) and isinstance(row.get("turnover"), (int, float))
    ]
    max_single_two_way = (
        2.0 * max(rebalance_turnovers)
        if rebalance_turnovers
        else (float(metrics_metrics["turnover_average"]) * 2.0 if isinstance(metrics_metrics.get("turnover_average"), (int, float)) else None)
    )
    if one_way_annualized is not None:
        two_way_annualized = 2.0 * one_way_annualized
    else:
        two_way_annualized = None
    turnover_definitions = {
        "annualised_one_way": one_way_annualized,
        "annualised_two_way": two_way_annualized,
        "max_single_rebalance_two_way": max_single_two_way,
    }
    turnover_text = (
        "Turnover: annualised one-way "
        f"{_fmt_number(one_way_annualized)}; two-way is {_fmt_number(two_way_annualized)}; "
        "the maximum single-rebalance two-way turnover is "
        f"{_fmt_number(max_single_two_way)}."
    )
    risk_budget["turnover_definitions"] = turnover_definitions
    risk_budget["turnover_text"] = turnover_text
    risk_budget["note"] = (
        "Use exactly these turnover definitions and values; do not report a different "
        "turnover measure."
    )
    data_quality = _read_upstream_json(ctx, "data_prep", "data_quality")
    pit_audit = _read_upstream_json(ctx, "data_prep", "pit_audit")
    data_limitations = {
        "data_quality": _excerpt(
            data_quality if isinstance(data_quality, Mapping) else {}, max_depth=2, max_items=12, max_str=400
        ),
        "pit_audit": _excerpt(
            pit_audit if isinstance(pit_audit, Mapping) else {}, max_depth=2, max_items=12, max_str=400
        ),
        "survivorship_note": (
            "The eligible universe is the current CSI 300 membership from the frozen snapshot; "
            "historical membership is not reconstructed, so results carry a survivorship caveat "
            "that must be disclosed. Fundamentals are point-in-time: only rows whose notice_date "
            "is on or before the decision date are used."
        ),
        "information_cutoff": date_str(as_of),
    }
    reporting_requirements = {
        "must_disclose_failed_robustness_checks": failed_check_names,
        "must_not_claim_all_checks_passed": bool(failed_robustness_checks),
        "robustness_check_counts_exact": robustness_counts_text,
        "turnover_definitions_exact": turnover_text,
        "must_use_exact_counts_wording": (
            "In the Robustness section, state exactly the robustness-suite counts above "
            "and name every failed check; never mix independent-verification check counts "
            "with robustness-suite counts and never restate a different total. In the "
            "Portfolio construction and risk section, use exactly the turnover definitions "
            "above (annualised one-way, annualised two-way, maximum single-rebalance "
            "two-way)."
        ),
        "must_discuss_placebo_if_concern": any(
            "placebo" in str(concern).lower()
            for concern in (robustness_summary_document.get("concerns") or [])
        )
        or any("placebo" in name.lower() for name in failed_check_names),
        "must_disclose_realized_vs_target_tracking_error": {
            "realized_tracking_error": realized_te,
            "ex_ante_tracking_error": ex_ante_te,
            "te_target": te_target,
        },
        "must_state_limitations": ["survivorship caveat", "point-in-time fundamentals"],
        "information_cutoff": date_str(as_of),
    }
    factor_excerpt = _factor_report_excerpt(factor_report if isinstance(factor_report, Mapping) else {})
    hypothesis_excerpt = _excerpt(hypothesis, max_depth=2, max_items=10)
    assets = load_role_assets("report_writer", ctx.config)

    metrics_table = _render_metrics_table(
        metrics=metrics,
        verification=verification,
        as_of=as_of,
        hypothesis=hypothesis,
    )
    allowed_numbers = build_allowed_numbers(
        verification_excerpt,
        metrics_excerpt,
        robustness_excerpt,
        risk_excerpt,
        factor_excerpt,
        hypothesis_excerpt,
        metrics_table,
        backtest_summary,
        robustness_summary,
        robustness_disclosure,
        risk_budget,
        data_limitations,
        reporting_requirements,
    )

    def check_verified_number(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or args.get("metric") or "").strip()
        if not name:
            raise ValueError("pass a path, e.g. {\"name\": \"recomputed.metrics.annualized_excess_return\"}")
        sources = {
            "verification": verification,
            "recomputed": verification.get("recomputed") or {},
            "metrics": metrics,
            "robustness": robustness if isinstance(robustness, Mapping) else {},
            "risk_report": risk_report or {},
        }
        for source_name, source in sources.items():
            try:
                return {
                    "name": name,
                    "source": source_name,
                    "value": _dig(source, name),
                }
            except ValueError:
                continue
        raise ValueError(f"verified number path {name!r} not found in verified artifacts")

    def check_list_verified_checks(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "checks": _excerpt(verification.get("checks") or [], max_depth=2, max_items=30),
            "discrepancies": _excerpt(verification.get("discrepancies") or [], max_depth=2),
        }

    def check_artifact_hash(args: dict[str, Any]) -> dict[str, Any]:
        prefix = str(args.get("prefix") or "").strip().lower()
        hashes = verification.get("artifact_hashes") or {}
        if not prefix:
            return {"artifact_hashes": _excerpt(hashes, max_depth=2, max_items=20)}
        matches = {
            name: value
            for name, value in hashes.items()
            if str(name).lower().startswith(prefix) or str(value).lower().startswith(prefix)
        }
        if not matches:
            raise ValueError(f"no artifact hash matches prefix {prefix!r}")
        return {"matches": matches}

    toolbox = _toolbox(
        ctx,
        blobs={
            "verification": verification_excerpt,
            "metrics": metrics_excerpt,
            "risk_budget": risk_budget,
            "robustness": robustness_excerpt,
            "robustness_disclosure": robustness_disclosure,
            "data_limitations": data_limitations,
            "reporting_requirements": reporting_requirements,
            "risk_report": risk_excerpt,
            "factor_report": factor_excerpt,
            "hypothesis": hypothesis_excerpt,
            "metrics_table": metrics_table,
        },
        checks={
            "verified_number": check_verified_number,
            "list_verified_checks": check_list_verified_checks,
            "artifact_hash": check_artifact_hash,
        },
        write_paths={
            "research_note": ctx.artifact_path(stage, "research_note.md"),
            "metrics_table": ctx.artifact_path(stage, "metrics_table.md"),
        },
    )

    user_prompt = _build_user_prompt(
        title="Stage 10_report — verified research note for human approval",
        intro=(
            "Write the research note for the approval package. Independent verification "
            "has PASSed. Cite only verified numbers from the artifacts below and state the "
            "information cutoff. This is research documentation, not investment advice."
        ),
        as_of=as_of,
        sections=[
            ("verification", verification_excerpt),
            ("reporting_requirements", reporting_requirements),
            ("metrics", metrics_excerpt),
            ("risk_budget_realized_vs_target", risk_budget),
            ("robustness_disclosure", robustness_disclosure),
            ("data_limitations", data_limitations),
            ("robustness_evidence", robustness_excerpt),
            ("portfolio_risk_report", risk_excerpt),
            ("factor_report", factor_excerpt),
            ("hypothesis", hypothesis_excerpt),
            ("companion_metrics_table", metrics_table),
        ],
        instructions=(
            "Return Markdown (submit_markdown) of at least ~900 characters with sections:\n"
            "## Executive summary\n## Hypothesis\n## Data and point-in-time discipline\n"
            "## Factor construction\n## Backtest results\n## Robustness\n"
            "## Portfolio construction and risk\n## Independent verification\n"
            "## Limitations and information cutoff\n"
            f"- state the information cutoff {date_str(as_of)} explicitly;\n"
            "- cite at least five numbers, each taken from the verified artifacts above and "
            "attributed to its source artifact; verification.recomputed and metrics are "
            "authoritative;\n"
            f"- report the verification verdict {verdict!r} and any discrepancies verbatim;\n"
            "- follow reporting_requirements exactly: list EVERY failed robustness check by "
            "name with its value/threshold and never claim that all checks passed; use the "
            "exact robustness-suite counts and turnover definitions given (never mix "
            "robustness counts with independent-verification counts); discuss "
            "the placebo result when must_discuss_placebo_if_concern is true; state the "
            "realised tracking error against the ex-ante/target tracking error; state the "
            "survivorship caveat and the point-in-time fundamental limitation;\n"
            "- reference metrics_table.md as the companion table;\n"
            "- no forecasts, price targets, external references or wall-clock timestamps."
        ),
    )

    def note_validator(text: str) -> list[str]:
        errors: list[str] = []
        lowered = text.lower()
        if verdict.lower() not in lowered:
            errors.append(
                f"report_writer.submission: the verification verdict {verdict!r} must be "
                "reported explicitly"
            )
        if "metrics_table" not in lowered:
            errors.append(
                "report_writer.submission: reference metrics_table.md as the companion table"
            )
        if "not investment advice" not in lowered and "research documentation" not in lowered:
            errors.append(
                "report_writer.submission: state that the note is research documentation, "
                "not investment advice"
            )
        return errors

    runtime = _make_runtime(
        ctx,
        assets,
        toolbox,
        as_of=as_of,
        output_kind="markdown",
        text_validators=(note_validator,),
        allowed_numbers=allowed_numbers,
        min_cited=int(assets.spec.min_cited_numbers),
        min_chars=int(assets.spec.min_chars),
        required_phrases=assets.spec.required_phrases,
    )
    report_mode = "agent"
    fallback_reason = ""
    try:
        note = str(runtime.run(user_prompt))
    except ReplayCacheMiss:
        # Deterministic replay must never silently degrade to a non-recorded artifact.
        raise
    except NonRetryableError as exc:
        # The producing agent is not trusted after exhausted repairs. The harness must
        # still publish a truthful, reviewable note; compose it deterministically from
        # the verified artifacts and mark the stage output accordingly.
        report_mode = "deterministic_fallback"
        fallback_reason = str(exc)[:1500]
        note = _compose_deterministic_report(
            as_of=as_of,
            verdict=verdict,
            verification=verification,
            metrics=metrics,
            robustness_checks=robustness_checks,
            robustness_summary=robustness_summary_document,
            risk_report=risk_report,
            factor_report=factor_report,
            hypothesis=hypothesis,
            data_quality=data_quality,
            pit_audit=pit_audit,
            risk_budget=risk_budget,
            metrics_table=metrics_table,
        )
        fallback_errors = _validate_deterministic_report(
            note, as_of=as_of, verdict=verdict, failed_check_names=failed_check_names
        )
        if fallback_errors:
            raise ValidationError(
                "deterministic fallback report failed its internal completeness check: "
                + "; ".join(fallback_errors[:8])
            ) from exc
        _event(
            ctx,
            "report_fallback_used",
            level="warn",
            stage=stage,
            reason=fallback_reason[:500],
            information_cutoff=date_str(as_of),
        )
    note = _normalise_markdown(note, as_of)
    note, disclosure_warnings = _augment_report_disclosure(
        note,
        failed_checks=failed_robustness_checks,
        concerns=robustness_summary_document.get("concerns") or [],
        realized_te=realized_te,
        ex_ante_te=ex_ante_te,
        te_target=te_target,
    )
    if disclosure_warnings:
        _event(
            ctx,
            "report_disclosure_added",
            stage=stage,
            warnings=disclosure_warnings,
            information_cutoff=date_str(as_of),
        )
    note_path = runtime.write_markdown("research_note", note)
    table_path = runtime.write_markdown("metrics_table", metrics_table)

    chart_paths, chart_warnings = _make_report_charts(ctx)
    artifacts: dict[str, str] = {
        "research_note": _rel(note_path),
        "metrics_table": _rel(table_path),
    }
    for key, path in chart_paths.items():
        artifacts[key] = _rel(path)
    summary_paths = [artifacts["research_note"], artifacts["metrics_table"]] + [
        artifacts[key] for key in chart_paths
    ]
    _event(
        ctx,
        "agent_stage_completed",
        stage=stage,
        role="report_writer",
        verdict=verdict,
        charts=len(chart_paths),
        information_cutoff=date_str(as_of),
        stats=_agent_summary(runtime),
    )
    return StageOutput(
        artifacts=artifacts,
        summary={
            "paths": summary_paths,
            "verdict": verdict,
            "report_mode": report_mode,
            "fallback_reason": fallback_reason,
            "charts": sorted(chart_paths),
            "chart_warnings": chart_warnings,
            "failed_robustness_checks": failed_check_names,
            "disclosure_warnings": disclosure_warnings,
            "information_cutoff": date_str(as_of),
            "agent": _agent_summary(runtime),
        },
    )


def _render_metrics_table(
    *,
    metrics: Mapping[str, Any],
    verification: Mapping[str, Any],
    as_of: dt.date,
    hypothesis: Mapping[str, Any] | None = None,
) -> str:
    recomputed = verification.get("recomputed") or {}
    strategy_metrics: Mapping[str, Any] = {}
    benchmark_metrics: Mapping[str, Any] = {}
    if isinstance(recomputed, Mapping):
        candidate = recomputed.get("metrics")
        if isinstance(candidate, Mapping):
            strategy_metrics = candidate
        candidate = recomputed.get("benchmark_metrics")
        if isinstance(candidate, Mapping):
            benchmark_metrics = candidate
    if not strategy_metrics:
        candidate = metrics.get("metrics")
        if isinstance(candidate, Mapping):
            strategy_metrics = candidate
    if not benchmark_metrics:
        candidate = metrics.get("benchmark_metrics")
        if isinstance(candidate, Mapping):
            benchmark_metrics = candidate
    by_year = metrics.get("by_year")
    constraints = metrics.get("constraints") or (hypothesis or {}).get("constraints") or {}

    lines = [
        "# Metrics table",
        "",
        f"Information cutoff: {date_str(as_of)}. Numbers are the independently "
        "recomputed values where available; otherwise the backtest artifact is used.",
        "",
        "| metric | strategy | benchmark |",
        "|---|---|---|",
    ]
    keys = sorted(set(strategy_metrics) | set(benchmark_metrics))
    emitted = 0
    for key in keys:
        strategy_value = strategy_metrics.get(key)
        benchmark_value = benchmark_metrics.get(key)
        if not isinstance(strategy_value, (int, float)) and not isinstance(
            benchmark_value, (int, float)
        ):
            continue
        lines.append(
            f"| {key} | {_fmt_number(strategy_value)} | {_fmt_number(benchmark_value)} |"
        )
        emitted += 1
    if emitted == 0:
        lines.append("| (no numeric metrics were supplied) | | |")
    if isinstance(by_year, Mapping) and by_year:
        lines.extend(["", "## By year", "", "| year | strategy | benchmark |", "|---|---|---|"])
        for year in sorted(by_year, key=str)[:20]:
            row = by_year[year]
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| {year} | "
                f"{_fmt_number(row.get('annualized_return', row.get('return')))} | "
                f"{_fmt_number(row.get('benchmark_annualized_return', row.get('benchmark_return')))} |"
            )
    elif isinstance(by_year, list) and by_year:
        lines.extend(["", "## By year", "", "| year | strategy | benchmark |", "|---|---|---|"])
        for row in by_year[:20]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| {row.get('year', row.get('date', ''))} | "
                f"{_fmt_number(row.get('strategy', row.get('strategy_return')))} | "
                f"{_fmt_number(row.get('benchmark', row.get('benchmark_return')))} |"
            )
    if isinstance(constraints, Mapping) and constraints:
        lines.extend(["", "## Constraints", "", "| constraint | value |", "|---|---|"])
        for key in sorted(constraints):
            value = constraints[key]
            if isinstance(value, (int, float)):
                lines.append(f"| {key} | {_fmt_number(value)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report charts (deterministic, best-effort)
# ---------------------------------------------------------------------------


def _configure_matplotlib_fonts(matplotlib_module: Any) -> None:
    """Prefer a CJK-capable font so industry labels in charts are legible."""

    try:
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001 - font probing is best effort
        available = set()
    for candidate in (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ):
        if candidate in available:
            matplotlib_module.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    matplotlib_module.rcParams["axes.unicode_minus"] = False


def _make_report_charts(ctx: RunContext) -> tuple[dict[str, Path], list[str]]:
    charts: dict[str, Path] = {}
    warnings: list[str] = []
    stage_dir = ctx.stage_dir("report")
    charts_dir = stage_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        _configure_matplotlib_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - charts are optional evidence, not gating
        return {}, [f"matplotlib unavailable: {exc}"]

    equity_curve = _try_upstream_path(ctx, "backtest", "equity_curve")
    if equity_curve is not None:
        for name, builder in (
            ("cumulative_excess", _chart_cumulative_excess),
            ("drawdown", _chart_drawdown),
        ):
            path = charts_dir / f"{name}.png"
            try:
                created = builder(equity_curve, path, plt)
                if created:
                    charts[f"chart_{name}"] = path
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{name}: {exc}")
    factor_report_path = _try_upstream_path(ctx, "factor_build", "factor_report")
    if factor_report_path is not None:
        path = charts_dir / "factor_ic.png"
        try:
            payload = read_json(factor_report_path)
            if _chart_factor_ic(payload, path, plt):
                charts["chart_factor_ic"] = path
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"factor_ic: {exc}")
    weights_path = _try_upstream_path(ctx, "portfolio", "target_weights") or _try_upstream_path(
        ctx, "backtest", "holdings"
    )
    if weights_path is not None:
        path = charts_dir / "exposures.png"
        try:
            if _chart_exposures(weights_path, path, plt):
                charts["chart_exposures"] = path
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"exposures: {exc}")
    return charts, warnings


def _try_upstream_path(ctx: RunContext, stage: str, logical: str) -> Path | None:
    if stage not in ctx.stage_outputs:
        return None
    try:
        return ctx.upstream_artifact(stage, logical)
    except WorkflowError:
        return None


def _read_equity_curve(path: Path) -> dict[str, list[Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[Any]] = {}
        for row in reader:
            for key, value in row.items():
                columns.setdefault(str(key), []).append(value)
        return columns


def _float_series(columns: Mapping[str, Sequence[Any]], key: str) -> list[float] | None:
    raw = columns.get(key)
    if not raw:
        return None
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return values


def _chart_cumulative_excess(equity_curve: Path, path: Path, plt: Any) -> bool:
    columns = _read_equity_curve(equity_curve)
    nav = _float_series(columns, "nav")
    benchmark = _float_series(columns, "benchmark_nav")
    if not nav or not benchmark:
        return False
    excess = [a / b - 1.0 if b else 0.0 for a, b in zip(nav, benchmark)]
    fig, axis = plt.subplots(figsize=(8.0, 3.6))
    axis.plot(excess, linewidth=1.4, color="#1f4e79")
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_title("Cumulative excess return (verified backtest)")
    axis.set_ylabel("excess vs benchmark")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _chart_drawdown(equity_curve: Path, path: Path, plt: Any) -> bool:
    columns = _read_equity_curve(equity_curve)
    drawdown = _float_series(columns, "drawdown")
    if not drawdown:
        return False
    fig, axis = plt.subplots(figsize=(8.0, 3.0))
    axis.fill_between(range(len(drawdown)), drawdown, 0.0, color="#c0504d", alpha=0.45)
    axis.plot(drawdown, color="#7f1d1d", linewidth=0.9)
    axis.set_title("Strategy drawdown")
    axis.set_ylabel("drawdown")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _chart_factor_ic(factor_report: Mapping[str, Any], path: Path, plt: Any) -> bool:
    rank_ic = factor_report.get("rank_ic")
    values: dict[str, float] = {}
    if isinstance(rank_ic, Mapping):
        for key, value in rank_ic.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[str(key)] = float(value)
            elif isinstance(value, Mapping):
                for candidate in ("mean", "ic", "rank_ic"):
                    if isinstance(value.get(candidate), (int, float)):
                        values[str(key)] = float(value[candidate])
                        break
    elif isinstance(rank_ic, list):
        for item in rank_ic:
            if isinstance(item, Mapping):
                name = item.get("factor") or item.get("id") or item.get("name")
                for candidate in ("mean", "ic", "rank_ic", "value"):
                    if isinstance(item.get(candidate), (int, float)):
                        values[str(name)] = float(item[candidate])
                        break
    if not values:
        return False
    items = sorted(values.items(), key=lambda pair: pair[1])
    fig, axis = plt.subplots(figsize=(8.0, 0.28 * len(items) + 1.4))
    axis.barh([name for name, _ in items], [value for _, value in items], color="#4f81bd")
    axis.axvline(0.0, color="#888888", linewidth=0.8)
    axis.set_title("Factor RankIC (mean)")
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _chart_exposures(weights_path: Path, path: Path, plt: Any) -> bool:
    from aicwf.util.jsonio import read_parquet

    frame = read_parquet(weights_path)
    if "industry" not in getattr(frame, "columns", []) or "active_weight" not in getattr(
        frame, "columns", []
    ):
        return False
    grouped = frame.groupby("industry", dropna=True)["active_weight"].mean()
    grouped = grouped[grouped.abs() > 0]
    if grouped.empty:
        return False
    grouped = grouped.sort_values()
    fig, axis = plt.subplots(figsize=(8.0, 0.3 * len(grouped) + 1.4))
    axis.barh(grouped.index.astype(str), grouped.values, color="#9bbb59")
    axis.axvline(0.0, color="#888888", linewidth=0.8)
    axis.set_title("Mean active industry exposure")
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Verification helper: independent verifier note
# ---------------------------------------------------------------------------


def _failed_checks(checks: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Checks whose lowercase status is ``fail`` (the verifier's convention)."""

    return [
        check
        for check in checks
        if isinstance(check, Mapping) and str(check.get("status", "")).lower() == "fail"
    ]


def generate_verifier_note(ctx: RunContext, verification_dict: Mapping[str, Any]) -> str:
    """Return the verifier agent's Markdown note for a verification dictionary.

    Called by the verification stage. The verification *result* is produced by
    deterministic recomputation code, never by the model; the model only writes the
    narrative note. The note is read-only, offline in ``replay``/``null`` mode, cites only
    numbers present in ``verification_dict`` and falls back to a deterministic template
    if the model cannot produce a valid note (the note is supporting evidence, not a gate).
    """

    errors = validate_verification_dict(verification_dict)
    if errors:
        raise ValidationError(
            "refusing to generate a verifier note for an invalid verification result: "
            + "; ".join(errors[:6])
        )
    as_of_raw = verification_dict.get("as_of")
    as_of = parse_date(str(as_of_raw)) if as_of_raw else _as_of(ctx)
    verdict = str(verification_dict.get("verdict", ""))
    checks = list(verification_dict.get("checks") or [])
    failed = _failed_checks(checks)
    recomputed = verification_dict.get("recomputed") or {}
    discrepancies = list(verification_dict.get("discrepancies") or [])
    hashes = verification_dict.get("artifact_hashes") or {}
    allowed_numbers = numbers_from(verification_dict)
    prompt_verification = _scrub_prompt_metadata(verification_dict)
    excerpt = _excerpt(prompt_verification, max_depth=3, max_items=30, max_str=600)

    def check_verdict(_args: dict[str, Any]) -> dict[str, Any]:
        return {"verdict": verdict}

    def check_list_checks(_args: dict[str, Any]) -> dict[str, Any]:
        return {"checks": _excerpt(checks, max_depth=2, max_items=40)}

    def check_failed_checks(_args: dict[str, Any]) -> dict[str, Any]:
        return {"failed_checks": _excerpt(failed, max_depth=2)}

    def check_check_detail(args: dict[str, Any]) -> dict[str, Any]:
        check_id = str(args.get("check_id") or args.get("id") or "").strip()
        for check in checks:
            if str(check.get("id")) == check_id:
                return {"check": _excerpt(check, max_depth=2)}
        raise ValueError(f"check {check_id!r} is not in the verification result")

    def check_recomputed_number(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name") or args.get("path") or "").strip()
        if not name:
            raise ValueError("pass a path, e.g. {\"name\": \"metrics.annualized_excess_return\"}")
        return {"name": name, "value": _dig(recomputed, name)}

    def check_hash_lookup(args: dict[str, Any]) -> dict[str, Any]:
        prefix = str(args.get("prefix") or "").strip().lower()
        matches = {
            name: value
            for name, value in hashes.items()
            if str(name).lower().startswith(prefix) or str(value).lower().startswith(prefix)
        }
        if not matches:
            raise ValueError(f"no artifact hash matches prefix {prefix!r}")
        return {"matches": matches}

    fallback = _deterministic_verifier_note(verification_dict, as_of)
    try:
        assets = load_role_assets("verifier", ctx.config)
        toolbox = _toolbox(
            ctx,
            blobs={"verification": excerpt},
            checks={                "verdict": check_verdict,
                "list_checks": check_list_checks,
                "failed_checks": check_failed_checks,
                "check_detail": check_check_detail,
                "recomputed_number": check_recomputed_number,
                "hash_lookup": check_hash_lookup,
            },
        )
        user_prompt = _build_user_prompt(
            title="Stage 09_verification — independent verifier note",
            intro=(
                "The deterministic verification result is below. Write the narrative "
                "verifier note. You may not modify the result or any artifact; report "
                "failures and discrepancies plainly."
            ),
            as_of=as_of,
            sections=[
                ("verification_result", excerpt),
                (
                    "failed_checks",
                    _excerpt(failed, max_depth=2, max_items=30),
                ),
                ("discrepancies", _excerpt(discrepancies, max_depth=2, max_items=20)),
            ],
            instructions=(
                "Return Markdown (submit_markdown) of at least ~400 characters with "
                "sections: ## Verdict, ## Checks performed, ## Recomputed results, "
                "## Discrepancies, ## Reproducibility, ## Information cutoff.\n"
                f"- state the information cutoff {date_str(as_of)} explicitly;\n"
                "- cite at least two numbers from the verification result;\n"
                f"- the verdict is {verdict!r}: report it verbatim even if it is FAIL;\n"
                "- refer to artifact hashes by short prefixes; do not invent checks."
            ),
        )

        def note_validator(text: str) -> list[str]:
            if verdict and verdict.lower() not in text.lower():
                return [f"verifier.submission: report the verdict {verdict!r} verbatim"]
            return []

        runtime = _make_runtime(
            ctx,
            assets,
            toolbox,
            as_of=as_of,
            output_kind="markdown",
            text_validators=(note_validator,),
            allowed_numbers=allowed_numbers,
            min_cited=int(assets.spec.min_cited_numbers),
            min_chars=int(assets.spec.min_chars),
            required_phrases=assets.spec.required_phrases,
        )
        note = str(runtime.run(user_prompt))
        _event(
            ctx,
            "verifier_note_ready",
            verdict=verdict,
            checks=len(checks),
            failed=len(failed),
            stats=_agent_summary(runtime),
        )
        return _normalise_markdown(note, as_of)
    except NonRetryableError as exc:
        _event(
            ctx,
            "agent_fallback",
            level="warn",
            role="verifier",
            reason=str(exc)[:500],
        )
        return fallback


def _deterministic_verifier_note(
    verification_dict: Mapping[str, Any], as_of: dt.date
) -> str:
    verdict = str(verification_dict.get("verdict", "UNKNOWN"))
    checks = list(verification_dict.get("checks") or [])
    failed = _failed_checks(checks)
    recomputed = verification_dict.get("recomputed") or {}
    discrepancies = list(verification_dict.get("discrepancies") or [])
    hashes = verification_dict.get("artifact_hashes") or {}
    lines = [
        "# Independent verification note",
        "",
        f"Information cutoff: {date_str(as_of)}.",
        "",
        "## Verdict",
        "",
        (
            f"Verification verdict: **{verdict}**. "
            + (
                "The independently recomputed artifacts match the produced artifacts within "
                "the configured tolerances."
                if verdict == "PASS"
                else "At least one independent check failed; the run must be revised or "
                "escalated before approval."
            )
        ),
        "",
        "## Checks performed",
        "",
    ]
    if checks:
        for check in checks:
            lines.append(
                f"- {check.get('id')}: {str(check.get('status')).upper()} - "
                f"{str(check.get('detail', '')).strip()}"
            )
        if failed:
            lines.extend(["", f"Failed checks: {len(failed)} of {len(checks)}."])
    else:
        lines.append("- No checks were reported in the verification result.")
    lines.extend(["", "## Recomputed results", ""])
    numeric: list[tuple[str, Any]] = []
    if isinstance(recomputed, Mapping):
        for key in sorted(recomputed):
            value = recomputed[key]
            if isinstance(value, Mapping):
                for sub_key in sorted(value):
                    sub_value = value[sub_key]
                    if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                        numeric.append((f"{key}.{sub_key}", sub_value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.append((key, value))
    if numeric:
        for name, value in numeric[:20]:
            lines.append(f"- {name}: {_fmt_number(value)}")
    else:
        lines.append("- No recomputed numeric values were reported.")
    lines.extend(["", "## Discrepancies", ""])
    if discrepancies:
        for item in discrepancies[:20]:
            lines.append(f"- {str(item)}")
    else:
        lines.append("- None found within the configured tolerances.")
    lines.extend(["", "## Reproducibility", ""])
    if hashes:
        for name in sorted(hashes)[:12]:
            digest = str(hashes[name])
            lines.append(f"- {name}: sha256 {digest[:16]}...")
        lines.append(
            "Artifact versions are identified by these sha256 digests; the verification "
            "stage recomputes every number independently of the producing code."
        )
    else:
        lines.append("- No artifact hashes were reported.")
    lines.extend(
        [
            "",
            "## Information cutoff",
            "",
            f"The verified run uses only information available on or before {date_str(as_of)}. "
            "No live or post-cutoff information enters the verification.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "generate_verifier_note",
    "run_factor_review",
    "run_hypothesis",
    "run_report",
    "run_risk_review",
]
