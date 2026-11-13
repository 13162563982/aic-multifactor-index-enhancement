"""A/B evaluation of a candidate run against a frozen baseline.

Hard gates (from ``config/gates.yaml: change_control.hard_gates``) must all pass:
``leakage_tests_pass``, ``verification_passes``, ``deterministic_replay`` and
``budget_within_limit``.  Then the soft metric tolerances
(``change_control.metric_acceptance``) decide whether the candidate may replace
the baseline.  The result is a decision record with every gate, every metric
diff, the config delta and a written rationale.

Used by ``python -m aicwf eval`` (through the pipeline/CLI) and by the Lead to
produce the accepted/rejected/rollback evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from aicwf.contracts import PROJECT_ROOT, date_str
from aicwf.eval.baseline import (
    INFO_METRIC_KEYS,
    SOFT_METRIC_KEYS,
    TOLERANCE_SEMANTICS,
    DEFAULT_BASELINE_PATH,
    DEFAULT_GATES_PATH,
    baseline_metrics,
    hard_gates,
    load_baseline,
    load_gates,
    metric_acceptance_tolerances,
)
from aicwf.eval.cases import config_overrides_applied, load_case
from aicwf.util.jsonio import read_json, read_yaml, write_json
from aicwf.util.timeutil import utc_now_iso

EVAL_SCHEMA_VERSION = "1.0.0"
_WATCH_CONFIG_KEYS = (
    "strategy.factors",
    "strategy.backtest",
    "prompt",
    "prompts",
    "strategy.demo",
)


# ---------------------------------------------------------------------------
# Small readers (all best-effort: a missing evidence file is a gate failure)
# ---------------------------------------------------------------------------


def _read_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _run_request(run_dir: Path) -> dict[str, Any]:
    return _read_optional(run_dir / "00_intake" / "run_request.json") or {}


def _metrics_doc(run_dir: Path) -> dict[str, Any]:
    return _read_optional(run_dir / "05_backtest" / "metrics.json") or {}


def _verification(run_dir: Path) -> dict[str, Any] | None:
    return _read_optional(run_dir / "09_verification" / "verification.json")


def _run_summary(run_dir: Path) -> dict[str, Any]:
    return _read_optional(run_dir / "run_summary.json") or {}


def _pit_audit(run_dir: Path) -> dict[str, Any]:
    return _read_optional(run_dir / "02_data" / "pit_audit.json") or {}


def _run_config(run_dir: Path) -> dict[str, Any]:
    bundle = _read_optional(run_dir / "config_bundle.json")
    if isinstance(bundle, dict):
        return bundle
    config: dict[str, Any] = {}
    for name in ("workflow", "strategy", "gates", "budgets", "model", "data"):
        path = PROJECT_ROOT / "config" / f"{name}.yaml"
        if path.exists():
            config[name] = read_yaml(path) or {}
    return config


def _as_of(run_dir: Path) -> str | None:
    request = _run_request(run_dir)
    if request.get("as_of"):
        return date_str(request["as_of"])
    return None


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def gate_leakage_tests_pass(run_dir: Path, override: bool | None = None) -> dict[str, Any]:
    """PIT/leakage evidence: clean pit_audit, no artifact dated after as_of."""

    if override is not None:
        return _gate("leakage_tests_pass", bool(override), "explicit override from the evaluator")
    as_of = _as_of(run_dir)
    audit = _pit_audit(run_dir)
    problems: list[str] = []
    if not audit:
        problems.append("02_data/pit_audit.json is missing")
    else:
        violations = audit.get("violations") or []
        if violations:
            problems.append(f"pit_audit reports {len(violations)} violation(s)")
        if as_of and audit.get("max_panel_date") and date_str(audit["max_panel_date"]) > as_of:
            problems.append(f"panel extends to {audit['max_panel_date']} after as_of {as_of}")
        if as_of and audit.get("as_of") and date_str(audit["as_of"]) != as_of:
            problems.append(f"pit_audit.as_of {audit['as_of']} != run as_of {as_of}")
    verification = _verification(run_dir) or {}
    pit_checks = [
        check
        for check in (verification.get("checks") or [])
        if str(check.get("id", "")).startswith("pit_") or str(check.get("id")) in {"forbidden_securities", "t1_tradability"}
    ]
    failed = [check["id"] for check in pit_checks if check.get("status") != "pass"]
    if failed:
        problems.append(f"verification PIT checks failed: {', '.join(failed)}")
    return _gate(
        "leakage_tests_pass",
        not problems,
        "pit_audit clean and verification PIT checks pass" if not problems else "; ".join(problems),
    )


def gate_verification_passes(run_dir: Path, override: bool | None = None) -> dict[str, Any]:
    if override is not None:
        return _gate("verification_passes", bool(override), "explicit override from the evaluator")
    verification = _verification(run_dir)
    if not verification:
        return _gate("verification_passes", False, "09_verification/verification.json is missing")
    verdict = str(verification.get("verdict", "FAIL")).upper()
    failed = [
        check.get("id") for check in (verification.get("checks") or []) if check.get("status") != "pass"
    ]
    detail = f"verification verdict {verdict}"
    if failed:
        detail += f"; failed checks: {', '.join(str(item) for item in failed[:6])}"
    return _gate("verification_passes", verdict == "PASS", detail, evidence={"verdict": verdict, "failed_checks": failed})


def gate_deterministic_replay(run_dir: Path, override: bool | None = None) -> dict[str, Any]:
    if override is not None:
        return _gate("deterministic_replay", bool(override), "explicit override from the evaluator")
    evidence = _read_optional(run_dir / "replay_evidence.json")
    if isinstance(evidence, dict):
        identical = bool(evidence.get("identical"))
        n_artifacts = evidence.get("n_artifacts")
        detail = (
            f"two independent builds produced identical content hashes for {n_artifacts} artifacts"
            if identical
            else "replay artifact hashes differ: the run is not reproducible"
        )
        return _gate("deterministic_replay", identical, detail, evidence={"n_artifacts": n_artifacts})
    # Controller state exports may carry per-step input/output hashes; accept an
    # explicit replay marker if present.
    for name in ("state_snapshot.json", "run_summary.json"):
        payload = _read_optional(run_dir / name)
        if isinstance(payload, dict) and payload.get("deterministic_replay") is not None:
            return _gate(
                "deterministic_replay",
                bool(payload["deterministic_replay"]),
                f"deterministic_replay flag read from {name}",
            )
    return _gate(
        "deterministic_replay",
        False,
        "no replay evidence: re-run the pipeline twice (controller hash comparison) or provide replay_evidence.json",
    )


def gate_budget_within_limit(run_dir: Path, override: bool | None = None) -> dict[str, Any]:
    if override is not None:
        return _gate("budget_within_limit", bool(override), "explicit override from the evaluator")
    summary = _run_summary(run_dir)
    if not summary:
        return _gate("budget_within_limit", False, "run_summary.json is missing; cannot prove the run stayed within budget")
    problems: list[str] = []
    if summary.get("status") not in {None, "completed"}:
        problems.append(f"run status {summary.get('status')!r} is not completed")
    budget = summary.get("budget") or {}
    for used_key, limit_key, label in (
        ("llm_calls", "max_llm_calls", "LLM calls"),
        ("llm_tokens", "max_tokens_total", "tokens"),
        ("cost_usd", "max_cost_usd", "cost"),
    ):
        used = summary.get(used_key)
        limit = budget.get(limit_key)
        if used is not None and limit is not None and float(used) > float(limit) * 1.0000001:
            problems.append(f"{label} {used} exceeded limit {limit}")
    return _gate(
        "budget_within_limit",
        not problems,
        "run completed within budget" if not problems else "; ".join(problems),
        evidence={"llm_calls": summary.get("llm_calls"), "llm_tokens": summary.get("llm_tokens")},
    )


_HARD_GATE_FUNCTIONS = {
    "leakage_tests_pass": gate_leakage_tests_pass,
    "verification_passes": gate_verification_passes,
    "deterministic_replay": gate_deterministic_replay,
    "budget_within_limit": gate_budget_within_limit,
}


def _gate(gate_id: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    payload = {"id": gate_id, "status": "pass" if ok else "fail", "detail": detail}
    payload.update(extra)
    return payload


def run_hard_gates(run_dir: Path, gates_config: Mapping[str, Any] | None, overrides: Mapping[str, bool] | None) -> list[dict[str, Any]]:
    overrides = dict(overrides or {})
    results: list[dict[str, Any]] = []
    for gate_id in hard_gates(gates_config):
        function = _HARD_GATE_FUNCTIONS.get(gate_id)
        if function is None:
            results.append(_gate(gate_id, False, f"unknown hard gate {gate_id!r} in config/gates.yaml"))
            continue
        try:
            results.append(function(run_dir, overrides.get(gate_id)))
        except Exception as exc:  # noqa: BLE001 - a broken gate fails closed
            results.append(_gate(gate_id, False, f"gate crashed: {type(exc).__name__}: {exc}"))
    return results


# ---------------------------------------------------------------------------
# Soft metric change control
# ---------------------------------------------------------------------------


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


#: Alias resolution so the change-control gates work with the production metric
#: naming (the backtest reports ``turnover_average``/``turnover_annualized``).
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "turnover": ("turnover", "turnover_annualized", "turnover_average"),
    "annualized_excess_return": ("annualized_excess_return",),
    "information_ratio": ("information_ratio",),
    "tracking_error": ("tracking_error",),
    "max_drawdown": ("max_drawdown",),
}


def metric_value(metrics: Mapping[str, Any], key: str) -> Any:
    """Resolve a headline metric, tolerating the production JSON naming."""

    for alias in _METRIC_ALIASES.get(key, (key,)):
        if alias in metrics and metrics[alias] is not None:
            return metrics[alias]
    return None


def compare_soft_metric(
    metric: str,
    baseline_value: Any,
    candidate_value: Any,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    spec = TOLERANCE_SEMANTICS[metric]
    tolerance_key = spec["tolerance"]
    tolerance = float(tolerances.get(tolerance_key, 0.0))
    baseline_number = _numeric(baseline_value)
    candidate_number = _numeric(candidate_value)
    record: dict[str, Any] = {
        "metric": metric,
        "tolerance_key": tolerance_key,
        "tolerance": tolerance,
        "direction": spec["direction"],
        "baseline": baseline_number,
        "candidate": candidate_number,
    }
    if baseline_number is None or candidate_number is None:
        record.update(
            {
                "status": "skip",
                "delta": None,
                "detail": (
                    f"cannot evaluate {metric}: "
                    f"{'baseline ' if baseline_number is None else ''}"
                    f"{'candidate ' if candidate_number is None else ''}value missing"
                ),
            }
        )
        return record
    delta = candidate_number - baseline_number
    record["delta"] = delta
    if spec["direction"] == "higher_better":
        passed = delta >= -tolerance
        detail = (
            f"{metric}: candidate {candidate_number:.6g} vs baseline {baseline_number:.6g} "
            f"(delta {delta:+.4g}, max allowed degradation {tolerance:g})"
        )
    elif spec["direction"] == "lower_better":
        passed = delta <= tolerance
        detail = (
            f"{metric}: candidate {candidate_number:.6g} vs baseline {baseline_number:.6g} "
            f"(delta {delta:+.4g}, max allowed increase {tolerance:g})"
        )
    else:  # absolute band
        passed = abs(delta) <= tolerance
        detail = (
            f"{metric}: candidate {candidate_number:.6g} vs baseline {baseline_number:.6g} "
            f"(|delta| {abs(delta):.4g} <= tolerance {tolerance:g})"
        )
    record.update({"status": "pass" if passed else "fail", "detail": detail})
    return record


def compare_info_metric(metric: str, baseline_value: Any, candidate_value: Any) -> dict[str, Any]:
    baseline_number = _numeric(baseline_value)
    candidate_number = _numeric(candidate_value)
    delta = None
    if baseline_number is not None and candidate_number is not None:
        delta = candidate_number - baseline_number
    return {
        "metric": metric,
        "baseline": baseline_number,
        "candidate": candidate_number,
        "delta": delta,
        "status": "info",
        "detail": f"{metric}: candidate {candidate_number} vs baseline {baseline_number} (context only)",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _display_path(path: Path | str | None) -> str | None:
    """Project-relative POSIX path when possible (keeps evidence portable)."""

    if path is None:
        return None
    candidate = Path(path)
    try:
        from aicwf.contracts import relpath

        if candidate.is_absolute():
            return relpath(candidate)
    except Exception:  # noqa: BLE001
        pass
    return candidate.as_posix()


def resolve_candidate_run(candidate: Path | str | None) -> Path:
    """Resolve a candidate target (run dir, case dir, run id or registry version)."""

    if candidate is None or str(candidate) == "":
        raise FileNotFoundError("no candidate run directory supplied")
    target = Path(str(candidate))
    if target.exists():
        if (target / "00_intake" / "run_request.json").exists():
            return target
        for child in ("run", "candidate_run"):
            nested = target / child
            if (nested / "00_intake" / "run_request.json").exists():
                return nested
        case_file = target / "case.json"
        if case_file.exists():
            try:
                case = read_json(case_file)
                nested = Path(target) / str(case.get("candidate_run") or "")
                if (nested / "00_intake" / "run_request.json").exists():
                    return nested
            except Exception:  # noqa: BLE001
                pass
    by_run_id = PROJECT_ROOT / "outputs" / "runs" / str(candidate)
    if (by_run_id / "00_intake" / "run_request.json").exists():
        return by_run_id
    try:  # a registry version can be re-evaluated from its recorded run id
        from aicwf.registry.store import find_entry

        entry = find_entry(str(candidate), kind="release")
    except Exception:  # noqa: BLE001
        entry = None
    if entry and entry.get("run_id"):
        recorded = PROJECT_ROOT / "outputs" / "runs" / str(entry["run_id"])
        if (recorded / "00_intake" / "run_request.json").exists():
            return recorded
        raise FileNotFoundError(
            f"registry version {candidate!r} refers to run {entry['run_id']!r}, but that run directory is gone; "
            "evaluate the run directory directly"
        )
    raise FileNotFoundError(
        f"cannot resolve candidate {candidate!r} to a run directory "
        "(expected a run dir containing 00_intake/run_request.json, a case dir, "
        "outputs/runs/<run_id>, or a promoted registry version)"
    )


def resolve_baseline_path(baseline: Path | str | None) -> Path:
    """Resolve a baseline target (path, or ``eval/<name>.json`` shorthand)."""

    if baseline is None or str(baseline) == "":
        return DEFAULT_BASELINE_PATH
    target = Path(str(baseline))
    candidates = [
        target,
        PROJECT_ROOT / target,
        PROJECT_ROOT / "eval" / f"{target}.json",
        PROJECT_ROOT / "eval" / f"baseline_{target}.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"eval baseline not found: {baseline!r}; freeze one with "
        "`python -m aicwf.eval --baseline-refresh --run <run_dir>`"
    )


def run_eval(
    *,
    run_dir: Path | str | None = None,
    candidate: Path | str | None = None,
    baseline_path: Path | str | None = None,
    baseline: Path | str | None = None,
    gates_path: Path | str | None = None,
    gates: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    out_path: Path | str | None = None,
    case_id: str | None = None,
    case_dir: Path | str | None = None,
    candidate_prompt_dir: Path | str | None = None,
    candidate_config_overrides: Mapping[str, Any] | None = None,
    hard_gate_overrides: Mapping[str, bool] | None = None,
    created_at_utc: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate one candidate run against a frozen baseline; returns the decision record.

    Accepts both the subsystem spelling (``run_dir`` / ``baseline_path``) and the
    pipeline/CLI spelling (``candidate`` / ``baseline`` / ``config``).
    """

    target = candidate if candidate is not None else run_dir
    candidate_dir = resolve_candidate_run(target)
    resolved_baseline = resolve_baseline_path(baseline_path if baseline_path is not None else baseline)
    if not candidate_dir.exists():
        raise FileNotFoundError(f"candidate run directory does not exist: {candidate_dir}")
    baseline = load_baseline(resolved_baseline)
    baseline_path = str(resolved_baseline)
    if gates is None and config is not None and isinstance(config.get("gates"), Mapping):
        gates = dict(config["gates"])
    gates_config = dict(gates) if gates is not None else load_gates(gates_path or DEFAULT_GATES_PATH)
    tolerances = dict(baseline.get("tolerances") or {})
    tolerances.update(metric_acceptance_tolerances(gates_config))
    case: dict[str, Any] = {}
    if case_dir:
        case = load_case(case_dir)
        case_id = case_id or case.get("case_id")
        if candidate_prompt_dir is None:
            candidate_prompt_dir = case.get("_prompt_dir")
        if candidate_config_overrides is None:
            candidate_config_overrides = case.get("config_overrides")
    request = _run_request(candidate_dir)
    metrics_doc = _metrics_doc(candidate_dir)
    candidate_metrics = dict(metrics_doc.get("metrics") or {})
    verification = _verification(candidate_dir)
    baseline_values = baseline_metrics(baseline)

    hard_gates_results = run_hard_gates(candidate_dir, gates_config, hard_gate_overrides)

    metric_diffs: list[dict[str, Any]] = []
    for metric in SOFT_METRIC_KEYS:
        metric_diffs.append(
            compare_soft_metric(
                metric,
                metric_value(baseline_values, metric),
                metric_value(candidate_metrics, metric),
                tolerances,
            )
        )
    info_diffs = [
        compare_info_metric(metric, metric_value(baseline_values, metric), metric_value(candidate_metrics, metric))
        for metric in INFO_METRIC_KEYS
    ]

    config_changes: list[dict[str, Any]] = []
    baseline_config = baseline.get("config") if isinstance(baseline.get("config"), dict) else {}
    candidate_config = _run_config(candidate_dir)
    if baseline_config:
        config_changes = config_overrides_applied(baseline_config, candidate_config, _WATCH_CONFIG_KEYS)

    prompt_hashes: dict[str, str] = {}
    from aicwf.eval.baseline import prompt_hashes as _prompt_hashes

    if candidate_prompt_dir:
        prompt_hashes = _prompt_hashes(candidate_prompt_dir)

    reasons: list[str] = []
    hard_failures = [gate for gate in hard_gates_results if gate["status"] != "pass"]
    soft_failures = [diff for diff in metric_diffs if diff["status"] == "fail"]
    evaluated = [diff for diff in metric_diffs if diff["status"] != "skip"]
    if hard_failures:
        reasons.extend(f"hard gate {gate['id']} failed: {gate['detail']}" for gate in hard_failures)
    if soft_failures:
        reasons.extend(f"change-control metric failed: {diff['detail']}" for diff in soft_failures)
    decision = "accept" if not hard_failures and not soft_failures else "reject"
    if not reasons:
        if evaluated:
            reasons.append(
                f"all {len(hard_gates_results)} hard gates passed and {len(evaluated)} "
                f"change-control metric(s) stayed within the baseline tolerances"
            )
        else:
            reasons.append(
                "all hard gates passed; the baseline has no comparable metric values "
                "(refresh the baseline with `python -m aicwf.eval --baseline-refresh`)"
            )
    status_word = "ACCEPTED" if decision == "accept" else "REJECTED"
    rationale = (
        f"Candidate {request.get('run_id', candidate_dir.name)} ({request.get('candidate_version') or 'unversioned'}) "
        f"was {status_word} against {baseline.get('baseline_id')} ({baseline.get('version')}). "
        + (f"{len(hard_gates_results)} hard gates: {sum(1 for gate in hard_gates_results if gate['status'] == 'pass')} passed, "
           f"{len(hard_failures)} failed. ")
        + (f"{len(evaluated)} change-control metrics evaluated: {len(soft_failures)} failed. "
           if evaluated
           else "No change-control metrics were available in the baseline. ")
        + "Reasons: " + "; ".join(reasons)
    )
    record: dict[str, Any] = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "decision_id": f"eval-{case_id or 'candidate'}-{request.get('run_id', candidate_dir.name)}",
        "case_id": case_id,
        "created_at_utc": created_at_utc or utc_now_iso(),
        "decision": decision,
        "reasons": reasons,
        "rationale": rationale,
        "candidate": {
            "run_id": request.get("run_id"),
            "run_dir": _display_path(candidate_dir),
            "version": request.get("candidate_version") or request.get("variant"),
            "as_of": _as_of(candidate_dir),
            "snapshot_hash": request.get("snapshot_hash"),
            "config_hash": request.get("config_hash") or metrics_doc.get("config_hash"),
            "llm_mode": request.get("llm_mode"),
            "verification_verdict": (verification or {}).get("verdict"),
            "prompt_dir": _display_path(candidate_prompt_dir),
            "prompt_hashes": prompt_hashes,
            "config_overrides": dict(candidate_config_overrides or {}),
            "metrics": candidate_metrics,
        },
        "baseline": {
            "baseline_id": baseline.get("baseline_id"),
            "version": baseline.get("version"),
            "data_scope": baseline.get("data_scope"),
            "evidence_mode": baseline.get("evidence_mode"),
            "frozen": baseline.get("frozen"),
            "metrics": baseline_values,
            "_path": _display_path(baseline.get("_path")),
        },
        "hard_gates": hard_gates_results,
        "metric_diffs": metric_diffs,
        "info_metric_diffs": info_diffs,
        "config_changes": config_changes,
        "tolerances": tolerances,
        "evidence": {
            "candidate_run_dir": _display_path(candidate_dir),
            "baseline_path": _display_path(baseline_path),
            "case_dir": _display_path(case_dir),
            "hard_gate_overrides": dict(hard_gate_overrides or {}),
        },
    }
    if write:
        target = Path(out_path) if out_path else candidate_dir / "eval_decision.json"
        write_json(target, record, indent=2, sort_keys=True)
        record["decision_path"] = _display_path(target)
    return record


# ---------------------------------------------------------------------------
# CLI: python -m aicwf.eval
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aicwf.eval",
        description="A/B evaluate a candidate run against the frozen eval baseline",
    )
    parser.add_argument("--run", "--run-dir", "--candidate", dest="run_dir", default=None, help="candidate run directory")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH), help="baseline JSON path")
    parser.add_argument("--gates", default=None, help="config/gates.yaml path (default: project config)")
    parser.add_argument("--out", default=None, help="decision record output path")
    parser.add_argument("--case", dest="case_id", default=None, help="case id recorded in the decision")
    parser.add_argument("--case-dir", default=None, help="case directory (loads case.json defaults)")
    parser.add_argument("--prompt-dir", default=None, help="candidate prompt directory (hashed into the record)")
    parser.add_argument("--json", action="store_true", help="print the decision record as JSON")
    parser.add_argument("--expect", choices=("accept", "reject"), default=None, help="assert the expected decision")
    parser.add_argument("--baseline-refresh", action="store_true", help="freeze/refresh the baseline from --run")
    parser.add_argument("--baseline-id", default="baseline_v1.0.0")
    parser.add_argument("--version", default=None, help="baseline version when refreshing")
    parser.add_argument("--data-scope", default=None, help="baseline data scope label when refreshing")
    parser.add_argument("--evidence-mode", choices=("fixture", "real"), default=None)
    parser.add_argument("--created-at", default=None, help="fixed created_at_utc for reproducible evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.baseline_refresh:
            from aicwf.eval.baseline import refresh_baseline

            if not args.run_dir:
                sys.stderr.write("--baseline-refresh requires --run <verified run directory>\n")
                return 2
            baseline = refresh_baseline(
                args.run_dir,
                out_path=args.baseline,
                baseline_id=args.baseline_id,
                version=args.version,
                prompt_dir=args.prompt_dir,
                data_scope=args.data_scope,
                evidence_mode=args.evidence_mode,
                created_at_utc=args.created_at,
            )
            sys.stdout.write(
                f"baseline written: {args.baseline} "
                f"(id={baseline['baseline_id']}, version={baseline['version']}, "
                f"evidence_mode={baseline.get('evidence_mode')})\n"
            )
            return 0
        run_dir = args.run_dir
        if not run_dir and args.case_dir:
            case = load_case(args.case_dir)
            run_dir = case.get("_candidate_run")
            args.case_id = args.case_id or case.get("case_id")
            args.prompt_dir = args.prompt_dir or case.get("_prompt_dir")
        if not run_dir:
            sys.stderr.write("provide --run <run_dir> (or --case-dir with a candidate_run)\n")
            return 2
        record = run_eval(
            run_dir=run_dir,
            baseline_path=args.baseline,
            gates_path=args.gates,
            out_path=args.out,
            case_id=args.case_id,
            case_dir=args.case_dir,
            candidate_prompt_dir=args.prompt_dir,
            created_at_utc=args.created_at,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"eval failed: {type(exc).__name__}: {exc}\n")
        return 2
    if args.json:
        sys.stdout.write(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n")
    else:
        sys.stdout.write(f"decision: {record['decision'].upper()}  ({record.get('decision_path', 'not written')})\n")
        for gate in record["hard_gates"]:
            sys.stdout.write(f"  hard gate {gate['id']}: {gate['status']}\n")
        for diff in record["metric_diffs"]:
            sys.stdout.write(
                f"  metric {diff['metric']}: {diff['status']} "
                f"(baseline={diff.get('baseline')}, candidate={diff.get('candidate')}, tolerance={diff.get('tolerance')})\n"
            )
        for reason in record["reasons"]:
            sys.stdout.write(f"  reason: {reason}\n")
    if args.expect and record["decision"] != args.expect:
        sys.stderr.write(f"expected decision {args.expect!r} but got {record['decision']!r}\n")
        return 3
    return 0 if record["decision"] == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
