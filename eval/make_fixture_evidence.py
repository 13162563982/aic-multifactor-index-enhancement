"""Regenerate the committed fixture-backed evidence for verification/eval/registry.

This is a repository maintenance helper (offline, deterministic).  It rebuilds:

* ``eval/runs/baseline_v1.0.0/run``               -- reference baseline run
* ``eval/baseline_v1.0.0.json``                   -- frozen baseline document
* ``eval/cases/accepted_v1.1.0/``                 -- accepted candidate + decision record
* ``eval/cases/rejected_v1.2.0/``                 -- rejected candidate + decision record
* ``eval/registry_roundtrip/``                    -- promote -> promote -> rollback transcript

The baseline is fixture-backed: after the Lead integrates the real snapshot and
the quant stack, replace it with the real run via

    python -m aicwf.eval --baseline-refresh --run outputs/runs/<real_run_id> \\
        --data-scope real_csi300 --evidence-mode real

The mechanism (baseline schema, hard gates, tolerances, decision records,
registry) is identical for fixture and real evidence; only the metrics differ.

Usage:
    python eval/make_fixture_evidence.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicwf.eval.baseline import refresh_baseline  # noqa: E402
from aicwf.eval.fixtures import build_demo_run  # noqa: E402
from aicwf.eval.runner import run_eval  # noqa: E402
from aicwf.registry.promote import promote_run  # noqa: E402
from aicwf.registry.rollback import rollback  # noqa: E402
from aicwf.registry.store import verify_bundle  # noqa: E402
from aicwf.util.jsonio import read_json, write_json, write_text  # noqa: E402

FIXED_TIME = "2025-01-01T00:00:00+00:00"
EVAL_ROOT = PROJECT_ROOT / "eval"
CASES_ROOT = EVAL_ROOT / "cases"
RUNS_ROOT = EVAL_ROOT / "runs"
ROUNDTRIP_ROOT = EVAL_ROOT / "registry_roundtrip"

PROMPT_ACCEPTED = """# Researcher prompt - candidate v1.1.0 (accepted)

Role: quantitative researcher proposing a monthly CSI 300 index-enhancement view.

## Change in this candidate
The short-horizon reversal signal uses a **16 trading-day** lookback instead of 15.
Everything else (rebalance cadence, execution at the next close, constraints and
information cutoff) is unchanged.  The hypothesis must still state the
information cutoff explicitly and may only cite data available on or before it.

## Output contract
Return the hypothesis JSON with `thesis`, `factor_weights`, `rebalance`,
`constraints`, `falsification`, `information_cutoff` and `source_limits`.
Never invent numbers: every value must come from the supplied artifacts.
"""

PROMPT_REJECTED = """# Researcher prompt - candidate v1.2.0 (rejected)

Role: quantitative researcher proposing a monthly CSI 300 index-enhancement view.

## Change in this candidate
This candidate proposes **weekly** active rebalancing to react faster to the
reversal signal.  The change is rejected by change control because the turnover
increase and the realized excess-return degradation exceed the frozen baseline
tolerances in config/gates.yaml, even though the pipeline itself verifies cleanly.

## Output contract
Return the hypothesis JSON with `thesis`, `factor_weights`, `rebalance`,
`constraints`, `falsification`, `information_cutoff` and `source_limits`.
Never invent numbers: every value must come from the supplied artifacts.
"""


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_evidence() -> dict:
    summary: dict = {}
    _reset(RUNS_ROOT)
    _reset(CASES_ROOT)
    _reset(ROUNDTRIP_ROOT)

    # 1. Baseline reference run + frozen baseline ------------------------
    baseline_bundle = RUNS_ROOT / "baseline_v1.0.0"
    baseline_run = build_demo_run(baseline_bundle, variant="baseline")
    baseline_path = EVAL_ROOT / "baseline_v1.0.0.json"
    baseline = refresh_baseline(
        baseline_run,
        out_path=baseline_path,
        baseline_id="baseline_v1.0.0",
        version="v1.0.0",
        data_scope="mini_fixture",
        evidence_mode="fixture",
        created_at_utc=FIXED_TIME,
        notes=[
            "Fixture-backed placeholder baseline built on tests/fixtures/mini_snapshot.",
            "Replace with the real-data baseline after integration:",
            "python -m aicwf.eval --baseline-refresh --run outputs/runs/<real_run_id> "
            "--data-scope real_csi300 --evidence-mode real",
        ],
    )
    summary["baseline"] = {
        "path": str(baseline_path.relative_to(PROJECT_ROOT)),
        "snapshot_hash": baseline["frozen"]["snapshot_hash"],
        "annualized_excess_return": baseline["metrics"].get("annualized_excess_return"),
        "tracking_error": baseline["metrics"].get("tracking_error"),
    }

    # 2. Accepted and rejected candidate cases --------------------------
    cases = {
        "accepted_v1.1.0": {
            "variant": "accepted",
            "version": "v1.1.0",
            "prompt": PROMPT_ACCEPTED,
            "expected": "accept",
            "description": (
                "Candidate v1.1.0 refines the reversal-signal lookback from 15 to 16 trading days; "
                "every change-control metric stays inside the frozen tolerances."
            ),
            "config_overrides": {"strategy.demo.lookback": 16},
            "change_summary": "Reversal signal lookback 15 -> 16 trading days.",
        },
        "rejected_v1.2.0": {
            "variant": "rejected",
            "version": "v1.2.0",
            "prompt": PROMPT_REJECTED,
            "expected": "reject",
            "description": (
                "Candidate v1.2.0 proposes weekly rebalancing; verification passes, but turnover increases "
                "far beyond the change-control tolerance and annualized excess return degrades."
            ),
            "config_overrides": {"strategy.demo.rebalance": "weekly"},
            "change_summary": "Rebalance cadence monthly -> weekly.",
        },
    }
    summary["cases"] = {}
    for case_id, spec in cases.items():
        case_dir = CASES_ROOT / case_id
        _reset(case_dir)
        prompt_dir = case_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        write_text(prompt_dir / "researcher.md", spec["prompt"])
        candidate_run = build_demo_run(case_dir, variant=spec["variant"], run_name="candidate_run")
        case = {
            "case_id": case_id,
            "candidate_version": spec["version"],
            "baseline": "../../baseline_v1.0.0.json",
            "description": spec["description"],
            "change_summary": spec["change_summary"],
            "prompt_overrides": {"researcher": "prompts/researcher.md"},
            "config_overrides": spec["config_overrides"],
            "candidate_run": "candidate_run",
            "expected_decision": spec["expected"],
            "evidence_mode": "fixture",
            "data_scope": "mini_fixture",
            "notes": (
                "Fixture-backed evidence built by eval/make_fixture_evidence.py on "
                "tests/fixtures/mini_snapshot. The real evidence is generated by running the "
                "pipeline with these prompt/config overrides and then "
                "`python -m aicwf.eval --case-dir eval/cases/<case>`."
            ),
        }
        write_json(case_dir / "case.json", case, indent=2, sort_keys=True)
        decision = run_eval(
            run_dir=candidate_run,
            baseline_path=baseline_path,
            out_path=case_dir / "decision.json",
            case_id=case_id,
            case_dir=case_dir,
            candidate_prompt_dir=prompt_dir,
            created_at_utc=FIXED_TIME,
        )
        summary["cases"][case_id] = {
            "dir": str(case_dir.relative_to(PROJECT_ROOT)),
            "decision": decision["decision"],
            "expected": spec["expected"],
            "metric_diffs": {
                diff["metric"]: {"status": diff["status"], "delta": diff.get("delta")}
                for diff in decision["metric_diffs"]
            },
        }
        if decision["decision"] != spec["expected"]:
            raise RuntimeError(
                f"fixture evidence inconsistent: {case_id} produced {decision['decision']} "
                f"but expected {spec['expected']}"
            )

    # 3. Registry promote/promote/rollback transcript -------------------
    history_path = ROUNDTRIP_ROOT / "registry" / "versions.json"
    releases_path = ROUNDTRIP_ROOT / "registry" / "releases"
    restored_root = ROUNDTRIP_ROOT / "restored"
    config_v100 = json.loads((baseline_run / "config_bundle.json").read_text(encoding="utf-8"))
    accepted_run = CASES_ROOT / "accepted_v1.1.0" / "candidate_run"
    config_v110 = json.loads((accepted_run / "config_bundle.json").read_text(encoding="utf-8"))

    entry_v100 = promote_run(
        baseline_run,
        version="v1.0.0",
        history_path=history_path,
        releases=releases_path,
        project_root=PROJECT_ROOT,
        config=config_v100,
        gates=config_v100.get("gates"),
        rationale="Initial accepted release v1.0.0 (fixture evidence).",
        git_commit=None,
        created_at_utc=FIXED_TIME,
    )
    entry_v110 = promote_run(
        accepted_run,
        version="v1.1.0",
        history_path=history_path,
        releases=releases_path,
        project_root=PROJECT_ROOT,
        config=config_v110,
        gates=config_v110.get("gates"),
        rationale="Accepted candidate v1.1.0 (fixture evidence).",
        git_commit=None,
        created_at_utc=FIXED_TIME,
    )
    rollback_result = rollback(
        to_version="v1.0.0",
        history_path=history_path,
        releases=releases_path,
        project_root=PROJECT_ROOT,
        restore_targets={
            "config": restored_root / "config",
            "prompts": restored_root / "prompts",
            "skills": restored_root / "skills",
        },
        rationale=(
            "Rolled v1.1.0 back to v1.0.0 in the fixture transcript to demonstrate the append-only "
            "rollback record with before/after metrics and a smoke evaluation."
        ),
        created_at_utc=FIXED_TIME,
    )
    summary["registry"] = {
        "dir": str(ROUNDTRIP_ROOT.relative_to(PROJECT_ROOT)),
        "history": str(history_path.relative_to(PROJECT_ROOT)),
        "release_v1.0.0_hash": entry_v100["entry_hash"],
        "release_v1.1.0_hash": entry_v110["entry_hash"],
        "rollback_entry_hash": (rollback_result.get("entry") or {}).get("entry_hash"),
        "rollback_smoke": (rollback_result.get("smoke_eval") or {}).get("status"),
        "bundle_v1.0.0_intact": not verify_bundle(releases_path / "v1.0.0"),
        "bundle_v1.1.0_intact": not verify_bundle(releases_path / "v1.1.0"),
        "restored_config_files": len(
            list((restored_root / "config").glob("*.yaml")) if (restored_root / "config").exists() else []
        ),
    }
    restored_strategy = restored_root / "config" / "strategy.yaml"
    if restored_strategy.exists():
        import yaml

        restored = yaml.safe_load(restored_strategy.read_text(encoding="utf-8")) or {}
        summary["registry"]["restored_variant"] = (restored.get("demo") or {}).get("variant") or (
            (restored.get("strategy") or {}).get("demo") or {}
        ).get("variant")
    return summary


def main() -> int:
    summary = build_evidence()
    write_json(EVAL_ROOT / "evidence_summary.json", summary, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
