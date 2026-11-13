"""Candidate-change test cases (accepted / rejected examples).

A case is a directory with a ``case.json`` file, a prompt-override tree and, for
offline evidence, a pre-built candidate run directory.  The real pipeline runs
the case's prompt/config overrides through the workflow and then calls
``aicwf.eval.runner.run_eval`` on the resulting run; the committed fixture cases
let the same decision logic be demonstrated without network or a full backtest.

``case.json`` shape::

    {
      "case_id": "accepted_v1.1.0",
      "candidate_version": "v1.1.0",
      "baseline": "eval/baseline_v1.0.0.json",
      "description": "...",
      "prompt_overrides": {"researcher": "prompts/researcher.md"},
      "config_overrides": {"strategy.factors.composite.default_weights.momentum_12_1": 0.20},
      "candidate_run": "candidate_run",
      "expected_decision": "accept",
      "evidence_mode": "fixture",
      "changed_files": ["prompts/researcher.md"],
      "change_summary": "..."
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aicwf.contracts import PROJECT_ROOT
from aicwf.util.jsonio import read_json, write_json

CASE_FILE = "case.json"
CASES_ROOT = PROJECT_ROOT / "eval" / "cases"


def load_case(case_dir: Path | str) -> dict[str, Any]:
    """Load one case definition and resolve its paths."""

    root = Path(case_dir)
    path = root / CASE_FILE
    if not path.exists():
        raise FileNotFoundError(f"case definition not found: {path}")
    case = read_json(path)
    if not isinstance(case, dict):
        raise ValueError(f"case definition {path} is not a JSON object")
    for key in ("case_id", "candidate_version", "expected_decision"):
        if key not in case:
            case[key] = root.name if key == "case_id" else ("accept" if key == "expected_decision" else "v0.0.0")
    case["_dir"] = str(root)
    baseline = case.get("baseline")
    case["_baseline_path"] = str(root / baseline) if baseline else None
    candidate_run = case.get("candidate_run")
    case["_candidate_run"] = str(root / candidate_run) if candidate_run else None
    prompt_overrides = case.get("prompt_overrides") or {}
    case["_prompt_dir"] = str(root / "prompts") if prompt_overrides or (root / "prompts").exists() else None
    return case


def list_cases(cases_root: Path | str = CASES_ROOT) -> list[dict[str, Any]]:
    root = Path(cases_root)
    if not root.exists():
        return []
    cases = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / CASE_FILE).exists():
            try:
                cases.append(load_case(path))
            except Exception:  # noqa: BLE001 - a broken case should not hide the others
                continue
    return cases


def write_case(case_dir: Path | str, case: dict[str, Any]) -> Path:
    """Write a case definition (used by the evidence-generation helper)."""

    root = Path(case_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / CASE_FILE
    write_json(path, case, indent=2, sort_keys=True)
    return path


def case_prompt_hashes(case_dir: Path | str) -> dict[str, str]:
    """Content hashes of a case's prompt-override files."""

    from aicwf.eval.baseline import prompt_hashes

    root = Path(case_dir)
    prompt_root = root / "prompts"
    if not prompt_root.exists():
        return {}
    return prompt_hashes(prompt_root)


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dotted keys (used to describe overrides)."""

    flat: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, dotted))
        else:
            flat[dotted] = value
    return flat


def config_overrides_applied(
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
    watch_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Dotted-path differences between the baseline and candidate config bundles."""

    flat_baseline = flatten_config(baseline_config)
    flat_candidate = flatten_config(candidate_config)
    changes: list[dict[str, Any]] = []
    for key in sorted(set(flat_baseline) | set(flat_candidate)):
        before = flat_baseline.get(key, "<absent>")
        after = flat_candidate.get(key, "<absent>")
        if before != after:
            changes.append(
                {
                    "key": key,
                    "baseline": before,
                    "candidate": after,
                    "watched": any(key == watch or key.endswith("." + watch) for watch in watch_keys),
                    "expected": any(key == watch or key.endswith("." + watch) for watch in watch_keys),
                }
            )
    changes.sort(key=lambda item: (not item["watched"], item["key"]))
    return changes
