"""Eval harness: frozen baseline A/B comparison and change-control decisions.

Public entry points:
    aicwf.eval.runner.run_eval(...)          -- compare a candidate run with the baseline
    aicwf.eval.runner.main(argv)             -- ``python -m aicwf.eval`` CLI
    aicwf.eval.baseline.refresh_baseline(...)-- freeze/refresh eval/baseline_v1.0.0.json
    aicwf.eval.fixtures.build_demo_run(...)  -- deterministic offline demo runs (tests/evidence)

The harness is what backs ``python -m aicwf eval`` through the pipeline/CLI.
"""

from aicwf.eval.baseline import (  # noqa: F401 - re-exported public surface
    build_baseline_from_run,
    load_baseline,
    refresh_baseline,
)
from aicwf.eval.replay import build_replay_evidence, load_replay_evidence  # noqa: F401
from aicwf.eval.runner import run_eval  # noqa: F401

__all__ = [
    "build_baseline_from_run",
    "build_replay_evidence",
    "load_baseline",
    "load_replay_evidence",
    "refresh_baseline",
    "run_eval",
]
