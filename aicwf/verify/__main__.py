"""``python -m aicwf.verify`` -- re-run independent verification on a run directory.

Examples::

    python -m aicwf.verify --run-dir outputs/runs/<run_id>
    python -m aicwf.verify --run-dir outputs/runs/<run_id> --json

By default the project config bundle (config/*.yaml) is loaded, so the weight /
active / industry bounds and the verification tolerances are the real published
ones.  Use ``--config-dir`` / ``--gates`` for isolated runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aicwf.util.jsonio import read_yaml
from aicwf.util.paths import load_config_bundle
from aicwf.verify.stage import (
    build_verification,
    is_blocking,
    render_verifier_note,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aicwf.verify", description="Independent verification of a run")
    parser.add_argument("--run-dir", required=True, help="run directory (outputs/runs/<run_id>)")
    parser.add_argument("--as-of", default=None, help="decision date (default: run_request.as_of)")
    parser.add_argument("--snapshot", default=None, help="frozen snapshot directory override")
    parser.add_argument("--config-dir", default=None, help="config directory override (default: project config/)")
    parser.add_argument("--gates", default=None, help="gates.yaml override path")
    parser.add_argument("--json", action="store_true", help="print the full verification JSON")
    return parser


def _load_config(config_dir: str | None) -> dict:
    if not config_dir:
        return load_config_bundle()
    directory = Path(config_dir)
    paths = {
        name: directory / f"{name}.yaml"
        for name in ("workflow", "strategy", "gates", "budgets", "model", "data")
    }
    return load_config_bundle(paths)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config_dir)
        gates = read_yaml(args.gates) if args.gates else None
        verification = build_verification(
            run_dir=args.run_dir,
            as_of=args.as_of,
            config=config,
            gates=gates,
            snapshot_dir=args.snapshot,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"verification failed to run: {type(exc).__name__}: {exc}\n")
        return 2
    if args.json:
        sys.stdout.write(json.dumps(verification, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(render_verifier_note(verification))
    return 1 if is_blocking(verification) else 0


if __name__ == "__main__":
    raise SystemExit(main())
