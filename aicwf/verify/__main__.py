"""``python -m aicwf.verify`` -- re-run independent verification on a run directory.

Examples::

    python -m aicwf.verify --run-dir outputs/runs/<run_id>
    python -m aicwf.verify --run-dir outputs/runs/<run_id> --json
"""

from __future__ import annotations

import argparse
import json
import sys

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
    parser.add_argument("--json", action="store_true", help="print the full verification JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verification = build_verification(
            run_dir=args.run_dir,
            as_of=args.as_of,
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
