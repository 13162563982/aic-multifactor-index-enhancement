"""``python -m aicwf.registry`` -- inspect / promote / roll back versions.

Examples::

    python -m aicwf.registry versions
    python -m aicwf.registry verify
    python -m aicwf.registry promote --run-dir outputs/runs/<run_id> --version v1.1.0
    python -m aicwf.registry rollback --to v1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aicwf.contracts import PROJECT_ROOT
from aicwf.registry.promote import promote_run
from aicwf.registry.rollback import rollback
from aicwf.registry.stage import load_registry_paths, registry_status
from aicwf.registry.store import history_summary, load_history
from aicwf.util.jsonio import read_yaml, write_json


def _registry_paths() -> tuple[Path, Path]:
    gates_path = PROJECT_ROOT / "config" / "gates.yaml"
    gates = read_yaml(gates_path) if gates_path.exists() else {}
    return load_registry_paths({"gates": gates}, gates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aicwf.registry", description="Version registry operations")
    sub = parser.add_subparsers(dest="command", required=True)

    versions = sub.add_parser("versions", help="list the append-only version history")
    versions.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify", help="validate the history chain and every release bundle")
    verify.add_argument("--json", action="store_true")

    promote_cmd = sub.add_parser("promote", help="promote an approved, verified run")
    promote_cmd.add_argument("--run-dir", required=True)
    promote_cmd.add_argument("--version", required=True)
    promote_cmd.add_argument("--approval", default=None, help="approval.json path (default: run 11_approval)")
    promote_cmd.add_argument("--git-commit", default=None)
    promote_cmd.add_argument("--git-clean", choices=("yes", "no"), default=None)
    promote_cmd.add_argument("--rationale", default=None)
    promote_cmd.add_argument("--entry-out", default=None)

    rollback_cmd = sub.add_parser("rollback", help="restore a previous release bundle and append the event")
    rollback_cmd.add_argument("--to", required=True, dest="to_version")
    rollback_cmd.add_argument("--from", default=None, dest="from_version")
    rollback_cmd.add_argument("--dry-run", action="store_true")
    rollback_cmd.add_argument("--no-restore", action="store_true", help="record the event without copying files")
    rollback_cmd.add_argument("--rationale", default=None)
    rollback_cmd.add_argument("--entry-out", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history_path, releases_path = _registry_paths()
    try:
        if args.command == "versions":
            history = load_history(history_path, validate=False) if history_path.exists() else {"entries": []}
            if args.json:
                sys.stdout.write(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
            else:
                summary = history_summary(history_path) if history_path.exists() else []
                if not summary:
                    sys.stdout.write("registry is empty\n")
                for item in summary:
                    sys.stdout.write(
                        f"#{item['sequence']:02d} {item['kind']:8s} {item['version']:24s} "
                        f"decision={item['decision']:9s} verification={item['verification_verdict']} "
                        f"hash={item['entry_hash']}\n"
                    )
            return 0
        if args.command == "verify":
            report = registry_status(history_path=history_path, releases=releases_path)
            if args.json:
                sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            else:
                sys.stdout.write(
                    f"history: {report['entries']} entries, chain_valid={report['valid_chain']}, "
                    f"releases={len(report['releases'])}\n"
                )
                for release in report["releases"]:
                    sys.stdout.write(
                        f"  {release['version']}: bundle_intact={release['bundle_intact']}\n"
                    )
                for problem in report["problems"]:
                    sys.stderr.write(f"PROBLEM: {problem}\n")
            return 0 if report["valid_chain"] and not report["problems"] else 1
        if args.command == "promote":
            entry = promote_run(
                args.run_dir,
                version=args.version,
                approval=args.approval,
                git_commit=args.git_commit,
                git_clean=None if args.git_clean is None else args.git_clean == "yes",
                rationale=args.rationale,
                history_path=history_path,
                releases=releases_path,
                write_entry=args.entry_out,
            )
            sys.stdout.write(
                f"promoted {entry['version']} (sequence {entry['sequence']}, entry_hash {entry['entry_hash'][:12]})\n"
            )
            return 0
        if args.command == "rollback":
            result = rollback(
                to_version=args.to_version,
                from_version=args.from_version,
                history_path=history_path,
                releases=releases_path,
                restore=not args.no_restore,
                rationale=args.rationale,
                dry_run=args.dry_run,
                write_entry=args.entry_out,
            )
            smoke = result.get("smoke_eval") or {}
            sys.stdout.write(
                f"rollback {result['status']}: {result['from_version']} -> {result['to_version']} "
                f"(smoke={smoke.get('status')}, files_restored={len(result['restored_files'])})\n"
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"registry {args.command} failed: {type(exc).__name__}: {exc}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
