#!/usr/bin/env python3
"""Scan the project folder and the complete Git history for credentials.

The assignment requires that credentials, private data and other secrets are absent
from *both* the submitted files and the commit history. ``scripts/package_submission.py``
scans the current working tree; this script closes the second gap by scanning every Git
object (reachable and unreachable), every file name ever committed, and the Git metadata
files, in addition to the working tree.

What is scanned
---------------
* the complete project folder (all files a reviewer could receive, hidden files included,
  Python caches excluded), including ignored-but-packaged run outputs;
* every object in ``.git`` via ``git cat-file --batch-all-objects``: blobs (all file
  versions that ever existed, including deleted files and dangling objects), trees
  (file names) and commits (messages);
* the Git metadata text files (``.git/config``, reflogs, refs, hooks);
* every file name present in the history.

What is reported
----------------
Matches for credential-shaped strings: GitHub classic and fine-grained tokens, OpenAI-like
``sk-`` keys, AWS access keys, Google API keys, Slack tokens, private-key headers, bearer
tokens, URLs with embedded credentials, and non-empty secret environment assignments.
Values are masked in the output; the script never prints a full candidate secret.

Allowed template
----------------
``.env.example`` is a documented placeholder file whose secret values are empty (an empty
assignment never matches the non-empty-value patterns), so its name is allowed. If a value
were ever filled in, the content pattern would flag it.

Usage
-----
    python scripts/scan_secrets.py                 # scan this repository
    python scripts/scan_secrets.py --root <dir>    # scan another checkout / extraction
    python scripts/scan_secrets.py --json          # machine-readable output

Exit codes: 0 clean, 1 findings, 2 scan error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 60 * 1024 * 1024
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
}
ALLOWED_TEMPLATES = {".env.example"}

# (name, compiled pattern). Patterns are deliberately realistic so that normal prose,
# configuration keys and environment-variable *names* do not match.
PATTERNS = [
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("github_fine_grained_token", re.compile(rb"github_pat_[A-Za-z0-9_]{30,}")),
    ("openai_like_key", re.compile(rb"sk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{30,}")),
    ("slack_token", re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private_key_header", re.compile(rb"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----")),
    ("bearer_token", re.compile(rb"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("url_with_credentials", re.compile(rb"https?://[^/\s:@]{2,}:[^/\s@]{2,}@")),
    (
        "secret_env_assignment",
        re.compile(
            rb"(?m)^[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)"
            rb"[A-Z0-9_]*=\S{8,}\s*$"
        ),
    ),
    # Split so that this script's own source text never matches the pattern itself.
    (
        "aicwf_api_key_value",
        re.compile(b"AICWF_LLM_" + b"API_KEY=" + rb"\S+"),
    ),
    (
        "quoted_secret_assignment",
        re.compile(
            rb"(?i)(?:api[_-]?key|secret|password|passwd|token|credential)s?"
            rb"\s*[:=]\s*[\"'][^\"'\s]{16,}[\"']"
        ),
    ),
]

SENSITIVE_NAME = re.compile(
    r"(?i)(^|/)("
    r"\.env|\.env\.[A-Za-z0-9_.-]+|\.pem|\.p12|\.pfx|\.key|"
    r"id_rsa|id_ed25519|credentials\.(json|ya?ml)|secrets?\.(json|ya?ml)|"
    r"\.netrc|\.npmrc|\.pypirc|token\.json|auth\.json"
    r")$"
)


def mask(fragment: bytes) -> str:
    """Mask a matched value so a report never contains the candidate secret."""

    text = fragment.decode("utf-8", "replace")
    return text[:10] + f"...[{len(fragment)} bytes]"


def find_in_bytes(data: bytes) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(data):
            hits.append((name, mask(match.group(0))))
    return hits


def iter_working_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_BYTES:
                continue
            yield path


def scan_working_tree(root: Path) -> tuple[int, list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    files = 0
    for path in iter_working_files(root):
        try:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
        except OSError:
            continue
        files += 1
        for pattern_name, masked in find_in_bytes(data):
            findings.append(
                {"scope": "working_tree", "path": rel, "pattern": pattern_name, "match": masked}
            )
        if SENSITIVE_NAME.search(rel) and path.name not in ALLOWED_TEMPLATES:
            findings.append(
                {
                    "scope": "working_tree_name",
                    "path": rel,
                    "pattern": "sensitive_filename",
                    "match": path.name,
                }
            )
    return files, findings


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def scan_git_history(root: Path) -> dict[str, object]:
    """Scan every Git object, the history file names and the Git metadata files."""

    check = _run_git(root, "rev-parse", "--is-inside-work-tree")
    if check.returncode != 0 or check.stdout.strip() != "true":
        return {"available": False, "reason": check.stderr.strip()[:300]}

    head = _run_git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    commits = _run_git(root, "rev-list", "--count", "HEAD").stdout.strip()

    listing = _run_git(
        root, "cat-file", "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
    )
    if listing.returncode != 0:
        return {"available": False, "reason": listing.stderr.strip()[:300]}
    rows = [line.split() for line in listing.stdout.splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for _oid, object_type, _size in rows:
        counts[object_type] = counts.get(object_type, 0) + 1

    reachable: dict[str, str] = {}
    objects = _run_git(root, "rev-list", "--objects", "--all")
    for line in objects.stdout.splitlines():
        parts = line.split(" ", 1)
        if parts and parts[0] not in reachable:
            reachable[parts[0]] = parts[1] if len(parts) > 1 else "<commit>"

    findings: list[dict[str, str]] = []
    scanned = 0
    scanned_bytes = 0
    unreachable = 0
    process = subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    for object_id, object_type, size in rows:
        expected = int(size)
        if expected > MAX_BYTES:
            continue
        process.stdin.write((object_id + "\n").encode())
        process.stdin.flush()
        header = process.stdout.readline().split()
        if len(header) < 3:
            break
        length = int(header[2])
        data = process.stdout.read(length)
        process.stdout.read(1)
        scanned += 1
        scanned_bytes += length
        if object_id not in reachable:
            unreachable += 1
        hits = find_in_bytes(data)
        if hits:
            location = reachable.get(object_id, "<unreachable>")
            for pattern_name, masked in hits:
                findings.append(
                    {
                        "scope": "git_object",
                        "path": f"{location} ({object_id[:12]}, {object_type})",
                        "pattern": pattern_name,
                        "match": masked,
                    }
                )
    process.stdin.close()
    process.wait()

    names = _run_git(root, "log", "--all", "--name-only", "--format=")
    committed_names = sorted({line.strip() for line in names.stdout.splitlines() if line.strip()})
    for name in committed_names:
        if SENSITIVE_NAME.search(name) and Path(name).name not in ALLOWED_TEMPLATES:
            findings.append(
                {
                    "scope": "git_history_name",
                    "path": name,
                    "pattern": "sensitive_filename",
                    "match": Path(name).name,
                }
            )

    for rel in (
        ".git/config",
        ".git/logs/HEAD",
        ".git/COMMIT_EDITMSG",
        ".git/FETCH_HEAD",
        ".git/ORIG_HEAD",
        ".git/packed-refs",
        ".git/index",
    ):
        path = root / rel
        if not path.exists():
            continue
        data = path.read_bytes()
        for pattern_name, masked in find_in_bytes(data):
            findings.append(
                {
                    "scope": "git_metadata",
                    "path": rel,
                    "pattern": pattern_name,
                    "match": masked,
                }
            )
    hooks = root / ".git" / "hooks"
    if hooks.is_dir():
        for hook in sorted(hooks.iterdir()):
            if not hook.is_file():
                continue
            for pattern_name, masked in find_in_bytes(hook.read_bytes()):
                findings.append(
                    {
                        "scope": "git_metadata",
                        "path": f".git/hooks/{hook.name}",
                        "pattern": pattern_name,
                        "match": masked,
                    }
                )

    return {
        "available": True,
        "head": head,
        "commits": int(commits) if commits.isdigit() else 0,
        "object_counts": counts,
        "objects_scanned": scanned,
        "bytes_scanned": scanned_bytes,
        "unreachable_objects": unreachable,
        "committed_names": len(committed_names),
        "findings": findings,
    }


def run_scan(
    root: Path,
    *,
    working_tree: bool = True,
    git_history: bool = True,
) -> dict[str, object]:
    package_root = Path(root).resolve()
    summary: dict[str, object] = {"root": package_root.name}
    findings: list[dict[str, str]] = []

    files = 0
    if working_tree:
        files, tree_findings = scan_working_tree(package_root)
        findings.extend(tree_findings)
    summary["working_tree_files"] = files

    if git_history:
        summary["git"] = scan_git_history(package_root)
        git_info = summary["git"]
        if isinstance(git_info, dict) and git_info.get("available"):
            findings.extend(git_info.get("findings", []))

    summary["findings"] = findings
    summary["total_findings"] = len(findings)
    return summary


def format_report(summary: dict[str, object]) -> str:
    git_info = summary.get("git") if isinstance(summary.get("git"), dict) else {}
    available = bool(git_info.get("available")) if isinstance(git_info, dict) else False
    lines: list[str] = []
    lines.append("# Repository credential scan")
    lines.append("")
    lines.append(
        "Generated by `python scripts/scan_secrets.py`; re-run the command to reproduce."
    )
    lines.append("")
    if available:
        lines.append(f"- Git commit: `{git_info.get('head')}` ({git_info.get('commits')} commits)")
        counts = git_info.get("object_counts") or {}
        counts_text = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
        lines.append(f"- Git objects scanned: {git_info.get('objects_scanned')} ({counts_text})")
        lines.append(
            f"- Git object bytes scanned: {git_info.get('bytes_scanned'):,} "
            f"(unreachable objects: {git_info.get('unreachable_objects')})"
        )
        lines.append(
            f"- File names ever committed: {git_info.get('committed_names')}"
        )
    else:
        lines.append("- Git history scan: not available in this location")
    lines.append(
        f"- Working-tree files scanned: {summary.get('working_tree_files')} "
        "(all files except `.git/` and Python caches, including ignored-but-packaged outputs)"
    )
    lines.append(
        "- Patterns: GitHub tokens (`gh*_`, `github_pat_`), OpenAI-style `sk-` keys, "
        "AWS access keys, Google API keys, Slack tokens, private-key headers, bearer "
        "tokens, URLs with embedded credentials, non-empty secret environment assignments, "
        "and `AICWF_LLM_API_KEY` with a non-empty value"
    )
    lines.append(
        "- Allowed template: `.env.example` (placeholder with empty values only); any "
        "non-empty value there would be flagged by the content patterns"
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- Findings: {summary.get('total_findings')}")
    findings = summary.get("findings") or []
    if findings:
        lines.append("")
        for finding in findings:  # type: ignore[union-attr]
            lines.append(
                f"- `{finding['path']}` ({finding['scope']}, {finding['pattern']}): "
                f"{finding['match']}"
            )
        lines.append("")
        lines.append(
            "VERDICT: REVIEW REQUIRED - inspect the findings above before publishing."
        )
    else:
        lines.append("")
        lines.append(
            "VERDICT: CLEAN - no API keys, tokens, private keys, passwords or other "
            "credentials were found in the submitted files or the commit history."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a project and its Git history for credentials")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root to scan")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2
    summary = run_scan(root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_report(summary), end="")
    return 0 if summary["total_findings"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
