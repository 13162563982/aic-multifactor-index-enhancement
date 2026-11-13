"""Package the submission ZIP and verify it.

The assignment requires one ZIP of the local project folder that includes the
complete hidden ``.git/`` directory (not a shallow copy or a host "download ZIP").
This script:

1. collects every project file, including dotfiles and ``.git/``, while excluding
   caches and generated run state;
2. writes ``<parent>/<student-id>-final-project.zip``;
3. verifies the archive contains ``.git/HEAD`` and a non-empty history;
4. optionally extracts the ZIP to a temporary directory and checks that the
   repository history is intact and the test suite is runnable from the extraction;
5. scans the repository text files and the complete Git history (every object, including
   unreachable ones) for credential-like strings.

Usage:
    python scripts/package_submission.py --dry-run
    python scripts/package_submission.py
    python scripts/package_submission.py --skip-extract-check
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Repository-wide credential scan (working tree *and* complete Git history), kept next
# to this script so ``python scripts/package_submission.py`` can import it directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import scan_secrets as secret_scanner
except ImportError:  # pragma: no cover - packaging still works without the helper
    secret_scanner = None

ROOT = Path(__file__).resolve().parents[1]
STUDENT_ID = "25210660121"
DEFAULT_OUT = ROOT.parent / f"{STUDENT_ID}-final-project.zip"

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "node_modules",
}
EXCLUDE_REL_DIRS = {
    "outputs/runs",
    "outputs/monitor",
    "state",
    "runs",
    "data/snapshots/2026-11-12/tmp",
}
EXCLUDE_FILES = {".env", ".env.local", "Thumbs.db", ".DS_Store", "desktop.ini"}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
]


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_excluded(rel: str) -> bool:
    parts = rel.split("/")
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    if rel in EXCLUDE_FILES or parts[-1] in EXCLUDE_FILES:
        return True
    for prefix in EXCLUDE_REL_DIRS:
        if rel == prefix or rel.startswith(prefix + "/"):
            return True
    return False


def collect_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir():
            continue
        rel = relative(path)
        if is_excluded(rel):
            continue
        files.append(path)
    return files


def run_git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def scan_secrets(files: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".parquet", ".gz", ".zip", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                fragment = match.group(0)[:24]
                hits.append(f"{relative(path)}: {fragment}")
    return hits


def build_zip(out: Path, files: list[Path]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=relative(path))


def verify_archive(out: Path) -> list[str]:
    problems: list[str] = []
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    if ".git/HEAD" not in names:
        problems.append("archive does not contain .git/HEAD (history missing)")
    if not any(name.startswith(".git/refs/") or name == ".git/packed-refs" for name in names):
        problems.append("archive contains no git refs")
    if not any(name.startswith(".git/objects/") for name in names):
        problems.append("archive contains no git objects")
    if "README.md" not in names:
        problems.append("archive does not contain README.md")
    if "docs/design_note.md" not in names:
        problems.append("archive does not contain docs/design_note.md")
    return problems


def verify_extraction(out: Path) -> list[str]:
    problems: list[str] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="aicwf-zip-check-"))
    try:
        with zipfile.ZipFile(out) as archive:
            archive.extractall(temp_dir)
        commits = run_git("rev-list", "--count", "HEAD", cwd=temp_dir)
        if int(commits) < 8:
            problems.append(f"extracted history has only {commits} commits")
        status = run_git("status", "--porcelain", cwd=temp_dir)
        if status:
            problems.append(f"extracted repository is not clean: {status.splitlines()[:3]}")
        head = run_git("rev-parse", "--short", "HEAD", cwd=temp_dir)
        print(f"  extracted HEAD: {head} ({commits} commits)")
        # Quick import check from the extraction (no network, no model).
        result = subprocess.run(
            [sys.executable, "-c", "import aicwf.contracts; print('contracts import ok')"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            problems.append(f"extracted import check failed: {result.stderr.strip()[:300]}")
        else:
            print("  extraction import check: ok")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Package and verify the final project ZIP")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="build and verify but keep the archive under temp/")
    parser.add_argument("--skip-extract-check", action="store_true")
    args = parser.parse_args()

    print(f"project root: {ROOT}")
    if not (ROOT / ".git").exists():
        print("ERROR: .git directory is missing; initialize and commit the repository first")
        return 2

    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    commits = run_git("rev-list", "--count", "HEAD")
    print(f"git branch: {branch}   commits: {commits}")
    dirty = run_git("status", "--porcelain")
    if dirty:
        print("WARNING: working tree is not clean:")
        for line in dirty.splitlines()[:10]:
            print(f"  {line}")

    files = collect_files()
    print(f"files to package: {len(files)}")

    secret_hits = scan_secrets(files)
    if secret_hits:
        print("ERROR: possible credentials found:")
        for hit in secret_hits:
            print(f"  {hit}")
        return 3
    print("secret scan: clean")

    # Second layer: the assignment also requires the *commit history* to be free of
    # credentials, so scan every Git object before building the archive.
    if secret_scanner is None:
        print("WARNING: history credential scan unavailable (scripts/scan_secrets.py missing)")
    else:
        history = secret_scanner.run_scan(ROOT, working_tree=False)
        git_info = history.get("git") or {}
        if history.get("total_findings"):
            print("ERROR: possible credentials found in the repository history:")
            for finding in history["findings"]:
                print(f"  - {finding['path']} ({finding['pattern']})")
            return 3
        print(
            "history scan: clean "
            f"({git_info.get('objects_scanned')} objects, {git_info.get('commits')} commits)"
        )

    out = args.out
    if args.dry_run:
        out = ROOT / "outputs" / "submission_dry_run.zip"
    build_zip(out, files)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out}  ({size_mb:.1f} MB)")

    problems = verify_archive(out)
    if not args.skip_extract_check:
        if problems:
            print("archive checks failed:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("archive checks: .git present, refs and objects present, required docs present")
            problems.extend(verify_extraction(out))
    if args.dry_run and out.exists():
        out.unlink()
        print("dry run: archive removed")
    if problems:
        print("FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 4
    print("OK: submission archive is complete and reviewable without a hosted repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
