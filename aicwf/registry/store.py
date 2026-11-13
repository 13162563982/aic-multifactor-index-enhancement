"""Append-only version registry storage and release bundles.

``registry/versions.json`` is the single history file.  It is *append only*:
every mutation goes through :func:`append_entry`, which assigns the sequence
number, links the entry to its predecessor with a digest chain
(``previous_entry_hash`` -> ``entry_hash``) and atomically rewrites the file.
Nothing is ever edited or deleted, so a rollback is another appended event and
the audit trail cannot be rewritten without breaking the chain.

``registry/releases/<version>/`` holds the approved configuration, prompts and
skills as a rollback bundle plus a ``bundle_manifest.json`` with a content hash
for every file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from aicwf.contracts import PROJECT_ROOT, relpath
from aicwf.util.hashing import hash_file, hash_json, hash_text
from aicwf.util.jsonio import read_json, read_yaml, write_json, write_text, write_yaml
from aicwf.util.timeutil import utc_now_iso

REGISTRY_SCHEMA_VERSION = "1.0.0"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "registry" / "versions.json"
DEFAULT_RELEASES_DIR = PROJECT_ROOT / "registry" / "releases"
BUNDLE_MANIFEST = "bundle_manifest.json"
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json")
_PROMPT_SUFFIXES = (".md", ".txt")


class RegistryError(Exception):
    """Base error for registry operations."""


class RegistryIntegrityError(RegistryError):
    """The append-only history or a release bundle has been tampered with."""


class RegistryConflict(RegistryError):
    """An append-only constraint would be violated (existing version, frozen bundle)."""


# ---------------------------------------------------------------------------
# History file
# ---------------------------------------------------------------------------


def history_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else DEFAULT_HISTORY_PATH


def releases_dir(path: Path | str | None = None) -> Path:
    return Path(path) if path else DEFAULT_RELEASES_DIR


def _empty_history() -> dict[str, Any]:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": []}


def ensure_history(path: Path | str | None = None) -> dict[str, Any]:
    """Create an empty history file if it does not exist and return it."""

    target = history_path(path)
    if not target.exists():
        write_json(target, _empty_history(), indent=2, sort_keys=True)
    return load_history(target)


def load_history(path: Path | str | None = None, *, validate: bool = True) -> dict[str, Any]:
    """Load the append-only history, optionally verifying its digest chain."""

    target = history_path(path)
    if not target.exists():
        return _empty_history()
    payload = read_json(target)
    if isinstance(payload, list):  # tolerate a bare list written by an earlier tool
        payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": payload}
    if not isinstance(payload, dict):
        raise RegistryIntegrityError(f"registry history {target} is not a JSON object")
    entries = payload.get("entries")
    if entries is None:
        payload["entries"] = []
    elif not isinstance(entries, list):
        raise RegistryIntegrityError(f"registry history {target} has a non-list 'entries' field")
    payload.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
    if validate:
        validate_history(payload)
    return payload


def validate_history(history: Mapping[str, Any]) -> None:
    """Verify sequence numbers and the entry digest chain (append-only integrity)."""

    entries = list(history.get("entries") or [])
    previous_hash = ""
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RegistryIntegrityError(f"history entry {index} is not an object")
        expected_sequence = index + 1
        if entry.get("sequence") != expected_sequence:
            raise RegistryIntegrityError(
                f"history entry {index} has sequence {entry.get('sequence')!r}, expected {expected_sequence}"
            )
        if str(entry.get("previous_entry_hash", "")) != previous_hash:
            raise RegistryIntegrityError(f"history entry {index} breaks the digest chain")
        recorded = str(entry.get("entry_hash", ""))
        recomputed = _entry_hash(entry)
        if recorded != recomputed:
            raise RegistryIntegrityError(
                f"history entry {index} (version {entry.get('version')!r}) was modified after being appended"
            )
        previous_hash = recorded


def _entry_hash(entry: Mapping[str, Any]) -> str:
    return hash_json({key: value for key, value in entry.items() if key != "entry_hash"})


def append_entry(
    entry: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Append one immutable entry to the history and return the stored record."""

    target = history_path(path)
    history = load_history(target) if target.exists() else _empty_history()
    entries: list[dict[str, Any]] = list(history.get("entries") or [])
    record = {str(key): value for key, value in entry.items()}
    record.pop("entry_hash", None)
    record.pop("sequence", None)
    record.pop("previous_entry_hash", None)
    record.setdefault("kind", "release")
    record.setdefault("decision", "accept")
    record["created_at_utc"] = str(record.get("created_at_utc") or created_at_utc or utc_now_iso())
    record["sequence"] = len(entries) + 1
    record["previous_entry_hash"] = str(entries[-1]["entry_hash"]) if entries else ""
    record["entry_hash"] = _entry_hash(record)
    entries.append(record)
    history["entries"] = entries
    write_json(target, history, indent=2, sort_keys=True)
    return record


def find_entry(
    version: str,
    *,
    path: Path | str | None = None,
    kind: str | None = None,
) -> dict[str, Any] | None:
    """Latest history entry for ``version`` (optionally restricted to a kind)."""

    history = load_history(path, validate=False)
    for entry in reversed(history.get("entries") or []):
        if str(entry.get("version")) != str(version):
            continue
        if kind is not None and str(entry.get("kind")) != kind:
            continue
        return entry
    return None


def versions_of_kind(kind: str, *, path: Path | str | None = None) -> list[dict[str, Any]]:
    history = load_history(path, validate=False)
    return [entry for entry in history.get("entries") or [] if str(entry.get("kind")) == kind]


def latest_release(path: Path | str | None = None) -> dict[str, Any] | None:
    history = load_history(path, validate=False)
    for entry in reversed(history.get("entries") or []):
        if str(entry.get("kind")) == "release" and str(entry.get("decision")) == "accept":
            return entry
    return None


def release_versions(path: Path | str | None = None) -> list[str]:
    return [str(entry.get("version")) for entry in versions_of_kind("release", path=path)]


def history_summary(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Compact view of the history for CLI/evidence output."""

    history = load_history(path, validate=False)
    summary = []
    for entry in history.get("entries") or []:
        summary.append(
            {
                "sequence": entry.get("sequence"),
                "kind": entry.get("kind"),
                "version": entry.get("version"),
                "created_at_utc": entry.get("created_at_utc"),
                "decision": entry.get("decision"),
                "previous_version": entry.get("previous_version"),
                "to_version": entry.get("to_version"),
                "from_version": entry.get("from_version"),
                "verification_verdict": entry.get("verification_verdict"),
                "entry_hash": str(entry.get("entry_hash", ""))[:12],
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Release bundles
# ---------------------------------------------------------------------------


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        result[Path(path).as_posix()] = hash_file(path)
    return result


def _copy_into(bundle_dir: Path, files: Mapping[str, Path], created_at_utc: str, version: str) -> dict[str, Any]:
    """Copy files into a fresh bundle directory and write its manifest."""

    if bundle_dir.exists():
        raise RegistryConflict(
            f"release bundle {bundle_dir} already exists; registry history is append-only"
        )
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}.", dir=bundle_dir.parent))
    try:
        hashes: dict[str, str] = {}
        for relative, source in sorted(files.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(source).read_bytes())
            hashes[relative.replace("\\", "/")] = hash_file(destination)
        config_hashes = {
            name: value for name, value in hashes.items() if name.startswith("config/")
        }
        prompt_hashes = {
            name: value for name, value in hashes.items() if name.startswith("prompts/")
        }
        skill_hashes = {
            name: value for name, value in hashes.items() if name.startswith("skills/")
        }
        manifest = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "version": version,
            "created_at_utc": created_at_utc,
            "files": dict(sorted(hashes.items())),
            "config_hashes": dict(sorted(config_hashes.items())),
            "prompt_hashes": dict(sorted(prompt_hashes.items())),
            "skill_hashes": dict(sorted(skill_hashes.items())),
            "config_hash": hash_json(dict(sorted(config_hashes.items()))),
            "bundle_hash": hash_json(dict(sorted(hashes.items()))),
        }
        write_json(staging / BUNDLE_MANIFEST, manifest, indent=2, sort_keys=True)
        if bundle_dir.exists():
            raise RegistryConflict(f"release bundle {bundle_dir} appeared concurrently")
        os.replace(staging, bundle_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _collect_config_files(config: Mapping[str, Any] | None, config_dir: Path | None) -> dict[str, Path]:
    if config_dir is not None:
        root = Path(config_dir)
        files: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _CONFIG_SUFFIXES:
                files[f"config/{path.relative_to(root).as_posix()}"] = path
        return files
    files = {}
    if config:
        staging = Path(tempfile.mkdtemp(prefix=".registry-config."))
        for name, value in sorted(config.items()):
            target = staging / f"{name}.yaml"
            write_yaml(target, value)
            files[f"config/{name}.yaml"] = target
    return files


def _collect_tree(root: Path, prefix: str, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not root.exists():
        return {}
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            files[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    return files


def create_bundle(
    version: str,
    *,
    config: Mapping[str, Any] | None = None,
    config_dir: Path | str | None = None,
    prompt_dir: Path | str | None = None,
    skills_dir: Path | str | None = None,
    destination: Path | str | None = None,
    created_at_utc: str | None = None,
    extra_files: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Create ``registry/releases/<version>/`` from config/prompts/skills and hash it."""

    bundle_dir = Path(destination) if destination else releases_dir() / version
    stamp = created_at_utc or utc_now_iso()
    files = _collect_config_files(config, Path(config_dir) if config_dir else None)
    files.update(_collect_tree(Path(prompt_dir) if prompt_dir else PROJECT_ROOT / "prompts", "prompts", _PROMPT_SUFFIXES))
    files.update(_collect_tree(Path(skills_dir) if skills_dir else PROJECT_ROOT / "skills", "skills", _PROMPT_SUFFIXES))
    for relative, source in (extra_files or {}).items():
        files[str(relative).replace("\\", "/")] = Path(source)
    manifest = _copy_into(bundle_dir, files, stamp, version)
    cleanup: list[Path] = []
    for path in files.values():
        if ".registry-config." in path.as_posix():
            cleanup.append(path.parent)
    for directory in set(cleanup):
        shutil.rmtree(directory, ignore_errors=True)
    return manifest


def load_bundle_manifest(bundle_dir: Path | str) -> dict[str, Any]:
    path = Path(bundle_dir) / BUNDLE_MANIFEST
    if not path.exists():
        raise RegistryIntegrityError(f"bundle manifest missing: {path}")
    manifest = read_json(path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise RegistryIntegrityError(f"bundle manifest {path} is malformed")
    return manifest


def verify_bundle(bundle_dir: Path | str, manifest: Mapping[str, Any] | None = None) -> list[str]:
    """Recompute every bundle file hash; returns a list of problems (empty = intact)."""

    root = Path(bundle_dir)
    problems: list[str] = []
    if not root.exists():
        return [f"release bundle {root} does not exist"]
    try:
        manifest = dict(manifest) if manifest is not None else load_bundle_manifest(root)
    except RegistryIntegrityError as exc:
        return [str(exc)]
    files = manifest.get("files") or {}
    if not files:
        return [f"bundle manifest for {root} lists no files"]
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.exists():
            problems.append(f"bundle file missing: {relative}")
            continue
        actual = hash_file(path)
        if actual != str(expected):
            problems.append(f"bundle file modified after release: {relative}")
    if not problems:
        recomputed = hash_json(dict(sorted((str(name), str(value)) for name, value in files.items())))
        if recomputed != str(manifest.get("bundle_hash")):
            problems.append("bundle_hash does not reproduce from the file hash manifest")
    return problems


def restore_bundle(
    bundle_dir: Path | str,
    *,
    targets: Mapping[str, Path | str] | None = None,
    project_root: Path | str = PROJECT_ROOT,
    manifest: Mapping[str, Any] | None = None,
) -> list[str]:
    """Restore a verified bundle into the working tree; returns restored paths."""

    root = Path(bundle_dir)
    problems = verify_bundle(root, manifest)
    if problems:
        raise RegistryIntegrityError("refusing to restore a corrupt bundle: " + "; ".join(problems))
    manifest = dict(manifest) if manifest is not None else load_bundle_manifest(root)
    root_project = Path(project_root)
    restored: list[str] = []
    for relative in sorted((manifest.get("files") or {})):
        rel = str(relative).replace("\\", "/")
        top = rel.split("/", 1)[0]
        destination_root = Path(targets[top]) if targets and top in targets else root_project / top
        destination = destination_root / rel.split("/", 1)[1] if "/" in rel else destination_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / rel).read_bytes())
        restored.append(relpath(destination, root_project) if _under(destination, root_project) else destination.as_posix())
    return restored


def _under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def bundle_dir_for(version: str, *, releases: Path | str | None = None) -> Path:
    return releases_dir(releases) / version


def entry_bundle_dir(entry: Mapping[str, Any], *, project_root: Path | str = PROJECT_ROOT) -> Path:
    """Resolve the bundle directory recorded on a release entry."""

    bundle = entry.get("bundle") or {}
    recorded = bundle.get("path")
    if recorded:
        path = Path(str(recorded))
        if not path.is_absolute():
            path = Path(project_root) / path
        return path
    return releases_dir() / str(entry.get("version"))


def prompt_bundle_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    return dict(manifest.get("prompt_hashes") or {})
