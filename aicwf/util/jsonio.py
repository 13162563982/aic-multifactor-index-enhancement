"""Atomic JSON / text / YAML IO used across the workflow.

Writes go to a sibling temporary file, are flushed and fsynced, and only then
replaced, so an interrupted process never leaves a truncated artifact behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def write_text(path: Path | str, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: Path | str, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    text = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=sort_keys, default=str)
    return write_text(path, text + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path | str, obj: Any) -> Path:
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    return write_text(path, text)


def write_parquet_atomic(path: Path | str, frame: Any, **kwargs: Any) -> Path:
    """Write a pandas DataFrame via a temporary file and an atomic replace."""

    import pandas as pd  # local import keeps this module light for non-data tests

    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False, **kwargs)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def read_parquet(path: Path | str) -> Any:
    import pandas as pd

    return pd.read_parquet(path)
