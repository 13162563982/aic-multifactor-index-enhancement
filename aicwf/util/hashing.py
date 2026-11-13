"""Deterministic, canonical hashing helpers.

Every hash that ends up in a manifest, a state database row or a version entry is
produced here so that replay runs on Windows and Linux agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys, no whitespace and UTF-8 preserved.

    ``default=str`` keeps date/datetime/Path payloads hashable; callers that need
    strict types should convert them first.
    """

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def hash_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def hash_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: str, length: int = 8) -> str:
    return value[:length]
