"""Deterministic content hashes for artifacts.

Raw parquet bytes depend on the writer version, so determinism claims use a
canonical projection of the tabular content instead.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from aicwf.util.hashing import sha256_hex


def _normalise_value(value: Any, digits: int) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return round(value, digits)
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        if isinstance(value, pd.Timestamp) and pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "item"):
        try:
            return _normalise_value(value.item(), digits)
        except Exception:  # pragma: no cover - exotic numpy scalars
            pass
    return str(value)


def hash_dataframe(
    frame: pd.DataFrame,
    *,
    digits: int = 10,
    sort_by: Iterable[str] | None = None,
) -> str:
    """Canonical SHA-256 of a DataFrame independent of row/column order."""

    if frame is None:
        return sha256_hex("null")
    columns = sorted(frame.columns)
    ordered = frame.loc[:, columns]
    keys = list(sort_by) if sort_by is not None else columns
    keys = [k for k in keys if k in ordered.columns]
    if keys:
        ordered = ordered.sort_values(keys, kind="mergesort").reset_index(drop=True)
    rows = [
        {column: _normalise_value(value, digits) for column, value in zip(columns, row)}
        for row in ordered.itertuples(index=False, name=None)
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(payload)


def hash_json_artifact(path: Path | str, *, digits: int = 10) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return sha256_hex(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))


def hash_artifact(path: Path | str, *, digits: int = 10) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return hash_dataframe(pd.read_parquet(path), digits=digits)
    if suffix == ".json":
        return hash_json_artifact(path, digits=digits)
    return sha256_hex(path.read_bytes())


def hash_artifacts(paths: Iterable[Path | str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        result[path.name] = hash_artifact(path)
    return result
