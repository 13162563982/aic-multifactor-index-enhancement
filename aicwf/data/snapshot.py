"""Frozen snapshot layout: loading, writing, manifest and deterministic hashing.

The on-disk layout follows ``docs/artifact_contract.md`` section 1::

    data/snapshots/<as_of>/
        prices.parquet
        fundamentals.parquet
        universe.parquet
        benchmark.parquet
        calendar.parquet
        manifest.json

Design rules (from the artifact contract):

* Dates inside parquet/JSON artifacts are ISO ``YYYY-MM-DD`` strings.
* Tabular artifacts are sorted by ``(date, code)`` unless stated otherwise.
* Deterministic content hashes come from :mod:`aicwf.util.artifact_hash` and are
  never computed from raw parquet bytes (parquet embeds writer metadata).
* ``manifest.snapshot_hash`` is the canonical hash of the per-file content
  hashes stored in ``manifest.files``.

The module is deliberately network-free so the deterministic mini fixture can
be produced offline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from aicwf.contracts import PROJECT_ROOT, ValidationError, parse_date
from aicwf.util.artifact_hash import hash_artifact
from aicwf.util.hashing import hash_file, hash_json
from aicwf.util.jsonio import (
    ensure_dir,
    read_json,
    read_parquet,
    write_json,
    write_parquet_atomic,
)

SCHEMA_VERSION = "1.1.0"
MANIFEST_FILE = "manifest.json"

PRICES_FILE = "prices.parquet"
FUNDAMENTALS_FILE = "fundamentals.parquet"
UNIVERSE_FILE = "universe.parquet"
BENCHMARK_FILE = "benchmark.parquet"
CALENDAR_FILE = "calendar.parquet"


@dataclass(frozen=True)
class FrameSpec:
    """Structural description of one snapshot table."""

    name: str
    filename: str
    columns: tuple[str, ...]
    sort_by: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    bool_columns: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = ()
    range_date_column: str | None = None


FRAME_SPECS: dict[str, FrameSpec] = {
    "prices": FrameSpec(
        name="prices",
        filename=PRICES_FILE,
        columns=(
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "board",
        ),
        sort_by=("date", "code"),
        date_columns=("date",),
        text_columns=("code", "board"),
        range_date_column="date",
    ),
    "fundamentals": FrameSpec(
        name="fundamentals",
        filename=FUNDAMENTALS_FILE,
        columns=(
            "code",
            "report_date",
            "notice_date",
            "eps",
            "bps",
            "roe",
            "gross_margin",
            "revenue",
            "revenue_yoy",
            "net_profit",
            "net_profit_yoy",
            "cfo_per_share",
            "notice_date_estimated",
        ),
        sort_by=("code", "report_date"),
        date_columns=("report_date", "notice_date"),
        bool_columns=("notice_date_estimated",),
        text_columns=("code",),
        range_date_column="report_date",
    ),
    "universe": FrameSpec(
        name="universe",
        filename=UNIVERSE_FILE,
        columns=(
            "code",
            "name",
            "industry",
            "weight",
            "free_cap",
            "total_cap",
            "listed_date",
            "is_st",
            "board",
        ),
        sort_by=("code",),
        date_columns=("listed_date",),
        bool_columns=("is_st",),
        text_columns=("code", "name", "industry", "board"),
        range_date_column="listed_date",
    ),
    "benchmark": FrameSpec(
        name="benchmark",
        filename=BENCHMARK_FILE,
        columns=("date", "close", "ret"),
        sort_by=("date",),
        date_columns=("date",),
        range_date_column="date",
    ),
    "calendar": FrameSpec(
        name="calendar",
        filename=CALENDAR_FILE,
        columns=("date",),
        sort_by=("date",),
        date_columns=("date",),
        range_date_column="date",
    ),
}

FRAME_ORDER: tuple[str, ...] = tuple(FRAME_SPECS)
SNAPSHOT_FILES: tuple[str, ...] = tuple(spec.filename for spec in FRAME_SPECS.values())


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _date_str_series(series: pd.Series) -> pd.Series:
    """Coerce a column to ISO date strings (``NaT``/None -> ``None``)."""

    parsed = pd.to_datetime(series, errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%d")
    out = formatted.astype(object)
    return out.where(parsed.notna(), None)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)

    def _convert(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "t"}
        try:
            return bool(value)
        except Exception:  # pragma: no cover - exotic scalar
            return False

    return series.map(_convert).astype(bool)


def _text_series(series: pd.Series) -> pd.Series:
    def _convert(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if value is pd.NA:
            return None
        return str(value)

    return series.map(_convert).astype(object)


def require_columns(name: str, frame: pd.DataFrame) -> None:
    """Raise :class:`ValidationError` when required columns are missing."""

    spec = FRAME_SPECS[name]
    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise ValidationError(
            f"snapshot frame {name!r} is missing required columns {missing}; "
            f"expected {list(spec.columns)}"
        )


def normalise_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` projected on the contract columns with canonical dtypes."""

    spec = FRAME_SPECS[name]
    require_columns(name, frame)
    out = frame.loc[:, list(spec.columns)].copy()
    for column in spec.date_columns:
        out[column] = _date_str_series(out[column])
    for column in spec.bool_columns:
        out[column] = _bool_series(out[column])
    for column in spec.text_columns:
        out[column] = _text_series(out[column])
    remaining = [
        column
        for column in spec.columns
        if column not in spec.date_columns
        and column not in spec.bool_columns
        and column not in spec.text_columns
    ]
    for column in remaining:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if spec.sort_by:
        out = out.sort_values(list(spec.sort_by), kind="mergesort")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def file_hashes(
    snapshot_dir: Path | str,
    files: Sequence[str] | Iterable[str] | None = None,
) -> dict[str, str]:
    """Raw SHA-256 of every snapshot file (freeze/tamper evidence).

    These are the hashes stored in ``manifest.files``: verification recomputes
    them from the bytes on disk to detect any post-freeze modification.
    """

    snapshot_dir = Path(snapshot_dir)
    names = list(files) if files is not None else list(SNAPSHOT_FILES)
    hashes: dict[str, str] = {}
    for name in names:
        path = snapshot_dir / name
        if not path.exists():
            raise FileNotFoundError(f"snapshot file missing: {path}")
        hashes[name] = hash_file(path)
    return hashes


def content_hashes(
    snapshot_dir: Path | str,
    files: Sequence[str] | Iterable[str] | None = None,
) -> dict[str, str]:
    """Deterministic content hashes (``aicwf.util.artifact_hash``).

    Stored in ``manifest.content_hashes`` so two rebuilds with different
    parquet writers can still be compared for content equality.
    """

    snapshot_dir = Path(snapshot_dir)
    names = list(files) if files is not None else list(SNAPSHOT_FILES)
    hashes: dict[str, str] = {}
    for name in names:
        path = snapshot_dir / name
        if not path.exists():
            raise FileNotFoundError(f"snapshot file missing: {path}")
        hashes[name] = hash_artifact(path)
    return hashes


def compute_snapshot_hash(hashes: Mapping[str, str]) -> str:
    """Canonical hash of the per-file hashes (``manifest.snapshot_hash``).

    Deterministic for a fixed set of file hashes and reproducible by anyone:
    ``hash_json(manifest["files"])`` must equal ``manifest["snapshot_hash"]``.
    """

    return hash_json({str(name): str(value) for name, value in hashes.items()})


def _date_range(frame: pd.DataFrame, column: str | None) -> list[str] | None:
    if not column or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return [str(values.min()), str(values.max())]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_snapshot(
    snapshot_dir: Path | str,
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of: str | dt.date | dt.datetime,
    schema_version: str = SCHEMA_VERSION,
    source_urls: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
    created_at_utc: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra_json_files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a complete, hashed snapshot directory and return its manifest.

    ``frames`` must contain every key in :data:`FRAME_ORDER`. Frames are
    normalised (column projection, ISO date strings, dtype coercion, sorting)
    before being written, so callers may pass loosely typed frames.
    ``extra_json_files`` are written before hashing (e.g. ``coverage_report.json``
    for the real snapshot) and therefore covered by ``snapshot_hash``.
    """

    snapshot_dir = ensure_dir(snapshot_dir)
    missing = [name for name in FRAME_ORDER if name not in frames]
    if missing:
        raise ValidationError(f"snapshot writer is missing frames: {missing}")
    as_of_date = parse_date(str(as_of))

    row_counts: dict[str, int] = {}
    date_ranges: dict[str, list[str]] = {}
    for name in FRAME_ORDER:
        spec = FRAME_SPECS[name]
        normalised = normalise_frame(name, frames[name])
        write_parquet_atomic(snapshot_dir / spec.filename, normalised)
        row_counts[name] = int(len(normalised))
        date_range = _date_range(normalised, spec.range_date_column)
        if date_range is not None:
            date_ranges[name] = date_range

    for extra_name, extra_obj in (extra_json_files or {}).items():
        write_json(snapshot_dir / extra_name, extra_obj, indent=2, sort_keys=True)

    # Hash every file that belongs to the snapshot except the manifest itself:
    # ``files`` is a flat filename -> raw sha256 mapping and snapshot_hash is the
    # canonical hash of that mapping (reproducible by verification).
    hashed_names = sorted(
        path.name
        for path in snapshot_dir.iterdir()
        if path.is_file() and path.name != MANIFEST_FILE and not path.name.endswith(".tmp")
    )
    hashes = file_hashes(snapshot_dir, files=hashed_names)
    content = content_hashes(snapshot_dir, files=hashed_names)
    snapshot_hash = compute_snapshot_hash(hashes)
    manifest: dict[str, Any] = {
        "schema_version": str(schema_version),
        "as_of": as_of_date.isoformat(),
        "created_at_utc": created_at_utc or _utc_now_iso(),
        "source_urls": dict(source_urls or {}),
        "row_counts": row_counts,
        "date_ranges": date_ranges,
        "files": hashes,
        "content_hashes": content,
        "snapshot_hash": snapshot_hash,
        "notes": list(notes or []),
    }
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    write_json(snapshot_dir / MANIFEST_FILE, manifest)
    return manifest


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Loading / verification
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """A frozen snapshot loaded from disk."""

    path: Path
    manifest: dict[str, Any]
    frames: dict[str, pd.DataFrame]

    @property
    def as_of(self) -> dt.date:
        return parse_date(self.manifest["as_of"])

    @property
    def snapshot_hash(self) -> str:
        return str(self.manifest.get("snapshot_hash", ""))

    def frame(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            raise KeyError(f"snapshot has no frame {name!r}; available: {sorted(self.frames)}")
        return self.frames[name]

    @property
    def prices(self) -> pd.DataFrame:
        return self.frames["prices"]

    @property
    def fundamentals(self) -> pd.DataFrame:
        return self.frames["fundamentals"]

    @property
    def universe(self) -> pd.DataFrame:
        return self.frames["universe"]

    @property
    def benchmark(self) -> pd.DataFrame:
        return self.frames["benchmark"]

    @property
    def calendar(self) -> pd.DataFrame:
        return self.frames["calendar"]

    def declared_hashes(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.manifest.get("files", {}).items()}

    def declared_content_hashes(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.manifest.get("content_hashes", {}).items()}

    def manifest_hash_consistent(self) -> bool:
        declared = self.declared_hashes()
        if not declared:
            return False
        return compute_snapshot_hash(declared) == self.snapshot_hash


def load_snapshot(snapshot_dir: Path | str) -> Snapshot:
    """Load a snapshot directory, validating the manifest and required columns."""

    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValidationError(f"snapshot manifest is not an object: {manifest_path}")
    for key in ("as_of", "snapshot_hash", "files"):
        if key not in manifest:
            raise ValidationError(f"snapshot manifest is missing {key!r}: {manifest_path}")
    frames: dict[str, pd.DataFrame] = {}
    for spec in FRAME_SPECS.values():
        frame_path = snapshot_dir / spec.filename
        if not frame_path.exists():
            raise FileNotFoundError(f"snapshot file missing: {frame_path}")
        frame = read_parquet(frame_path)
        require_columns(spec.name, frame)
        frames[spec.name] = frame
    return Snapshot(path=snapshot_dir, manifest=manifest, frames=frames)


def verify_snapshot(
    snapshot_dir: Path | str,
    *,
    deep: bool = True,
) -> dict[str, Any]:
    """Check a snapshot directory against its manifest.

    ``deep=False`` only recomputes ``snapshot_hash`` from the hashes recorded in
    ``manifest.files`` (cheap). ``deep=True`` re-hashes every file on disk.
    """

    snapshot_dir = Path(snapshot_dir)
    manifest = read_json(snapshot_dir / MANIFEST_FILE)
    declared = {str(k): str(v) for k, v in manifest.get("files", {}).items()}
    result: dict[str, Any] = {
        "snapshot_dir": str(snapshot_dir),
        "declared_snapshot_hash": manifest.get("snapshot_hash"),
        "manifest_hash_consistent": compute_snapshot_hash(declared) == manifest.get("snapshot_hash"),
        "file_mismatches": {},
        "missing_files": [],
        "checked_files": sorted(declared),
        "ok": False,
    }
    if not deep:
        result["ok"] = bool(result["manifest_hash_consistent"])
        return result
    try:
        actual = file_hashes(snapshot_dir, files=sorted(declared))
    except FileNotFoundError as exc:
        result["missing_files"] = [str(exc)]
        return result
    recomputed_snapshot_hash = compute_snapshot_hash(actual)
    mismatches = {
        name: {"declared": declared[name], "actual": actual[name]}
        for name in sorted(declared)
        if declared[name] != actual.get(name)
    }
    content_mismatches: dict[str, dict[str, str]] = {}
    declared_content = {str(k): str(v) for k, v in manifest.get("content_hashes", {}).items()}
    if declared_content:
        actual_content = content_hashes(snapshot_dir, files=sorted(declared_content))
        content_mismatches = {
            name: {"declared": declared_content[name], "actual": actual_content[name]}
            for name in sorted(declared_content)
            if declared_content[name] != actual_content.get(name)
        }
    result.update(
        {
            "recomputed_snapshot_hash": recomputed_snapshot_hash,
            "file_mismatches": mismatches,
            "content_mismatches": content_mismatches,
            "ok": (
                not mismatches
                and not content_mismatches
                and recomputed_snapshot_hash == manifest.get("snapshot_hash")
                and result["manifest_hash_consistent"]
            ),
        }
    )
    return result


def snapshot_dir_for(
    as_of: str | dt.date | dt.datetime,
    root: str | Path = "data/snapshots",
) -> Path:
    """``<root>/<as_of>`` resolved against the project root when relative."""

    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    return root_path / parse_date(str(as_of)).isoformat()


def default_snapshot_root() -> Path:
    return PROJECT_ROOT / "data" / "snapshots"
