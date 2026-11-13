"""SQLite-backed persistent state for the AICWF controller.

The controller is the only component that opens ``state/runs.db``. Every other
subsystem talks to the run state through this module (or through the files the
controller writes next to it), so there is exactly one writer.

Design notes
------------
* The database is a durable checkpoint: a run may be killed at any point
  (including ``os._exit``) and resumed; committed rows survive, uncommitted work
  is simply repeated by the stage that was interrupted.
* SQLite is opened in autocommit mode (``isolation_level=None``) plus explicit
  ``BEGIN IMMEDIATE`` transactions for multi-row mutations (stage completion,
  failure recording, budget snapshots). This keeps crash recovery simple: every
  completed statement is durable.
* Timestamps are UTC ISO-8601 and live only in state/events, never in numeric
  artifacts. The state schema itself never feeds a content hash.
* ``export_run`` / ``export_state`` produce the JSON used for
  ``state_snapshot.json`` and the ``aicwf state-export`` command.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from aicwf.contracts import PROJECT_ROOT, RUN_STATUSES, STEP_STATUSES, WorkflowError
from aicwf.util.jsonio import ensure_dir, write_json
from aicwf.util.logging import EventLog
from aicwf.util.timeutil import utc_now_iso

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "StateError",
    "RunRecord",
    "StepRecord",
    "AttemptRecord",
    "ArtifactRecord",
    "BudgetRecord",
    "ApprovalRecord",
    "EventRecord",
    "RunEventSink",
    "StateStore",
    "default_db_path",
]


class StateError(WorkflowError):
    """The persistent state store rejected an operation."""


def default_db_path(project_root: Path | str = PROJECT_ROOT) -> Path:
    """Return the canonical SQLite location (``state/runs.db``, gitignored)."""

    return Path(project_root) / "state" / "runs.db"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return utc_now_iso()


def _as_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {key: row[key] for key in row.keys()}
    return dict(row)


_STEP_STATUS_SQL = ", ".join(f"'{status}'" for status in sorted(STEP_STATUSES))
_RUN_STATUS_SQL = ", ".join(f"'{status}'" for status in sorted(RUN_STATUSES))

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_RUN_STATUS_SQL})),
    run_dir TEXT NOT NULL DEFAULT '',
    llm_mode TEXT NOT NULL DEFAULT 'replay',
    seed INTEGER NOT NULL DEFAULT 0,
    workflow_version TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    snapshot_hash TEXT NOT NULL DEFAULT '',
    snapshot_dir TEXT NOT NULL DEFAULT '',
    trigger TEXT NOT NULL DEFAULT '',
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    current_stage TEXT,
    error TEXT,
    summary_json TEXT NOT NULL DEFAULT '{{}}'
);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_STEP_STATUS_SQL})),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 0,
    input_hash TEXT,
    output_hash TEXT,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    duration_s REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    error_type TEXT,
    summary_json TEXT NOT NULL DEFAULT '{{}}',
    skip_count INTEGER NOT NULL DEFAULT 0,
    last_disposition TEXT NOT NULL DEFAULT 'pending',
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, stage),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attempts (
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    duration_s REAL NOT NULL DEFAULT 0.0,
    error_type TEXT,
    error TEXT,
    PRIMARY KEY (run_id, stage, attempt),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    logical_name TEXT NOT NULL,
    relpath TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, logical_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    kind TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    payload_json TEXT NOT NULL DEFAULT '{{}}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS budgets (
    run_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    limit_value REAL NOT NULL,
    used_value REAL NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, metric),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    action TEXT NOT NULL,
    decision TEXT NOT NULL,
    approver TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'run',
    artifact_hashes_json TEXT NOT NULL DEFAULT '{{}}',
    note TEXT NOT NULL DEFAULT '',
    decided_at_utc TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_run_stage ON attempts(run_id, stage);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id, id);
"""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One workflow run (one research cycle for one ``as_of`` date)."""

    run_id: str
    as_of: str
    status: str = "pending"
    run_dir: str = ""
    llm_mode: str = "replay"
    seed: int = 0
    workflow_version: str = ""
    config_hash: str = ""
    snapshot_hash: str = ""
    snapshot_dir: str = ""
    trigger: str = ""
    created_at_utc: str = ""
    updated_at_utc: str = ""
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    current_stage: str | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "as_of": self.as_of,
            "status": self.status,
            "run_dir": self.run_dir,
            "llm_mode": self.llm_mode,
            "seed": self.seed,
            "workflow_version": self.workflow_version,
            "config_hash": self.config_hash,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_dir": self.snapshot_dir,
            "trigger": self.trigger,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "current_stage": self.current_stage,
            "error": self.error,
            "summary": dict(self.summary),
        }
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "RunRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            as_of=data["as_of"],
            status=data["status"],
            run_dir=data.get("run_dir") or "",
            llm_mode=data.get("llm_mode") or "replay",
            seed=int(data.get("seed") or 0),
            workflow_version=data.get("workflow_version") or "",
            config_hash=data.get("config_hash") or "",
            snapshot_hash=data.get("snapshot_hash") or "",
            snapshot_dir=data.get("snapshot_dir") or "",
            trigger=data.get("trigger") or "",
            created_at_utc=data.get("created_at_utc") or "",
            updated_at_utc=data.get("updated_at_utc") or "",
            started_at_utc=data.get("started_at_utc"),
            finished_at_utc=data.get("finished_at_utc"),
            current_stage=data.get("current_stage"),
            error=data.get("error"),
            summary=_loads(data.get("summary_json"), {}) or {},
        )


@dataclass
class StepRecord:
    """One DAG node's execution state inside a run."""

    run_id: str
    stage: str
    status: str = "pending"
    attempt: int = 0
    max_attempts: int = 0
    input_hash: str | None = None
    output_hash: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    duration_s: float = 0.0
    error: str | None = None
    error_type: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    skip_count: int = 0
    last_disposition: str = "pending"
    updated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_s": self.duration_s,
            "error": self.error,
            "error_type": self.error_type,
            "summary": dict(self.summary),
            "skip_count": self.skip_count,
            "last_disposition": self.last_disposition,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "StepRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            stage=data["stage"],
            status=data["status"],
            attempt=int(data.get("attempt") or 0),
            max_attempts=int(data.get("max_attempts") or 0),
            input_hash=data.get("input_hash"),
            output_hash=data.get("output_hash"),
            started_at_utc=data.get("started_at_utc"),
            finished_at_utc=data.get("finished_at_utc"),
            duration_s=float(data.get("duration_s") or 0.0),
            error=data.get("error"),
            error_type=data.get("error_type"),
            summary=_loads(data.get("summary_json"), {}) or {},
            skip_count=int(data.get("skip_count") or 0),
            last_disposition=data.get("last_disposition") or "pending",
            updated_at_utc=data.get("updated_at_utc") or "",
        )


@dataclass
class AttemptRecord:
    """One stage attempt (retries and fault-injected crashes get their own rows)."""

    run_id: str
    stage: str
    attempt: int
    status: str = "running"
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    duration_s: float = 0.0
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "attempt": self.attempt,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "error": self.error,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "AttemptRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            stage=data["stage"],
            attempt=int(data["attempt"]),
            status=data.get("status") or "running",
            started_at_utc=data.get("started_at_utc"),
            finished_at_utc=data.get("finished_at_utc"),
            duration_s=float(data.get("duration_s") or 0.0),
            error_type=data.get("error_type"),
            error=data.get("error"),
        )


@dataclass
class ArtifactRecord:
    """A stage output registered with its deterministic content hash."""

    run_id: str
    stage: str
    logical_name: str
    relpath: str
    content_hash: str
    size_bytes: int = 0
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "logical_name": self.logical_name,
            "relpath": self.relpath,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "ArtifactRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            stage=data["stage"],
            logical_name=data["logical_name"],
            relpath=data["relpath"],
            content_hash=data["content_hash"],
            size_bytes=int(data.get("size_bytes") or 0),
            created_at_utc=data.get("created_at_utc") or "",
        )


@dataclass
class BudgetRecord:
    """Persisted budget usage for one metric of one run."""

    run_id: str
    metric: str
    limit_value: float
    used_value: float
    updated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "limit_value": self.limit_value,
            "used_value": self.used_value,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "BudgetRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            metric=data["metric"],
            limit_value=float(data["limit_value"]),
            used_value=float(data["used_value"]),
            updated_at_utc=data.get("updated_at_utc") or "",
        )


@dataclass
class ApprovalRecord:
    """A human approval decision, bound to the artifact hashes it approved."""

    run_id: str
    action: str
    decision: str = "approved"
    approver: str = ""
    scope: str = "run"
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    note: str = ""
    decided_at_utc: str = ""
    approval_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.approval_id,
            "run_id": self.run_id,
            "action": self.action,
            "decision": self.decision,
            "approver": self.approver,
            "scope": self.scope,
            "artifact_hashes": dict(self.artifact_hashes),
            "note": self.note,
            "decided_at_utc": self.decided_at_utc,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "ApprovalRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            action=data["action"],
            decision=data["decision"],
            approver=data.get("approver") or "",
            scope=data.get("scope") or "run",
            artifact_hashes=_loads(data.get("artifact_hashes_json"), {}) or {},
            note=data.get("note") or "",
            decided_at_utc=data.get("decided_at_utc") or "",
            approval_id=data.get("id"),
        )


@dataclass
class EventRecord:
    """One structured event; mirrored to ``events.jsonl`` for human review."""

    run_id: str
    ts_utc: str
    kind: str
    level: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "run_id": self.run_id,
            "ts_utc": self.ts_utc,
            "kind": self.kind,
            "level": self.level,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, Any]) -> "EventRecord":
        data = _as_dict(row)
        return cls(
            run_id=data["run_id"],
            ts_utc=data["ts_utc"],
            kind=data["kind"],
            level=data.get("level") or "info",
            payload=_loads(data.get("payload_json"), {}) or {},
            event_id=data.get("id"),
        )


# ---------------------------------------------------------------------------
# Event sink
# ---------------------------------------------------------------------------


class RunEventSink:
    """Dual event trail: SQLite rows for queries, JSONL for crash-proof review.

    The object satisfies the ``ctx.events.emit(kind, *, level, run_id, payload)``
    convention used by :class:`aicwf.contracts.RunContext`.
    """

    def __init__(self, state: "StateStore", log_path: Path | str, run_id: str, *, echo: bool = False) -> None:
        self.state = state
        self.run_id = run_id
        self.log = EventLog(log_path, echo=echo)

    def emit(
        self,
        kind: str,
        *,
        level: str = "info",
        run_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_run = run_id or self.run_id
        body = dict(payload or {})
        record = self.state.add_event(target_run, kind, level=level, payload=body)
        self.log.emit(kind=kind, level=level, run_id=target_run, payload=body)
        return record.to_dict()


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------

_RUN_UPDATABLE = frozenset(
    {
        "status",
        "run_dir",
        "llm_mode",
        "seed",
        "workflow_version",
        "config_hash",
        "snapshot_hash",
        "snapshot_dir",
        "trigger",
        "updated_at_utc",
        "started_at_utc",
        "finished_at_utc",
        "current_stage",
        "error",
        "summary",
    }
)


class StateStore:
    """SQLite state store for runs, steps, attempts, artifacts, budgets, approvals.

    Parameters
    ----------
    db_path:
        SQLite file location; defaults to ``<project_root>/state/runs.db``.
    project_root:
        Project root used only to resolve the default database path.
    """

    def __init__(self, db_path: Path | str | None = None, *, project_root: Path | str = PROJECT_ROOT) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path(project_root)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.initialize()

    # -- connection management ----------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                ensure_dir(self.db_path.parent)
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    isolation_level=None,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                self._conn = conn
            return self._conn

    @property
    def path(self) -> Path:
        return self.db_path

    def initialize(self) -> None:
        """Create the schema if needed (idempotent, safe on every open)."""

        with self._lock:
            conn = self.connection
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit write transaction; rolls back on any exception."""

        with self._lock:
            conn = self.connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    # -- runs ----------------------------------------------------------
    def create_run(self, record: RunRecord) -> RunRecord:
        if record.status not in RUN_STATUSES:
            raise StateError(f"invalid run status {record.status!r}")
        now = record.created_at_utc or _now()
        record.created_at_utc = now
        record.updated_at_utc = record.updated_at_utc or now
        with self._lock:
            try:
                self.connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, as_of, status, run_dir, llm_mode, seed, workflow_version,
                        config_hash, snapshot_hash, snapshot_dir, trigger, created_at_utc,
                        updated_at_utc, started_at_utc, finished_at_utc, current_stage,
                        error, summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.as_of,
                        record.status,
                        record.run_dir,
                        record.llm_mode,
                        int(record.seed),
                        record.workflow_version,
                        record.config_hash,
                        record.snapshot_hash,
                        record.snapshot_dir,
                        record.trigger,
                        record.created_at_utc,
                        record.updated_at_utc,
                        record.started_at_utc,
                        record.finished_at_utc,
                        record.current_stage,
                        record.error,
                        _dumps(record.summary),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateError(f"run {record.run_id!r} already exists") from exc
        return record

    def update_run(self, run_id: str, **fields: Any) -> RunRecord:
        unknown = set(fields) - _RUN_UPDATABLE
        if unknown:
            raise StateError(f"cannot update run fields: {sorted(unknown)}")
        payload = dict(fields)
        payload.setdefault("updated_at_utc", _now())
        if "summary" in payload:
            payload["summary_json"] = _dumps(payload.pop("summary"))
        assignments = ", ".join(f"{key} = ?" for key in payload)
        values = list(payload.values()) + [run_id]
        with self._lock:
            cursor = self.connection.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?", values)
            if cursor.rowcount == 0:
                raise StateError(f"unknown run {run_id!r}")
        record = self.get_run(run_id)
        assert record is not None
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.from_row(row) if row is not None else None

    def list_runs(self, *, limit: int = 100, status: str | None = None) -> list[RunRecord]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at_utc DESC, run_id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [RunRecord.from_row(row) for row in rows]

    def latest_run(self) -> RunRecord | None:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    # -- steps ---------------------------------------------------------
    def _upsert_step_conn(self, conn: sqlite3.Connection, record: StepRecord) -> None:
        record.updated_at_utc = record.updated_at_utc or _now()
        conn.execute(
            """
            INSERT INTO steps (
                run_id, stage, status, attempt, max_attempts, input_hash, output_hash,
                started_at_utc, finished_at_utc, duration_s, error, error_type,
                summary_json, skip_count, last_disposition, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, stage) DO UPDATE SET
                status = excluded.status,
                attempt = excluded.attempt,
                max_attempts = excluded.max_attempts,
                input_hash = excluded.input_hash,
                output_hash = excluded.output_hash,
                started_at_utc = excluded.started_at_utc,
                finished_at_utc = excluded.finished_at_utc,
                duration_s = excluded.duration_s,
                error = excluded.error,
                error_type = excluded.error_type,
                summary_json = excluded.summary_json,
                skip_count = excluded.skip_count,
                last_disposition = excluded.last_disposition,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                record.run_id,
                record.stage,
                record.status,
                int(record.attempt),
                int(record.max_attempts),
                record.input_hash,
                record.output_hash,
                record.started_at_utc,
                record.finished_at_utc,
                float(record.duration_s),
                record.error,
                record.error_type,
                _dumps(record.summary),
                int(record.skip_count),
                record.last_disposition,
                record.updated_at_utc,
            ),
        )

    def upsert_step(self, record: StepRecord) -> StepRecord:
        if record.status not in STEP_STATUSES:
            raise StateError(f"invalid step status {record.status!r}")
        with self._lock:
            self._upsert_step_conn(self.connection, record)
        return record

    def get_step(self, run_id: str, stage: str) -> StepRecord | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM steps WHERE run_id = ? AND stage = ?", (run_id, stage)
            ).fetchone()
        return StepRecord.from_row(row) if row is not None else None

    def list_steps(self, run_id: str) -> list[StepRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY rowid ASC", (run_id,)
            ).fetchall()
        return [StepRecord.from_row(row) for row in rows]

    # -- attempts ------------------------------------------------------
    def _insert_attempt_conn(self, conn: sqlite3.Connection, record: AttemptRecord) -> None:
        conn.execute(
            """
            INSERT INTO attempts (
                run_id, stage, attempt, status, started_at_utc, finished_at_utc,
                duration_s, error_type, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, stage, attempt) DO UPDATE SET
                status = excluded.status,
                started_at_utc = excluded.started_at_utc,
                finished_at_utc = excluded.finished_at_utc,
                duration_s = excluded.duration_s,
                error_type = excluded.error_type,
                error = excluded.error
            """,
            (
                record.run_id,
                record.stage,
                int(record.attempt),
                record.status,
                record.started_at_utc,
                record.finished_at_utc,
                float(record.duration_s),
                record.error_type,
                record.error,
            ),
        )

    def record_attempt(self, record: AttemptRecord) -> AttemptRecord:
        with self._lock:
            self._insert_attempt_conn(self.connection, record)
        return record

    def update_attempt(self, run_id: str, stage: str, attempt: int, **fields: Any) -> None:
        allowed = {"status", "started_at_utc", "finished_at_utc", "duration_s", "error_type", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise StateError(f"cannot update attempt fields: {sorted(unknown)}")
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [run_id, stage, int(attempt)]
        with self._lock:
            self.connection.execute(
                f"UPDATE attempts SET {assignments} WHERE run_id = ? AND stage = ? AND attempt = ?",
                values,
            )

    def list_attempts(self, run_id: str, stage: str | None = None) -> list[AttemptRecord]:
        sql = "SELECT * FROM attempts WHERE run_id = ?"
        params: list[Any] = [run_id]
        if stage is not None:
            sql += " AND stage = ?"
            params.append(stage)
        sql += " ORDER BY rowid ASC"
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [AttemptRecord.from_row(row) for row in rows]

    # -- artifacts -----------------------------------------------------
    def _replace_artifacts_conn(
        self, conn: sqlite3.Connection, run_id: str, stage: str, records: Sequence[ArtifactRecord]
    ) -> None:
        conn.execute("DELETE FROM artifacts WHERE run_id = ? AND stage = ?", (run_id, stage))
        for record in records:
            conn.execute(
                """
                INSERT INTO artifacts (
                    run_id, stage, logical_name, relpath, content_hash, size_bytes, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    stage,
                    record.logical_name,
                    record.relpath,
                    record.content_hash,
                    int(record.size_bytes),
                    record.created_at_utc or _now(),
                ),
            )

    def replace_artifacts(self, run_id: str, stage: str, records: Sequence[ArtifactRecord]) -> None:
        with self._lock:
            self._replace_artifacts_conn(self.connection, run_id, stage, records)

    def list_artifacts(self, run_id: str, stage: str | None = None) -> list[ArtifactRecord]:
        sql = "SELECT * FROM artifacts WHERE run_id = ?"
        params: list[Any] = [run_id]
        if stage is not None:
            sql += " AND stage = ?"
            params.append(stage)
        sql += " ORDER BY stage ASC, logical_name ASC"
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [ArtifactRecord.from_row(row) for row in rows]

    def get_artifact(self, run_id: str, stage: str, logical_name: str) -> ArtifactRecord | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND stage = ? AND logical_name = ?",
                (run_id, stage, logical_name),
            ).fetchone()
        return ArtifactRecord.from_row(row) if row is not None else None

    def delete_artifacts(self, run_id: str, stage: str | None = None) -> None:
        sql = "DELETE FROM artifacts WHERE run_id = ?"
        params: list[Any] = [run_id]
        if stage is not None:
            sql += " AND stage = ?"
            params.append(stage)
        with self._lock:
            self.connection.execute(sql, params)

    def artifact_hashes(self, run_id: str, stage: str | None = None) -> dict[str, str]:
        """Map declared artifact relpath -> canonical content hash."""

        return {record.relpath: record.content_hash for record in self.list_artifacts(run_id, stage)}

    # -- events --------------------------------------------------------
    def add_event(
        self,
        run_id: str,
        kind: str,
        *,
        level: str = "info",
        payload: Mapping[str, Any] | None = None,
        ts_utc: str | None = None,
    ) -> EventRecord:
        record = EventRecord(
            run_id=run_id,
            ts_utc=ts_utc or _now(),
            kind=kind,
            level=level,
            payload=dict(payload or {}),
        )
        with self._lock:
            cursor = self.connection.execute(
                "INSERT INTO events (run_id, ts_utc, kind, level, payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, record.ts_utc, record.kind, record.level, _dumps(record.payload)),
            )
            record.event_id = int(cursor.lastrowid or 0)
        return record

    def list_events(self, run_id: str, *, limit: int | None = None) -> list[EventRecord]:
        sql = "SELECT * FROM events WHERE run_id = ? ORDER BY id ASC"
        params: list[Any] = [run_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        # LIMIT applies to the head; for "recent events" callers can slice after
        # fetching a bounded tail, so keep the query deterministic here.
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [EventRecord.from_row(row) for row in rows]

    def recent_events(self, run_id: str, *, limit: int = 20) -> list[EventRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, int(limit)),
            ).fetchall()
        return [EventRecord.from_row(row) for row in reversed(rows)]

    # -- budgets -------------------------------------------------------
    def upsert_budgets(
        self, run_id: str, records: Sequence[BudgetRecord] | Mapping[str, tuple[float, float]]
    ) -> None:
        if isinstance(records, Mapping):
            items = [
                BudgetRecord(run_id=run_id, metric=str(metric), limit_value=float(limit), used_value=float(used))
                for metric, (limit, used) in records.items()
            ]
        else:
            items = list(records)
        now = _now()
        with self.transaction() as conn:
            for record in items:
                conn.execute(
                    """
                    INSERT INTO budgets (run_id, metric, limit_value, used_value, updated_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, metric) DO UPDATE SET
                        limit_value = excluded.limit_value,
                        used_value = excluded.used_value,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        run_id,
                        record.metric,
                        float(record.limit_value),
                        float(record.used_value),
                        record.updated_at_utc or now,
                    ),
                )

    def list_budgets(self, run_id: str) -> list[BudgetRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM budgets WHERE run_id = ? ORDER BY metric ASC", (run_id,)
            ).fetchall()
        return [BudgetRecord.from_row(row) for row in rows]

    # -- approvals -----------------------------------------------------
    def add_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        record.decided_at_utc = record.decided_at_utc or _now()
        with self._lock:
            cursor = self.connection.execute(
                """
                INSERT INTO approvals (
                    run_id, action, decision, approver, scope, artifact_hashes_json,
                    note, decided_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.action,
                    record.decision,
                    record.approver,
                    record.scope,
                    _dumps(record.artifact_hashes),
                    record.note,
                    record.decided_at_utc,
                ),
            )
            record.approval_id = int(cursor.lastrowid or 0)
        return record

    def list_approvals(
        self,
        run_id: str,
        *,
        action: str | None = None,
        decision: str | None = None,
    ) -> list[ApprovalRecord]:
        sql = "SELECT * FROM approvals WHERE run_id = ?"
        params: list[Any] = [run_id]
        if action is not None:
            sql += " AND action = ?"
            params.append(action)
        if decision is not None:
            sql += " AND decision = ?"
            params.append(decision)
        sql += " ORDER BY id ASC"
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [ApprovalRecord.from_row(row) for row in rows]

    # -- composite transitions ----------------------------------------
    def complete_stage(
        self,
        *,
        step: StepRecord,
        attempt: AttemptRecord,
        artifacts: Sequence[ArtifactRecord],
    ) -> None:
        """Atomically register artifacts and mark a stage attempt completed."""

        if step.status != "completed":
            raise StateError("complete_stage requires a completed step record")
        with self.transaction() as conn:
            self._replace_artifacts_conn(conn, step.run_id, step.stage, artifacts)
            self._upsert_step_conn(conn, step)
            self._insert_attempt_conn(conn, attempt)

    def fail_stage(self, *, step: StepRecord, attempt: AttemptRecord) -> None:
        """Atomically persist a failed attempt and the resulting step state."""

        with self.transaction() as conn:
            self._upsert_step_conn(conn, step)
            self._insert_attempt_conn(conn, attempt)

    def invalidate_step(self, run_id: str, stage: str, *, reason: str = "invalidated") -> None:
        """Reset one step to ``pending`` and drop its stale artifact registrations."""

        step = self.get_step(run_id, stage)
        record = step or StepRecord(run_id=run_id, stage=stage)
        record.status = "pending"
        record.input_hash = None
        record.output_hash = None
        record.summary = {}
        record.error = None
        record.error_type = None
        record.last_disposition = f"invalidated:{reason}"
        record.updated_at_utc = _now()
        with self.transaction() as conn:
            self._upsert_step_conn(conn, record)
            conn.execute("DELETE FROM artifacts WHERE run_id = ? AND stage = ?", (run_id, stage))

    def reset_running_steps(self, run_id: str) -> list[str]:
        """Return steps left ``running`` by a crash back to ``pending`` (for resume)."""

        reset: list[str] = []
        for step in self.list_steps(run_id):
            if step.status == "running":
                step.status = "pending"
                step.last_disposition = "resumed_after_interruption"
                step.updated_at_utc = _now()
                self.upsert_step(step)
                reset.append(step.stage)
        return reset

    # -- export --------------------------------------------------------
    def export_run(self, run_id: str) -> dict[str, Any]:
        """State snapshot for one run (used by ``state_snapshot.json``)."""

        run = self.get_run(run_id)
        if run is None:
            raise StateError(f"unknown run {run_id!r}")
        steps = self.list_steps(run_id)
        attempts = self.list_attempts(run_id)
        artifacts = self.list_artifacts(run_id)
        budgets = self.list_budgets(run_id)
        approvals = self.list_approvals(run_id)
        events = self.list_events(run_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "run": run.to_dict(),
            "steps": [step.to_dict() for step in steps],
            "attempts": [attempt.to_dict() for attempt in attempts],
            "artifacts": [artifact.to_dict() for artifact in artifacts],
            "budgets": [budget.to_dict() for budget in budgets],
            "approvals": [approval.to_dict() for approval in approvals],
            "events": [event.to_dict() for event in events],
        }

    def export_state(self, run_id: str | None = None) -> dict[str, Any]:
        """Exported controller state, either for one run or every stored run."""

        run_ids = [run_id] if run_id is not None else [run.run_id for run in self.list_runs(limit=10_000)]
        return {
            "schema_version": SCHEMA_VERSION,
            "exported_at_utc": _now(),
            "runs": [self.export_run(rid) for rid in run_ids],
        }

    def write_state_export(self, path: Path | str, run_id: str | None = None) -> Path:
        """Atomically write a state export JSON file."""

        return write_json(path, self.export_state(run_id), sort_keys=True)
