"""Structured event logging (JSONL) and plain console logging."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class EventLog:
    """Append-only JSONL event sink for one run.

    The controller owns one instance per run. Events are flushed and fsynced so a
    crashed run still leaves a reviewable audit trail.
    """

    def __init__(self, path: Path | str, *, echo: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def emit(self, kind: str, *, level: str = "info", run_id: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "ts_utc": _utc_now_iso(),
            "kind": kind,
            "level": level,
            "run_id": run_id,
            "payload": payload or {},
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
        if self.echo:
            sys.stderr.write(f"[{record['ts_utc']}] {level.upper():7s} {kind} {json.dumps(payload or {}, ensure_ascii=False)}\n")
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events


def get_logger(name: str = "aicwf") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
