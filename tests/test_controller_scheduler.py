"""Tests for cron parsing, monthly/new-data triggers and trigger records."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from aicwf.contracts import ConfigError
from aicwf.controller import (
    CronSpec,
    evaluate_monthly_trigger,
    evaluate_new_data_trigger,
    evaluate_triggers,
    write_trigger_record,
)
from aicwf.controller.scheduler import resolve_timezone
from test_controller_helpers import workflow_section

TZ = resolve_timezone("Asia/Shanghai")


def _monthly_config(cron: str = "0 18 1 * *", timezone: str = "Asia/Shanghai") -> dict:
    workflow = workflow_section([])
    workflow["triggers"] = {"monthly": {"cron": cron, "timezone": timezone}}
    return {"workflow": workflow}


def test_cron_parse_and_matching() -> None:
    spec = CronSpec.parse("0 18 1 * *")
    assert spec.minutes == frozenset({0})
    assert spec.hours == frozenset({18})
    assert spec.days_of_month == frozenset({1})
    assert spec.months is None
    assert spec.matches(dt.datetime(2026, 11, 1, 18, 0, tzinfo=TZ))
    assert not spec.matches(dt.datetime(2026, 11, 1, 18, 1, tzinfo=TZ))
    assert not spec.matches(dt.datetime(2026, 11, 2, 18, 0, tzinfo=TZ))

    complex_spec = CronSpec.parse("*/15 9-17 * * 1-5")
    assert complex_spec.minutes == frozenset({0, 15, 30, 45})
    assert complex_spec.hours == frozenset(range(9, 18))
    assert complex_spec.matches(dt.datetime(2026, 11, 13, 10, 30, tzinfo=TZ))  # Friday
    assert not complex_spec.matches(dt.datetime(2026, 11, 15, 10, 30, tzinfo=TZ))  # Sunday
    assert not complex_spec.matches(dt.datetime(2026, 11, 13, 10, 31, tzinfo=TZ))

    wildcard = CronSpec.parse("* * * * *")
    assert wildcard.minutes is None and wildcard.hours is None
    assert wildcard.matches(dt.datetime(2026, 11, 13, 0, 0, tzinfo=TZ))


@pytest.mark.parametrize("expression", ["0 18 1 *", "61 18 1 * *", "0 24 1 * *", "0 18 32 * *", "0 18 1 13 *", "a 18 1 * *", "*/0 18 1 * *"])
def test_cron_parse_rejects_invalid_expressions(expression: str) -> None:
    with pytest.raises(ConfigError):
        CronSpec.parse(expression)


def test_cron_occurrence_search() -> None:
    spec = CronSpec.parse("0 18 1 * *")
    now = dt.datetime(2026, 11, 13, 10, 0, tzinfo=TZ)
    assert spec.most_recent_before(now) == dt.datetime(2026, 11, 1, 18, 0, tzinfo=TZ)
    assert spec.next_after(now) == dt.datetime(2026, 12, 1, 18, 0, tzinfo=TZ)
    assert spec.most_recent_before(dt.datetime(2026, 12, 1, 18, 0, tzinfo=TZ)) == dt.datetime(
        2026, 12, 1, 18, 0, tzinfo=TZ
    )
    with pytest.raises(ConfigError):
        spec.most_recent_before(dt.datetime(2026, 11, 13, 10, 0))  # naive


def test_monthly_trigger_due_when_never_run() -> None:
    decision = evaluate_monthly_trigger(
        _monthly_config(), now=dt.datetime(2026, 11, 13, 10, 0, tzinfo=TZ), last_run_at_utc=None
    )
    assert decision.trigger == "monthly"
    assert decision.due is True
    assert decision.reason == "no_previous_run"
    assert decision.scheduled_for_utc == "2026-11-01T10:00:00+00:00"  # 18:00 +08
    assert decision.as_of == "2026-11-13"


def test_monthly_trigger_due_or_already_ran() -> None:
    now = dt.datetime(2026, 11, 13, 10, 0, tzinfo=TZ)
    latest = evaluate_monthly_trigger(_monthly_config(), now=now, last_run_at_utc="2026-11-02T00:00:00+00:00")
    assert latest.due is False
    assert latest.reason == "already_ran_this_period"
    stale = evaluate_monthly_trigger(_monthly_config(), now=now, last_run_at_utc="2026-10-05T00:00:00+00:00")
    assert stale.due is True
    assert stale.reason == "new_period_since_last_run"
    # naive datetimes are interpreted in the configured timezone
    naive = evaluate_monthly_trigger(_monthly_config(), now=dt.datetime(2026, 11, 13, 10, 0), last_run_at_utc=None)
    assert naive.due is True


def test_new_data_trigger() -> None:
    assert evaluate_new_data_trigger(current_snapshot_hash=None).due is False
    first = evaluate_new_data_trigger(current_snapshot_hash="abc")
    assert first.due is True and first.reason == "no_previous_snapshot"
    unchanged = evaluate_new_data_trigger(current_snapshot_hash="abc", previous_snapshot_hash="abc")
    assert unchanged.due is False and unchanged.reason == "snapshot_hash_unchanged"
    changed = evaluate_new_data_trigger(current_snapshot_hash="def", previous_snapshot_hash="abc")
    assert changed.due is True and changed.reason == "snapshot_hash_changed"
    assert changed.snapshot_hash == "def"
    assert changed.previous_snapshot_hash == "abc"


def test_evaluate_triggers_returns_both() -> None:
    config = _monthly_config()
    config["workflow"]["triggers"]["new_data"] = {"watch": "data/snapshots/*/manifest.json"}
    decisions = evaluate_triggers(
        config,
        now=dt.datetime(2026, 11, 13, 10, 0, tzinfo=TZ),
        last_run_at_utc=None,
        current_snapshot_hash="abc",
        previous_snapshot_hash="abc",
        as_of="2026-11-13",
    )
    assert [decision.trigger for decision in decisions] == ["monthly", "new_data"]
    assert decisions[0].due is True
    assert decisions[1].due is False
    assert decisions[1].details["watch_glob"] == "data/snapshots/*/manifest.json"
    payload = decisions[0].to_dict()
    assert payload["trigger"] == "monthly"
    assert set(payload) >= {"due", "reason", "scheduled_for_utc", "details"}


def test_write_trigger_record(tmp_path: Path) -> None:
    decision = evaluate_new_data_trigger(
        current_snapshot_hash="def", previous_snapshot_hash="abc", as_of="2026-11-13"
    )
    path = write_trigger_record(decision, state_dir=tmp_path / "state")
    assert path.is_file()
    assert path.parent == tmp_path / "state" / "triggers"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trigger"] == "new_data"
    assert payload["due"] is True
    assert payload["snapshot_hash"] == "def"

    with_extra = write_trigger_record(decision, state_dir=tmp_path / "state", extra={"run_id": "run-1"})
    assert json.loads(with_extra.read_text(encoding="utf-8"))["extra"] == {"run_id": "run-1"}
