"""Scheduler trigger helpers for monthly monitoring and new-data detection.

The controller does not install cron itself; ``scripts/scheduled_monitor.ps1`` (or
the equivalent cron entry) calls the CLI, and these helpers decide whether the
trigger is actually due and write a small ``state/triggers/*.json`` record for the
run log. Two triggers are defined in ``config/workflow.yaml``:

* ``monthly`` — a five-field cron expression with a timezone, e.g.
  ``0 18 1 * *`` in ``Asia/Shanghai``, compared against the last monitor run;
* ``new_data`` — watch ``data/snapshots/*/manifest.json`` and trigger when the
  latest snapshot hash differs from the previous monitor run.

A cron expression is evaluated with a bounded day scan (4 years), which keeps the
implementation dependency-free (no croniter) while supporting ``*``, single
values, ranges, lists and ``*/step``.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from aicwf.contracts import PROJECT_ROOT, ConfigError
from aicwf.util.hashing import hash_json, short_hash
from aicwf.util.jsonio import write_json
from aicwf.util.timeutil import MARKET_TZ

__all__ = [
    "CronSpec",
    "TriggerDecision",
    "evaluate_monthly_trigger",
    "evaluate_new_data_trigger",
    "evaluate_triggers",
    "resolve_timezone",
    "write_trigger_record",
]

_CRON_LOOKBACK_DAYS = 366 * 4 + 1
_STEP_RE = re.compile(r"^(?P<base>[^/]+)/(?P<step>\d+)$")


# ---------------------------------------------------------------------------
# Cron expression
# ---------------------------------------------------------------------------


def resolve_timezone(name: str | None) -> dt.tzinfo:
    """Resolve an IANA timezone name, falling back to Asia/Shanghai fixed offset."""

    if not name:
        return MARKET_TZ
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(str(name))
    except Exception:  # pragma: no cover - zoneinfo data missing on stripped systems
        return MARKET_TZ


def _parse_cron_field(expr: str, *, low: int, high: int, name: str) -> frozenset[int] | None:
    expr = expr.strip()
    if expr in {"*", "*/1"}:
        return None
    values: set[int] = set()

    def _add_token(token: str) -> None:
        token = token.strip()
        if not token:
            raise ConfigError(f"empty token in cron {name} field")
        step = 1
        match = _STEP_RE.match(token)
        if match:
            token = match.group("base")
            step = int(match.group("step"))
            if step <= 0:
                raise ConfigError(f"cron {name} field step must be positive: {expr!r}")
        if token == "*":
            start, end = low, high
        elif "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ConfigError(f"invalid cron {name} range: {token!r}")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                raise ConfigError(f"invalid cron {name} value in {token!r}") from None
        else:
            try:
                start = end = int(token)
            except ValueError:
                raise ConfigError(f"invalid cron {name} value {token!r} in {expr!r}") from None
        if not (low <= start <= high and low <= end <= high) or start > end:
            raise ConfigError(
                f"cron {name} value {token!r} outside [{low}, {high}] or inverted range"
            )
        values.update(range(start, end + 1, step))

    for token in expr.split(","):
        _add_token(token)
    if not values:
        raise ConfigError(f"cron {name} field resolved to no values: {expr!r}")
    return frozenset(values)


@dataclass(frozen=True)
class CronSpec:
    """A validated five-field cron expression (``minute hour dom month dow``)."""

    expression: str
    minutes: frozenset[int] | None
    hours: frozenset[int] | None
    days_of_month: frozenset[int] | None
    months: frozenset[int] | None
    days_of_week: frozenset[int] | None  # 0 = Sunday, 6 = Saturday (cron convention)

    @classmethod
    def parse(cls, expression: str) -> "CronSpec":
        parts = str(expression).split()
        if len(parts) != 5:
            raise ConfigError(f"cron expression must have 5 fields, got {expression!r}")
        minute, hour, dom, month, dow = parts
        return cls(
            expression=str(expression).strip(),
            minutes=_parse_cron_field(minute, low=0, high=59, name="minute"),
            hours=_parse_cron_field(hour, low=0, high=23, name="hour"),
            days_of_month=_parse_cron_field(dom, low=1, high=31, name="day-of-month"),
            months=_parse_cron_field(month, low=1, high=12, name="month"),
            days_of_week=_parse_cron_field(dow, low=0, high=6, name="day-of-week"),
        )

    # -- matching ------------------------------------------------------
    def _day_matches(self, day: dt.date) -> bool:
        if self.months is not None and day.month not in self.months:
            return False
        dom_matches = self.days_of_month is None or day.day in self.days_of_month
        cron_dow = (day.weekday() + 1) % 7  # Python Monday=0 -> cron Sunday=0
        dow_matches = self.days_of_week is None or cron_dow in self.days_of_week
        if self.days_of_month is not None and self.days_of_week is not None:
            return dom_matches or dow_matches
        return dom_matches and dow_matches

    def _times_of_day(self, *, reverse: bool) -> list[tuple[int, int]]:
        hours: Iterable[int] = range(24) if self.hours is None else sorted(self.hours)
        minutes: Iterable[int] = range(60) if self.minutes is None else sorted(self.minutes)
        hours = list(hours)
        minutes = list(minutes)
        if reverse:
            hours = list(reversed(hours))
            minutes = list(reversed(minutes))
        return [(hour, minute) for hour in hours for minute in minutes]

    def _matches(self, moment: dt.datetime) -> bool:
        if not self._day_matches(moment.date()):
            return False
        if self.hours is not None and moment.hour not in self.hours:
            return False
        if self.minutes is not None and moment.minute not in self.minutes:
            return False
        return moment.second == 0

    # -- occurrence search --------------------------------------------
    def most_recent_before(self, now: dt.datetime, *, lookback_days: int = _CRON_LOOKBACK_DAYS) -> dt.datetime | None:
        """Latest scheduled occurrence at or before ``now`` (timezone-aware)."""

        if now.tzinfo is None:
            raise ConfigError("most_recent_before requires a timezone-aware datetime")
        for offset in range(lookback_days + 1):
            day = now.date() - dt.timedelta(days=offset)
            if not self._day_matches(day):
                continue
            for hour, minute in self._times_of_day(reverse=True):
                candidate = dt.datetime(
                    day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo
                )
                if candidate <= now:
                    return candidate
        return None

    def next_after(self, now: dt.datetime, *, horizon_days: int = _CRON_LOOKBACK_DAYS) -> dt.datetime | None:
        """Earliest scheduled occurrence strictly after ``now`` (timezone-aware)."""

        if now.tzinfo is None:
            raise ConfigError("next_after requires a timezone-aware datetime")
        for offset in range(horizon_days + 1):
            day = now.date() + dt.timedelta(days=offset)
            if not self._day_matches(day):
                continue
            for hour, minute in self._times_of_day(reverse=False):
                candidate = dt.datetime(
                    day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo
                )
                if candidate > now:
                    return candidate
        return None

    def matches(self, moment: dt.datetime) -> bool:
        return self._matches(moment)

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression}


# ---------------------------------------------------------------------------
# Trigger decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerDecision:
    """Whether a monitor/evaluation trigger is due, with the evidence why."""

    trigger: str
    due: bool
    reason: str
    as_of: str | None = None
    scheduled_for_utc: str | None = None
    last_run_id: str | None = None
    last_run_at_utc: str | None = None
    snapshot_hash: str | None = None
    previous_snapshot_hash: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "due": self.due,
            "reason": self.reason,
            "as_of": self.as_of,
            "scheduled_for_utc": self.scheduled_for_utc,
            "last_run_id": self.last_run_id,
            "last_run_at_utc": self.last_run_at_utc,
            "snapshot_hash": self.snapshot_hash,
            "previous_snapshot_hash": self.previous_snapshot_hash,
            "details": dict(self.details),
        }


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            raise ConfigError(f"cannot parse trigger timestamp {value!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _monthly_config(config: Mapping[str, Any] | None) -> tuple[str, str]:
    config = dict(config or {})
    workflow = config.get("workflow") if isinstance(config.get("workflow"), Mapping) else {}
    triggers = dict(workflow.get("triggers") or {}) if isinstance(workflow, Mapping) else {}
    monthly = triggers.get("monthly") or {}
    if not isinstance(monthly, Mapping):
        raise ConfigError("workflow.triggers.monthly must be a mapping")
    cron = str(monthly.get("cron") or "0 18 1 * *")
    timezone = str(monthly.get("timezone") or "Asia/Shanghai")
    return cron, timezone


def evaluate_monthly_trigger(
    config: Mapping[str, Any] | None = None,
    *,
    cron: str | None = None,
    timezone: str | None = None,
    now: dt.datetime | None = None,
    last_run_at_utc: Any = None,
    last_run_id: str | None = None,
) -> TriggerDecision:
    """Decide whether the monthly monitor trigger is due.

    Due when the latest scheduled cron occurrence at/before ``now`` is later than
    the last monitor run (or when no previous run is recorded).
    """

    configured_cron, configured_tz = _monthly_config(config)
    spec = CronSpec.parse(cron or configured_cron)
    tz = resolve_timezone(timezone or configured_tz)
    moment = now or dt.datetime.now(tz)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    local_now = moment.astimezone(tz)
    scheduled = spec.most_recent_before(local_now)
    if scheduled is None:
        return TriggerDecision(
            trigger="monthly",
            due=True,
            reason="no_previous_occurrence_found",
            scheduled_for_utc=None,
            last_run_id=last_run_id,
            details={"cron": spec.expression, "timezone": str(tz)},
        )
    scheduled_utc = scheduled.astimezone(dt.timezone.utc).isoformat()
    last_run_dt = _parse_timestamp(last_run_at_utc)
    if last_run_dt is None:
        due = True
        reason = "no_previous_run"
    else:
        due = last_run_dt.astimezone(tz) < scheduled
        reason = "new_period_since_last_run" if due else "already_ran_this_period"
    return TriggerDecision(
        trigger="monthly",
        due=due,
        reason=reason,
        as_of=local_now.date().isoformat(),
        scheduled_for_utc=scheduled_utc,
        last_run_id=last_run_id,
        last_run_at_utc=last_run_dt.astimezone(dt.timezone.utc).isoformat() if last_run_dt else None,
        details={"cron": spec.expression, "timezone": str(tz)},
    )


def evaluate_new_data_trigger(
    *,
    current_snapshot_hash: str | None,
    previous_snapshot_hash: str | None = None,
    last_run_id: str | None = None,
    as_of: str | None = None,
    watch_glob: str | None = None,
) -> TriggerDecision:
    """Decide whether a new-data trigger is due (snapshot hash change)."""

    details: dict[str, Any] = {}
    if watch_glob:
        details["watch_glob"] = watch_glob
    if current_snapshot_hash is None:
        return TriggerDecision(
            trigger="new_data",
            due=False,
            reason="no_snapshot_available",
            as_of=as_of,
            last_run_id=last_run_id,
            previous_snapshot_hash=previous_snapshot_hash,
            details=details,
        )
    if previous_snapshot_hash is None:
        due = True
        reason = "no_previous_snapshot"
    elif str(current_snapshot_hash) != str(previous_snapshot_hash):
        due = True
        reason = "snapshot_hash_changed"
    else:
        due = False
        reason = "snapshot_hash_unchanged"
    return TriggerDecision(
        trigger="new_data",
        due=due,
        reason=reason,
        as_of=as_of,
        last_run_id=last_run_id,
        snapshot_hash=str(current_snapshot_hash),
        previous_snapshot_hash=str(previous_snapshot_hash) if previous_snapshot_hash else None,
        details=details,
    )


def evaluate_triggers(
    config: Mapping[str, Any] | None = None,
    *,
    now: dt.datetime | None = None,
    last_run_at_utc: Any = None,
    last_run_id: str | None = None,
    current_snapshot_hash: str | None = None,
    previous_snapshot_hash: str | None = None,
    as_of: str | None = None,
) -> list[TriggerDecision]:
    """Evaluate both configured triggers (monthly first, new-data second)."""

    config = dict(config or {})
    workflow = config.get("workflow") if isinstance(config.get("workflow"), Mapping) else {}
    triggers = dict(workflow.get("triggers") or {}) if isinstance(workflow, Mapping) else {}
    new_data_cfg = triggers.get("new_data") or {}
    watch_glob = None
    if isinstance(new_data_cfg, Mapping):
        watch_glob = new_data_cfg.get("watch")
    return [
        evaluate_monthly_trigger(
            config,
            now=now,
            last_run_at_utc=last_run_at_utc,
            last_run_id=last_run_id,
        ),
        evaluate_new_data_trigger(
            current_snapshot_hash=current_snapshot_hash,
            previous_snapshot_hash=previous_snapshot_hash,
            last_run_id=last_run_id,
            as_of=as_of,
            watch_glob=str(watch_glob) if watch_glob else None,
        ),
    ]


def write_trigger_record(
    decision: TriggerDecision,
    *,
    state_dir: Path | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a trigger decision under ``state/triggers/`` and return the path.

    ``state/`` is gitignored: trigger records are operational evidence, not
    committed artifacts.
    """

    root = Path(state_dir) if state_dir is not None else PROJECT_ROOT / "state"
    payload = decision.to_dict()
    if extra:
        payload["extra"] = dict(extra)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = short_hash(hash_json(payload), 10)
    path = root / "triggers" / f"{stamp}-{decision.trigger}-{digest}.json"
    write_json(path, payload, sort_keys=True)
    return path
