"""Eastmoney public-endpoint client and parsers.

Three token-less public endpoints are used (verified 2026-11-13):

1. CSI 300 constituents (board ``b:BK0500``)::

       https://{host}/api/qt/clist/get?pn={1..3}&pz=100&po=1&np=1&fltt=2&invt=2
           &fid=f3&fs=b:BK0500&fields=f12,f13,f14,f2,f3,f8,f9,f20,f21,f23,f26,f100

   ``f12`` code, ``f13`` market id (1=SH, 0=SZ), ``f14`` name, ``f2`` price,
   ``f3`` pct change, ``f8`` turnover, ``f9`` PE(dynamic), ``f20`` total market
   cap, ``f21`` free-float market cap, ``f23`` PB, ``f26`` listing date
   (YYYYMMDD int), ``f100`` industry. Page size is capped at 100, total is 300.

2. Daily K-line (qfq-adjusted, ``fqt=1``)::

       https://{host}/api/qt/stock/kline/get?secid=1.000300|1.600519|0.000001
           &fields1=f1,f2,f3,f4,f5,f6
           &fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61
           &klt=101&fqt=1&beg=20160101&end=20261112&ut=<public token>

   A historical-data host must be used (``push2his.eastmoney.com``); the
   ``push2`` hosts often answer HTTP 200 with an empty ``klines`` list for
   historical ranges. The public ``ut`` token is a constant used by Eastmoney's
   own web client (no login/API key). Each ``klines`` item is an 11-field CSV::

       f51 date, f52 open, f53 close, f54 high, f55 low, f56 volume,
       f57 amount, f58 amplitude(%), f59 pct_chg(%), f60 change, f61 turnover(%)

   ``f51..f58`` alone is not enough (no turnover/pct_chg for the contract), so
   the client requests all eleven fields.

3. Fundamentals history (data centre)::

       https://datacenter-web.eastmoney.com/api/data/v1/get
           ?reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL
           &filter=(SECUCODE="600519.SH")&pageSize=60
           &sortColumns=REPORT_DATE&sortTypes=-1

   Mapped fields: ``SECUCODE``->code, ``REPORT_DATE``->report_date,
   ``NOTICE_DATE``->notice_date, ``EPSJB``->eps, ``BPS``->bps, ``ROEJQ``->roe,
   ``XSMLL``->gross_margin, ``TOTALOPERATEREVE``->revenue,
   ``TOTALOPERATEREVETZ``->revenue_yoy, ``PARENTNETPROFIT``->net_profit,
   ``PARENTNETPROFITTZ``->net_profit_yoy, ``MGJYXJJE``->cfo_per_share.
   ``MLR`` is the gross-profit amount, not the margin, so ``XSMLL`` (销售毛利率)
   is used for ``gross_margin``.

All parser functions are pure (payload in, DataFrame out) so they are unit
tested against the small saved payloads in ``tests/fixtures/raw/``.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from aicwf.contracts import PROJECT_ROOT, RetryableError, parse_date
from aicwf.data.http import HttpClient, RequestsTransport
from aicwf.data.snapshot import FRAME_SPECS
from aicwf.util.jsonio import read_yaml

# Public constant token used by Eastmoney's own web K-line widget; not a secret.
PUBLIC_UT = "fa5fd1943c7b386f172d6893dbfba10b"

DEFAULT_HOSTS: tuple[str, ...] = (
    "push2his.eastmoney.com",
    "push2.eastmoney.com",
    "82.push2.eastmoney.com",
    "1.push2.eastmoney.com",
)
DEFAULT_LIST_FIELDS = "f12,f13,f14,f2,f3,f8,f9,f20,f21,f23,f26,f100"
DEFAULT_KLINE_FIELDS1 = "f1,f2,f3,f4,f5,f6"
DEFAULT_KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"

# 11 CSV fields returned by fields2=f51..f61, in order.
KLINE_FIELDS: tuple[str, ...] = (
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
)

CONSTITUENT_COLUMNS = (
    "code",
    "name",
    "industry",
    "price",
    "pct_chg",
    "turnover",
    "pe",
    "pb",
    "total_cap",
    "free_cap",
    "listed_date",
    "board",
)

FUNDAMENTAL_COLUMNS = tuple(FRAME_SPECS["fundamentals"].columns)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "None", "null", "nan", "NaN"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _date(value: Any) -> str | None:
    """Normalise ``2024-09-30 00:00:00`` / ``19960927`` / ``2024-09-30``."""

    if value is None:
        return None
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text or text in {"-", "--", "None", "null", "nan", "NaN"}:
        return None
    match = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return None


def board_for_code(code: str) -> str:
    """Exchange board classification used by prices/universe frames."""

    digits = code.split(".")[0].zfill(6)
    if digits.startswith(("688", "689")):
        return "star"
    if digits.startswith(("300", "301")):
        return "chinext"
    if digits.startswith(("4", "8")) or code.upper().endswith(".BJ"):
        return "bse"
    return "main"


def normalize_code(raw: Any, market: Any = None) -> str:
    """Return ``600519.SH`` style codes (suffix from market id or code prefix)."""

    text = str(raw).strip().upper()
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", text):
        return text
    if "." in text:
        digits, _, suffix = text.partition(".")
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{digits.zfill(6)}.{suffix}"
    digits = re.sub(r"\D", "", text).zfill(6)[-6:]
    suffix = ""
    if market is not None:
        try:
            market_id = int(market)
            suffix = {1: "SH", 0: "SZ"}.get(market_id, "")
        except (TypeError, ValueError):
            suffix = ""
    if not suffix:
        if digits.startswith("6"):
            suffix = "SH"
        elif digits.startswith(("4", "8")):
            suffix = "BJ"
        else:
            suffix = "SZ"
    return f"{digits}.{suffix}"


def secid_for_code(code: str) -> str:
    """``600519.SH`` -> ``1.600519``; ``000001.SZ`` -> ``0.000001``."""

    normalized = normalize_code(code)
    digits, _, suffix = normalized.partition(".")
    prefix = {"SH": "1", "SZ": "0", "BJ": "0"}.get(suffix, "0")
    return f"{prefix}.{digits}"


def code_from_secid(secid: str) -> str:
    prefix, _, digits = str(secid).partition(".")
    suffix = {"1": "SH", "0": "SZ"}.get(prefix, "SZ")
    return f"{digits.zfill(6)}.{suffix}"


@dataclass
class EastmoneyClient:
    """Token-less Eastmoney client built from ``config/data.yaml``."""

    config: Mapping[str, Any]
    http: HttpClient
    http_kline: HttpClient | None = None
    hosts: tuple[str, ...] = DEFAULT_HOSTS
    kline_hosts: tuple[str, ...] = DEFAULT_HOSTS
    ut: str = PUBLIC_UT
    fields1: str = DEFAULT_KLINE_FIELDS1
    fields2: str = DEFAULT_KLINE_FIELDS2
    index_secid: str = "1.000300"
    csi300_board: str = "b:BK0500"
    page_size: int = 100
    list_fields: str = DEFAULT_LIST_FIELDS
    report_name: str = "RPT_F10_FINANCE_MAINFINADATA"
    fundamentals_page_size: int = 60

    # -- construction --------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        transport: Any = None,
        sleep: Any = None,
        rng: Any = None,
    ) -> "EastmoneyClient":
        config = dict(config or {})
        em = dict(config.get("sources", {}).get("eastmoney", config.get("eastmoney", {})))
        snapshot_cfg = dict(config.get("snapshot", {}))
        _ = snapshot_cfg  # kept for API symmetry; builder reads it separately

        hosts = tuple(em.get("hosts") or DEFAULT_HOSTS)
        kline_cfg = dict(em.get("kline") or {})
        kline_hosts = tuple(kline_cfg.get("hosts") or hosts)
        retry_cfg = dict(em.get("retry") or {})
        kline_retry = dict(em.get("kline_retry") or retry_cfg)
        fundamentals_retry = dict(em.get("fundamentals_retry") or retry_cfg)
        polite = float(em.get("polite_delay_seconds", 0.15) or 0.0)
        headers = {
            "User-Agent": str(em.get("user_agent", "Mozilla/5.0 AICWF academic research")),
            "Referer": "https://www.eastmoney.com/",
            "Accept": "*/*",
        }
        transport = transport or RequestsTransport(
            headers=headers,
            ipv4_only=bool(em.get("ipv4_only", True)),
        )
        import random

        base_rng = rng if rng is not None else random.Random(20261112)

        def _client(cfg: Mapping[str, Any]) -> HttpClient:
            kwargs: dict[str, Any] = {
                "transport": transport,
                "attempts": int(cfg.get("attempts", retry_cfg.get("attempts", 4))),
                "backoff_seconds": tuple(float(x) for x in (cfg.get("backoff_seconds") or retry_cfg.get("backoff_seconds") or (0.5, 1.0, 2.0, 4.0))),
                "jitter_fraction": float(cfg.get("jitter_fraction", retry_cfg.get("jitter_fraction", 0.3))),
                "timeout_s": float(em.get("timeout_s", 20)),
                "polite_delay_s": polite,
            }
            if sleep is not None:
                kwargs["sleep"] = sleep
            return HttpClient(**kwargs)

        default_client = _client(retry_cfg)
        if rng is None:
            default_client.rng = base_rng
        kline_client = _client(kline_retry)
        if rng is None:
            kline_client.rng = random.Random(base_rng.randint(0, 10**9))

        fundamentals_cfg = dict(em.get("fundamentals") or {})
        return cls(
            config=config,
            http=default_client,
            http_kline=kline_client,
            hosts=hosts,
            kline_hosts=kline_hosts,
            ut=str(kline_cfg.get("ut") or PUBLIC_UT),
            fields1=str(kline_cfg.get("fields1") or DEFAULT_KLINE_FIELDS1),
            fields2=str(kline_cfg.get("fields2") or DEFAULT_KLINE_FIELDS2),
            index_secid=str(em.get("index_secid") or "1.000300"),
            csi300_board=str(em.get("csi300_board") or "b:BK0500"),
            page_size=int(em.get("page_size") or 100),
            list_fields=str(em.get("list_fields") or DEFAULT_LIST_FIELDS),
            report_name=str(fundamentals_cfg.get("report") or em.get("report") or "RPT_F10_FINANCE_MAINFINADATA"),
            fundamentals_page_size=int(fundamentals_cfg.get("page_size") or 60),
        )

    # -- URLs ----------------------------------------------------------
    @property
    def list_urls(self) -> list[str]:
        template = self._url("list_url", "https://{host}/api/qt/clist/get")
        return [template.format(host=host) for host in self.hosts]

    @property
    def kline_urls(self) -> list[str]:
        template = self._url("kline_url", "https://{host}/api/qt/stock/kline/get")
        return [template.format(host=host) for host in self.kline_hosts]

    @property
    def datacenter_url(self) -> str:
        em = dict(self.config.get("sources", {}).get("eastmoney", {}))
        return str(em.get("datacenter_url") or "https://datacenter-web.eastmoney.com/api/data/v1/get")

    def _url(self, key: str, default: str) -> str:
        em = dict(self.config.get("sources", {}).get("eastmoney", {}))
        return str(em.get(key) or default)

    # -- fetchers ------------------------------------------------------
    def fetch_constituents(self, *, max_pages: int = 6) -> pd.DataFrame:
        """Fetch every CSI 300 constituent page and return a parsed frame."""

        rows: list[dict[str, Any]] = []
        total: int | None = None
        page = 1
        while page <= max_pages:
            params = {
                "pn": page,
                "pz": self.page_size,
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f3",
                "fs": self.csi300_board,
                "fields": self.list_fields,
            }
            payload = self.http.get_json(
                self.list_urls,
                params=params,
                validator=_constituents_validator,
                describe=f"constituents page {page}",
            )
            data = payload.get("data") or {}
            chunk = _diff_rows(data.get("diff"))
            total = int(data.get("total") or total or len(rows) + len(chunk))
            rows.extend(chunk)
            if len(rows) >= total or not chunk:
                break
            page += 1
        if not rows:
            raise RetryableError("constituents endpoint returned no rows")
        return parse_constituents({"data": {"total": total, "diff": rows}})

    def fetch_kline(self, secid: str, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        """Daily qfq bars for one security; returns the parsed K-line frame."""

        beg = parse_date(str(start)).strftime("%Y%m%d")
        fin = parse_date(str(end)).strftime("%Y%m%d")
        params = {
            "secid": str(secid),
            "fields1": self.fields1,
            "fields2": self.fields2,
            "klt": 101,
            "fqt": 1,
            "beg": beg,
            "end": fin,
            "ut": self.ut,
        }
        client = self.http_kline or self.http
        payload = client.get_json(
            self.kline_urls,
            params=params,
            validator=_kline_validator,
            describe=f"kline {secid}",
        )
        return parse_kline(payload)

    def fetch_index_kline(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        return self.fetch_kline(self.index_secid, start, end)

    def fetch_fundamentals(self, secucode: str) -> pd.DataFrame:
        """Fundamentals history for one security (one row per report period)."""

        normalized = normalize_code(secucode)
        params = {
            "reportName": self.report_name,
            "columns": "ALL",
            "filter": f'(SECUCODE="{normalized}")',
            "pageSize": self.fundamentals_page_size,
            "sortColumns": "REPORT_DATE",
            "sortTypes": -1,
        }
        payload = self.http.get_json(
            [self.datacenter_url],
            params=params,
            validator=_fundamentals_validator,
            describe=f"fundamentals {normalized}",
        )
        return parse_fundamentals(payload)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _diff_rows(diff: Any) -> list[dict[str, Any]]:
    if diff is None:
        return []
    if isinstance(diff, Mapping):
        return [dict(value) for value in diff.values() if isinstance(value, Mapping)]
    if isinstance(diff, list):
        return [dict(value) for value in diff if isinstance(value, Mapping)]
    return []


def _constituents_validator(payload: Any) -> bool:
    data = (payload or {}).get("data") or {}
    return bool(_diff_rows(data.get("diff")))


def _kline_validator(payload: Any) -> bool:
    data = (payload or {}).get("data") or {}
    return bool(data.get("klines"))


def _fundamentals_validator(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    success = payload.get("success")
    if isinstance(success, str):
        success = success.strip().lower() in {"true", "1", "yes"}
    result = payload.get("result") or {}
    data = result.get("data") if isinstance(result, Mapping) else None
    return bool(success) and bool(data)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_constituents(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Parse a ``clist/get`` payload into a constituent frame."""

    data = (payload or {}).get("data") or {}
    rows = _diff_rows(data.get("diff"))
    records: list[dict[str, Any]] = []
    for row in rows:
        market = row.get("f13")
        code = normalize_code(row.get("f12"), market)
        name = str(row.get("f14") or "").strip()
        records.append(
            {
                "code": code,
                "name": name,
                "industry": str(row.get("f100") or "").strip() or None,
                "price": _num(row.get("f2")),
                "pct_chg": _num(row.get("f3")),
                "turnover": _num(row.get("f8")),
                "pe": _num(row.get("f9")),
                "pb": _num(row.get("f23")),
                "total_cap": _num(row.get("f20")),
                "free_cap": _num(row.get("f21")),
                "listed_date": _date(row.get("f26")),
                "board": board_for_code(code),
            }
        )
    frame = pd.DataFrame(records, columns=list(CONSTITUENT_COLUMNS))
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    return frame


def parse_kline(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Parse one ``kline/get`` payload into an 11-column bar frame."""

    data = (payload or {}).get("data") or {}
    lines = data.get("klines") or []
    width = len(KLINE_FIELDS)
    records: list[dict[str, Any]] = []
    for line in lines:
        parts = str(line).split(",")
        if len(parts) < 8:
            continue
        padded = parts + [None] * (width - len(parts))
        record: dict[str, Any] = {}
        for index, field in enumerate(KLINE_FIELDS):
            value = padded[index]
            if field == "date":
                record[field] = _date(value)
            else:
                record[field] = _num(value)
        records.append(record)
    frame = pd.DataFrame(records, columns=list(KLINE_FIELDS))
    if not frame.empty:
        frame = frame.dropna(subset=["date"])
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    return frame


def parse_fundamentals(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Parse a data-centre fundamentals payload into the contract columns."""

    result = (payload or {}).get("result") or {}
    rows = result.get("data") if isinstance(result, Mapping) else None
    rows = rows or []
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        code = normalize_code(row.get("SECUCODE"))
        report_date = _date(row.get("REPORT_DATE"))
        if report_date is None:
            continue
        records.append(
            {
                "code": code,
                "report_date": report_date,
                "notice_date": _date(row.get("NOTICE_DATE")),
                "eps": _num(row.get("EPSJB")),
                "bps": _num(row.get("BPS")),
                "roe": _num(row.get("ROEJQ")),
                "gross_margin": _num(row.get("XSMLL")),
                "revenue": _num(row.get("TOTALOPERATEREVE")),
                "revenue_yoy": _num(row.get("TOTALOPERATEREVETZ")),
                "net_profit": _num(row.get("PARENTNETPROFIT")),
                "net_profit_yoy": _num(row.get("PARENTNETPROFITTZ")),
                "cfo_per_share": _num(row.get("MGJYXJJE")),
                "notice_date_estimated": False,
            }
        )
    frame = pd.DataFrame(records, columns=list(FUNDAMENTAL_COLUMNS))
    if not frame.empty:
        frame = (
            frame.sort_values(["code", "report_date"], kind="mergesort")
            .drop_duplicates(subset=["code", "report_date"], keep="last")
            .reset_index(drop=True)
        )
    return frame


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/data.yaml`` (falls back to built-in defaults when absent)."""

    config_path = Path(path) if path else PROJECT_ROOT / "config" / "data.yaml"
    if config_path.exists():
        return dict(read_yaml(config_path) or {})
    return {
        "sources": {
            "eastmoney": {
                "hosts": list(DEFAULT_HOSTS),
                "index_secid": "1.000300",
                "csi300_board": "b:BK0500",
                "page_size": 100,
                "timeout_s": 20,
                "polite_delay_seconds": 0.15,
                "retry": {"attempts": 4, "backoff_seconds": [0.5, 1.0, 2.0, 4.0], "jitter_fraction": 0.3},
            }
        },
        "snapshot": {"root": "data/snapshots", "schema_version": "1.1.0", "min_universe_size": 250},
    }
