from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Optional
import json
import random
from datetime import datetime, timedelta

import httpx

from app.config import settings
from app.utils.crypto import sha256_hex
from app.adapters.ifind_http import IFindHTTPProvider, IFindHTTPResponse


class DataProvider(Protocol):
    """Minimal abstraction used by offline collectors & ML backfill.

    The project primarily uses iFinD HTTP (QuantAPI) but also supports:
      - THS_DATAAPI (10jqka dataapi)
      - CUSTOM_HTTP (your own gateway)
      - MOCK (offline deterministic)

    NOTE: We keep this protocol intentionally tiny: one `call()` method
    returning a standard IFindHTTPResponse-like envelope.
    """

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        raise NotImplementedError


def _mk_ifind_table_payload(table: dict[str, Any]) -> dict[str, Any]:
    # Mirror the common iFinD QuantAPI shape that our extractor expects:
    # {"tables":[{"table": {...}}]}
    return {"tables": [{"table": table}]}


def _json_loads_safe(s: str) -> dict[str, Any]:
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


@dataclass
class MockDataProvider:
    """Deterministic-ish mock provider for local/offline runs.

    It returns small, plausible numeric payloads for:
      - real_time_quotation
      - cmd_history_quotation
      - high_frequency

    This keeps the pipeline runnable without tokens/network access.
    """

    seed: int = 42

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        endpoint = (endpoint or "").strip()
        symbol = str(payload.get("symbol") or payload.get("ths_code") or payload.get("codes") or "")

        # Stable per-(endpoint,symbol) RNG so repeated calls are reproducible.
        h = sha256_hex(f"{endpoint}|{symbol}|{self.seed}".encode("utf-8"))
        rng = random.Random(int(h[:8], 16))

        now = datetime.utcnow()
        base_px = 5.0 + (int(h[8:12], 16) % 5000) / 1000.0  # ~5-10
        drift = (int(h[12:16], 16) % 2000 - 1000) / 100000.0
        px = max(0.01, base_px * (1.0 + drift))

        if endpoint in {"real_time_quotation", "realtime_quotation", "rtq"}:
            table = {
                "latest": [round(px, 4)],
                "open": [round(px * (1 - 0.003), 4)],
                "high": [round(px * (1 + 0.006), 4)],
                "low": [round(px * (1 - 0.006), 4)],
                "close": [round(px * (1 - 0.001), 4)],
                "volume": [int(100000 * rng.random())],
                "amount": [round(px * int(100000 * rng.random()), 2)],
                "time": [(now + timedelta(hours=8)).isoformat()],
            }
            raw = _mk_ifind_table_payload(table)

        elif endpoint in {"cmd_history_quotation", "history_quotation", "hq"}:
            n = int(payload.get("limit") or 60)
            n = max(5, min(500, n))
            closes, opens, highs, lows, vols, amts = [], [], [], [], [], []
            base = px
            for _i in range(n):
                shock = (rng.random() - 0.5) * 0.02
                base = max(0.01, base * (1 + shock))
                o = base * (1 + (rng.random() - 0.5) * 0.005)
                c = base * (1 + (rng.random() - 0.5) * 0.005)
                hi = max(o, c) * (1 + rng.random() * 0.01)
                lo = min(o, c) * (1 - rng.random() * 0.01)
                v = int(100000 * rng.random())
                a = float(v) * float(c)
                opens.append(round(o, 4))
                closes.append(round(c, 4))
                highs.append(round(hi, 4))
                lows.append(round(lo, 4))
                vols.append(v)
                amts.append(round(a, 2))
            table = {"open": opens, "close": closes, "high": highs, "low": lows, "volume": vols, "amount": amts}
            raw = _mk_ifind_table_payload(table)

        elif endpoint in {"high_frequency", "hf"}:
            table = {"latest": [round(px, 4)], "close": [round(px, 4)]}
            raw = _mk_ifind_table_payload(table)

        else:
            raw = {"errorcode": "UNSUPPORTED_ENDPOINT", "errmsg": f"mock endpoint not supported: {endpoint}"}

        raw_json = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        payload_sha256 = sha256_hex(raw_json.encode("utf-8"))

        errorcode = str(raw.get("errorcode", "0") or "0")
        errmsg = str(raw.get("errmsg", "") or "")
        return IFindHTTPResponse(
            http_status=200 if errorcode == "0" else 400,
            errorcode=errorcode,
            errmsg=errmsg,
            quota_context="",
            raw=raw,
            payload_sha256=payload_sha256,
        )


@dataclass
class THSDataAPIProvider:
    """10jqka dataapi provider.

    This is intentionally tolerant of response shapes: different tenants
    sometimes wrap the kline list differently.

    We normalize the response into an iFinD-like table shape:
      {"tables":[{"table": {"date": [...], "open": [...], ...}}]}

    Endpoint mapping:
      call("cmd_history_quotation", {codes,startdate,enddate,limit})
        -> GET/POST {THS_DATAAPI_BASE_URL}/dataapi/{THS_DAILY_ENDPOINT}
    """

    base_url: str = ""
    token: str = ""
    timeout_sec: float = 15.0

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or settings.THS_DATAAPI_BASE_URL).rstrip("/")
        self.token = (self.token or settings.THS_DATAAPI_TOKEN).strip()
        self._client = httpx.Client(timeout=float(self.timeout_sec))

    def _mk_url(self) -> str:
        base = self.base_url
        if "/dataapi" not in base:
            base = f"{base}/dataapi"
        ep = (settings.THS_DAILY_ENDPOINT or "kline/daily").lstrip("/")
        return f"{base}/{ep}"

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        endpoint = (endpoint or "").strip()
        url = self._mk_url()

        # We currently only support cmd_history_quotation for daily OHLCV backfill.
        if endpoint not in {"cmd_history_quotation", "history_quotation", "daily_ohlcv"}:
            raw = {"errorcode": "UNSUPPORTED_ENDPOINT", "errmsg": f"ths endpoint not supported: {endpoint}"}
            sha = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            return IFindHTTPResponse(http_status=400, errorcode=raw["errorcode"], errmsg=raw["errmsg"], quota_context="", raw=raw, payload_sha256=sha)

        codes = payload.get("codes") or payload.get("symbol") or payload.get("ths_code") or ""
        startdate = str(payload.get("startdate") or payload.get("start") or "")
        enddate = str(payload.get("enddate") or payload.get("end") or "")
        limit = payload.get("limit")

        params: dict[str, Any] = {"codes": codes}
        if startdate:
            params["startdate"] = startdate
        if enddate:
            params["enddate"] = enddate
        if limit:
            params["limit"] = limit

        headers = {"Accept": "application/json"}
        if self.token:
            # Some tenants use token as query; some as header.
            params.setdefault("access_token", self.token)
            headers.setdefault("Authorization", self.token)

        http_status: Optional[int] = None
        raw: dict[str, Any] = {}
        try:
            # Use GET by default (most dataapi endpoints accept query params)
            r = self._client.get(url, params=params, headers=headers)
            http_status = int(r.status_code)
            raw = r.json() if r.content else {}
            if not isinstance(raw, dict):
                raw = {"errorcode": "NON_JSON", "errmsg": "ths response is not JSON dict", "raw": repr(raw)}
        except Exception as e:
            raw = {"errorcode": "HTTP_EXCEPTION", "errmsg": str(e)}

        # Normalize
        norm = _normalize_daily_kline_payload(raw)
        payload_sha256 = sha256_hex(json.dumps(norm, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        errorcode = str(norm.get("errorcode", "0") or "0")
        errmsg = str(norm.get("errmsg", "") or "")
        return IFindHTTPResponse(
            http_status=http_status,
            errorcode=errorcode,
            errmsg=errmsg,
            quota_context="",
            raw=norm,
            payload_sha256=payload_sha256,
        )


@dataclass
class CustomHTTPProvider:
    """Your own OHLCV gateway.

    Configure:
      CUSTOM_DAILY_OHLCV_URL
      CUSTOM_DAILY_OHLCV_METHOD
      CUSTOM_DAILY_HEADERS_JSON
      CUSTOM_DAILY_BODY_JSON

    Expected response (any of the following):
      - {"data": [{"date":"20260115","open":...,"high":...}, ...]}
      - [{"date":...,"open":..., ...}, ...]

    We normalize into iFinD-like table payload for downstream code.
    """

    url: str = ""
    method: str = ""
    timeout_sec: float = 15.0

    def __post_init__(self) -> None:
        self.url = (self.url or settings.CUSTOM_DAILY_OHLCV_URL).strip()
        self.method = (self.method or settings.CUSTOM_DAILY_OHLCV_METHOD or "POST").upper().strip()
        self._client = httpx.Client(timeout=float(self.timeout_sec))

    def call(self, endpoint: str, payload: dict[str, Any]) -> IFindHTTPResponse:
        endpoint = (endpoint or "").strip()
        if endpoint not in {"cmd_history_quotation", "history_quotation", "daily_ohlcv"}:
            raw = {"errorcode": "UNSUPPORTED_ENDPOINT", "errmsg": f"custom endpoint not supported: {endpoint}"}
            sha = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            return IFindHTTPResponse(http_status=400, errorcode=raw["errorcode"], errmsg=raw["errmsg"], quota_context="", raw=raw, payload_sha256=sha)

        if not self.url:
            raw = {"errorcode": "NO_URL", "errmsg": "CUSTOM_DAILY_OHLCV_URL is empty"}
            sha = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            return IFindHTTPResponse(http_status=None, errorcode=raw["errorcode"], errmsg=raw["errmsg"], quota_context="", raw=raw, payload_sha256=sha)

        headers = _json_loads_safe(settings.CUSTOM_DAILY_HEADERS_JSON)
        body = _json_loads_safe(settings.CUSTOM_DAILY_BODY_JSON)
        # Merge caller payload (codes/start/end) into body
        body = {**body, **payload}

        http_status: Optional[int] = None
        raw: Any = None
        try:
            if self.method == "GET":
                r = self._client.get(self.url, params=body, headers=headers)
            else:
                r = self._client.post(self.url, json=body, headers=headers)
            http_status = int(r.status_code)
            raw = r.json() if r.content else {}
        except Exception as e:
            raw = {"errorcode": "HTTP_EXCEPTION", "errmsg": str(e)}

        if not isinstance(raw, (dict, list)):
            raw = {"errorcode": "NON_JSON", "errmsg": "custom response is not JSON", "raw": repr(raw)}

        norm = _normalize_daily_kline_payload(raw)
        payload_sha256 = sha256_hex(json.dumps(norm, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        errorcode = str(norm.get("errorcode", "0") or "0")
        errmsg = str(norm.get("errmsg", "") or "")
        return IFindHTTPResponse(http_status=http_status, errorcode=errorcode, errmsg=errmsg, quota_context="", raw=norm, payload_sha256=payload_sha256)


def _normalize_daily_kline_payload(raw: Any) -> dict[str, Any]:
    """Try to normalize various daily-kline response shapes.

    Output shape:
      {"tables":[{"table":{"date":[],"open":[],"high":[],"low":[],"close":[],"volume":[],"amount":[]}}]}

    On failure, returns {"errorcode":"PARSE_FAILED",...}.
    """

    # If already iFinD-like
    if isinstance(raw, dict):
        tables = raw.get("tables")
        if isinstance(tables, list) and tables and isinstance(tables[0], dict) and isinstance(tables[0].get("table"), dict):
            return raw

    items: list[dict[str, Any]] = []

    def _coerce_items(x: Any) -> None:
        nonlocal items
        if isinstance(x, list):
            for it in x:
                if isinstance(it, dict):
                    items.append(it)
        elif isinstance(x, dict):
            d = x.get("data")
            if isinstance(d, list):
                _coerce_items(d)
            elif isinstance(d, dict):
                # sometimes {data:{list:[]}}
                for k in ("list", "items", "rows"):
                    if isinstance(d.get(k), list):
                        _coerce_items(d.get(k))
                        return
            # other wrappers
            for k in ("rows", "items", "list"):
                if isinstance(x.get(k), list):
                    _coerce_items(x.get(k))
                    return

    _coerce_items(raw)

    # If still empty, maybe it's a dict of arrays
    if not items and isinstance(raw, dict):
        # try keys: date/open/high/low/close/volume/amount
        if any(k in raw for k in ("date", "trade_date", "day")) and any(k in raw for k in ("open", "high", "close")):
            table = {
                "date": raw.get("date") or raw.get("trade_date") or raw.get("day"),
                "open": raw.get("open"),
                "high": raw.get("high"),
                "low": raw.get("low"),
                "close": raw.get("close"),
                "volume": raw.get("volume") or raw.get("vol"),
                "amount": raw.get("amount"),
            }
            return _mk_ifind_table_payload(table)

    if not items:
        return {"errorcode": "PARSE_FAILED", "errmsg": "unable to normalize daily kline payload", "raw": raw}

    # Sort by date if possible
    def _get_date(it: dict[str, Any]) -> str:
        return str(it.get("date") or it.get("trade_date") or it.get("day") or "")

    items.sort(key=_get_date)

    dates, opens, highs, lows, closes, vols, amts = [], [], [], [], [], [], []
    for it in items:
        d = _get_date(it)
        if not d:
            continue
        dates.append(d.replace("-", ""))
        opens.append(it.get("open"))
        highs.append(it.get("high"))
        lows.append(it.get("low"))
        closes.append(it.get("close"))
        vols.append(it.get("volume") or it.get("vol"))
        amts.append(it.get("amount"))

    table = {"date": dates, "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols, "amount": amts}
    return _mk_ifind_table_payload(table)


def get_data_provider() -> DataProvider:
    mode = (settings.DATA_PROVIDER or "").upper().strip()
    if mode == "IFIND_HTTP":
        return IFindHTTPProvider()
    if mode == "THS_DATAAPI":
        return THSDataAPIProvider()
    if mode == "CUSTOM_HTTP":
        return CustomHTTPProvider()
    return MockDataProvider()
