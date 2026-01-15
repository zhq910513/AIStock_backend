from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
import hashlib
from urllib.parse import urlencode

from app.utils.time import now_shanghai

import requests

from app.config import settings
from app.utils.crypto import sha256_hex


@dataclass(frozen=True)
class PoolFetchResult:
    """Result of pulling the external limit-up candidate pool."""

    raw: Any
    raw_hash: str
    items: list[dict]


def _safe_json_loads(s: str) -> dict:
    try:
        obj = json.loads(s or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def fetch_limitup_pool(timeout_sec: int = 20) -> PoolFetchResult:
    """Pull external candidate pool.

    The external API shape is intentionally flexible. We treat the whole payload as raw,
    and try to find a list of items under common keys:
    - {"data": [...]} / {"items": [...]} / [...]

    Each item should be a dict; otherwise it will be ignored.
    """

    mode = str(getattr(settings, "POOL_FETCHER_MODE", "CUSTOM") or "CUSTOM").strip().upper()
    if mode == "THS_10JQKA":
        return _fetch_ths_10jqka(timeout_sec=timeout_sec)

    url = str(settings.POOL_FETCH_URL or "").strip()
    if not url:
        # Hard-fail: without URL we cannot fetch.
        raw = {"error": "POOL_FETCH_URL is empty and POOL_FETCHER_MODE is not THS_10JQKA"}
        raw_hash = sha256_hex(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
        return PoolFetchResult(raw=raw, raw_hash=raw_hash, items=[])

    method = str(settings.POOL_FETCH_METHOD or "GET").strip().upper() or "GET"
    headers = _safe_json_loads(settings.POOL_FETCH_HEADERS_JSON)
    body = _safe_json_loads(settings.POOL_FETCH_BODY_JSON)

    if method not in {"GET", "POST"}:
        method = "GET"

    resp = None
    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
    else:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_sec)

    # best-effort parse
    try:
        raw: Any = resp.json()
    except Exception:
        raw = {"text": (resp.text or "")[:2000], "http_status": resp.status_code}

    raw_hash = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    items: list[dict] = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, dict):
                items.append(x)
    elif isinstance(raw, dict):
        cand = raw.get("data")
        if isinstance(cand, list):
            items = [x for x in cand if isinstance(x, dict)]
        else:
            cand2 = raw.get("items")
            if isinstance(cand2, list):
                items = [x for x in cand2 if isinstance(x, dict)]

    return PoolFetchResult(raw=raw, raw_hash=raw_hash, items=items)


def _fetch_ths_10jqka(timeout_sec: int = 20) -> PoolFetchResult:
    """Fetch limit-up continuous limit pool from 10jqka (同花顺数据)."""
    date = now_shanghai().strftime("%Y%m%d")
    url = "https://data.10jqka.com.cn/dataapi/limit_up/continuous_limit_pool"
    params = {
        "page": "1",
        "limit": str(int(getattr(settings, "THS_POOL_LIMIT", 200) or 200)),
        "field": str(getattr(settings, "THS_POOL_FIELDS", "") or "").strip(),
        "filter": str(getattr(settings, "THS_POOL_FILTER", "") or "").strip(),
        "order_field": str(getattr(settings, "THS_POOL_ORDER_FIELD", "") or "").strip(),
        "order_type": str(getattr(settings, "THS_POOL_ORDER_TYPE", "0") or "0").strip(),
        "date": f"{date}",
    }

    # Minimal headers. This endpoint usually works without special tokens, but does verify UA.
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://data.10jqka.com.cn/",
    }

    try:
        # Note: their API expects query params already encoded.
        resp = requests.get(url=url, headers=headers, params=urlencode(params), timeout=timeout_sec, verify=False)
        raw: Any
        try:
            raw = resp.json()
        except Exception:
            raw = {"text": (resp.text or "")[:2000], "http_status": resp.status_code}

        raw_hash = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))

        items: list[dict] = []
        if isinstance(raw, dict):
            # Expected shape:
            # {"status_code":0,"status_msg":"success","data":{"info":[...]}}
            data = raw.get("data")
            if isinstance(data, dict):
                info = data.get("info")
                if isinstance(info, list):
                    items = [x for x in info if isinstance(x, dict)]

        # add audit fields per legacy script
        for it in items:
            try:
                code = str(it.get("code") or "").strip()
                if code:
                    hash_key = hashlib.md5((date + code).encode("utf8")).hexdigest()
                    it["hash_key"] = hash_key
                    it["date"] = date
            except Exception:
                continue

        return PoolFetchResult(raw=raw, raw_hash=raw_hash, items=items)
    except Exception as e:
        raw = {"error": f"THS_10JQKA fetch failed: {type(e).__name__}", "date": date}
        raw_hash = sha256_hex(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return PoolFetchResult(raw=raw, raw_hash=raw_hash, items=[])
