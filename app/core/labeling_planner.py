from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import hashlib
import json
from urllib.parse import quote_plus

from app.config import settings
from app.utils.time import now_shanghai


@dataclass
class PlannedRequest:
    symbol: str
    endpoint: str
    # Payload to provider (iFinD QuantAPI).
    payload: dict
    # Stable canonical string for dedupe & tracing (stored in DB, max 2048 chars).
    params_canonical: str
    purpose: str
    priority: int
    dedupe_key: str
    correlation_id: str
    deadline_sec: int | None = None

    # Backward-compat alias (some older code used `params`).
    @property
    def params(self) -> dict:
        return self.payload


@dataclass
class Plan:
    symbol: str
    trading_day: str
    stage: int
    requests: list[PlannedRequest]
    planner_state: dict[str, Any]


def _stage_from_hits(hit_count: int) -> int:
    hc = int(hit_count or 0)
    if hc >= 5:
        return 2
    if hc >= 2:
        return 1
    return 0


def _params_canonical(payload: dict) -> str:
    """Make a stable, compact string for request parameters.

    - Sort keys
    - JSON-dump nested dict/list values with sort_keys
    - URL-escape values so '&' and '=' don't break the format
    """
    items: list[tuple[str, str]] = []
    for k in sorted((payload or {}).keys()):
        v = (payload or {}).get(k)
        if isinstance(v, (dict, list)):
            vs = json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            vs = "" if v is None else str(v)
        items.append((str(k), quote_plus(vs)))
    s = "&".join([f"{k}={v}" for k, v in items])
    # DB column is 2048; keep a safe buffer.
    return s[:2000]


def _mk_dedupe_key(trading_day: str, endpoint: str, params_canonical: str) -> str:
    """DataRequest.dedupe_key max len is 128; use a short hash."""
    h = hashlib.sha256(f"{endpoint}|{params_canonical}".encode("utf-8")).hexdigest()[:16]
    ep = (endpoint or "").replace("/", "_")[:32]
    return f"LP:{trading_day}:{ep}:{h}"[:128]


def _mk_correlation_id(trading_day: str, stage: int, dedupe_key: str) -> str:
    h = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:10]
    return f"LP:{trading_day}:{stage}:{h}"


def build_plan(
    symbol: str,
    trading_day: str | None = None,
    hit_count: int = 0,
    planner_state: dict[str, Any] | None = None,
    **_: Any,
) -> Plan:
    """
    Rule-based planner (v1),兼容 router/pipeline 两种调用方式：
      - router:   build_plan(symbol=..., hit_count=..., planner_state={...})
      - pipeline: build_plan(symbol=..., trading_day=..., hit_count=...)

    返回 Plan 时一定带 planner_state；PlannedRequest 带 correlation_id，
    与 router 的 enqueue 入参对齐，避免 500。
    """
    symbol = (symbol or "").strip()
    now = now_shanghai()

    td = (trading_day or "").replace("-", "").strip()
    if len(td) != 8:
        td = now.strftime("%Y%m%d")

    stage = _stage_from_hits(hit_count)

    # history length grows with stage
    base_days = int(getattr(settings, "LABELING_HISTORY_DAYS_BASE", 60))
    expand_days = int(getattr(settings, "LABELING_HISTORY_DAYS_EXPAND", 180))
    max_days = int(getattr(settings, "LABELING_HISTORY_DAYS_MAX", 360))

    history_days = base_days if stage == 0 else (expand_days if stage == 1 else max_days)
    history_days = max(10, min(max_days, history_days))

    # high-frequency sample length
    hf_limit = int(getattr(settings, "LABELING_HF_LIMIT_BASE", 240))
    hf_limit = max(60, min(2000, hf_limit + stage * 120))

    # iFinD QuantAPI commonly uses `codes=...`; keep both to be robust.
    rt_payload = {
        "codes": symbol,
        "indicators": "open,high,low,latest,close,volume,amount",
        "symbol": symbol,
        "ths_code": symbol,
    }

    end_dt = now
    start_dt = now - timedelta(days=history_days)

    hist_payload = {
        "codes": symbol,
        "indicators": "open,high,low,close,volume,amount",
        "startdate": start_dt.strftime("%Y-%m-%d"),
        "enddate": end_dt.strftime("%Y-%m-%d"),
        "functionpara": {"Fill": "Blank"},
        "symbol": symbol,
        "ths_code": symbol,
    }

    hf_payload = {
        "codes": symbol,
        "day": td,
        "limit": hf_limit,
        "symbol": symbol,
        "ths_code": symbol,
    }

    reqs: list[PlannedRequest] = []

    # 1) real-time quote
    ep1 = "real_time_quotation"
    pc1 = _params_canonical(rt_payload)
    dk1 = _mk_dedupe_key(td, ep1, pc1)
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint=ep1,
            payload=rt_payload,
            params_canonical=pc1,
            purpose="LABELING_BASE",
            priority=100,
            dedupe_key=dk1,
            correlation_id=_mk_correlation_id(td, stage, dk1),
            deadline_sec=5,
        )
    )

    # 2) daily history
    ep2 = "cmd_history_quotation"
    pc2 = _params_canonical(hist_payload)
    dk2 = _mk_dedupe_key(td, ep2, pc2)
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint=ep2,
            payload=hist_payload,
            params_canonical=pc2,
            purpose="LABELING_BASE" if stage == 0 else "LABELING_EXPAND",
            priority=80,
            dedupe_key=dk2,
            correlation_id=_mk_correlation_id(td, stage, dk2),
            deadline_sec=8,
        )
    )

    # 3) high frequency
    ep3 = "high_frequency"
    pc3 = _params_canonical(hf_payload)
    dk3 = _mk_dedupe_key(td, ep3, pc3)
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint=ep3,
            payload=hf_payload,
            params_canonical=pc3,
            purpose="LABELING_BASE" if stage == 0 else "LABELING_REFRESH",
            priority=60,
            dedupe_key=dk3,
            correlation_id=_mk_correlation_id(td, stage, dk3),
            deadline_sec=10,
        )
    )

    # ---- planner_state：持久化到 watchlist ----
    st: dict[str, Any] = dict(planner_state or {})
    st["stage"] = stage
    st["last_planned_day"] = td
    st["history_days"] = history_days
    st["hf_limit"] = hf_limit

    return Plan(symbol=symbol, trading_day=td, stage=stage, requests=reqs, planner_state=st)


def calc_refresh_seconds(hit_count: int) -> int:
    """hit_count 越高，刷新越频繁。"""
    hc = int(hit_count or 0)
    base = int(getattr(settings, "LABELING_REFRESH_BASE_SEC", 21600))     # 6h
    active = int(getattr(settings, "LABELING_REFRESH_ACTIVE_SEC", 1800))  # 30min

    if hc >= 5:
        return max(60, min(base, active // 2))
    if hc >= 2:
        return max(60, min(base, active))
    return max(60, base)


def calc_refresh_sec(hit_count: int) -> int:
    return calc_refresh_seconds(hit_count)
