from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.utils.time import now_shanghai, trading_day_str
from app.utils.symbols import normalize_symbol


def _json_canonical(obj: Any) -> str:
    # Deterministic JSON used for dedupe/keying. Keep it compact and stable.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _params_canonical(payload: dict) -> str:
    return _json_canonical(payload or {})


def _mk_dedupe_key(trading_day: str, endpoint: str, params_canonical: str) -> str:
    h = hashlib.sha256(f"{endpoint}|{params_canonical}".encode("utf-8")).hexdigest()[:16]
    return f"LP:{trading_day}:{endpoint}:{h}"


def _mk_correlation_id(trading_day: str, stage: int, dedupe_key: str) -> str:
    # short correlation id, stable and human readable
    h = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:10]
    return f"LP:{trading_day}:{stage}:{h}"


@dataclass(frozen=True)
class PlannedRequest:
    dedupe_key: str
    correlation_id: str
    purpose: str
    endpoint: str
    params_canonical: str
    payload: dict
    deadline_sec: int


@dataclass(frozen=True)
class Plan:
    symbol: str
    trading_day: str
    requests: list[PlannedRequest]
    planner_state: dict


def build_plan(
    *,
    symbol: str,
    hit_count: int,
    planner_state: dict | None,
    trading_day: str | None = None,
) -> Plan:
    """Planner for labeling pipeline.

    The goal is simple and robust:
    - Always fetch RT quote + short history + small HF slice first.
    - Expand history/hf window as hit_count increases.
    - All requests are deduped by (endpoint, canonical params).
    """
    sym = normalize_symbol(symbol)
    td = (trading_day or trading_day_str(now_shanghai())).strip()
    if len(td) != 8 or not td.isdigit():
        # safety: re-derive from local time
        td = now_shanghai().strftime("%Y%m%d")

    hit = max(int(hit_count or 0), 0)
    st = dict(planner_state or {})
    version = int(st.get("planner_version") or 1)
    stage = int(st.get("stage") or 0) + 1

    # Expand windows with hit_count, bounded.
    # base: last ~10 trading days; expand up to ~60 days.
    hist_days = min(10 + hit * 5, 60)

    # High frequency: keep small by default; expand modestly.
    hf_limit = min(200 + hit * 100, 1200)

    # RT: always needed
    rt_payload = {"symbol": sym}
    rt_pc = _params_canonical(rt_payload)
    rt_dk = _mk_dedupe_key(td, "real_time_quotation", rt_pc)

    # HIST: include date window
    end_dt = datetime.strptime(td, "%Y%m%d")
    start_dt = end_dt - timedelta(days=hist_days)
    hist_payload = {
        "symbol": sym,
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
        "fields": ["open", "high", "low", "close", "volume", "amount"],
    }
    hist_pc = _params_canonical(hist_payload)
    hist_dk = _mk_dedupe_key(td, "cmd_history_quotation", hist_pc)

    # HF: keep bounded
    hf_payload = {"symbol": sym, "limit": hf_limit}
    hf_pc = _params_canonical(hf_payload)
    hf_dk = _mk_dedupe_key(td, "high_frequency", hf_pc)

    reqs = [
        PlannedRequest(
            dedupe_key=rt_dk,
            correlation_id=_mk_correlation_id(td, stage, rt_dk),
            purpose="LABELING_BASE",
            endpoint="real_time_quotation",
            params_canonical=rt_pc,
            payload=rt_payload,
            deadline_sec=5,
        ),
        PlannedRequest(
            dedupe_key=hist_dk,
            correlation_id=_mk_correlation_id(td, stage, hist_dk),
            purpose="LABELING_BASE",
            endpoint="cmd_history_quotation",
            params_canonical=hist_pc,
            payload=hist_payload,
            deadline_sec=8,
        ),
        PlannedRequest(
            dedupe_key=hf_dk,
            correlation_id=_mk_correlation_id(td, stage, hf_dk),
            purpose="LABELING_BASE",
            endpoint="high_frequency",
            params_canonical=hf_pc,
            payload=hf_payload,
            deadline_sec=10,
        ),
    ]

    # Refresh policy hint (seconds) for pipeline
    # Higher hit -> more frequent refresh, bounded.
    next_refresh_in_sec = max(30, 300 - hit * 20)
    next_refresh_in_sec = min(next_refresh_in_sec, 300)

    st.update(
        {
            "planner_version": version,
            "stage": stage,
            "last_td": td,
            "hist_days": hist_days,
            "hf_limit": hf_limit,
            "next_refresh_in_sec": next_refresh_in_sec,
        }
    )

    return Plan(symbol=sym, trading_day=td, requests=reqs, planner_state=st)
