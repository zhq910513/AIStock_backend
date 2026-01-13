from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.config import settings
from app.utils.time import now_shanghai


@dataclass
class PlannedRequest:
    symbol: str
    endpoint: str
    params: dict
    purpose: str
    priority: int
    dedupe_key: str


@dataclass
class Plan:
    symbol: str
    trading_day: str
    stage: int
    requests: list[PlannedRequest]


def _stage_from_hits(hit_count: int) -> int:
    hc = int(hit_count or 0)
    if hc >= 5:
        return 2
    if hc >= 2:
        return 1
    return 0


def _dedupe_key(symbol: str, endpoint: str, params: dict) -> str:
    items = sorted((k, str(v)) for k, v in (params or {}).items())
    joined = "&".join([f"{k}={v}" for k, v in items])
    return f"{symbol}|{endpoint}|{joined}"


def build_plan(
    symbol: str,
    trading_day: str | None = None,
    hit_count: int = 0,
    planner_state: dict[str, Any] | None = None,
    **_: Any,
) -> Plan:
    """
    Rule-based planner (v1),兼容两种调用方式：
      - API router: build_plan(symbol=..., hit_count=..., planner_state={...})
      - Pipeline:   build_plan(symbol=..., trading_day=..., hit_count=...)

    planner_state 当前版本先作为“未来扩维/去重/阶段记忆”的输入保留，
    v1 里不强依赖它（避免因为 state 格式变化导致启动/接口崩溃）。
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

    # iFinD HTTP: params里既放 ths_code，也放 symbol，便于你们后端做 adapter/兼容 mock
    rt_payload = {"symbol": symbol, "ths_code": symbol}

    end_dt = now
    start_dt = now - timedelta(days=history_days)

    hist_payload = {
        "symbol": symbol,
        "ths_code": symbol,
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
        "period": "D",
    }

    hf_payload = {
        "symbol": symbol,
        "ths_code": symbol,
        "day": td,
        "limit": hf_limit,
    }

    reqs: list[PlannedRequest] = []

    # 1) real-time quote
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint="IFIND_HTTP.real_time_quotation",
            params=rt_payload,
            purpose="LABELING_BASE",
            priority=100,
            dedupe_key=_dedupe_key(symbol, "IFIND_HTTP.real_time_quotation", rt_payload),
        )
    )

    # 2) daily history
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint="IFIND_HTTP.cmd_history_quotation",
            params=hist_payload,
            purpose="LABELING_BASE" if stage == 0 else "LABELING_EXPAND",
            priority=80,
            dedupe_key=_dedupe_key(symbol, "IFIND_HTTP.cmd_history_quotation", hist_payload),
        )
    )

    # 3) high frequency
    reqs.append(
        PlannedRequest(
            symbol=symbol,
            endpoint="IFIND_HTTP.high_frequency",
            params=hf_payload,
            purpose="LABELING_BASE" if stage == 0 else "LABELING_REFRESH",
            priority=60,
            dedupe_key=_dedupe_key(symbol, "IFIND_HTTP.high_frequency", hf_payload),
        )
    )

    # planner_state 预留：未来你要“同一股票持续扩维/持续更新”时会在这里用到
    _ = planner_state  # noqa: F841

    return Plan(symbol=symbol, trading_day=td, stage=stage, requests=reqs)


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


# 向后兼容别名：避免任何地方写旧名导致启动/接口崩溃
def calc_refresh_sec(hit_count: int) -> int:
    return calc_refresh_seconds(hit_count)
