from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.data_provider import get_data_provider
from app.database import models
from app.utils.time import now_shanghai


log = logging.getLogger(__name__)


def _parse_ifind_table(raw: dict[str, Any]) -> dict[str, list[Any]] | None:
    try:
        tabs = raw.get("tables") or raw.get("Tables")
        if not tabs or not isinstance(tabs, list):
            return None
        tab0 = tabs[0] or {}
        t = tab0.get("table") or tab0.get("Table")
        if isinstance(t, dict):
            return t
        return None
    except Exception:
        return None


def _pick_last_num(xs: list[Any] | None) -> float | None:
    if not xs:
        return None
    for v in reversed(xs):
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _infer_limit_ratio(symbol: str, name: str | None = None) -> float:
    """Best-effort涨跌停比例。

    NOTE:
    - 当前项目实际主要是 SZ 主板（0开头）=> 10%，所以这足够先闭环。
    - ST/退市等特殊规则需要额外数据源确认（缺条件会打 error_tag）。
    """
    sym = (symbol or "").upper()
    # ST (best-effort by name)
    if name and "ST" in name.upper():
        return 0.05

    # STAR / ChiNext 20%
    if sym.startswith("688") or sym.startswith("689"):
        return 0.20
    if sym.startswith("300") or sym.startswith("301"):
        return 0.20

    # default main board
    return 0.10


def _round_price(px: float) -> float:
    # A-share tick size is typically 0.01
    return round(float(px) + 1e-12, 2)


@dataclass
class LabelResult:
    labeled: int
    skipped: int
    failed: int
    details: list[dict[str, Any]]


def label_decisions_for_day(s: Session, *, label_day: str) -> LabelResult:
    """Label ModelDecision rows for a given label_day (YYYYMMDD).

    This is meant to be triggered around 15:30 on T (label_day), to label decisions produced for that day.
    """
    provider = get_data_provider()

    # fetch decisions for that day
    decisions = (
        s.execute(select(models.ModelDecision).where(models.ModelDecision.decision_day == label_day))
        .scalars()
        .all()
    )

    labeled = 0
    skipped = 0
    failed = 0
    details: list[dict[str, Any]] = []

    for d in decisions:
        # already labeled?
        existing = (
            s.execute(
                select(models.DecisionLabel).where(models.DecisionLabel.decision_id == d.decision_id).where(models.DecisionLabel.label_day == label_day)
            )
            .scalars()
            .first()
        )
        if existing is not None:
            skipped += 1
            continue

        # get snapshot evidence for name + extra context
        snap_ev = (
            s.execute(
                select(models.DecisionEvidence)
                .where(models.DecisionEvidence.decision_id == d.decision_id)
                .where(models.DecisionEvidence.reason_code == "RAW_SNAPSHOT")
            )
            .scalars()
            .first()
        )
        snap = dict((snap_ev.evidence_fields or {}) if snap_ev else {})
        name = str(snap.get("name") or snap.get("candidate_name") or "") or None

        # get T-1 close and T high/close/low via provider
        # We request 2 bars ending at label_day.
        prev_day = _prev_calendar_day(label_day)
        payload = {
            "symbol": d.symbol,
            "begin_date": prev_day,
            "end_date": label_day,
            "limit": 2,
            "indicators": "open,high,low,close,amount,volume,time",
        }

        error_tags: list[str] = []
        try:
            resp = provider.call("cmd_history_quotation", payload)
            raw = resp.raw or {}
            tab = _parse_ifind_table(raw) or {}
            closes = tab.get("close") or tab.get("Close")
            highs = tab.get("high") or tab.get("High")
            lows = tab.get("low") or tab.get("Low")

            close_t = _pick_last_num(closes)
            high_t = _pick_last_num(highs)
            low_t = _pick_last_num(lows)

            # prev close: second last
            prev_close = None
            if isinstance(closes, list) and len(closes) >= 2:
                try:
                    prev_close = float(closes[-2])
                except Exception:
                    prev_close = None
            if prev_close is None:
                error_tags.append("MISSING_PREV_CLOSE")
                # fallback: use close_t (not ideal)
                prev_close = close_t

            if prev_close is None or close_t is None or high_t is None:
                raise ValueError("missing OHLC for labeling")

            ratio = _infer_limit_ratio(d.symbol, name)
            if ratio == 0.05 and not name:
                error_tags.append("ST_RULE_GUESS_BY_NAME")

            limit_price = _round_price(prev_close * (1.0 + ratio))
            hit = bool(high_t >= (limit_price - 1e-9))

            close_ret = float(close_t / prev_close - 1.0) if prev_close else 0.0
            max_ret = float(high_t / prev_close - 1.0) if prev_close else 0.0
            dd = 0.0
            if low_t is not None and prev_close:
                dd = float(low_t / prev_close - 1.0)

            s.add(
                models.DecisionLabel(
                    decision_id=d.decision_id,
                    label_day=label_day,
                    hit_limitup=hit,
                    close_return=close_ret,
                    max_return=max_ret,
                    drawdown=dd,
                    error_tags=error_tags,
                )
            )
            labeled += 1
            details.append(
                {
                    "decision_id": d.decision_id,
                    "symbol": d.symbol,
                    "hit_limitup": hit,
                    "limit_price": limit_price,
                    "prev_close": prev_close,
                    "close": close_t,
                    "high": high_t,
                    "error_tags": error_tags,
                }
            )
        except Exception as e:
            failed += 1
            details.append({"decision_id": d.decision_id, "symbol": d.symbol, "error": repr(e)})
            # do not raise; continue

    return LabelResult(labeled=labeled, skipped=skipped, failed=failed, details=details)


def _prev_calendar_day(yyyymmdd: str) -> str:
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    return (dt - timedelta(days=1)).strftime("%Y%m%d")
