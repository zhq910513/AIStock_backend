from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models
from app.utils.trading_calendar import next_n_trading_days


@dataclass
class Label3DResult:
    instrument_id: str
    signal_day_T: date
    cutoff_ts: datetime

    entry_day: date
    entry_px: Decimal
    max_future_high_3d: Decimal

    label_tp5_3d: bool
    label_tp8_3d: bool

    label_liquidity_ok: Optional[bool]
    label_gap_risk: Optional[bool]
    label_limitup_lock: Optional[bool]

    raw_refs: dict


def _get_daily(session: Session, instrument_id: str, d: date) -> Optional[models.FactDailyOHLCV]:
    return session.execute(
        select(models.FactDailyOHLCV).where(
            models.FactDailyOHLCV.instrument_id == instrument_id,
            models.FactDailyOHLCV.trading_day == d,
        )
    ).scalar_one_or_none()


def compute_label_3d(
    session: Session,
    instrument_id: str,
    signal_day_T: date,
    cutoff_ts: datetime,
    label_version: str | None = None,
) -> Label3DResult:
    """Compute labels anchored at Open(T+1), using High(T+1..T+3).

    This follows the project's "physical law" exactly:
      entry_px = Open(T+1)
      max_future_high_3d = max(High(T+1), High(T+2), High(T+3))
      tp5/tp8 determined by whether achievable return crosses 5%/8%.

    Aux labels (optional but recommended):
      - gap_risk: Open(T+1) vs Close(T)
      - liquidity_ok: min amount threshold across T+1..T+3
      - limitup_lock: heuristic lock detection on T+1
    """

    lv = (label_version or settings.LABEL3D_VERSION or "v1").strip()

    # trading days after T: [T+1, T+2, T+3]
    future_days = next_n_trading_days(signal_day_T, 3, session=session)
    if len(future_days) != 3:
        raise ValueError("unable to resolve T+1..T+3 via trading calendar")

    d1, d2, d3 = future_days
    row_T = _get_daily(session, instrument_id, signal_day_T)
    row_1 = _get_daily(session, instrument_id, d1)
    row_2 = _get_daily(session, instrument_id, d2)
    row_3 = _get_daily(session, instrument_id, d3)

    if row_1 is None or row_2 is None or row_3 is None:
        missing = [str(x) for x, r in ((d1, row_1), (d2, row_2), (d3, row_3)) if r is None]
        raise ValueError(f"missing daily ohlcv for {instrument_id}: {','.join(missing)}")

    entry_px = Decimal(row_1.open)
    max_future_high = max(Decimal(row_1.high), Decimal(row_2.high), Decimal(row_3.high))

    if entry_px <= 0:
        raise ValueError("entry_px must be positive")

    r = (max_future_high / entry_px) - Decimal("1")
    label_tp5 = bool(r >= Decimal("0.05"))
    label_tp8 = bool(r >= Decimal("0.08"))

    # Aux labels
    liq_min = Decimal(str(settings.LABEL3D_LIQUIDITY_MIN_AMOUNT))
    amounts = [row_1.amount, row_2.amount, row_3.amount]
    if all(a is not None for a in amounts):
        label_liq = all(Decimal(a) >= liq_min for a in amounts)
    else:
        label_liq = None

    # gap risk uses Close(T) if present
    if row_T is not None and row_T.close and Decimal(row_T.close) > 0:
        gap = (Decimal(row_1.open) / Decimal(row_T.close)) - Decimal("1")
        label_gap = bool(gap > Decimal(str(settings.LABEL3D_GAP_RISK_THRESH)))
    else:
        label_gap = None

    # lock heuristic: if open==high==low or very tight range (within 0.02%) treat as lock
    # (strict limit-up detection requires limit_up_px / rule-set)
    try:
        o = Decimal(row_1.open)
        h = Decimal(row_1.high)
        l = Decimal(row_1.low)
        if o > 0:
            tight = (h - l) / o
            label_lock = bool((o == h == l) or (tight <= Decimal("0.0002")))
        else:
            label_lock = None
    except Exception:
        label_lock = None

    raw_refs = {
        "label_version": lv,
        "signal_day_T": signal_day_T.isoformat(),
        "entry_day": d1.isoformat(),
        "future_days": [d1.isoformat(), d2.isoformat(), d3.isoformat()],
        "daily_raw_hash": {
            str(signal_day_T): getattr(row_T, "raw_hash", None) if row_T else None,
            str(d1): getattr(row_1, "raw_hash", None),
            str(d2): getattr(row_2, "raw_hash", None),
            str(d3): getattr(row_3, "raw_hash", None),
        },
    }

    return Label3DResult(
        instrument_id=instrument_id,
        signal_day_T=signal_day_T,
        cutoff_ts=cutoff_ts,
        entry_day=d1,
        entry_px=entry_px,
        max_future_high_3d=max_future_high,
        label_tp5_3d=label_tp5,
        label_tp8_3d=label_tp8,
        label_liquidity_ok=label_liq,
        label_gap_risk=label_gap,
        label_limitup_lock=label_lock,
        raw_refs=raw_refs,
    )


def upsert_label_3d(
    session: Session,
    instrument_id: str,
    signal_day_T: date,
    cutoff_ts: datetime,
    label_version: str | None = None,
) -> models.ModelTrainingLabel3D:
    """Compute and upsert ModelTrainingLabel3D."""
    res = compute_label_3d(session, instrument_id, signal_day_T, cutoff_ts, label_version=label_version)
    lv = (label_version or settings.LABEL3D_VERSION or "v1").strip()

    existing = session.execute(
        select(models.ModelTrainingLabel3D).where(
            models.ModelTrainingLabel3D.instrument_id == instrument_id,
            models.ModelTrainingLabel3D.signal_day_T == signal_day_T,
            models.ModelTrainingLabel3D.cutoff_ts == cutoff_ts,
            models.ModelTrainingLabel3D.label_version == lv,
        )
    ).scalar_one_or_none()

    if existing is None:
        row = models.ModelTrainingLabel3D(
            instrument_id=instrument_id,
            signal_day_T=signal_day_T,
            cutoff_ts=cutoff_ts,
            entry_day=res.entry_day,
            entry_px=res.entry_px,
            max_future_high_3d=res.max_future_high_3d,
            label_tp5_3d=res.label_tp5_3d,
            label_tp8_3d=res.label_tp8_3d,
            label_liquidity_ok=res.label_liquidity_ok,
            label_gap_risk=res.label_gap_risk,
            label_limitup_lock=res.label_limitup_lock,
            label_version=lv,
            raw_refs=res.raw_refs,
        )
        session.add(row)
        session.flush()
        return row

    existing.entry_day = res.entry_day
    existing.entry_px = res.entry_px
    existing.max_future_high_3d = res.max_future_high_3d
    existing.label_tp5_3d = res.label_tp5_3d
    existing.label_tp8_3d = res.label_tp8_3d
    existing.label_liquidity_ok = res.label_liquidity_ok
    existing.label_gap_risk = res.label_gap_risk
    existing.label_limitup_lock = res.label_limitup_lock
    existing.raw_refs = res.raw_refs
    return existing
