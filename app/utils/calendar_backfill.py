from __future__ import annotations

"""Trading calendar backfill.

We treat the DB table `trading_calendar_day` as the source of truth.

When the table is empty (or missing a requested date range), we can
programmatically compute an exchange calendar and upsert it into DB.

For China A-share, we start with XSHG (SSE) as a practical approximation.
If you later want to be stricter, we can extend to XSHE or merge calendars.
"""

from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd
import pandas_market_calendars as mcal
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import models


def _to_date(d: str | date | datetime) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date) -> Iterable[date]:
    if end < start:
        start, end = end, start
    for d in pd.date_range(start=start, end=end, freq="D").date:
        yield d


def compute_trading_days(
    *,
    start_day: str | date | datetime,
    end_day: str | date | datetime,
    market_code: str = "XSHG",
) -> set[date]:
    """Compute trading days using pandas_market_calendars."""

    s = _to_date(start_day)
    e = _to_date(end_day)
    cal = mcal.get_calendar(market_code)
    sched = cal.schedule(start_date=s, end_date=e)
    # Some calendars (e.g. SSE) may not have future holiday data beyond a cutoff.
    # If schedule is empty for a non-empty range, fall back to weekday-only.
    if sched.empty:
        return {d for d in _daterange(s, e) if d.weekday() < 5}

    # schedule index is timezone-aware timestamps; normalize to dates
    return set(pd.to_datetime(sched.index).date)


def ensure_trading_calendar(
    session: Session,
    *,
    start_day: str | date | datetime,
    end_day: str | date | datetime,
    market_code: str = "XSHG",
    note: Optional[str] = None,
) -> dict[str, int]:
    """Ensure trading_calendar_day covers the range [start_day, end_day].

    Upserts rows in `TradingCalendarDay` for all calendar days in the range.
    Returns counts.
    """

    s = _to_date(start_day)
    e = _to_date(end_day)
    trading_days = compute_trading_days(start_day=s, end_day=e, market_code=market_code)

    # Preload existing
    existing = {
        r.day: r
        for r in session.execute(
            select(models.TradingCalendarDay).where(models.TradingCalendarDay.day.between(min(s, e), max(s, e)))
        ).scalars().all()
    }

    inserted = 0
    updated = 0
    for d in _daterange(s, e):
        is_open = d in trading_days
        row = existing.get(d)
        if row is None:
            session.add(models.TradingCalendarDay(day=d, is_open=bool(is_open), note=note))
            inserted += 1
        else:
            if row.is_open != bool(is_open) or (note and row.note != note):
                row.is_open = bool(is_open)
                if note:
                    row.note = note
                updated += 1

    return {"inserted": inserted, "updated": updated}
