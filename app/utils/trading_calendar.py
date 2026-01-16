from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

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


def is_trading_day(d: str | date | datetime, session: Optional[Session] = None) -> bool:
    """Return whether date is a trading day.

    If `trading_calendar_day` has data, it is authoritative.
    Otherwise fall back to weekend-only logic.
    """
    day = _to_date(d)
    if session is not None:
        row = session.execute(select(models.TradingCalendarDay).where(models.TradingCalendarDay.day == day)).scalar_one_or_none()
        if row is not None:
            return bool(row.is_open)
    # fallback: Mon..Fri
    return day.weekday() < 5


def next_trading_day(d: str | date | datetime, session: Optional[Session] = None) -> date:
    """Get the next trading day after d (strictly > d)."""
    day = _to_date(d)
    cur = day + timedelta(days=1)
    for _ in range(366):
        if is_trading_day(cur, session=session):
            return cur
        cur += timedelta(days=1)
    raise RuntimeError("No trading day found within 1 year; calendar data may be missing")


def next_n_trading_days(start_day: str | date | datetime, n: int, session: Optional[Session] = None) -> list[date]:
    """Return next n trading days after start_day (strictly after)."""
    if n <= 0:
        return []
    res: list[date] = []
    cur = _to_date(start_day)
    for _ in range(n):
        cur = next_trading_day(cur, session=session)
        res.append(cur)
    return res


def prev_trading_day(d: str | date | datetime, session: Optional[Session] = None) -> date:
    """Get the previous trading day before d (strictly < d)."""
    day = _to_date(d)
    cur = day - timedelta(days=1)
    for _ in range(366):
        if is_trading_day(cur, session=session):
            return cur
        cur -= timedelta(days=1)
    raise RuntimeError("No trading day found within 1 year; calendar data may be missing")


def iter_prev_trading_days(trading_day: str | date | datetime, n: int, session: Optional[Session] = None) -> list[str]:
    """Return up to n trading days ending at trading_day, in DESC order.

    Example: trading_day=20260116, n=3 -> ['20260116','20260115','20260114'] (skipping weekends/holidays).
    """
    if n <= 0:
        return []
    cur = _to_date(trading_day)
    out: list[str] = []
    # include trading_day if it's a trading day, otherwise start from previous trading day
    if not is_trading_day(cur, session=session):
        cur = prev_trading_day(cur, session=session)
    for _ in range(n):
        out.append(cur.strftime("%Y%m%d"))
        cur = prev_trading_day(cur, session=session)
    return out


def trading_days_between(start_day: str | date | datetime, end_day: str | date | datetime, session: Optional[Session] = None) -> list[date]:
    """Return trading days in [start_day, end_day] (inclusive), ascending."""
    s = _to_date(start_day)
    e = _to_date(end_day)
    if e < s:
        s, e = e, s
    out: list[date] = []
    cur = s
    # Hard cap to avoid infinite loops when calendar is broken
    for _ in range(3660):
        if cur > e:
            break
        if is_trading_day(cur, session=session):
            out.append(cur)
        cur = cur + timedelta(days=1)
    return out
