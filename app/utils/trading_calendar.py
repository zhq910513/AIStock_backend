from __future__ import annotations

from datetime import datetime, timedelta

from app.config import settings


def _holidays_set() -> set[str]:
    """Return configured holidays as YYYYMMDD strings.

    P0 contract (simple but useful):
    - Weekends are treated as non-trading days.
    - Additional holidays can be provided via settings.TRADING_HOLIDAYS
      as comma-separated YYYYMMDD.

    Later we can replace this with a proper exchange calendar (iFinD/交易所日历)
    to correctly handle make-up trading days.
    """
    raw = getattr(settings, "TRADING_HOLIDAYS", "")
    out: set[str] = set()
    for x in (raw or "").split(","):
        s = x.strip()
        if len(s) == 8 and s.isdigit():
            out.add(s)
    return out


def is_trading_day(day: str) -> bool:
    """Whether YYYYMMDD is a trading day (approx).

    Rule:
    - Weekend => False
    - In configured holidays => False
    - Otherwise True
    """
    if not (isinstance(day, str) and len(day) == 8 and day.isdigit()):
        return False
    dt = datetime.strptime(day, "%Y%m%d")
    if dt.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    if day in _holidays_set():
        return False
    return True


def prev_trading_day(day: str) -> str:
    """Return the previous trading day for a given YYYYMMDD (approx)."""
    dt = datetime.strptime(day, "%Y%m%d")
    for _ in range(366):
        dt = dt - timedelta(days=1)
        s = dt.strftime("%Y%m%d")
        if is_trading_day(s):
            return s
    return day


def iter_prev_trading_days(end_day: str, n: int) -> list[str]:
    """Return a list of last n trading days ending at end_day (inclusive if trading)."""
    out: list[str] = []
    cur = end_day
    if is_trading_day(cur):
        out.append(cur)
    while len(out) < max(0, int(n)):
        cur = prev_trading_day(cur)
        if cur == out[-1]:
            break
        out.append(cur)
    return out
