from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional


IST = ZoneInfo("Asia/Kolkata")

PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)
REGULAR_OPEN = time(9, 15)
REGULAR_CLOSE = time(15, 30)
CLOSING_SESSION_START = time(15, 40)
CLOSING_SESSION_END = time(16, 0)


TRADING_HOLIDAYS_2026: Dict[date, str] = {
    date(2026, 1, 15): "Municipal Corporation Election - Maharashtra",
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 3): "Holi",
    date(2026, 3, 26): "Shri Ram Navami",
    date(2026, 3, 31): "Shri Mahavir Jayanti",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 28): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 8): "Diwali Laxmi Pujan (Muhurat trading only; regular session closed)",
    date(2026, 11, 10): "Diwali-Balipratipada",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
    date(2026, 12, 25): "Christmas",
}


@dataclass(frozen=True)
class MarketClock:
    now_ist: str
    today: str
    is_trading_day: bool
    session: str
    holiday_name: Optional[str]
    next_trading_day: str
    next_regular_open: str


def _as_date(value: date | datetime | str) -> date:
    """Normalize a date-ish value to a :class:`~datetime.date` in IST.

    A tz-aware ``datetime`` is converted to IST before the date component is
    read. A *naive* ``datetime`` is assumed to already be in IST (the convention
    used by the runtime scheduler). Previously naive values were treated as
    "whatever timezone the caller happened to be in", which silently produced
    off-by-one-day errors when a UTC-naive timestamp (e.g. ``datetime.utcnow()``)
    was passed to :func:`is_trading_day` or :func:`holiday_name`.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=IST)
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def holiday_name(value: date | datetime | str) -> Optional[str]:
    return TRADING_HOLIDAYS_2026.get(_as_date(value))


def is_trading_day(value: date | datetime | str) -> bool:
    day = _as_date(value)
    return day.weekday() < 5 and day not in TRADING_HOLIDAYS_2026


def next_trading_day(value: date | datetime | str) -> date:
    day = _as_date(value)
    probe = day + timedelta(days=1)
    while not is_trading_day(probe):
        probe += timedelta(days=1)
    return probe


def market_clock(now: Optional[datetime] = None) -> MarketClock:
    now = (now or datetime.now(IST)).astimezone(IST)
    today = now.date()

    if not is_trading_day(today):
        session = "holiday" if holiday_name(today) else "weekend"
    else:
        current_t = now.time()
        if PRE_OPEN_START <= current_t < PRE_OPEN_END:
            session = "pre_open"
        elif REGULAR_OPEN <= current_t <= REGULAR_CLOSE:
            session = "regular"
        elif CLOSING_SESSION_START <= current_t <= CLOSING_SESSION_END:
            session = "closing_session"
        elif current_t < PRE_OPEN_START:
            session = "before_open"
        else:
            session = "after_close"

    next_day = today if is_trading_day(today) and now.time() < REGULAR_OPEN else next_trading_day(today)
    next_open = datetime.combine(next_day, REGULAR_OPEN, tzinfo=IST)

    return MarketClock(
        now_ist=now.isoformat(),
        today=today.isoformat(),
        is_trading_day=is_trading_day(today),
        session=session,
        holiday_name=holiday_name(today),
        next_trading_day=next_day.isoformat(),
        next_regular_open=next_open.isoformat(),
    )
