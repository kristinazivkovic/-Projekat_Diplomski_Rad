"""Jedini izvor istine za to šta se računa kao svećica iz redovnog
trgovinskog vremena (regular-trading-hours).

Zamenjuje fiksni vremenski prozor (npr. 13:30-20:00 UTC), za koji je
utvrđeno da propušta 70-81 svećicu posle zatvaranja u sesije sa ranijim
zatvaranjem (provereno: 2015-11-27. i 2015-12-24. svaka sadrži tačno 210
pravih svećica iz sesije plus rep tankih svećica sa tržišta posle
zatvaranja, sa ~40 puta manjim medijanom obima). Pravi berzanski kalendar
zna koje sesije imaju ranije zatvaranje i daje tačan prozor za svaki dan.
"""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd

from projekat.config import EXCHANGE_CALENDAR, EXPECTED_BARS_FULL, EXPECTED_BARS_HALF

_calendar = xcals.get_calendar(EXCHANGE_CALENDAR)


def is_session(day: str | date) -> bool:
    return bool(_calendar.is_session(pd.Timestamp(day)))


def session_window(day: str | date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Vremenske oznake (otvaranje, zatvaranje) u UTC za trgovinsku sesiju."""
    ts = pd.Timestamp(day)
    return _calendar.session_open(ts), _calendar.session_close(ts)


def is_half_day(day: str | date) -> bool:
    """True ako se sesija zatvara pre redovnog zatvaranja u 16:00 po istočnom vremenu."""
    open_ts, close_ts = session_window(day)
    duration_minutes = (close_ts - open_ts).total_seconds() / 60
    return duration_minutes < 389  # redovna sesija traje 390 minuta; zaštita zbog zaokruživanja


def expected_bar_count(day: str | date, grid: str) -> int:
    """Očekivan broj svećica sesije za ovaj konkretan dan na ovoj mreži,
    razlikujući pune dane (390/78/39) od dana sa ranijim zatvaranjem (210/42/21)."""
    table = EXPECTED_BARS_HALF if is_half_day(day) else EXPECTED_BARS_FULL
    return table[grid]


def sessions_between(start: str, end: str) -> pd.DatetimeIndex:
    return _calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
