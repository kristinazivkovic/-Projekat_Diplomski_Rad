"""Filtriranje sesija i čišćenje kvaliteta svećica.

Dve stvari koje fiksni vremenski prozor pogrešno radi, obe ovde ispravljene
korišćenjem pravog berzanskog kalendara (projekat.data.calendar):

1. Curenje posle zatvaranja (post-close leakage): fiksni prozor 13:30-20:00 UTC
   zadržava ~70-81 tankih svećica posle zatvaranja tržišta na dane sa ranijim
   zatvaranjem (proveren medijan obima 460 naspram 18.603 u pravoj sesiji --
   40 puta veća razlika). Filtriranje prema kalendarskom prozoru same sesije
   ih uklanja.
2. Grub filter "<80% od 390" ne bi pouzdano razlikovao pravo rano zatvaranje
   od rupe u podacima. Poređenje sa očekivanim brojem za taj konkretan dan
   čini da filter proverava pravu stvar.

JEDAN SKUP DANA ZA SVE MREŽE. Odluka o zadržavanju dana ne zavisi od mreže:
dan se zadržava samo ako prolazi proveru za SVAKU mrežu iz config.GRIDS.
Tako 1m, 5m i 15m imaju tačno isti skup dana po konstrukciji, bez obzira na
to koje se mreže u datom pokretanju grade:

  * skraćeni dani (rano zatvaranje) se izbacuju u celini (config.
    DROP_HALF_DAYS) -- 15m skraćeni dan daje svega 13 prinosa, pa bi ga 15m
    izgubio a 1m/5m zadržali, i A1 nivoi bi imali različit skup (baš
    mirnih) dana;
  * relativni prag (>= 80% očekivanog broja 1m svećica) se meri na 1m
    svećicama, iz kojih se grade sve mreže;
  * apsolutni prag (>= MIN_INTRADAY_OBS prinosa) se meri za SVAKU mrežu, na
    broju prinosa koje ta mreža zaista daje -- dan kome bilo koja mreža daje
    premalo prinosa ispada iz svih.

Prag se primenjuje za CELU mrežu, a ne po pojedinačnom ocenjivaču --
primena samo unutar pojedinačnih A2 ocenjivača ostavila bi BPV dizajne sa
više upotrebljivih dana nego MedRV/TrBPV dizajne na istoj mreži.

Svaki izbačen dan dobija RAZLOG (EXCLUSION_REASONS), koji build_panel.py
beleži za dijagnostiku.
"""

from __future__ import annotations

import pandas as pd

from projekat.config import DROP_HALF_DAYS, GRID_MINUTES, GRIDS, MIN_BAR_COVERAGE, MIN_INTRADAY_OBS
from projekat.data.calendar import expected_bar_count, is_half_day, is_session, session_window

#: razlozi izbacivanja dana, redom kojim se proveravaju
EXCLUSION_REASONS = (
    "not_a_session",            # svećice na vikend/praznik (dešava se kod dobavljača)
    "half_day",                 # rano zatvaranje, izbačeno sa svih mreža
    "insufficient_1m_coverage", # < 80% očekivanih 1m svećica (rupa u podacima)
    "below_min_intraday_obs",   # neka mreža daje < MIN_INTRADAY_OBS prinosa
)


def filter_session_bars(bars: pd.DataFrame, day: str) -> pd.DataFrame:
    """Zadrži samo svećice koje upadaju u kalendarski prozor sesije za ovaj
    konkretan dan. `bars` mora imati kolonu `timestamp` sa UTC vremenskom zonom."""
    if not is_session(day):
        return bars.iloc[0:0]
    open_ts, close_ts = session_window(day)
    mask = (bars["timestamp"] >= open_ts) & (bars["timestamp"] < close_ts)
    return bars.loc[mask]


def drop_thin_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Odbaci svećice sa nultim obimom I nultim prinosom (open == close).
    Svećica sa pravim obimom ali bez promene cene je legitiman miran trenutak;
    svećica bez obima i bez pomeranja je artefakt popune/odsustva trgovanja."""
    zero_volume = bars["volume"] == 0
    zero_return = bars["open"] == bars["close"]
    return bars.loc[~(zero_volume & zero_return)]


def coverage_failure(bars: pd.DataFrame, day: str, grids=GRIDS) -> str | None:
    """Razlog zbog kog dan NE prolazi proveru pokrivenosti, ili None.

    `bars` su UVEK 1-minutne svećice (čišćenje ide pre preuzorkovanja).
    Relativni prag se meri na 1m svećicama; apsolutni na broju PRINOSA koje
    svaka mreža iz `grids` zaista daje (isto što panel beleži kao n_obs)."""
    if len(bars) < MIN_BAR_COVERAGE * expected_bar_count(day, "1m"):
        return "insufficient_1m_coverage"
    if any(_n_grid_returns(bars, grid) < MIN_INTRADAY_OBS for grid in grids):
        return "below_min_intraday_obs"
    return None


def has_sufficient_coverage(bars: pd.DataFrame, day: str) -> bool:
    return coverage_failure(bars, day) is None


def _n_grid_returns(bars: pd.DataFrame, grid: str) -> int:
    """Broj prinosa koje measure/grids.py::resample_day + log_returns daju
    za ove 1m svećice: resample je closed="right", label="right", pa je
    interval svake svećice njen timestamp zaokružen NAVIŠE na mrežu."""
    minutes = GRID_MINUTES[grid]
    ts = bars["timestamp"]
    n_points = ts.nunique() if minutes == 1 else ts.dt.ceil(f"{minutes}min").nunique()
    return max(n_points - 1, 0)


def clean_day_with_reason(bars: pd.DataFrame, day: str) -> tuple[pd.DataFrame | None, str | None]:
    """Ceo pipeline čišćenja za 1m svećice jednog dana. Vraća (svećice,
    None) za zadržan dan ili (None, razlog) za izbačen -- ista odluka za
    svaku mrežu."""
    if not is_session(day):
        return None, "not_a_session"
    if DROP_HALF_DAYS and is_half_day(day):
        return None, "half_day"
    session_bars = filter_session_bars(bars, day)
    session_bars = drop_thin_bars(session_bars)
    session_bars = session_bars.drop_duplicates(subset="timestamp").sort_values("timestamp")
    reason = coverage_failure(session_bars, day)
    if reason is not None:
        return None, reason
    return session_bars, None


def clean_day(bars: pd.DataFrame, day: str) -> pd.DataFrame | None:
    """clean_day_with_reason bez razloga: očišćene svećice ili None."""
    cleaned, _reason = clean_day_with_reason(bars, day)
    return cleaned
