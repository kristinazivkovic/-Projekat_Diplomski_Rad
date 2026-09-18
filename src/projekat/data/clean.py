"""Filtriranje sesija i čišćenje kvaliteta svećica.

Dve stvari koje fiksni vremenski prozor pogrešno radi, obe ovde ispravljene
korišćenjem pravog berzanskog kalendara (projekat.data.calendar):

1. Curenje posle zatvaranja (post-close leakage): fiksni prozor 13:30-20:00 UTC
   zadržava ~70-81 tankih svećica posle zatvaranja tržišta na dane sa ranijim
   zatvaranjem (proveren medijan obima 460 naspram 18.603 u pravoj sesiji --
   40 puta veća razlika). Filtriranje prema kalendarskom prozoru same sesije
   ih uklanja.
2. Grub filter "<80% od 390" ne bi pouzdano razlikovao pravo rano zatvaranje
   od rupe u podacima, jer procurele svećice posle zatvaranja mogu podići
   sirovi broj svećica i do ~470. Poređenje sa očekivanim brojem za svaki
   konkretan dan (390 puni dan / 210 skraćeni dan) čini da filter proverava
   pravu stvar, a dani sa ranijim zatvaranjem se ispravno *zadržavaju* sa
   svojih pravih 210 svećica umesto da budu odbačeni.

Dan se odbacuje za CELU mrežu (grid), a ne po pojedinačnom ocenjivaču, kada
padne ispod MIN_INTRADAY_OBS (npr. 15-minutni skraćeni dan sa n=14) --
primena ovog praga samo unutar pojedinačnih A2 ocenjivača (RQ/MedRV/TrBPV)
ostavila bi BPV dizajne sa više upotrebljivih dana nego MedRV/TrBPV dizajne
na istoj mreži, što bi napravilo neuravnotežen faktorijalni dizajn i učinilo
udele varijanse po faktoru u ANOVA analizi neinterpretabilnim -- isti problem
zbog kog je TrBPV prag ostao fiksiran umesto da postane 4. dimenzija.
"""

from __future__ import annotations

import pandas as pd

from projekat.config import MIN_BAR_COVERAGE, MIN_INTRADAY_OBS
from projekat.data.calendar import expected_bar_count, is_session, session_window


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


def has_sufficient_coverage(bars: pd.DataFrame, day: str, grid: str) -> bool:
    """Relativni filter (>= 80% od očekivanog broja za taj konkretan dan) I
    apsolutni prag (>= MIN_INTRADAY_OBS), primenjeni zajedno tako da prag
    hvata slučajeve koje relativni filter ne može -- npr. 80% od punog
    dnevnog broja za 15m (26) je samo 20.8, pa bi 15m dan inače mogao proći
    sa svega 21 svećicom, dok ispravno zadržan 15m skraćeni dan (n=14)
    uopšte treba apsolutni prag da bi bio isključen."""
    expected = expected_bar_count(day, grid)
    return len(bars) >= MIN_BAR_COVERAGE * expected and len(bars) >= MIN_INTRADAY_OBS


def clean_day(bars: pd.DataFrame, day: str, grid: str) -> pd.DataFrame | None:
    """Ceo pipeline čišćenja za svećice jednog dana na jednoj mreži. Vraća
    None ako dan nije prava berzanska sesija (svećica od dobavljača može da
    upadne na vikend/praznik -- provereno da se to dešava, npr. na Novu
    godinu) ili ako treba biti odbačen zbog nedovoljne pokrivenosti."""
    if not is_session(day):
        return None
    session_bars = filter_session_bars(bars, day)
    session_bars = drop_thin_bars(session_bars)
    session_bars = session_bars.drop_duplicates(subset="timestamp").sort_values("timestamp")
    if not session_bars["timestamp"].is_monotonic_increasing:
        session_bars = session_bars.sort_values("timestamp")
    if not has_sufficient_coverage(session_bars, day, grid):
        return None
    return session_bars
