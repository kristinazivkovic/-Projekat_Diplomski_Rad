"""VIX podaci sa FRED-a (VIXCLS), pošto sopstvena VIX/USD serija iz LSE
vault-a počinje tek jula 2026 -- previše kratko za uzorak 2010-2025.

FRED za praznike na američkom tržištu ispisuje redove sa `.` kao znakom
nedostajuće vrednosti, umesto da ih izostavi (provereno: 2015. godina ima
260 redova, od kojih je samo 252 brojčano; tih 8 redova sa `.` odgovara
tačno 8 američkih praznika). Ako bi se `.` protumačio kao broj i pomerio za
jedan RED, praznik bi se tiho preslikao na sledeći pravi trgovinski dan --
ispravka je da se `.` tumači kao nedostajuća vrednost, da se serija
REINDEKSIRA na trgovinske datume iz panela, i tek onda pomeri za jedan
TRGOVINSKI dan.

Pomeraj (lag) je obavezan: korišćenje VIX(t) za predviđanje RV(t) predstavlja
unapredno gledanje (lookahead bias), što je eksplicitno navedeno kao zamka u
radu.
"""

from __future__ import annotations

import ssl
import urllib.request

import certifi
import pandas as pd

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd={start}&coed={end}"


def fetch_vix_daily(start: str, end: str) -> pd.Series:
    """Sirova VIXCLS serija, `.` protumačen kao NaN, indeksirana po datumu (bez pomeraja)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    url = _FRED_URL.format(start=start, end=end)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        raw = resp.read().decode()

    lines = [ln for ln in raw.strip().split("\n")[1:] if ln.strip()]
    dates, values = [], []
    for line in lines:
        date_str, value_str = line.split(",")
        value_str = value_str.strip()
        # FRED obeležava nedostajuću opservaciju (praznici na američkom
        # tržištu) ili doslovnim "." ili praznim poljem -- oba slučaja su
        # viđena u praksi, pa se svaki nebrojčani znak tumači kao nedostajuća
        # vrednost umesto da se pretpostavi samo jedna konvencija dobavljača.
        dates.append(date_str)
        values.append(float("nan") if value_str in ("", ".") else float(value_str))

    series = pd.Series(values, index=pd.to_datetime(dates), name="vix")
    return series.dropna()


def vix_lag1_on_calendar(vix_raw: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Reindeksiraj VIX na trgovinske datume iz panela (popunjavajući unapred
    svaki datum gde VIX privremeno nedostaje, npr. rupa kod dobavljača
    podataka koja nije praznik), pa TEK ONDA pomeri za jedan red -- što je
    sada pravi pomeraj od jednog trgovinskog dana jer je indeks trgovinski
    kalendar panela, a ne sirovi redosled redova iz FRED-a."""
    aligned = vix_raw.reindex(trading_dates).ffill()
    lagged = aligned.shift(1)
    return lagged


def assert_lag_not_stale(vix_lag1: pd.Series, trading_dates: pd.DatetimeIndex, max_days_stale: int = 1) -> None:
    """Striktna provera: nijedna vrednost VIX(t-1) ne sme biti starija od
    jednog trgovinskog dana u odnosu na trgovinski datum svog reda."""
    stale = vix_lag1.isna() & (pd.Series(trading_dates).rank() > max_days_stale)
    if stale.any():
        bad_dates = pd.Series(trading_dates)[stale].tolist()
        raise AssertionError(f"vix_lag1 is stale/missing on trading dates: {bad_dates[:5]}...")
