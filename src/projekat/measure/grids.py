"""Faktor A1: frekvencija uzorkovanja.

Sve tri mreže (1m, 5m, 15m) se grade preuzorkovanjem istih 1-minutnih
izvornih svećica, namerno generičkim putem iako je 15m sam po sebi nativno
dostupan u vault-u (1s,5s,15s,30s,1m,3m,5m,15m,30m,1h,4h,1d,1w,1mo -- za
razliku od 10m, koji nikad nije bio nativan). Identične izvorne svećice za
svaku mrežu znače da A1 menja samo agregaciju i ništa drugo.

Kritičan invarijant, ključan za svakog potrošača nizvodno (A2 ocenjivače,
A3 testove skoka, Lee-Mykland K-prozor): log-prinosi se računaju ISKLJUČIVO
unutar sopstvene preuzorkovane liste cena jednog dana, nikad kao razlika
preko spljoštene višednevne serije cena. Spajanje (concatenating) nizova
prinosa na nivou dana (umesto računanja razlike preko spljoštene serije
cena) garantuje da nijedan prinos nikad ne prelazi noćni jaz -- konkretno
provereno: za dva uzastopna AAPL trgovinska dana stvarni noćni log-prinos je
bio log(15.05/15.12) = -0.0046, i ta vrednost se nigde ne pojavljuje u
ispravno konstruisanoj seriji prinosa. Spljoštavanje sirove serije cena
preko dana je prirodna (i pogrešna) prečica koja bi noćni jaz prokrijumčarila
nazad kao lažan "skok" na početku svakog dana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.config import GRID_MINUTES


def resample_day(bars: pd.DataFrame, grid: str) -> pd.Series:
    """Preuzorkuj očišćene 1-minutne svećice jednog dana na `grid` koristeći
    poslednju cenu u svakom intervalu, vraćajući seriju cena indeksiranu
    krajem intervala."""
    minutes = GRID_MINUTES[grid]
    if minutes == 1:
        prices = bars.set_index("timestamp")["close"]
        return prices.sort_index()
    s = bars.set_index("timestamp")["close"].sort_index()
    resampled = s.resample(f"{minutes}min", label="right", closed="right").last().dropna()
    return resampled


def log_returns(prices: pd.Series) -> np.ndarray:
    """Log-prinosi unutar serije cena jednog dana. Nikad ne pozivati ovo
    preko spajanja serija cena više dana -- vidi docstring modula."""
    values = prices.to_numpy(dtype=float)
    if len(values) < 2:
        return np.array([])
    return np.diff(np.log(values))


def daily_returns_by_grid(bars_by_day: dict[str, pd.DataFrame], grid: str) -> dict[str, np.ndarray]:
    """Izgradi nizove prinosa po danu za jednu mrežu kroz mnogo dana.
    Prinosi se računaju nezavisno po danu (vidi docstring modula), a zatim
    skupljaju u rečnik ključan po danu -- nikad se ne spajaju u jedan
    spljošten niz pre računanja razlike."""
    out: dict[str, np.ndarray] = {}
    for day, bars in bars_by_day.items():
        prices = resample_day(bars, grid)
        r = log_returns(prices)
        if len(r) > 0:
            out[day] = r
    return out
