"""Zajedničko pravilo kliznog prozora (rolling-window) za nedeljne/mesečne
agregate u HAR stilu.

Koristi se identično za RV_w/RV_m, C_w/C_m, i J_w/J_m tako da pravilo
kliznog prozora ne može da se razlikuje između kolona (ranija verzija ovog
plana je čuvala samo dnevne C/J, što bi primoralo sloj za modelovanje da
sâm ponovo implementira ovo pravilo -- čime bi odluka o merenju procurila
izvan sloja za merenje).

Klizi preko DOSTUPNIH TRGOVINSKIH DANA (standardna HAR konvencija), nikad
preko kalendarskih dana, i vrednost ispisuje samo kada prozor ima svoj PUN
komplement opservacija (5 za nedeljni, 22 za mesečni) -- kratak prozor je
NaN, a ne pristrasan prosek preko manje dana nego što je predviđeno.
"""

from __future__ import annotations

import pandas as pd

WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22


def rolling_mean_full_window(series: pd.Series, window: int) -> pd.Series:
    """Klizni (trailing) prosek preko `window` dostupnih (već vremenski
    uređenih) redova, NaN osim ako je prozor potpuno pun."""
    return series.rolling(window=window, min_periods=window).mean()


def add_har_aggregates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Za datu dnevnu kolonu `{column}_d` sortiranu po datumu, dodaj
    `{column}_w` i `{column}_m` koristeći zajedničko pravilo kliznog
    prozora."""
    daily = df[f"{column}_d"]
    df[f"{column}_w"] = rolling_mean_full_window(daily, WEEKLY_WINDOW)
    df[f"{column}_m"] = rolling_mean_full_window(daily, MONTHLY_WINDOW)
    return df
