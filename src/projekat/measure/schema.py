"""Zamrznut (frozen) ugovor o izlazu za panel merenja, napisan pre
panel.py tako da sastavljanje panela mora da zadovolji ovaj ugovor, umesto
da se ugovor naknadno (reverse-engineered) izvodi iz onoga što panel.py
uzgred proizvede.

Ovo je granica interfejsa prema (odvojenom, kasnijem) sloju za modelovanje:
imena kolona, tipovi podataka, semantika indeksa i jedinice su ovde
fiksirani i tretiraju se kao API. Izmena ovog fajla predstavlja probojnu
(breaking) promenu te granice.
"""

from __future__ import annotations

import numpy as np

# ── tipovi kolona ────────────────────────────────────────────────────────
# Sve kolone realizovanih mera su u jedinicama varijanse (ne volatilnosti
# niti vol-of-vol jedinicama), osim ako je drugačije naznačeno. r_d i
# vix_lag1 su u svojim prirodnim jedinicama (log-prinos; VIX indeksni poeni).

PANEL_SCHEMA: dict[str, str] = {
    "date": "datetime64[ns]",       # trgovinski dan, bez vremenske zone (UTC datum sesije)
    "symbol": "string",
    "design_id": "string",          # MeasurementDesign.id -- ključ ANOVA faktora

    # HAR komponente, u jedinicama varijanse, klizne preko DOSTUPNIH
    # TRGOVINSKIH DANA (nikad kalendarskih dana), samo puni prozori -- NaN
    # kada prozor nema svoj pun komplement (1 / 5 / 22 trgovinska dana redom).
    "RV_d": "float64", "RV_w": "float64", "RV_m": "float64",
    "C_d": "float64", "C_w": "float64", "C_m": "float64",
    "J_d": "float64", "J_w": "float64", "J_m": "float64",

    "jump_flag": "boolean",         # značajno prema test_skoka dizajna
    "RS_pos": "float64", "RS_neg": "float64",   # RS_pos + RS_neg == RV_d tačno
    "SJ": "float64",                # RS_pos - RS_neg
    "RQ": "float64",                # realizovana kvartičnost, nestabilna na skraćene dane (vidi is_half_day)
    "r_d": "float64",               # dnevni log-prinos (leverage član)
    "vix_lag1": "float64",          # VIX(t-1), FRED VIXCLS, reindeksiran + pomeren

    "n_obs": "int64",               # stvarni broj intradnevnih svećica korišćen za RV_d
    "is_half_day": "boolean",       # sesija sa ranijim zatvaranjem; RQ je ovde nestabilan (0.05-0.12x medijana punog dana)
    "fdr_warmup": "boolean",        # true dok je lm_fdr degenerisao na obični lm (prvih W kliznih dana)
}

# intraday_times (JumpVerdict) namerno NIJE kolona panela: to je samo
# dijagnostika/Nivo 4 (Jaccard poređenja vremenskog usklađivanja, brojevi
# obeležja LM naspram lm_fdr) i nikad ne ulazi u RV/C/J niti u ijednu
# karakteristiku (feature) modela. Držanje van panela strukturno sprovodi
# tu granicu, umesto da se to radi samo konvencijom.

REQUIRED_COLUMNS = tuple(PANEL_SCHEMA.keys())


def empty_panel_frame():
    import pandas as pd

    return pd.DataFrame({col: pd.array([], dtype=dtype) for col, dtype in PANEL_SCHEMA.items()})


def validate_panel(df) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"panelu nedostaju obavezne kolone: {sorted(missing)}")
