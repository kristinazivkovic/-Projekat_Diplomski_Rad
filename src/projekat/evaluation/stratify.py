"""Naknadna (post-hoc) stratifikacija: modeli se treniraju jednom po
(design_id, prozor); rezultati se obeležavaju režimom i jump_day oznakom,
pa se tek onda filtriraju -- nikad se ne treniraju ponovo po stratumu.

Odluka C6: granice režima se računaju SAMO JEDNOM iz skupa za treniranje
prozora 1 (2010-2014) i drže se FIKSNE tokom cele studije -- u suprotnom bi
oznaka režima za dati dan mogla da se menja između "mirno" i "turbulentno"
kako se usidreni (anchored) prozor za treniranje širi, čineći stratifikovanu
tabelu nedefinisanom (isti kalendarski dan bi pripadao različitim
stratumima u zavisnosti od toga koji prozor je proizveo tu prognozu).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from projekat.config import PROCESSED_DIR

REGIME_BOUNDARY_PATH = PROCESSED_DIR / "regime_boundary.json"


def compute_and_persist_regime_boundary(window1_train_df: pd.DataFrame, *, quantile: float = 0.5) -> float:
    """Izračunato samo jednom, isključivo iz podataka za treniranje prozora
    1, i sačuvano na disk kako bi bilo proverljivo i da se ne bi tiho
    menjalo između pokretanja.

    Uz prag se zapisuje i PROVENIJENCIJA (broj redova, simboli, raspon
    datuma). Bez toga je granica izračunata na sitnom probnom panelu
    bajt-identična onoj sa punog panela od 30 simbola, pa bi se tiho
    koristila za obeležavanje cele studije -- greška koja ne podiže
    nijedan izuzetak i vidi se tek u stratifikovanoj tabeli."""
    threshold = float(window1_train_df["RV_m"].quantile(quantile))
    payload = {
        "quantile": quantile,
        "threshold": threshold,
        "n_rows": int(len(window1_train_df)),
        "n_symbols": int(window1_train_df["symbol"].nunique()) if "symbol" in window1_train_df else None,
        "symbols": sorted(window1_train_df["symbol"].unique().tolist()) if "symbol" in window1_train_df else None,
        "date_min": str(window1_train_df["date"].min()) if "date" in window1_train_df else None,
        "date_max": str(window1_train_df["date"].max()) if "date" in window1_train_df else None,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    REGIME_BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGIME_BOUNDARY_PATH.write_text(json.dumps(payload, indent=2))
    return threshold


def regime_boundary_provenance() -> dict | None:
    """Šta je granica zapravo videla kad je računata. Vrati None ako fajl
    ne postoji ili je star (bez polja provenijencije)."""
    if not REGIME_BOUNDARY_PATH.exists():
        return None
    payload = json.loads(REGIME_BOUNDARY_PATH.read_text())
    return payload if "n_rows" in payload else None


def load_regime_boundary() -> float:
    if not REGIME_BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"{REGIME_BOUNDARY_PATH} nije pronađen -- pozovi compute_and_persist_regime_boundary "
            "nad podacima za treniranje prozora 1 pre obeležavanja bilo kog režima."
        )
    return json.loads(REGIME_BOUNDARY_PATH.read_text())["threshold"]


def tag_regime(df: pd.DataFrame, *, threshold: float | None = None) -> pd.Series:
    """Mirno naspram turbulentno, u odnosu na FIKSNU granicu iz prozora 1 --
    nikad u odnosu na sopstvenu raspodelu prozora koji poziva funkciju."""
    t = threshold if threshold is not None else load_regime_boundary()
    return (df["RV_m"] > t).map({True: "turbulent", False: "calm"})


def tag_jump_day(df: pd.DataFrame) -> pd.Series:
    return df["jump_flag"].map({True: "jump", False: "no_jump"})


def stratified_table(results_df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """results_df: jedan red po prognozi, sa kolonom gubitka i već
    prikačenim oznakama regime/jump_day/horizon. Vraća prosečan gubitak
    grupisan po traženim stratumima (podrazumevano: horizont x režim x
    jump_day)."""
    cols = group_cols or ["horizon", "regime", "jump_day"]
    return results_df.groupby(cols, observed=True).mean(numeric_only=True).reset_index()
