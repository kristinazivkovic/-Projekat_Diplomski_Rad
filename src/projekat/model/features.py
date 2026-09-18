"""Matrice karakteristika (design matrices) po modelu, izgrađene iz šeme
panela (measure/schema.py) isključivo izborom kolona i pomeranjem (lag) --
nikad ponovnim računanjem bilo čega što je sloj za merenje već proizveo.

Svaka karakteristika (feature) koristi informaciju dostupnu na dan
prognoze ili pre njega: RV_d/w/m, C_d/w/m, J_d/w/m su realizovane vrednosti
istog dana t, dana porekla prognoze, a ciljna vrednost (targets.py) je
h-dnevni prosek RV_{t+1}..RV_{t+h} -- pa su karakteristike na redu t već
"pomeraj 1" (lag 1) u odnosu na cilj, po samoj konstrukciji cilja.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.targets import TARGET_COL  # noqa: F401  (ponovo izvezeno radi praktičnosti)


def _log(col: pd.Series) -> pd.Series:
    return np.log(col.clip(lower=1e-12))


def naive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["RV_d_lag1"] = _log(df["RV_d"])
    return out


def har_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["RV_d_lag1"] = _log(df["RV_d"])
    out["RV_w_lag1"] = _log(df["RV_w"])
    out["RV_m_lag1"] = _log(df["RV_m"])
    return out


def char_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["C_d_lag1"] = _log(df["C_d"])
    out["C_w_lag1"] = _log(df["C_w"])
    out["C_m_lag1"] = _log(df["C_m"])
    return out


def har_j_features(df: pd.DataFrame) -> pd.DataFrame:
    out = char_features(df)
    # J može biti tačno 0 (dan bez značajnog testa) -- log(0+eps) umesto log(0)
    out["J_d_lag1"] = np.log(df["J_d"].clip(lower=1e-12))
    return out


def shar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["RS_pos_lag1"] = _log(df["RS_pos"])
    out["RS_neg_lag1"] = _log(df["RS_neg"])
    out["RV_w_lag1"] = _log(df["RV_w"])
    out["RV_m_lag1"] = _log(df["RV_m"])
    return out


def harq_features(df: pd.DataFrame) -> pd.DataFrame:
    out = har_features(df)
    # Karakteristika koja definiše HARQ: RV_d u interakciji sa sqrt(RQ)
    # (Bollerslev, Patton & Quaedvlieg 2016 koriste kvadratni koren RQ, a ne
    # sam RQ, tako da je interakcioni član istog reda veličine kao RV_d).
    out["RV_d_x_sqrtRQ_lag1"] = df["RV_d"] * np.sqrt(df["RQ"].clip(lower=0))
    return out


def har_iv_features(df: pd.DataFrame) -> pd.DataFrame:
    out = har_features(df)
    out["vix_lag1"] = df["vix_lag1"]
    return out


def union_features(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM/XGBoost: unija svih karakteristika iz HAR porodice iznad,
    plus r_d (leverage član)."""
    parts = [
        har_features(df),
        char_features(df),
        har_j_features(df)[["J_d_lag1"]],
        shar_features(df)[["RS_pos_lag1", "RS_neg_lag1"]],
        harq_features(df)[["RV_d_x_sqrtRQ_lag1"]],
        df[["vix_lag1", "r_d"]],
    ]
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


SEQUENCE_COLUMNS = ["RV_d", "C_d", "J_d", "RS_pos", "RS_neg", "r_d"]
SEQUENCE_LENGTH = 60


def build_sequences(df: pd.DataFrame, *, seq_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Niz oblika (n_samples, seq_len, n_features) od SEQUENCE_COLUMNS za
    prethodnih seq_len dana (log-transformisano tamo gde je uvek pozitivno;
    J_d preko clip-a jer može biti 0; r_d ostavljen u svojim prirodnim
    jedinicama sa znakom, bez logaritmovanja), i niz indeksa redova koji
    označava kom originalnom redu df-a odgovara poreklo svakog uzorka
    (poslednji dan prozora) -- tako da se predikcije mogu naknadno spojiti
    sa datumima/ciljnim vrednostima."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    mat = np.column_stack(
        [
            _log(df["RV_d"]).to_numpy(),
            _log(df["C_d"]).to_numpy(),
            np.log(df["J_d"].clip(lower=1e-12)).to_numpy(),
            _log(df["RS_pos"]).to_numpy(),
            _log(df["RS_neg"]).to_numpy(),
            df["r_d"].to_numpy(),
        ]
    )
    symbols = df["symbol"].to_numpy()
    n = len(df)
    sequences = []
    origin_idx = []
    for i in range(seq_len - 1, n):
        # a window must never span two symbols -- since rows are sorted by
        # (symbol, date), that's true iff the first and last row of the
        # window share the same symbol.
        if symbols[i - seq_len + 1] != symbols[i]:
            continue
        window = mat[i - seq_len + 1 : i + 1]
        if np.isnan(window).any():
            continue
        sequences.append(window)
        origin_idx.append(i)
    if not sequences:
        return np.empty((0, seq_len, mat.shape[1])), np.array([], dtype=int)
    return np.stack(sequences), np.array(origin_idx)
