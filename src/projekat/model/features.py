"""Matrice karakteristika (design matrices) po modelu, izgrađene iz šeme
panela (measure/schema.py) isključivo izborom kolona i transformacijom --
nikad ponovnim računanjem bilo čega što je sloj za merenje već proizveo.

Svaka karakteristika koristi informaciju dostupnu na dan prognoze: RV_d/w/m,
C_d/w/m, J_d su realizovane vrednosti dana t (dana porekla prognoze), a cilj
(targets.py) je prosek RV_{t+1}..RV_{t+h} -- pa su karakteristike na redu t
već "pomeraj 1" u odnosu na cilj, po samoj konstrukciji cilja.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _log(col: pd.Series) -> pd.Series:
    return np.log(col.clip(lower=1e-12))


def log1p_jump(col: pd.Series) -> pd.Series:
    """ln(1 + J): skokovita komponenta u log prostoru.

    J = 0 na većini dana (uslovna dekompozicija), pa ln(J) nije definisan,
    a ln(max(J, eps)) bi na svakom danu bez skoka dao proizvoljnu konstantu
    ln(eps) (-27.6 za eps = 1e-12) -- koeficijent bi tada mešao "da li je
    bilo skoka" sa "koliki je bio" i zavisio od izbora eps. ln(1 + J) je 0
    tačno kad je J = 0 (Andersen, Bollerslev i Diebold 2007, log
    specifikacija HAR-RV-CJ). Jedini oblik J u log prostoru u kodu."""
    return np.log1p(col.clip(lower=0.0))


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
    out["J_d_lag1"] = log1p_jump(df["J_d"])
    return out


def shar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["RS_pos_lag1"] = _log(df["RS_pos"])
    out["RS_neg_lag1"] = _log(df["RS_neg"])
    out["RV_w_lag1"] = _log(df["RV_w"])
    out["RV_m_lag1"] = _log(df["RV_m"])
    return out


# ── HARQ ─────────────────────────────────────────────────────────────────


def harq_standardization(train: pd.DataFrame) -> tuple[float, float]:
    """(mean, sd) of sqrt(RQ) on the TRAINING rows of one refit -- frozen and
    reused for every forecast until the next refit (never re-estimated on
    the rows being forecast)."""
    s = np.sqrt(train["RQ"].clip(lower=0)).to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2 or s.std() == 0:
        return 0.0, 1.0
    return float(s.mean()), float(s.std())


def harq_features(df: pd.DataFrame, params: tuple[float, float]) -> pd.DataFrame:
    """log-HAR regressors plus the interaction z * ln(RV_d), where
    z = (sqrt(RQ) - mean) / sd with (mean, sd) from the training data.

    In log space sqrt(RQ) is ~1e-4 while ln(RV_d) is ~-9, so the raw product
    would sit orders of magnitude away from the base regressor; after
    standardization the coefficient on ln(RV_d) is (beta_d + beta_Q * z),
    i.e. it varies around beta_d with the day's measurement precision
    (Bollerslev, Patton & Quaedvlieg 2016 use the levels form; this is the
    log-space analogue)."""
    mean, sd = params
    out = har_features(df)
    z = (np.sqrt(df["RQ"].clip(lower=0)) - mean) / sd
    out["sqrtRQ_z_x_lnRV_d"] = z * out["RV_d_lag1"]
    return out


def har_iv_features(df: pd.DataFrame) -> pd.DataFrame:
    out = har_features(df)
    out["vix_lag1"] = df["vix_lag1"]
    return out


def union_features(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM/XGBoost: union of the HAR-family regressors plus r_d
    (leverage). The HARQ interaction enters unstandardized as
    sqrt(RQ) * ln(RV_d): standardization is a linear-model concern, and the
    trees also see ln(RV_d) itself."""
    parts = [
        har_features(df),
        char_features(df),
        har_j_features(df)[["J_d_lag1"]],
        shar_features(df)[["RS_pos_lag1", "RS_neg_lag1"]],
    ]
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out["sqrtRQ_x_lnRV_d"] = np.sqrt(df["RQ"].clip(lower=0)) * out["RV_d_lag1"]
    out["vix_lag1"] = df["vix_lag1"]
    out["r_d"] = df["r_d"]
    return out


# ── sequences (PatchTST) ─────────────────────────────────────────────────

SEQUENCE_COLUMNS = ["RV_d", "C_d", "J_d", "RS_pos", "RS_neg", "r_d"]
SEQUENCE_LENGTH = 60
JUMP_CHANNEL = SEQUENCE_COLUMNS.index("J_d")


def sequence_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_channels) channel values per row: logs where always
    positive, ln(1+J) for the jump channel, r_d signed in natural units."""
    return np.column_stack(
        [
            _log(df["RV_d"]).to_numpy(dtype=float),
            _log(df["C_d"]).to_numpy(dtype=float),
            log1p_jump(df["J_d"]).to_numpy(dtype=float),
            _log(df["RS_pos"]).to_numpy(dtype=float),
            _log(df["RS_neg"]).to_numpy(dtype=float),
            df["r_d"].to_numpy(dtype=float),
        ]
    )


def sequences_at(df: pd.DataFrame, positions: np.ndarray, *, seq_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """The seq_len-day window ENDING at each requested row of `df` (sorted
    by (symbol, date)), never crossing into another symbol. Returns
    (sequences[k, seq_len, channels], ok[k]); a window is not ok when the
    symbol has fewer than seq_len rows up to that row or the window holds a
    NaN (its sequence is then zeros and must not be used)."""
    positions = np.asarray(positions, dtype=np.int64)
    mat = sequence_matrix(df)
    n_ch = mat.shape[1]
    out = np.zeros((len(positions), seq_len, n_ch), dtype=np.float32)
    ok = np.zeros(len(positions), dtype=bool)
    if len(positions) == 0:
        return out, ok
    symbols = df["symbol"].to_numpy()
    groups = pd.Series(np.arange(len(df))).groupby(symbols, sort=False).agg(["min", "max"])
    first_row = groups["min"].reindex(symbols[positions]).to_numpy()
    enough = positions - seq_len + 1 >= first_row
    if len(df) >= seq_len:
        windows = sliding_window_view(mat, seq_len, axis=0)          # (n - L + 1, channels, L)
        idx = positions[enough] - seq_len + 1
        seqs = np.transpose(windows[idx], (0, 2, 1))                 # (k, L, channels)
        finite = np.isfinite(seqs).all(axis=(1, 2))
        take = np.flatnonzero(enough)
        out[take[finite]] = seqs[finite]
        ok[take[finite]] = True
    return out, ok


def build_sequences(df: pd.DataFrame, *, seq_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Every complete window of `df` (after sorting by symbol, date): the
    sequences and the positions (in the sorted frame) of their last row."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    seqs, ok = sequences_at(df, np.arange(len(df)), seq_len=seq_len)
    positions = np.flatnonzero(ok)
    return seqs[ok], positions
