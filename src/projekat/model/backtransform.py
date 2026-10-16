"""Zajednička korekcija povratne transformacije i Mincer-Zarnowitz
rekalibracija -- JEDINO mesto u kodu gde se one računaju.

KOREKCIJA IZ SOPSTVENIH VAN-UZORAČKIH GREŠAKA, STROGO PRE t
===========================================================
Nema posebnog validacionog preklopa. Za prognozu napravljenu na dan t,
korekcija se ocenjuje iz van-uzoračkih grešaka ISTOG modela (iste serije
model x dizajn x horizont x seme, objedinjeno preko simbola) čiji je cilj
REALIZOVAN u poslednjih W = config.CORRECTION_WINDOW_SESSIONS dana
trgovanja strogo pre t:

    greška prognoze sa porekla s ulazi u korekciju za dan t
    akko   t - W <= r(s) <= t - 1,

gde je r(s) indeks dana trgovanja na koji je cilj y_{s,h} realizovan
(targets.py, target_end). Za h = 22 to znači da greška prognoze sa dana s
postaje dostupna tek 22 dana posle s -- nikad ranije. Pre prve realizovane
greške (početak 2014, godine zagrevanja) korekcija je neutralna (a = 0,
m* = 1); ti dani se ne ocenjuju.

Tri vrste korekcije (Model.correction):

  "log"            y_hat = exp(p + a),  a = ln(mean(exp(e))),  e = ln y - p.
                   NEPARAMETARSKA (Duan 1983), ne Gaussova sigma^2/2: RV
                   reziduali nisu normalni, posebno kod stabala.
  "ratio_of_sums"  y_hat = m* * y_raw,  m* = sum(y) / sum(y_raw).
                   Za KOMBINOVANE prognoze (ttm_c_plus_j: C + J u nivoima;
                   log_har_ttm: kombinacija u log prostoru): JEDNA korekcija
                   na već kombinovanoj prognozi, nikad po komponenti --
                   Jensenova nejednakost nije aditivna preko komponenti.
                   Odnos zbirova, a ne prosek odnosa: prosek odnosa y/y_raw
                   eksplodira kad je y_raw blizu nule (ttm_c_plus_j seče J na
                   nuli), pa jedan dan preuzme ceo faktor.
  "none"           y_hat = p (nivoi; NNLS HAR i MEM -- pozitivnost je
                   strukturna).

MINCER-ZARNOWITZ po istom pravilu: y = alpha + beta * y_hat, OLS nad
parovima (y, y_hat) čiji je cilj realizovan u istom kliznom prozoru;
y_hat_recalibrated = alpha + beta * y_hat. Sekundarna figura, uvek pored
sirove, nikad umesto nje.
"""

from __future__ import annotations

import numpy as np

from projekat.config import CORRECTION_WINDOW_SESSIONS, MZ_MIN_PAIRS


def backtransform(log_y_hat: np.ndarray, correction) -> np.ndarray:
    """y_hat = exp(log_y_hat + a), gde je `a` neparametarska korekcija."""
    return np.exp(np.asarray(log_y_hat) + correction)


def residual_correction(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """a = ln(mean(exp(rezidual))) nad jednim skupom reziduala."""
    resid = np.asarray(y_true_log) - np.asarray(y_pred_log)
    return float(np.log(np.mean(np.exp(resid))))


def combined_forecast_correction(y_true_level: np.ndarray, combined_y_hat_level: np.ndarray) -> float:
    """m* = sum(y) / sum(y_raw) za već kombinovanu prognozu u nivoima --
    ODNOS ZBIROVA, ne prosek odnosa (vidi docstring modula)."""
    y = np.asarray(y_true_level, dtype=float)
    f = np.asarray(combined_y_hat_level, dtype=float)
    ok = np.isfinite(y) & np.isfinite(f)
    total = f[ok].sum()
    if not ok.any() or total <= 0:
        return 1.0
    return float(y[ok].sum() / total)


def apply_combined_correction(combined_y_hat_level: np.ndarray, correction) -> np.ndarray:
    return correction * np.asarray(combined_y_hat_level)


# ── klizne sume po danu realizacije ──────────────────────────────────────


def _trailing_sums(
    origin_idx: np.ndarray,
    realized_idx: np.ndarray,
    weights: list[np.ndarray],
    *,
    window: int,
) -> list[np.ndarray]:
    """Za svaki red (poreklo o): suma težina preko redova čiji je indeks
    realizacije r u [o - window, o - 1]. Redovi bez realizacije
    (realized_idx < 0) ili sa nekonačnom težinom ne ulaze."""
    origin_idx = np.asarray(origin_idx, dtype=np.int64)
    realized_idx = np.asarray(realized_idx, dtype=np.int64)
    ok = realized_idx >= 0
    for w in weights:
        ok &= np.isfinite(w)
    size = int(max(origin_idx.max(initial=0), realized_idx.max(initial=0))) + 2
    hi = origin_idx                                   # c[o] = suma za r <= o - 1
    lo = np.clip(origin_idx - window, 0, None)        # c[o - W] = suma za r <= o - W - 1
    out = []
    for w in weights:
        agg = np.bincount(realized_idx[ok], weights=w[ok], minlength=size)
        c = np.concatenate([[0.0], np.cumsum(agg)])
        out.append(c[hi] - c[lo])
    return out


def trailing_corrections(
    origin_idx: np.ndarray,
    realized_idx: np.ndarray,
    raw_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    kind: str,
    predicts_levels: bool,
    window: int = CORRECTION_WINDOW_SESSIONS,
) -> dict[str, np.ndarray]:
    """Korekcija za svaki red iz van-uzoračkih grešaka realizovanih u
    [o - window, o - 1] (vidi docstring modula).

    Vraća: log_y_hat (sirova log prognoza ili NaN), raw_y_hat (nivoi pre
    korekcije), backtransform_correction (a, samo za "log"),
    combined_correction (m*, samo za "ratio_of_sums"), y_hat."""
    raw_pred = np.asarray(raw_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    n = len(raw_pred)
    nan = np.full(n, np.nan)
    log_y_hat = nan.copy() if predicts_levels else raw_pred
    raw_level = raw_pred if predicts_levels else np.exp(raw_pred)

    if kind == "none":
        if not predicts_levels:
            raise ValueError("correction='none' važi samo za modele koji predviđaju nivoe")
        return {"log_y_hat": log_y_hat, "raw_y_hat": raw_level, "backtransform_correction": nan,
                "combined_correction": nan.copy(), "y_hat": raw_level}

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "log":
            if predicts_levels:
                raise ValueError("correction='log' zahteva log prognozu (predicts_levels=False)")
            w = np.exp(np.log(y_true) - raw_pred)
            s, cnt = _trailing_sums(origin_idx, realized_idx, [w, np.ones(n)], window=window)
            a = np.where(cnt > 0, np.log(s / np.where(cnt > 0, cnt, 1.0)), 0.0)
            return {"log_y_hat": log_y_hat, "raw_y_hat": raw_level, "backtransform_correction": a,
                    "combined_correction": nan, "y_hat": np.exp(raw_pred + a)}

        if kind == "ratio_of_sums":
            s_y, s_f = _trailing_sums(origin_idx, realized_idx, [y_true, raw_level], window=window)
            m = np.where(s_f > 0, s_y / np.where(s_f > 0, s_f, 1.0), 1.0)
            return {"log_y_hat": log_y_hat, "raw_y_hat": raw_level, "backtransform_correction": nan,
                    "combined_correction": m, "y_hat": m * raw_level}

    raise ValueError(f"nepoznata vrsta korekcije {kind!r}")


def trailing_mincer_zarnowitz(
    origin_idx: np.ndarray,
    realized_idx: np.ndarray,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    window: int = CORRECTION_WINDOW_SESSIONS,
    min_pairs: int = MZ_MIN_PAIRS,
) -> tuple[np.ndarray, np.ndarray]:
    """(odsečak, nagib) MZ regresije y = alpha + beta * y_hat za svaki red,
    OLS nad parovima realizovanim u [o - window, o - 1]; NaN gde je parova
    manje od `min_pairs` ili je y_hat u prozoru konstantan."""
    x = np.asarray(y_hat, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n_, sx, sy, sxx, sxy = _trailing_sums(
        origin_idx, realized_idx, [np.ones(len(x)), x, y, x * x, x * y], window=window
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = n_ * sxx - sx * sx
        ok = (n_ >= min_pairs) & (denom > 0)
        slope = np.where(ok, (n_ * sxy - sx * sy) / np.where(ok, denom, 1.0), np.nan)
        intercept = np.where(ok, (sy - slope * sx) / np.where(ok, n_, 1.0), np.nan)
    return intercept, slope
