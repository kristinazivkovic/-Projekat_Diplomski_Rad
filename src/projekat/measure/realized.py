"""Osnovni gradivni elementi realizovanih mera, zajednički svim A2
ocenjivačima: RV, RQ, semivarijanse sa znakom, i dnevni prinos r_d.
"""

from __future__ import annotations

import numpy as np


def realized_variance(r: np.ndarray) -> float:
    """RV = suma kvadrata intradnevnih log-prinosa."""
    return float(np.sum(r**2))


def realized_quarticity(r: np.ndarray) -> float:
    """RQ, standardni ocenjivač četvrtog momenta skaliran sa (n/3). Četvrti
    momenat: nestabilan pri malom n (provereno: RQ za skraćen dan iznosi
    0.05-0.12x medijana punog dana na 5m/15m mrežama -- najizraženije na
    15m, čiji skraćen dan sa n=14 pada ispod MIN_INTRADAY_OBS i potpuno se
    isključuje) -- vidi qa/checks.py za pripadajuću dijagnostiku, i primeti
    da je is_half_day uključen u šemu baš zato da bi ova nestabilnost mogla
    biti obeležena, a ne tiho apsorbovana."""
    n = len(r)
    if n == 0:
        return float("nan")
    return (n / 3.0) * float(np.sum(r**4))


def signed_semivariances(r: np.ndarray) -> tuple[float, float]:
    """(RS_pos, RS_neg): suma kvadrata prinosa razdvojena po znaku. Po
    konstrukciji RS_pos + RS_neg == RV tačno -- ovo je pravi invarijant koji
    se proverava (assert) u qa/checks.py (za razliku od C + J = RV, koje
    važi po konstrukciji same dekompozicije i ništa ne testira)."""
    rs_pos = float(np.sum(r[r > 0] ** 2))
    rs_neg = float(np.sum(r[r < 0] ** 2))
    return rs_pos, rs_neg


def signed_jump_variation(rs_pos: float, rs_neg: float) -> float:
    """SJ = RS_pos - RS_neg (Patton & Sheppard 2015)."""
    return rs_pos - rs_neg


def daily_return(r: np.ndarray) -> float:
    """r_d, ukupan log-prinos dana (suma intradnevnih log-prinosa)."""
    return float(np.sum(r))
