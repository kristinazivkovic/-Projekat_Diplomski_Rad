"""Test skoka po Barndorff-Nielsen & Shephard (2006).

Jedan test po danu: poredi RV sa ocenjivačem otpornim na skokove (prosleđen
kao `robust`, ocenjivač IZ DIZAJNA -- BPV, MedRV ili TrBPV), skaliranim
tripower kvartičnošću, naspram njegove asimptotske standardne normalne
raspodele pod nultom hipotezom o odsustvu skokova.

KONSTANTA theta ZAVISI OD OCENJIVAČA. Imenilac statistike je asimptotska
varijansa razlike (ocena - RV): (pi/2)^2 + pi - 5 = 0.609 za BPV i TrBPV,
0.96 za MedRV. Konstanta se čita iz registra (`ESTIMATORS[ime].bns_theta`)
po imenu ocenjivača koji panel.py prosleđuje kao `estimator=` -- ovaj modul
ne imenuje nijedan konkretan ocenjivač. Nepoznat ocenjivač bez bns_theta je
greška, nikad tiha zamena BPV-ovom konstantom.

Test je jednostran (p = 1 - Phi(Z)) na nivou config.BNS_ALPHA = 0.001.

Ovo je test po danu, za razliku od Lee-Mykland testa koji pokreće jedan
test po intradnevnom prinosu. FDR korekcija se sparuje sa lm, a ne sa bns
-- vidi jumps/lm_fdr.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from projekat.config import BNS_ALPHA
from projekat.measure.protocols import JumpVerdict
from projekat.measure.registry import ESTIMATORS, register_jump_test

_MU1_INV_3 = (2.0 / math.pi) ** -1.5  # mu_1^-3, za tripower kvartičnost

# Konstanta za BPV; koristi se SAMO kad pozivalac ne navede ocenjivač
# (izolovani testovi nad BPV-om). Pipeline (panel.py) uvek navodi ocenjivač.
_BPV_THETA = (math.pi / 2) ** 2 + math.pi - 5


def _tripower_quarticity(r: np.ndarray) -> float:
    n = len(r)
    if n < 3:
        return float("nan")
    abs_r = np.abs(r)
    a, b, c = abs_r[:-2], abs_r[1:-1], abs_r[2:]
    terms = (a ** (4 / 3)) * (b ** (4 / 3)) * (c ** (4 / 3))
    return n * _MU1_INV_3 * float(np.sum(terms)) * (n / (n - 2))


def theta_for(estimator: str | None) -> float:
    """BNS konstanta asimptotske varijanse za dati ocenjivač (iz registra)."""
    if estimator is None:
        return _BPV_THETA
    from projekat.measure import estimators  # noqa: F401  (registracija, ne konkretan modul)

    impl = ESTIMATORS.get(estimator)
    theta = getattr(impl, "bns_theta", None) if impl is not None else None
    if theta is None:
        raise ValueError(f"ocenjivač {estimator!r} ne deklariše bns_theta -- BNS test nije definisan za njega")
    return float(theta)


def bns(
    r: np.ndarray,
    rv: float,
    robust: float,
    *,
    estimator: str | None = None,
    alpha: float = BNS_ALPHA,
    **kwargs,
) -> JumpVerdict:
    n = len(r)
    if n < 3 or rv <= 0 or robust <= 0:
        return JumpVerdict(significant=False, statistic=0.0, p_value=1.0)

    theta = theta_for(estimator)
    tpq = _tripower_quarticity(r)
    relative_jump = (rv - robust) / rv
    # Normalizacija sa n*ocena^2 (ne RV^2) daje ispravnu asimptotsku N(0,1)
    # skalu -- ranija verzija je izostavljala faktor n i koristila RV^2, pa
    # je ubačen skok od 20 sigma davao statistiku od svega 0.69 (posle
    # ispravke ~8.0).
    denom = math.sqrt(theta * max(tpq, 1e-18) / (n * robust**2))
    statistic = relative_jump / denom if denom > 0 else 0.0
    p_value = float(1 - stats.norm.cdf(statistic))
    significant = p_value < alpha
    return JumpVerdict(significant=significant, statistic=float(statistic), p_value=p_value)


bns.name = "bns"
register_jump_test(bns)
