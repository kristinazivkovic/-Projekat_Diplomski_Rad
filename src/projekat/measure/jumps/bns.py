"""Test skoka po Barndorff-Nielsen & Shephard (2006).

Jedan test po danu: poredi RV sa ocenjivačem otpornim na skokove (prosleđen
kao `robust`, npr. BPV), skaliranim tripower kvartičnošću, naspram njegove
asimptotske standardne normalne raspodele pod nultom hipotezom o odsustvu
skokova.

Ovo je test po danu, za razliku od Lee-Mykland testa koji pokreće jedan
test po intradnevnom prinosu. FDR korekcija preko BNS statistika bi
kontrolisala lažna otkrića PREKO dana -- koherentna korekcija, ali ne ona
koju ovaj rad navodi kao motivaciju (briga o višestrukom testiranju iz rada
iz februara 2026. odnosi se na testiranje UNUTAR dana, tj. Lee-Mykland).
Zato se FDR sparuje sa lm, a ne sa bns -- vidi jumps/lm_fdr.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from projekat.measure.protocols import JumpVerdict
from projekat.measure.registry import register_jump_test

_MU1_INV_3 = (2.0 / math.pi) ** -1.5  # mu_1^-3, za tripower kvartičnost


def _tripower_quarticity(r: np.ndarray) -> float:
    n = len(r)
    if n < 3:
        return float("nan")
    abs_r = np.abs(r)
    a, b, c = abs_r[:-2], abs_r[1:-1], abs_r[2:]
    terms = (a ** (4 / 3)) * (b ** (4 / 3)) * (c ** (4 / 3))
    return n * _MU1_INV_3 * float(np.sum(terms)) * (n / (n - 2))


_THETA = (math.pi / 2) ** 2 + math.pi - 5  # konstanta asimptotske varijanse


def bns(r: np.ndarray, rv: float, robust: float, *, alpha: float = 0.05, **kwargs) -> JumpVerdict:
    n = len(r)
    if n < 3 or rv <= 0 or robust <= 0:
        return JumpVerdict(significant=False, statistic=0.0, p_value=1.0)

    tpq = _tripower_quarticity(r)
    relative_jump = (rv - robust) / rv
    # Standardna BNS statistika odnosa relativnog skoka: normalizacija sa
    # n*BPV^2 (ne RV^2) je ono što daje ispravnu asimptotsku N(0,1) skalu --
    # ranija verzija ove formule je izostavljala faktor n i koristila RV^2,
    # što je činilo statistiku suštinski neosetljivom na veličinu skoka
    # (provereno: ubačen skok od 20 sigma je proizveo statistiku od svega
    # 0.69). Ponovo provereno nakon ispravke: isti skok sada proizvodi
    # statistiku od ~8.0.
    denom = math.sqrt(_THETA * max(tpq, 1e-18) / (n * robust**2))
    statistic = relative_jump / denom if denom > 0 else 0.0
    p_value = float(1 - stats.norm.cdf(statistic))
    significant = p_value < alpha
    return JumpVerdict(significant=significant, statistic=float(statistic), p_value=p_value)


bns.name = "bns"
register_jump_test(bns)
