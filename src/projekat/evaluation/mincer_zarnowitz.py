"""Mincer-Zarnowitz regresija: regresija realizovane vrednosti na prognozu;
testira da li je odsečak (intercept)=0 i nagib (slope)=1 -- hvata
sistematsku pristrasnost koju QLIKE ne izdvaja posebno.

Na h=5 i h=22 ciljne vrednosti su preklapajući h-dnevni proseci
(targets.py), pa su reziduali po konstrukciji autokorelisani -- OLS tačkaste
ocene ostaju konzistentne, ali obične standardne greške nisu. Ovde se
koriste Newey-West HAC standardne greške sa pomerajem h-1, u skladu sa
sopstvenim HAC tretmanom istog preklapanja u DM testu."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


@dataclass(frozen=True)
class MZResult:
    intercept: float
    slope: float
    intercept_pvalue: float
    slope_pvalue: float  # p-vrednost za H0: nagib == 1


def mincer_zarnowitz(y_true: np.ndarray, y_hat: np.ndarray, *, horizon: int = 1) -> MZResult:
    X = sm.add_constant(y_hat)
    nw_lags = max(horizon - 1, 0)
    if nw_lags > 0:
        model = sm.OLS(y_true, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    else:
        model = sm.OLS(y_true, X).fit()
    intercept, slope = model.params
    intercept_p = model.pvalues[0]
    slope_p = float(model.t_test("x1 = 1").pvalue)
    return MZResult(
        intercept=float(intercept),
        slope=float(slope),
        intercept_pvalue=float(intercept_p),
        slope_pvalue=slope_p,
    )
