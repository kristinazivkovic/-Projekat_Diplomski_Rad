"""Naivan test praga: dan je dan sa skokom ako njegov ukupan dnevni prinos
prelazi fiksiran broj standardnih devijacija. Nije statistički test --
koristi se kao donji prag kvaliteta, ne kao ozbiljna metoda (prema radu).

Sigma se ocenjuje iz RV/robust (BPV) ocenjivača prosleđenih pozivaocem, a
NE iz np.std(r) po celom danu -- ta ocena bi uključivala sam skok koji se
testira u svom sopstvenom imeniocu, tako da dovoljno velik skok naduva
sopstveni prag i izbegne obeležavanje (samoreferentna kontaminacija)."""

from __future__ import annotations

import numpy as np

from projekat.measure.protocols import JumpVerdict
from projekat.measure.registry import register_jump_test

_N_SIGMA = 3.0


def naive(r: np.ndarray, rv: float, robust: float, *, local_sigma: float | None = None, **kwargs) -> JumpVerdict:
    daily_return = float(np.sum(r))
    # robust (jump-robust continuous-variance estimator, e.g. BPV) is
    # already the day's total variance in the same units as rv/RV_d, and
    # excludes the jump's own contribution by construction -- unlike
    # np.std(r)*sqrt(n) over the whole day, which includes the jump in its
    # own denominator and can inflate sigma enough to hide the jump it's
    # supposed to detect.
    sigma = local_sigma if local_sigma is not None else float(np.sqrt(max(robust, 0.0)))
    if sigma <= 0:
        return JumpVerdict(significant=False, statistic=0.0, p_value=1.0)
    statistic = abs(daily_return) / sigma
    significant = statistic > _N_SIGMA
    return JumpVerdict(significant=significant, statistic=statistic, p_value=float("nan"))


naive.name = "naive"
register_jump_test(naive)
