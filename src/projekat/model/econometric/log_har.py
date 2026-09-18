"""Log-HAR (Corsi 2009): RV regressed on log daily/weekly/monthly lagged RV
components, in log space, back-transformed via the shared non-parametric
Jensen correction (backtransform.py). This is the HEADLINE BENCHMARK for
relative QLIKE (evaluation/losses.py::relative_qlike) -- the standard form
of HAR reported throughout the literature. Depends only on grid -- RV_d/w/m
exist identically under every estimator/jump_test.

Deliberately kept distinct from `har.py` (NNLS-in-levels): the two use
different positivity mechanisms (log+correction here vs. non-negative
coefficients there) and must never be unified into one implementation --
see har.py's docstring for why."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import har_features
from projekat.model.registry import register_model


class _LogHAR:
    name = "log_har"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(har_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return har_features(df)


log_har = _LogHAR()
register_model(log_har)
