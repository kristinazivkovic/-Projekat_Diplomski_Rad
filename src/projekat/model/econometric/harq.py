"""HARQ (Bollerslev, Patton & Quaedvlieg 2016): HAR with the daily RV
coefficient interacted with sqrt(realized quarticity) -- when measurement
was noisy that day (high RQ), the model automatically downweights it.
Depends only on grid -- RQ is a Factor A2 output computed identically
regardless of jump_test."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import harq_features
from projekat.model.registry import register_model


class _HARQ:
    name = "harq"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(harq_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return harq_features(df)


harq = _HARQ()
register_model(harq)
