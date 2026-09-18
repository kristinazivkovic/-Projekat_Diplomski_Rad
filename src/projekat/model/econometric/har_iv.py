"""HAR-IV: HAR extended with VIX(t-1) -- tests whether the options market
carries information not already in the price history. Depends only on
grid -- vix_lag1 is identical across estimator/jump_test."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import har_iv_features
from projekat.model.registry import register_model


class _HARIV:
    name = "har_iv"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(har_iv_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return har_iv_features(df)


har_iv = _HARIV()
register_model(har_iv)
