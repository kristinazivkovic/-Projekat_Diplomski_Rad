"""Naive baseline: y_hat = RV(t) (today's RV, used as the flat forecast
of the h-day average). No fitting -- the absolute floor every other
model must beat. Depends only on grid (RV_d exists identically under
every estimator/jump_test at a given grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.backtransform import residual_correction
from projekat.model.features import naive_features
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL


class _FittedNaive:
    def __init__(self, backtransform_correction: float):
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["RV_d_lag1"].to_numpy()


class _Naive:
    name = "naive"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None) -> _FittedNaive:
        X_val = naive_features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid = X_val.notna().all(axis=1) & y_val.notna()
        if valid.any():
            pred = X_val.loc[valid, "RV_d_lag1"].to_numpy()
            correction = residual_correction(y_val.loc[valid].to_numpy(), pred)
        else:
            correction = 0.0
        return _FittedNaive(correction)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return naive_features(df)


naive = _Naive()
register_model(naive)
