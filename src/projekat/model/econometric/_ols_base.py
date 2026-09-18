"""Shared OLS fit/predict scaffolding for the HAR family, so the
regression mechanics (add constant, fit, predict, validation-fold
backtransform correction) are implemented once rather than per model file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from projekat.model.backtransform import residual_correction
from projekat.model.targets import TARGET_COL


class FittedOLS:
    def __init__(self, result, feature_cols: list[str], backtransform_correction: float):
        self._result = result
        self._feature_cols = feature_cols
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_ = sm.add_constant(X[self._feature_cols], has_constant="add")
        return self._result.predict(X_).to_numpy()


def fit_ols(features_fn, fit_train: pd.DataFrame, val_fold: pd.DataFrame) -> FittedOLS:
    """Fit on fit_train (excludes the validation fold), then estimate the
    backtransform correction from OUT-OF-FOLD residuals on val_fold --
    never in-sample."""
    X_train = features_fn(fit_train)
    y_train = fit_train[TARGET_COL]
    valid = X_train.notna().all(axis=1) & y_train.notna()
    X_with_const = sm.add_constant(X_train.loc[valid], has_constant="add")
    result = sm.OLS(y_train.loc[valid], X_with_const).fit()

    X_val = features_fn(val_fold)
    y_val = val_fold[TARGET_COL]
    valid_val = X_val.notna().all(axis=1) & y_val.notna()
    if valid_val.any():
        X_val_const = sm.add_constant(X_val.loc[valid_val], has_constant="add")
        val_pred = result.predict(X_val_const).to_numpy()
        correction = residual_correction(y_val.loc[valid_val].to_numpy(), val_pred)
    else:
        correction = 0.0

    return FittedOLS(result, list(X_train.columns), correction)
