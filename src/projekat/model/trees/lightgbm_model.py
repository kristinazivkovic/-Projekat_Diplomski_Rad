"""LightGBM on the union feature set. Hyperparameters fixed and stated,
not tuned per window. Depends on grid, estimator, and jump_test since the
union includes C/J (the conditional decomposition's output)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from projekat.model.backtransform import residual_correction
from projekat.model.features import union_features
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL

_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=20,
    verbose=-1,
)


class _FittedLightGBM:
    def __init__(self, booster, feature_cols, backtransform_correction: float):
        self._booster = booster
        self._feature_cols = feature_cols
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._booster.predict(X[self._feature_cols])


class _LightGBM:
    name = "lightgbm"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        X = union_features(fit_train)
        y = fit_train[TARGET_COL]
        valid = X.notna().all(axis=1) & y.notna()
        model = lgb.LGBMRegressor(random_state=seed or 0, **_PARAMS)
        model.fit(X.loc[valid], y.loc[valid])

        X_val = union_features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid_val = X_val.notna().all(axis=1) & y_val.notna()
        if valid_val.any():
            val_pred = model.predict(X_val.loc[valid_val])
            correction = residual_correction(y_val.loc[valid_val].to_numpy(), val_pred)
        else:
            correction = 0.0

        return _FittedLightGBM(model, list(X.columns), correction)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return union_features(df)


lightgbm_model = _LightGBM()
register_model(lightgbm_model)
