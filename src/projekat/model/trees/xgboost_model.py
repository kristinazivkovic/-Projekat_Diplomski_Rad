"""XGBoost on the union feature set. Same fixed-hyperparameter and
validation-fold backtransform-correction policy as LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from projekat.model.backtransform import residual_correction
from projekat.model.features import union_features
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL

_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=20,
    verbosity=0,
)


class _FittedXGBoost:
    def __init__(self, model, feature_cols, backtransform_correction: float):
        self._model = model
        self._feature_cols = feature_cols
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self._feature_cols])


class _XGBoost:
    name = "xgboost"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        X = union_features(fit_train)
        y = fit_train[TARGET_COL]
        valid = X.notna().all(axis=1) & y.notna()
        model = xgb.XGBRegressor(random_state=seed or 0, **_PARAMS)
        model.fit(X.loc[valid], y.loc[valid])

        X_val = union_features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid_val = X_val.notna().all(axis=1) & y_val.notna()
        if valid_val.any():
            val_pred = model.predict(X_val.loc[valid_val])
            correction = residual_correction(y_val.loc[valid_val].to_numpy(), val_pred)
        else:
            correction = 0.0

        return _FittedXGBoost(model, list(X.columns), correction)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return union_features(df)


xgboost_model = _XGBoost()
register_model(xgboost_model)
