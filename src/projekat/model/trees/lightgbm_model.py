"""LightGBM on the union feature set, one booster per horizon per refit.
Hyperparameters fixed and stated, never tuned (tuning would make the
model's variance share reflect search effort, not architecture). Depends
on grid, estimator, and jump_test since the union includes C/J."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from projekat.model.features import union_features
from projekat.model.registry import register_model
from projekat.model.targets import target_columns

_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=20,
    verbose=-1,
    n_jobs=1,
)


class FittedTrees:
    """Shared by LightGBM and XGBoost: one regressor per horizon."""

    def __init__(self, models: dict, feature_cols: list[str]):
        self._models = models
        self._feature_cols = feature_cols

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        X = union_features(history.iloc[origin_idx])[self._feature_cols]
        valid = X.notna().all(axis=1).to_numpy()
        out = {}
        for h in horizons:
            pred = np.full(len(X), np.nan)
            model = self._models.get(h)
            if model is not None and valid.any():
                pred[valid] = model.predict(X.loc[valid])
            out[h] = pred
        return out


def fit_trees(make_model, train: pd.DataFrame, horizons) -> FittedTrees:
    X = union_features(train)
    finite_x = X.notna().all(axis=1)
    models = {}
    for h in horizons:
        log_col, _, _ = target_columns(h)
        y = train[log_col]
        valid = finite_x & y.notna()
        if valid.sum() < 50:
            models[h] = None
            continue
        model = make_model()
        model.fit(X.loc[valid], y.loc[valid])
        models[h] = model
    return FittedTrees(models, list(X.columns))


class _LightGBM:
    name = "lightgbm"
    stochastic = False
    predicts_levels = False
    refit = "every_21"
    correction = "log"
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> FittedTrees:
        return fit_trees(lambda: lgb.LGBMRegressor(random_state=seed or 0, **_PARAMS), train, horizons)


lightgbm_model = _LightGBM()
register_model(lightgbm_model)
