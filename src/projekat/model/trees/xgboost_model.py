"""XGBoost on the union feature set. Same fixed-hyperparameter policy and
the same refit/predict mechanics as LightGBM (lightgbm_model.fit_trees)."""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from projekat.model.registry import register_model
from projekat.model.trees.lightgbm_model import FittedTrees, fit_trees

_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=20,
    verbosity=0,
    n_jobs=1,
)


class _XGBoost:
    name = "xgboost"
    stochastic = False
    predicts_levels = False
    refit = "every_21"
    correction = "log"
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> FittedTrees:
        return fit_trees(lambda: xgb.XGBRegressor(random_state=seed or 0, **_PARAMS), train, horizons)


xgboost_model = _XGBoost()
register_model(xgboost_model)
