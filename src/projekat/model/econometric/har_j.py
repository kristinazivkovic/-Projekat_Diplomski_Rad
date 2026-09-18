"""HAR-J: RV decomposed into continuous (C) and jump (J) components,
entering as separate regressors -- tests whether jumps have different
persistence than the continuous part. Depends on grid, estimator, and
jump_test (both C and J come from the conditional decomposition)."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import har_j_features
from projekat.model.registry import register_model


class _HARJ:
    name = "har_j"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(har_j_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return har_j_features(df)


har_j = _HARJ()
register_model(har_j)
