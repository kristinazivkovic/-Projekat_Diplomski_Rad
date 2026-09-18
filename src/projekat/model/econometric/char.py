"""CHAR: HAR structure but using only the continuous component C (skokovi
isključeni). Depends on grid, estimator, AND jump_test -- C is the
conditional decomposition's output, which depends on both A2 and A3."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import char_features
from projekat.model.registry import register_model


class _CHAR:
    name = "char"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(char_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return char_features(df)


char = _CHAR()
register_model(char)
