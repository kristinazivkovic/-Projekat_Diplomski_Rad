"""SHAR: positive/negative semivariance decomposition (Patton & Sheppard
2015) plus weekly/monthly components -- captures asymmetry between
negative and positive shocks. Depends only on grid -- RS_pos/RS_neg are
computed directly from returns (Factor A2), not from the A2/A3
conditional decomposition."""

from __future__ import annotations

import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import shar_features
from projekat.model.registry import register_model


class _SHAR:
    name = "shar"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        return fit_ols(shar_features, fit_train, val_fold)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return shar_features(df)


shar = _SHAR()
register_model(shar)
