"""ttm_rv: zero-shot TTM prognoza log(RV_d), prikovan checkpoint.

Zavisi samo od mreže (grid) -- RV_d je identičan pod svakom A2/A3
kombinacijom na datoj mreži. Normalan log put: serija se log-transformiše,
TTM prognozira u log prostoru, i prognoza prolazi kroz deljenu
neparametarsku back-transform korekciju (backtransform.py) kao svaki drugi
log-model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.backtransform import residual_correction
from projekat.model.foundation._ttm_base import (
    forecast_per_symbol,
    predict_from_map,
    symbol_only_features,
)
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL


class _FittedTTMRV:
    def __init__(self, per_symbol: dict[str, float], backtransform_correction: float):
        self._per_symbol = per_symbol
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return predict_from_map(X, self._per_symbol)


class _TTMRV:
    name = "ttm_rv"
    stochastic = False          # zero-shot, deterministički -- nema semena
    is_sequence_model = False
    predicts_levels = False     # log put
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        per_symbol = forecast_per_symbol(fit_train, "RV_d", horizon, log_space=True)
        fitted = _FittedTTMRV(per_symbol, 0.0)

        X_val = symbol_only_features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid = X_val["symbol"].isin(per_symbol) & y_val.notna()
        if valid.any():
            pred = fitted.predict(X_val.loc[valid])
            ok = ~np.isnan(pred)
            if ok.any():
                fitted.backtransform_correction = residual_correction(
                    y_val.loc[valid].to_numpy()[ok], pred[ok]
                )
        return fitted

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return symbol_only_features(df)


ttm_rv = _TTMRV()
register_model(ttm_rv)
