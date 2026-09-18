"""ttm_c: zero-shot TTM prognoza log(C_d) -- samo kontinuirana komponenta,
foundation analogon CHAR-a.

Zavisi od grid, estimator I jump_test: C je izlaz uslovne dekompozicije
(measure/decompose.py), koja zavisi i od A2 (koji ocenjivač) i od A3 (koji
test skoka obeležava dan). Normalan log put -- C_d je strogo pozitivan
(C = RV - J, a J <= RV po konstrukciji), za razliku od J (vidi
ttm_c_plus_j.py)."""

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


class _FittedTTMC:
    def __init__(self, per_symbol: dict[str, float], backtransform_correction: float):
        self._per_symbol = per_symbol
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return predict_from_map(X, self._per_symbol)


class _TTMC:
    name = "ttm_c"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        per_symbol = forecast_per_symbol(fit_train, "C_d", horizon, log_space=True)
        fitted = _FittedTTMC(per_symbol, 0.0)

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


ttm_c = _TTMC()
register_model(ttm_c)
