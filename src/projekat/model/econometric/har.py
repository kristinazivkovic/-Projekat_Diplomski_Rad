"""HAR (Corsi 2009), fit by NNLS directly on LEVELS -- one of three
deliberately distinct positivity mechanisms in this codebase (the others:
log+correction in log_har.py/char.py/etc., and the structural
multiplicative form in mem.py). Never unify these.

Non-negative least squares on y_{t,h} ~ RV_d + RV_w + RV_m (levels, no
intercept dropped -- NNLS here includes a non-negative intercept as a
constant regressor) guarantees a non-negative forecast by construction,
since every fitted coefficient is >= 0 and every regressor (RV_d/w/m) is
already non-negative: no forecast can go negative, so there is no need for
either a log transform or a back-transform correction. This is the
structural counterpart to log_har.py's log-space fit -- same HAR feature
set, different fitting/positivity mechanism, deliberately kept as a
SEPARATE model rather than folded into log_har.py, so the thesis can
compare whether the positivity mechanism itself (not just the feature set)
affects forecast accuracy.

Depends only on grid -- RV_d/w/m exist identically under every
estimator/jump_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from projekat.model.registry import register_model
from projekat.model.targets import LEVEL_TARGET_COL


def _har_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same HAR feature set as log_har.py's har_features, but in LEVELS
    (no log) -- NNLS needs non-negative regressors to guarantee a
    non-negative fitted value, and RV_d/w/m are already non-negative by
    construction (they're variances)."""
    out = pd.DataFrame(index=df.index)
    out["const"] = 1.0
    out["RV_d_lag1"] = df["RV_d"]
    out["RV_w_lag1"] = df["RV_w"]
    out["RV_m_lag1"] = df["RV_m"]
    return out


class _FittedHARLevels:
    def __init__(self, coef: np.ndarray, feature_cols: list[str]):
        self._coef = coef
        self._feature_cols = feature_cols
        # NNLS coefficients are non-negative and every regressor is
        # non-negative, so the fitted value can never be negative --
        # structural positivity, no correction needed.
        self.backtransform_correction = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self._feature_cols].to_numpy() @ self._coef)


class _HARLevels:
    name = "har"
    stochastic = False
    is_sequence_model = False
    predicts_levels = True
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None) -> _FittedHARLevels:
        X_train = _har_level_features(fit_train)
        y_train = fit_train[LEVEL_TARGET_COL]
        valid = X_train.notna().all(axis=1) & y_train.notna()

        coef, _residual_norm = nnls(X_train.loc[valid].to_numpy(), y_train.loc[valid].to_numpy())
        return _FittedHARLevels(coef, list(X_train.columns))

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return _har_level_features(df)


har = _HARLevels()
register_model(har)
