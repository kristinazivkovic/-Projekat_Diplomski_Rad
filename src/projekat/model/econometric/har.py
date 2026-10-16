"""HAR (Corsi 2009), fit by NNLS directly on LEVELS -- one of the
deliberately distinct positivity mechanisms in this codebase (the others:
log+correction in log_har.py/char.py/etc., the structural multiplicative
form in mem.py, and zero-clipping plus one combined correction in
ttm_c_plus_j.py). Never unify these.

Non-negative least squares on y_{t,h} ~ const + RV_d + RV_w + RV_m (levels,
the non-negative intercept enters as a constant regressor) guarantees a
non-negative forecast by construction, since every fitted coefficient is
>= 0 and every regressor (RV_d/w/m) is already non-negative -- so there is
no log transform and no back-transform correction (correction = "none").
Deliberately kept as a SEPARATE model from log_har.py, so the thesis can
compare whether the positivity mechanism itself (not just the feature set)
affects forecast accuracy.

Depends only on grid -- RV_d/w/m exist identically under every
estimator/jump_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from projekat.model.registry import register_model
from projekat.model.targets import target_columns

_COLUMNS = ("RV_d", "RV_w", "RV_m")


def _level_design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in _COLUMNS])


class _FittedHARLevels:
    def __init__(self, coefs: dict[int, np.ndarray | None]):
        self._coefs = coefs

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        X = _level_design(history.iloc[origin_idx])
        valid = np.isfinite(X).all(axis=1)
        out = {}
        for h in horizons:
            pred = np.full(len(X), np.nan)
            coef = self._coefs.get(h)
            if coef is not None:
                pred[valid] = X[valid] @ coef
            out[h] = pred
        return out


class _HARLevels:
    name = "har"
    stochastic = False
    predicts_levels = True
    refit = "every_21"
    correction = "none"
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedHARLevels:
        X = _level_design(train)
        finite_x = np.isfinite(X).all(axis=1)
        coefs = {}
        for h in horizons:
            _, level_col, _ = target_columns(h)
            y = train[level_col].to_numpy(dtype=float)
            valid = finite_x & np.isfinite(y)
            coefs[h] = nnls(X[valid], y[valid])[0] if valid.sum() > X.shape[1] else None
        return _FittedHARLevels(coefs)


har = _HARLevels()
register_model(har)
