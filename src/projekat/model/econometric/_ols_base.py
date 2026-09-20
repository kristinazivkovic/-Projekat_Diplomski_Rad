"""Shared OLS fit/predict for the log-space HAR family (log_har, char,
har_j, shar, harq, har_iv, the HAR part of log_har_ttm): one regression of
log y_{t,h} per horizon, fit on the rows the walk-forward hands in (all
strictly before the refit date, NaN targets where not yet realized), and a
forecast for each origin row from that row's own regressors (all known on
the origin day). No correction here -- walkforward.py estimates it from the
model's own out-of-sample errors.

`features_fn(df)` builds the regressors. A model whose regressors need
parameters estimated on the training data (HARQ's standardized sqrt(RQ))
passes `prepare(train) -> params` and a `features_fn(df, params)`; the
params are frozen at the refit and reused for every forecast until the
next one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.targets import target_columns


class FittedOLS:
    def __init__(self, coefs: dict[int, np.ndarray | None], feature_cols: list[str], features_fn, params=None):
        self._coefs = coefs
        self._feature_cols = feature_cols
        self._features_fn = features_fn
        self._params = params

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._features_fn(df) if self._params is None else self._features_fn(df, self._params)
        return X[self._feature_cols]

    def predict_rows(self, rows: pd.DataFrame, horizons) -> dict[int, np.ndarray]:
        X = self.features(rows)
        values = X.to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        design = np.column_stack([np.ones(len(values)), values])
        out = {}
        for h in horizons:
            pred = np.full(len(values), np.nan)
            coef = self._coefs.get(h)
            if coef is not None and valid.any():
                pred[valid] = design[valid] @ coef
            out[h] = pred
        return out

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        return self.predict_rows(history.iloc[origin_idx], horizons)


def fit_ols(features_fn, train: pd.DataFrame, horizons, *, prepare=None) -> FittedOLS:
    params = prepare(train) if prepare is not None else None
    X = features_fn(train) if params is None else features_fn(train, params)
    values = X.to_numpy(dtype=float)
    finite_x = np.isfinite(values).all(axis=1)
    design = np.column_stack([np.ones(len(values)), values])
    coefs: dict[int, np.ndarray | None] = {}
    for h in horizons:
        log_col, _, _ = target_columns(h)
        y = train[log_col].to_numpy(dtype=float)
        valid = finite_x & np.isfinite(y)
        if valid.sum() <= design.shape[1]:
            coefs[h] = None
            continue
        coefs[h], *_ = np.linalg.lstsq(design[valid], y[valid], rcond=None)
    return FittedOLS(coefs, list(X.columns), features_fn, params)


class OLSModel:
    """Base for the stateless log-space HAR-family models: every refit is
    one OLS per horizon on `features_fn`'s regressors."""

    stochastic = False
    predicts_levels = False
    refit = "every_21"
    correction = "log"

    features_fn = None  # set by subclasses (a plain function of the frame)

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> FittedOLS:
        return fit_ols(type(self).features_fn, train, horizons)
