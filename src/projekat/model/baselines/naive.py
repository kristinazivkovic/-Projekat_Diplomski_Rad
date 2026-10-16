"""Naive baseline: log y_hat = ln RV(t) (today's RV as the flat forecast of
the h-day average), then the same trailing out-of-sample correction every
log model gets. No fitting -- the absolute floor every other model must
beat. Depends only on grid (RV_d is identical under every
estimator/jump_test at a given grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.registry import register_model


class _FittedNaive:
    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        value = np.log(history["RV_d"].to_numpy(dtype=float)[origin_idx].clip(min=1e-12))
        return {h: value.copy() for h in horizons}


class _Naive:
    name = "naive"
    stochastic = False
    predicts_levels = False
    refit = "never"
    correction = "log"
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedNaive:
        return _FittedNaive()


naive = _Naive()
register_model(naive)
