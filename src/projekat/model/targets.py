"""Forecast target construction (Decision C2 in revision 2 of the plan):
y_{t,h} = (1/h) * sum_{i=1..h} RV_{t+i} -- the h-day AVERAGE future RV,
modelled in logs for most models. This is Corsi's own HAR construction and
what every HAR paper reports; a single-day target RV_{t+h} would be a
different, much noisier problem and incomparable with the literature.

TARGET_COL (log space) is the target used by every log+backtransform-
correction model. LEVEL_TARGET_COL is the same y_{t,h}, in levels, kept
alongside it for the small number of models whose positivity is
STRUCTURAL rather than achieved via a log transform (e.g. har.py's NNLS
fit, mem.py's multiplicative form) -- those models fit directly on levels
and must never go through backtransform.py (see model/protocols.py's
`predicts_levels` flag)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "log_y_target"
LEVEL_TARGET_COL = "y_target"


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    # grouped by symbol: a panel slice spans every symbol in the universe,
    # so shifting on the whole frame would pull RV_{t+i} from a different
    # symbol's row whenever two symbols' dates interleave near a boundary.
    rv_by_symbol = df.groupby("symbol")["RV_d"]
    future_sum = pd.Series(0.0, index=df.index)
    any_nan = pd.Series(False, index=df.index)
    for i in range(1, horizon + 1):
        shifted = rv_by_symbol.shift(-i)
        any_nan = any_nan | shifted.isna()
        future_sum = future_sum + shifted.fillna(0.0)
    y = future_sum / horizon
    y[any_nan] = np.nan
    df[LEVEL_TARGET_COL] = y
    df[TARGET_COL] = np.log(y.clip(lower=1e-12))
    return df
