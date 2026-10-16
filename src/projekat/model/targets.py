"""Forecast target construction: y_{t,h} = (1/h) * sum_{i=1..h} RV_{t+i} --
the h-day AVERAGE future RV, modelled in logs for most models. This is
Corsi's own HAR construction and what every HAR paper reports.

For each horizon h, `add_targets` writes three columns:

    y_target_h{h}       y_{t,h} in levels (models with structural positivity
                        fit on this: har.py NNLS, and the ratio-of-sums
                        corrections use it)
    log_y_target_h{h}   log y_{t,h}
    target_end_h{h}     the DATE of the last RV entering y_{t,h}, i.e. the
                        day the target is realized

`target_end` is what makes the walk-forward causal: a row may train a model
refit on date R only if its target_end < R, and its out-of-sample error may
enter the correction for day t only if its target_end < t. Rows are the
panel's own trading days per symbol, so the target spans the next h
AVAILABLE days of that symbol (a dropped day is skipped, not counted).

Grouped by symbol: a panel slice spans every symbol, so shifting the whole
frame would pull RV_{t+i} from another symbol wherever two symbols' dates
interleave."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_columns(horizon: int) -> tuple[str, str, str]:
    """(log target, level target, realization date) column names for h."""
    return f"log_y_target_h{horizon}", f"y_target_h{horizon}", f"target_end_h{horizon}"


def add_targets(df: pd.DataFrame, horizons) -> pd.DataFrame:
    """Sort by (symbol, date) and add the target columns for every horizon."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    by_symbol_rv = df.groupby("symbol", sort=False)["RV_d"]
    by_symbol_date = df.groupby("symbol", sort=False)["date"]
    for horizon in horizons:
        log_col, level_col, end_col = target_columns(horizon)
        future_sum = pd.Series(0.0, index=df.index)
        any_nan = pd.Series(False, index=df.index)
        for i in range(1, horizon + 1):
            shifted = by_symbol_rv.shift(-i)
            any_nan = any_nan | shifted.isna()
            future_sum = future_sum + shifted.fillna(0.0)
        y = future_sum / horizon
        y[any_nan] = np.nan
        df[level_col] = y
        df[log_col] = np.log(y.clip(lower=1e-12)).where(y.notna())
        end = by_symbol_date.shift(-horizon)
        df[end_col] = end.where(y.notna())
    return df
