"""Anchored walk-forward window generator. Training start is fixed; only
the training end expands. Never rolling -- eliminates lookahead.

11 windows over 2010-2025 (not 10): window 1 trains 2010-2014/tests 2015,
window 11 trains 2010-2024/tests 2025 -- 2015-2025 is eleven test years,
so there is no leftover year to hold out as a buffer.

Each window's training data is further split: the last year is carved out
as a VALIDATION FOLD, used only to estimate the non-parametric Jensen
back-transform correction (never for hyperparameter tuning, never for
evaluation) -- see backtransform.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp  # start of the last-year validation fold, within [train_start, train_end]
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def anchored_windows(
    *,
    sample_start: str = "2010-01-01",
    initial_train_years: int = 5,
    n_windows: int = 11,
) -> list[Window]:
    start = pd.Timestamp(sample_start)
    windows = []
    for w in range(n_windows):
        train_end = start + pd.DateOffset(years=initial_train_years + w) - pd.Timedelta(days=1)
        val_start = train_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        windows.append(
            Window(train_start=start, train_end=train_end, val_start=val_start, test_start=test_start, test_end=test_end)
        )
    return windows


def split(df: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) -- train includes the validation fold; use
    split_with_validation if the fold must be excluded from fitting."""
    train = df[(df["date"] >= window.train_start) & (df["date"] <= window.train_end)]
    test = df[(df["date"] >= window.test_start) & (df["date"] <= window.test_end)]
    return train, test


def split_with_validation(df: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fit_train, val_fold, test) -- fit_train excludes the last training
    year (the validation fold), which is used only to estimate the
    non-parametric Jensen back-transform correction from out-of-fold
    residuals."""
    fit_train = df[(df["date"] >= window.train_start) & (df["date"] < window.val_start)]
    val_fold = df[(df["date"] >= window.val_start) & (df["date"] <= window.train_end)]
    test = df[(df["date"] >= window.test_start) & (df["date"] <= window.test_end)]
    return fit_train, val_fold, test
