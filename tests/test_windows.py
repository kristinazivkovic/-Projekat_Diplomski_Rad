"""Anchored walk-forward window tests: no leakage between train/validation/
test spans, and the window generator produces the documented 11 anchored
windows with a fixed training start."""

from __future__ import annotations

import pandas as pd

from projekat.model.windows import anchored_windows, split_with_validation


def test_eleven_anchored_windows_fixed_start():
    windows = anchored_windows()
    assert len(windows) == 11
    # training start is fixed across every window -- anchored, never rolling
    assert all(w.train_start == windows[0].train_start for w in windows)
    # only the training end expands, strictly increasing window-over-window
    train_ends = [w.train_end for w in windows]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_no_test_observation_in_train_or_validation_span():
    """For every anchored window, no date in the test span may also fall
    inside that window's training span (fit_train) or its validation fold
    -- the walk-forward split must never leak future data into fitting or
    correction estimation."""
    windows = anchored_windows()
    dates = pd.date_range("2010-01-01", "2026-01-01", freq="D")
    df = pd.DataFrame({"date": dates})

    for window in windows:
        fit_train, val_fold, test = split_with_validation(df, window)
        train_dates = set(fit_train["date"])
        val_dates = set(val_fold["date"])
        test_dates = set(test["date"])

        assert train_dates.isdisjoint(test_dates), f"window {window}: train/test overlap"
        assert val_dates.isdisjoint(test_dates), f"window {window}: validation/test overlap"
        # fit_train and val_fold together must also never leak into test,
        # and fit_train must exclude the validation fold entirely
        assert train_dates.isdisjoint(val_dates), f"window {window}: train/validation overlap"


def test_validation_fold_is_last_year_of_training_span():
    windows = anchored_windows()
    for window in windows:
        assert window.val_start > window.train_start
        assert window.val_start <= window.train_end
        # the validation fold is exactly the window's own last training year
        assert (window.train_end - window.val_start).days in range(363, 367)


def test_no_leftover_gap_between_train_end_and_test_start():
    """test_start must immediately follow train_end -- no buffer year is
    held out, since 2015-2025 is exactly eleven test years for eleven
    windows."""
    for window in anchored_windows():
        assert window.test_start == window.train_end + pd.Timedelta(days=1)
