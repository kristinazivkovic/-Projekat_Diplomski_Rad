"""The expanding-window, daily-forecast validation scheme.

The lookahead tests are the core: every refit and every correction may use
only data strictly before the date it serves, and no forecast for day t
may change when anything after t changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.config import EVAL_START
from projekat.model.registry import MODELS
from projekat.model.runner import run_one  # noqa: F401  (registers every model)
from projekat.model.targets import add_targets, target_columns
from projekat.model.walkforward import (
    finalize_forecasts,
    refit_dates,
    run_walk_forward,
    trading_sessions,
    training_frame,
)
from tests.conftest import make_panel

FIRST, LAST = "2014-01-01", "2015-03-31"
HORIZONS = (1, 5, 22)
FAST_MODELS = ["naive", "har", "log_har", "shar", "harq", "har_iv", "char", "har_j",
               "lightgbm", "xgboost", "mem", "arfima", "ttm_rv", "ttm_c", "ttm_c_plus_j", "log_har_ttm"]


@pytest.fixture(autouse=True)
def fake_ttm(monkeypatch):
    """No checkpoint in tests: a deterministic stand-in with the real
    output shape, so the TTM models' daily-context plumbing is exercised."""
    from projekat.model.foundation import _ttm_base

    def fake_batch(contexts):
        level = contexts[:, -20:].mean(axis=1, keepdims=True)
        return level + 0.001 * np.arange(1, 193)[None, :]

    monkeypatch.setattr(_ttm_base, "ttm_forecast_batch", fake_batch)


@pytest.fixture(scope="module")
def panel():
    return make_panel(design_ids=("5m__bpv__bns",))


# ── schedule and training slice ──────────────────────────────────────────


def test_refits_every_21_sessions_and_yearly():
    sessions = trading_sessions()
    starts = refit_dates("every_21", first="2014-01-01", last="2014-12-31")
    positions = sessions.searchsorted(pd.DatetimeIndex(starts))
    assert starts[0] == pd.Timestamp("2014-01-02")
    assert set(np.diff(positions)) == {21}

    yearly = refit_dates("yearly", first="2014-01-01", last="2016-12-31")
    assert yearly == [pd.Timestamp("2014-01-02"), pd.Timestamp("2015-01-02"), pd.Timestamp("2016-01-04")]


def test_training_frame_masks_targets_not_realized_before_the_refit(panel):
    df = add_targets(panel, HORIZONS)
    refit = pd.Timestamp("2014-06-02")
    train = training_frame(df, refit, HORIZONS)
    assert train["date"].max() < refit
    for h in HORIZONS:
        log_col, level_col, end_col = target_columns(h)
        realized = train[log_col].notna()
        assert (train.loc[realized, end_col] < refit).all()
        # the last h rows of each symbol before the refit lose their target
        assert train.groupby("symbol").tail(h)[log_col].isna().all()


# ── lookahead: refits ────────────────────────────────────────────────────


class _Spy:
    """Delegates to a real model, recording every training frame."""

    def __init__(self, model):
        self._model = model
        self.trains: list[pd.DataFrame] = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def fit(self, train, horizons, *, seed=None):
        self.trains.append(train.copy())
        return self._model.fit(train, horizons, seed=seed)


@pytest.mark.parametrize("model_name", FAST_MODELS)
def test_every_refit_uses_only_data_strictly_before_the_refit_date(panel, model_name):
    spy = _Spy(MODELS[model_name])
    run_walk_forward(panel, spy, horizons=HORIZONS, first=FIRST, last=LAST)
    starts = refit_dates(spy.refit, first=FIRST, last=LAST)
    expected_fits = 1 if spy.refit == "never" else len(starts)
    assert len(spy.trains) == expected_fits
    for refit, train in zip(starts, spy.trains):
        assert train["date"].max() < refit
        for h in HORIZONS:
            log_col, level_col, end_col = target_columns(h)
            used = train[log_col].notna() | train[level_col].notna()
            assert (train.loc[used, end_col] < refit).all(), f"{model_name} h={h} saw an unrealized target"


# ── lookahead: corrections ───────────────────────────────────────────────


def test_correction_and_mz_at_t_ignore_every_error_realized_on_or_after_t(panel):
    raw = run_walk_forward(panel, MODELS["log_har"], horizons=HORIZONS, first=FIRST, last=LAST, finalize=False)
    base = finalize_forecasts(raw, MODELS["log_har"])

    t0 = pd.Timestamp("2014-09-15")
    perturbed = raw.copy()
    late = perturbed["target_end"] >= t0
    perturbed.loc[late, "y_true"] *= 50.0          # huge errors, all realized on/after t0
    after = finalize_forecasts(perturbed, MODELS["log_har"])

    key = ["date", "symbol", "horizon"]
    merged = base.merge(after, on=key, suffixes=("", "_p"))
    early = merged["date"] <= t0
    for col in ("backtransform_correction", "y_hat", "mz_intercept", "mz_slope"):
        a, b = merged.loc[early, col].to_numpy(), merged.loc[early, f"{col}_p"].to_numpy()
        assert np.allclose(a, b, equal_nan=True), col
    # ...and the perturbation does reach the corrections after t0
    later = merged["date"] > t0 + pd.Timedelta(days=40)
    assert not np.allclose(merged.loc[later, "backtransform_correction"], merged.loc[later, "backtransform_correction_p"])


def test_a_22_day_error_waits_22_days_before_entering_the_correction(panel):
    raw = run_walk_forward(panel, MODELS["log_har"], horizons=(22,), first=FIRST, last=LAST, finalize=False)
    out = finalize_forecasts(raw, MODELS["log_har"])
    first_day = out["date"].min()
    first_realized = out.loc[out["date"] == first_day, "target_end"].min()
    # nothing is realized before the first target_end, so the correction
    # stays neutral (a = 0) up to and including that day
    neutral = out[out["date"] <= first_realized]
    assert (neutral["backtransform_correction"] == 0.0).all()
    assert (out.loc[out["date"] > first_realized + pd.Timedelta(days=3), "backtransform_correction"] != 0.0).any()


# ── lookahead: forecasts (causality of predict) ──────────────────────────


def _perturb_after(panel_df: pd.DataFrame, t0: pd.Timestamp, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = panel_df.copy()
    late = out["date"] > t0
    factor = rng.uniform(0.2, 5.0, late.sum())
    for col in ("RV_d", "RV_w", "RV_m", "C_d", "C_w", "C_m", "J_d", "J_w", "J_m", "RS_pos", "RS_neg"):
        out.loc[late, col] = out.loc[late, col].to_numpy() * factor
    out.loc[late, "RQ"] = out.loc[late, "RQ"].to_numpy() * factor**2
    out.loc[late, "r_d"] = -out.loc[late, "r_d"].to_numpy()
    out.loc[late, "jump_flag"] = ~out.loc[late, "jump_flag"].astype(bool)
    out.loc[late, "vix_lag1"] = out.loc[late, "vix_lag1"] * 2
    return out


@pytest.mark.parametrize("model_name", FAST_MODELS)
def test_changing_anything_after_t_never_changes_the_forecast_for_t(panel, model_name):
    # t0 sits in the MIDDLE of a 21-day block, so rows after t0 are inside
    # the block's own history -- a leaking predict() would see them
    t0 = pd.Timestamp("2014-07-15")
    model = MODELS[model_name]
    a = run_walk_forward(panel, model, horizons=HORIZONS, first=FIRST, last="2014-09-30", finalize=False)
    b = run_walk_forward(_perturb_after(panel, t0), model, horizons=HORIZONS, first=FIRST,
                         last="2014-09-30", finalize=False)
    key = ["date", "symbol", "horizon"]
    merged = a.merge(b, on=key, suffixes=("", "_p"))
    early = merged["date"] <= t0
    assert early.sum() > 100
    assert np.allclose(merged.loc[early, "raw_pred"], merged.loc[early, "raw_pred_p"], rtol=1e-9, equal_nan=True)


def test_patchtst_is_causal_and_refits_yearly():
    small = make_panel(symbols=("AAA",), start="2011-06-01", end="2015-02-27")
    spy = _Spy(MODELS["patchtst"])
    t0 = pd.Timestamp("2014-05-15")
    a = run_walk_forward(small, spy, horizons=(1,), seed=0, first=FIRST, last="2014-08-29", finalize=False)
    assert len(spy.trains) == 1                                  # one yearly refit inside 2014
    assert spy.trains[0]["date"].max() < pd.Timestamp("2014-01-02")
    b = run_walk_forward(_perturb_after(small, t0), MODELS["patchtst"], horizons=(1,), seed=0,
                         first=FIRST, last="2014-08-29", finalize=False)
    merged = a.merge(b, on=["date", "symbol", "horizon"], suffixes=("", "_p"))
    early = merged["date"] <= t0
    assert np.allclose(merged.loc[early, "raw_pred"], merged.loc[early, "raw_pred_p"], rtol=1e-5, atol=1e-6)


# ── daily state, warm-up, output shape ───────────────────────────────────


@pytest.mark.parametrize("model_name", ["mem", "arfima", "ttm_rv", "ttm_c_plus_j"])
def test_state_models_update_their_forecast_every_day(panel, model_name):
    """Regression for the old scheme's flaw: MEM/ARFIMA/TTM gave ONE number
    per symbol for a whole test year. Within one 21-day block (fixed
    parameters) the forecast must still move with the data."""
    out = run_walk_forward(panel, MODELS[model_name], horizons=(1,), first=FIRST, last="2014-01-31", finalize=False)
    one = out[out["symbol"] == "AAA"]
    assert one["raw_pred"].nunique() == len(one) > 15


def test_warm_up_year_is_flagged_and_output_covers_every_horizon(panel):
    out = run_walk_forward(panel, MODELS["log_har"], horizons=HORIZONS, first=FIRST, last=LAST)
    assert out["date"].min() >= pd.Timestamp(FIRST)
    assert set(out["horizon"]) == set(HORIZONS)
    assert (out["is_warmup"] == (out["date"] < pd.Timestamp(EVAL_START))).all()
    for col in ("y_hat", "y_hat_recalibrated", "backtransform_correction", "mz_slope", "r_d_next", "target_end"):
        assert col in out.columns


def test_r_d_next_is_the_same_symbols_next_session_return(panel):
    out = run_walk_forward(panel, MODELS["naive"], horizons=(1,), first=FIRST, last="2014-03-31", finalize=False)
    ref = panel.sort_values(["symbol", "date"]).copy()
    ref["expected"] = ref.groupby("symbol")["r_d"].shift(-1)
    merged = out.merge(ref[["symbol", "date", "expected"]], on=["symbol", "date"])
    assert np.allclose(merged["r_d_next"], merged["expected"])
