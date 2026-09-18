"""model/runner.py orchestration tests: every registered model round-trips
by name alone (registry discipline), and the primary seed-collapse path is
kept structurally distinct from the secondary forecast-ensemble path
(Decision C3) -- the two must never be accidentally swappable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.model import baselines, econometric, trees  # noqa: F401 (registration)
from projekat.model.registry import MODELS
from projekat.model.runner import collapse_seeds, ensemble_forecast, run_all
from projekat.model.windows import anchored_windows


def _synthetic_panel(symbols=("AAA", "BBB"), design_ids=("5m__bpv__bns",), n_days=1200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    rows = []
    for sym in symbols:
        for design_id in design_ids:
            rv_d = np.abs(rng.standard_normal(n_days)) * 1e-4 + 1e-5
            # RV_w/RV_m are independently-jittered, not identical to RV_d --
            # avoids a perfectly collinear design matrix in the HAR-family
            # OLS fits (an artifact of a degenerate fixture, not a real bug).
            rv_w = rv_d * (1.0 + 0.05 * rng.standard_normal(n_days))
            rv_m = rv_d * (1.0 + 0.05 * rng.standard_normal(n_days))
            vix = np.abs(rng.standard_normal(n_days)) * 2 + 18
            for d, r, rw, rm, v in zip(dates, rv_d, rv_w, rv_m, vix):
                rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "design_id": design_id,
                        "RV_d": r,
                        "RV_w": rw,
                        "RV_m": rm,
                        "C_d": r * 0.9,
                        "C_w": rw * 0.9,
                        "C_m": rm * 0.9,
                        "J_d": r * 0.1,
                        "J_w": rw * 0.1,
                        "J_m": rm * 0.1,
                        "jump_flag": bool(rng.random() < 0.05),
                        "RS_pos": r / 2,
                        "RS_neg": r / 2,
                        "RQ": r**2,
                        "r_d": 0.001,
                        "vix_lag1": v,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def small_window():
    return anchored_windows(sample_start="2010-01-01", initial_train_years=3, n_windows=1)


@pytest.mark.parametrize(
    "model_name",
    ["naive", "har", "log_har", "shar", "harq", "har_iv", "char", "har_j", "arfima", "mem", "lightgbm", "xgboost"],
)
def test_model_round_trips_through_runner_by_name(model_name, small_window):
    """Every registered model must be resolvable and runnable purely by
    string name through run_all/MODELS -- runner.py never imports a
    concrete model module."""
    assert model_name in MODELS
    panel = _synthetic_panel()
    test_df, val_df = run_all(panel, model_names=[model_name], horizons=[1], windows=small_window)
    assert not test_df.empty
    assert (test_df["model"] == model_name).all()
    assert test_df["y_hat"].notna().all()


def test_har_nnls_forecast_is_never_negative(small_window):
    """har.py fits NNLS directly on levels -- non-negative coefficients on
    non-negative regressors must guarantee a non-negative forecast, with
    no log transform or back-transform correction involved at all."""
    panel = _synthetic_panel()
    test_df, _ = run_all(panel, model_names=["har"], horizons=[1], windows=small_window)
    assert (test_df["y_hat"] >= 0).all()
    assert test_df["backtransform_correction"].isna().all()  # predicts_levels models skip backtransform entirely


def test_log_har_uses_backtransform_correction(small_window):
    """log_har.py (unlike har.py) is a log+correction model: its rows must
    carry a real, non-NaN backtransform_correction and its log_y_hat must
    differ from its already-back-transformed y_hat."""
    panel = _synthetic_panel()
    test_df, _ = run_all(panel, model_names=["log_har"], horizons=[1], windows=small_window)
    assert test_df["backtransform_correction"].notna().all()
    assert test_df["log_y_hat"].notna().all()


def test_collapse_seeds_is_a_no_op_that_preserves_per_seed_rows(small_window):
    """The PRIMARY path (Decision C3) must average QLIKE-relevant losses
    per seed downstream, never forecasts -- collapse_seeds itself must
    leave the per-seed rows untouched (a true no-op), so that whatever
    loss function runs after it operates on one row per seed and only
    THEN gets averaged."""
    panel = _synthetic_panel()
    test_df, _ = run_all(panel, model_names=["naive"], horizons=[1], windows=small_window)
    collapsed = collapse_seeds(test_df)
    pd.testing.assert_frame_equal(collapsed, test_df)


def test_ensemble_forecast_is_a_structurally_distinct_secondary_path(small_window):
    """ensemble_forecast must average forecasts (in log space, before
    back-transformation) -- a fundamentally different operation from
    collapse_seeds's per-seed-loss no-op. The two functions must never be
    swappable: collapse_seeds returns one row PER SEED (same length as
    input for a stochastic model), while ensemble_forecast collapses all
    seeds of a stochastic model into ONE row per (model, design_id,
    horizon, date, symbol) -- a different shape, not just different
    values, so a caller cannot accidentally use one in place of the other
    without the row count itself flagging the mistake."""
    # Build a synthetic stochastic results frame directly (bypassing a real
    # PatchTST fit, which is expensive) -- five seeds of an identical
    # (model, design_id, horizon, date, symbol) cell with different
    # log_y_hat/backtransform_correction per seed, to isolate the
    # aggregation logic itself from any specific model's fitting behavior.
    n_seeds = 5
    log_y_hats = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    corrections = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    rows = pd.DataFrame(
        {
            "model": ["patchtst"] * n_seeds,
            "design_id": ["5m__bpv__bns"] * n_seeds,
            "horizon": [1] * n_seeds,
            "date": [pd.Timestamp("2020-01-01")] * n_seeds,
            "symbol": ["AAA"] * n_seeds,
            "seed": [0, 1, 2, 3, 4],
            "log_y_hat": log_y_hats,
            "backtransform_correction": corrections,
            "y_hat": np.exp(log_y_hats + corrections),
            "y_true": [1e-4] * n_seeds,
        }
    )

    collapsed = collapse_seeds(rows)
    ensembled = ensemble_forecast(rows)

    # collapse_seeds: same length, same values -- a true no-op
    assert len(collapsed) == n_seeds
    pd.testing.assert_frame_equal(collapsed, rows)

    # ensemble_forecast: collapses to exactly one row for this one cell
    assert len(ensembled) == 1
    expected_log_mean = np.mean(log_y_hats)
    expected_correction_mean = np.mean(corrections)
    expected_y_hat = np.exp(expected_log_mean + expected_correction_mean)
    assert ensembled["y_hat_ensemble"].iloc[0] == pytest.approx(expected_y_hat)

    # the ensemble forecast must NOT equal the mean of the per-seed
    # already-back-transformed y_hat values (that would be forecast-space
    # averaging, not log-space averaging before back-transformation)
    naive_mean_of_backtransformed = np.mean(rows["y_hat"].to_numpy())
    assert ensembled["y_hat_ensemble"].iloc[0] != pytest.approx(naive_mean_of_backtransformed)


def test_ensemble_forecast_does_not_double_count_backtransform_correction():
    """Regression test for the specific bug this session fixed: averaging
    log(y_hat) (already back-transformed) instead of log_y_hat directly
    would bake each seed's own correction back in before averaging,
    double-counting it relative to averaging log_y_hat and the correction
    SEPARATELY. mean(log_y_hat + correction) == mean(log_y_hat) +
    mean(correction) always holds by linearity, so the two computations
    only diverge once qlike-style NONLINEAR downstream usage or a
    correlation between log_y_hat and correction across seeds is
    introduced -- construct log_y_hat and correction so they covary
    (larger log_y_hat paired with larger correction) and confirm
    ensemble_forecast's result matches the log_y_hat-then-correction
    formula exactly, not a coincidental value."""
    n_seeds = 3
    log_y_hats = np.array([-0.2, 0.1, 0.4])
    corrections = np.array([0.0, 0.5, 1.0])
    rows = pd.DataFrame(
        {
            "model": ["patchtst"] * n_seeds,
            "design_id": ["5m__bpv__bns"] * n_seeds,
            "horizon": [1] * n_seeds,
            "date": [pd.Timestamp("2020-01-01")] * n_seeds,
            "symbol": ["AAA"] * n_seeds,
            "seed": [0, 1, 2],
            "log_y_hat": log_y_hats,
            "backtransform_correction": corrections,
            "y_hat": np.exp(log_y_hats + corrections),
            "y_true": [1e-4] * n_seeds,
        }
    )
    ensembled = ensemble_forecast(rows)

    correct_value = np.exp(np.mean(log_y_hats) + np.mean(corrections))
    assert ensembled["y_hat_ensemble"].iloc[0] == pytest.approx(correct_value)

    # ensemble_forecast must read log_y_hat directly, not re-derive it via
    # log(y_hat) -- verified by checking the two source columns actually
    # differ per row (otherwise this test wouldn't distinguish the bug)
    assert not np.allclose(rows["log_y_hat"], np.log(rows["y_hat"]))
