"""Back-transform corrections: the non-parametric log correction, the
ratio-of-sums combined correction, and their TRAILING out-of-sample forms
(window of realizations strictly before t), checked against brute force."""

from __future__ import annotations

import numpy as np
import pytest

from projekat.model.backtransform import (
    backtransform,
    combined_forecast_correction,
    residual_correction,
    trailing_corrections,
    trailing_mincer_zarnowitz,
)


def test_correction_recovers_sigma2_over_2_for_gaussian_residuals():
    rng = np.random.default_rng(0)
    sigma, n = 0.3, 200_000
    resid = rng.normal(0.0, sigma, n)
    assert residual_correction(resid, np.zeros(n)) == pytest.approx(sigma**2 / 2, rel=0.02)


def test_correction_diverges_from_gaussian_formula_for_skewed_residuals():
    rng = np.random.default_rng(1)
    scale, n = 0.5, 200_000
    resid = rng.exponential(scale, n) - scale
    assert abs(residual_correction(resid, np.zeros(n)) - scale**2 / 2) > 0.03


def test_backtransform_applies_additive_correction_in_log_space():
    log_y_hat = np.array([0.0, 1.0, 2.0])
    assert backtransform(log_y_hat, 0.5) == pytest.approx(np.exp(log_y_hat + 0.5))
    assert backtransform(log_y_hat, 0.0) == pytest.approx(np.exp(log_y_hat))


def test_combined_correction_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """One near-zero forecast (e.g. after ttm_c_plus_j's zero clip) blows a
    mean of ratios up; the ratio of sums stays put."""
    y = np.array([1.0, 1.0, 1.0, 1.0])
    f = np.array([1.0, 1.0, 1.0, 1e-9])
    m = combined_forecast_correction(y, f)
    assert m == pytest.approx(y.sum() / f.sum())
    assert m < 1.5                                   # mean of ratios would be ~2.5e8
    assert np.sum(m * f) == pytest.approx(np.sum(y))  # unbiased in sums
    assert combined_forecast_correction(y, np.zeros(4)) == 1.0


# ── trailing forms against brute force ───────────────────────────────────


def _random_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    origin = np.sort(rng.integers(0, 400, n))
    horizon = rng.integers(1, 23, n)
    realized = origin + horizon
    realized[rng.random(n) < 0.05] = -1               # unrealized targets
    raw_log = rng.normal(-9, 0.5, n)
    y = np.exp(raw_log + rng.normal(0.1, 0.4, n))
    return origin, realized, raw_log, y


def test_trailing_log_correction_matches_brute_force():
    origin, realized, raw, y = _random_series()
    out = trailing_corrections(origin, realized, raw, y, kind="log", predicts_levels=False, window=50)
    for i in range(len(origin)):
        use = (realized >= origin[i] - 50) & (realized <= origin[i] - 1) & (realized >= 0)
        expected = np.log(np.mean(y[use] / np.exp(raw[use]))) if use.any() else 0.0
        assert out["backtransform_correction"][i] == pytest.approx(expected)
    assert out["y_hat"] == pytest.approx(np.exp(raw + out["backtransform_correction"]))


def test_trailing_ratio_correction_matches_brute_force():
    origin, realized, raw, y = _random_series(seed=1)
    out = trailing_corrections(origin, realized, raw, y, kind="ratio_of_sums", predicts_levels=False, window=30)
    level = np.exp(raw)
    for i in range(0, len(origin), 7):
        use = (realized >= origin[i] - 30) & (realized <= origin[i] - 1) & (realized >= 0)
        expected = y[use].sum() / level[use].sum() if use.any() else 1.0
        assert out["combined_correction"][i] == pytest.approx(expected)
    assert np.isnan(out["backtransform_correction"]).all()


def test_levels_model_takes_no_correction():
    origin, realized, _raw, y = _random_series(seed=2)
    levels = y * 0.9
    out = trailing_corrections(origin, realized, levels, y, kind="none", predicts_levels=True)
    assert np.array_equal(out["y_hat"], levels)
    assert np.isnan(out["log_y_hat"]).all()


def test_trailing_mincer_zarnowitz_matches_polyfit_on_the_window():
    origin, realized, raw, y = _random_series(n=2000, seed=3)
    y_hat = np.exp(raw)
    a, b = trailing_mincer_zarnowitz(origin, realized, y_hat, y, window=120, min_pairs=30)
    checked = 0
    for i in range(0, len(origin), 25):
        use = (realized >= origin[i] - 120) & (realized <= origin[i] - 1) & (realized >= 0)
        if use.sum() < 30:
            assert np.isnan(a[i]) and np.isnan(b[i])
            continue
        slope, intercept = np.polyfit(y_hat[use], y[use], 1)
        assert b[i] == pytest.approx(slope, rel=1e-6)
        assert a[i] == pytest.approx(intercept, rel=1e-6, abs=1e-12)
        checked += 1
    assert checked > 10
