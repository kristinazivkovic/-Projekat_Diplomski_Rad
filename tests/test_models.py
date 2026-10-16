"""Model-specific changes: HARQ's standardized interaction, the ARFIMA
bandwidth and daily filter, the MEM recursion, PatchTST's yearly training
details (early-stopping purge, jump-channel standardization)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.econometric.arfima import _SymbolARFIMA, _binomial_weights, _fractional_diff, lw_bandwidth
from projekat.model.econometric.mem import _mu_path
from projekat.model.features import harq_features, harq_standardization, log1p_jump
from projekat.model.sequence.patchtst import early_stopping_split, jump_standardization
from projekat.model.targets import add_targets, target_columns
from projekat.model.walkforward import training_frame
from tests.conftest import make_panel


@pytest.fixture(scope="module")
def panel():
    return make_panel(symbols=("AAA",), end="2014-12-31")


# ── HARQ ─────────────────────────────────────────────────────────────────


def test_harq_interaction_is_standardized_sqrt_rq_times_ln_rv(panel):
    mean, sd = harq_standardization(panel)
    s = np.sqrt(panel["RQ"])
    assert mean == pytest.approx(s.mean()) and sd == pytest.approx(s.std(ddof=0))
    X = harq_features(panel, (mean, sd))
    expected = ((s - mean) / sd) * np.log(panel["RV_d"])
    assert np.allclose(X["sqrtRQ_z_x_lnRV_d"], expected)
    assert abs(X["sqrtRQ_z_x_lnRV_d"].mean()) < 10     # order of the base regressor, not 1e-8


def test_harq_standardization_is_frozen_at_the_refit(panel):
    df = add_targets(panel, (1,))
    train = training_frame(df, pd.Timestamp("2013-01-02"), (1,))
    fitted = fit_ols(harq_features, train, (1,), prepare=harq_standardization)
    assert fitted._params == pytest.approx(harq_standardization(train))
    # later rows are standardized with the TRAINING constants, not their own
    later = df[df["date"] >= pd.Timestamp("2014-01-02")]
    assert fitted._params != pytest.approx(harq_standardization(later))


# ── ARFIMA ───────────────────────────────────────────────────────────────


def test_local_whittle_bandwidth_is_n_to_the_065():
    assert lw_bandwidth(4000) == int(4000 ** 0.65) == 219     # the old n/4 rule gave 1000
    assert lw_bandwidth(1000) == int(1000 ** 0.65)


def test_reintegration_is_the_exact_inverse_of_the_differencing():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    u = _fractional_diff(x, 0.35)
    v = _binomial_weights(0.35, 500, inverse=True)
    rebuilt = np.array([np.dot(v[: t + 1], u[t::-1]) for t in range(500)])
    assert np.allclose(rebuilt, x, atol=1e-8)


def test_arfima_numerical_blow_up_becomes_nan_not_inf():
    """With a near-unit MA root the innovation filter can amplify over a
    long history and hand back a log forecast tens of orders of magnitude
    away from the data (measured: +55,848 -> exp() = inf). That is a
    numerical failure and must surface as NaN, never as inf."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(800) * 0.4 - 9
    exploding = _SymbolARFIMA(mu=-9.0, d=0.45, const=0.0, ar=np.array([0.9]), ma=np.array([-0.999999]))
    out = exploding.forecast(x, np.array([700, 799]), (1, 22))
    for h in (1, 22):
        assert not np.isfinite(out[h]).any() or np.nanmax(np.abs(out[h])) < abs(x.min()) + 20
        assert not np.isinf(out[h]).any()

    sane = _SymbolARFIMA(mu=-9.0, d=0.3, const=0.0, ar=np.array([0.3]), ma=np.array([0.1]))
    assert np.isfinite(sane.forecast(x, np.array([700]), (1,))[1]).all()


def test_arfima_forecast_uses_only_the_past():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(400)) * 0.05 - 9
    model = _SymbolARFIMA(mu=-9.0, d=0.3, const=0.0, ar=np.array([0.4]), ma=np.array([0.2]))
    a = model.forecast(x, np.array([200, 250]), (1, 5))
    x2 = x.copy()
    x2[251:] += 5.0
    b = model.forecast(x2, np.array([200, 250]), (1, 5))
    for h in (1, 5):
        assert np.allclose(a[h], b[h], rtol=1e-9)


# ── MEM ──────────────────────────────────────────────────────────────────


def test_mem_recursion_matches_the_explicit_loop():
    rng = np.random.default_rng(2)
    rv = rng.lognormal(-9, 0.4, 300)
    omega, alpha, beta, mu0 = 2e-6, 0.3, 0.6, float(rv.mean())
    mu = _mu_path(rv, omega, alpha, beta, mu0)
    ref = np.empty(300)
    ref[0] = mu0
    for t in range(1, 300):
        ref[t] = omega + alpha * rv[t - 1] + beta * ref[t - 1]
    assert np.allclose(mu, ref)


# ── PatchTST ─────────────────────────────────────────────────────────────


def test_patchtst_early_stopping_is_the_last_10pct_with_purged_targets(panel):
    df = add_targets(panel, (5,))
    train = training_frame(df, pd.Timestamp("2014-01-02"), (5,))
    fit_rows, stop_rows = early_stopping_split(train, 5, np.ones(len(train), bool))
    dates = np.sort(train["date"].unique())
    cut = dates[int(len(dates) * 0.9)]
    assert (train.loc[stop_rows, "date"] >= cut).all()
    assert stop_rows.sum() == pytest.approx(0.1 * len(dates), abs=6)
    _, _, end_col = target_columns(5)
    assert (train.loc[fit_rows, end_col] < cut).all()          # no fit target reaches into the stop set
    assert not (fit_rows & stop_rows).any()


def test_patchtst_jump_channel_uses_training_mean_and_sd(panel):
    mean, sd = jump_standardization(panel)
    j = log1p_jump(panel["J_d"])
    assert mean == pytest.approx(j.mean()) and sd == pytest.approx(j.std(ddof=0))
