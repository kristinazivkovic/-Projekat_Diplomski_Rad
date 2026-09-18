"""Tests for measure/jumps/_gumbel.py -- the closed-form Gumbel critical
value for max|L| (Revision 9). See the module docstring for why the
Revision 8 "simulated table" comparison was itself flawed (it compared
max|Z| quantiles against a statistic that is actually Z/mu_1): this test
file was written against Revision 8's now-superseded SIMULATED_CRITICAL_VALUES
table and has been updated to match the corrected, documented Revision 9
closed-form behavior instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from projekat.measure.jumps._gumbel import (
    critical_value,
    exceedance_p_value,
    gumbel_c_and_s,
)

TABULATED_N = [14, 26, 42, 78, 210, 390]


def _closed_form_expected(n: int, confidence: float = 0.95) -> float:
    beta_star = -math.log(-math.log(confidence))
    c_n, s_n = gumbel_c_and_s(n)
    return c_n + s_n * beta_star


def test_critical_value_matches_closed_form():
    """critical_value() is exactly the closed-form Gumbel formula -- no
    simulated table backs it (Revision 9)."""
    for n in TABULATED_N:
        assert critical_value(n) == pytest.approx(_closed_form_expected(n), abs=1e-9)


def test_critical_value_rises_with_n():
    """Larger n -> larger critical value (more intraday tests means the
    max of more standard normals is expected to be larger)."""
    values = [critical_value(n) for n in TABULATED_N]
    assert values == sorted(values)
    assert values[-1] > values[-2] > values[-3] > values[-4] > values[-5] > values[0]


@pytest.mark.parametrize("n", TABULATED_N)
def test_simulated_size_is_near_nominal(n):
    """Regression test for the Revision 9 fix: simulating the FULL
    procedure (independent Gaussian path, local_sigma_series's own
    trailing-window estimator with a large fixed K=270 warm-up window --
    matching the module docstring's own verification setup, since a
    K scaled down to n gives a poor sigma estimate and skews the size)
    against the closed-form critical value must give an empirical
    exceedance rate near the nominal 5% -- not the 0.01-0.08% a naive
    max|Z| threshold comparison would give if the 1/mu_1 rescaling were
    dropped."""
    from projekat.measure.jumps._gumbel import local_sigma_series

    crit = critical_value(n)
    k = 270
    rng = np.random.default_rng(999 + n)
    trials = 2_000
    exceed = 0
    for _ in range(trials):
        r = rng.standard_normal(k + n)
        sigma = local_sigma_series(r, k)
        tested = np.abs(r[k:]) / sigma[k:]  # exactly n indices, all with full K-history
        max_l = np.max(tested)
        if max_l > crit:
            exceed += 1
    size = exceed / trials
    assert 0.02 < size < 0.10, f"n={n}: simulated size {size:.4f} not near nominal 5%"


def test_fallback_for_untabulated_n_is_reasonable():
    """An n not among any special-cased set (e.g. a ragged day) must still
    produce a sensible, monotonically-consistent critical value via the
    closed-form formula."""
    crit_100 = critical_value(100)
    assert critical_value(78) < crit_100 < critical_value(210)


def test_exceedance_p_value_consistent_with_critical_value():
    """A statistic exactly at the critical value should have a p-value
    near (but not exactly) the nominal 5% level; well below it, p should
    be much larger; well above it, p should be much smaller."""
    n = 78
    crit = critical_value(n)
    p_at_crit = exceedance_p_value(crit, n)
    p_low = exceedance_p_value(1.0, n)
    p_high = exceedance_p_value(crit + 2.0, n)
    assert p_low > p_at_crit > p_high
