"""VaR backtests at the degenerate breach counts that do occur on real data
(first e2e run: 9 of 492 cells had zero breaches in ~990 days).

Regression: Kupiec returned LR=0, p=1 for zero breaches -- reporting an
over-conservative VaR as a perfect one -- and DQ returned a negative F from
a rank-deficient regression."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from projekat.evaluation.var import dynamic_quantile_test, kupiec_test


def test_kupiec_zero_breaches_rejects_over_conservative_var():
    n, level = 1000, 0.01
    res = kupiec_test(np.zeros(n, dtype=int), level=level)
    expected_lr = -2 * n * np.log(1 - level)
    assert res.lr_statistic == pytest.approx(expected_lr)
    assert res.p_value == pytest.approx(1 - stats.chi2.cdf(expected_lr, df=1))
    assert res.p_value < 1e-4


def test_kupiec_exact_rate_does_not_reject():
    b = np.zeros(1000, dtype=int)
    b[::100] = 1  # exactly 1%
    res = kupiec_test(b, level=0.01)
    assert res.lr_statistic == pytest.approx(0.0, abs=1e-9)
    assert res.p_value == pytest.approx(1.0)


def test_kupiec_matches_textbook_formula_in_the_interior():
    b = np.zeros(1000, dtype=int)
    b[:25] = 1
    x, n, p = 25, 1000, 0.01
    lr = -2 * ((x * np.log(p) + (n - x) * np.log(1 - p))
               - (x * np.log(x / n) + (n - x) * np.log(1 - x / n)))
    assert kupiec_test(b, level=p).lr_statistic == pytest.approx(lr)


def test_dq_is_undefined_without_breach_variation():
    rng = np.random.default_rng(0)
    res = dynamic_quantile_test(np.zeros(500, dtype=int), rng.uniform(0.01, 0.03, 500), level=0.01)
    assert np.isnan(res.f_statistic) and np.isnan(res.p_value)


def test_dq_is_finite_and_non_negative_with_breaches():
    rng = np.random.default_rng(1)
    b = (rng.uniform(size=1000) < 0.02).astype(int)
    res = dynamic_quantile_test(b, rng.uniform(0.01, 0.03, 1000), level=0.01)
    assert np.isfinite(res.f_statistic) and res.f_statistic >= 0
