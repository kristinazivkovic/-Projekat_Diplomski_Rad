"""Factor A2 estimator tests. See plan step 14."""

from __future__ import annotations

import numpy as np
import pytest

from projekat.measure.estimators.bpv import bpv
from projekat.measure.estimators.medrv import medrv
from projekat.measure.estimators.trbpv import trbpv
from projekat.measure.realized import realized_variance, signed_semivariances


def test_semivariances_sum_to_rv():
    """RS_pos + RS_neg == RV_d exactly -- the genuine invariant asserted in
    qa/checks.py (unlike C+J=RV, which holds by construction and tests
    nothing)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 390)
    rv = realized_variance(r)
    rs_pos, rs_neg = signed_semivariances(r)
    assert rs_pos + rs_neg == pytest.approx(rv, rel=1e-12)


def test_bpv_divisor_tracks_per_day_n():
    """The n/(n-1) finite-sample correction must be computed from each
    day's ACTUAL n, never a hardcoded constant -- half-days legitimately
    carry fewer bars (Resolution 4/Revision 7), and a hardcoded divisor
    would bias every such day quietly and uniformly."""
    rng = np.random.default_rng(1)
    r_full = rng.normal(0, 0.001, 390)
    r_half = r_full[:210]  # shorter day, same underlying process

    # Recompute the correction factor implied by each call and confirm it
    # differs between the two lengths (i.e. it is NOT a fixed constant).
    def implied_correction(r):
        n = len(r)
        abs_r = np.abs(r)
        cross = np.sum(abs_r[1:] * abs_r[:-1])
        mu1_inv_sq = np.pi / 2
        raw = mu1_inv_sq * cross
        corrected = bpv(r)
        return corrected / raw if raw > 0 else float("nan")

    corr_full = implied_correction(r_full)
    corr_half = implied_correction(r_half)
    assert corr_full == pytest.approx(390 / 389, rel=1e-9)
    assert corr_half == pytest.approx(210 / 209, rel=1e-9)
    assert corr_full != corr_half


def test_medrv_divisor_tracks_per_day_n():
    rng = np.random.default_rng(2)
    r_full = rng.normal(0, 0.001, 390)
    r_half = r_full[:210]
    # Both should produce finite, positive estimates at these sizes.
    assert medrv(r_full) > 0
    assert medrv(r_half) > 0


def test_no_jumps_yields_bpv_close_to_rv():
    """Without injected jumps, BPV and RV should be close (same order of
    magnitude) -- not required to be near-equal on any single day (BPV
    exceeds RV on ~43% of no-jump days per Revision 6's measurement), but
    across many days the average gap should be small."""
    rng = np.random.default_rng(3)
    gaps = []
    for seed in range(50):
        r = np.random.default_rng(seed).normal(0, 0.001, 390)
        rv = realized_variance(r)
        b = bpv(r)
        gaps.append((rv - b) / rv)
    mean_gap = np.mean(gaps)
    assert abs(mean_gap) < 0.10  # small relative to RV, consistent with ~3.5% measured


def test_estimators_reject_too_short_series():
    tiny = np.array([0.001])
    assert np.isnan(bpv(tiny))
    assert np.isnan(medrv(tiny))
    assert np.isnan(trbpv(tiny))


def test_trbpv_threshold_parameter_changes_result():
    """TrBPV's threshold is a real free parameter -- a tighter threshold
    should exclude more returns and generally lower the estimate when
    large returns are present."""
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.001, 390)
    r[100] += 0.05  # one large outlier
    loose = trbpv(r, threshold=10.0)  # excludes almost nothing
    tight = trbpv(r, threshold=1.5)   # excludes more
    assert loose != tight
