"""WI-7/WI-8: contamination check, the J-in-levels exception, and the
one-correction-on-the-combined-forecast rule.

These tests never call the real TTM checkpoint (no network in tests) --
they exercise the surrounding logic, which is where the invariants live."""

from __future__ import annotations

import json

import numpy as np
import pytest

from projekat.model.backtransform import (
    apply_combined_correction,
    combined_forecast_correction,
    residual_correction,
)
from projekat.model.foundation.contamination import (
    CONTAMINATION_FILENAME,
    check_and_write,
    check_contamination,
)


def test_contamination_detected_and_quantified_when_test_precedes_cutoff():
    report = check_contamination("2015-01-01", "2016-12-31")
    assert report.contaminated
    assert report.overlap_days > 0
    assert report.overlap_share_of_test == pytest.approx(1.0)
    assert "CONTAMINATED" in report.note
    # the model must be KEPT IN, clearly labelled -- never silently dropped
    assert "KEPT IN" in report.note


def test_no_contamination_when_test_starts_after_cutoff():
    report = check_contamination("2030-01-01", "2030-12-31")
    assert not report.contaminated
    assert report.overlap_days == 0
    assert "CLEAN" in report.note


def test_contamination_report_is_written_either_way(tmp_path):
    """The artefact must land in results/ regardless of outcome -- no ttm_*
    number is reported without it."""
    for start, end in [("2015-01-01", "2016-12-31"), ("2030-01-01", "2030-12-31")]:
        report = check_and_write(start, end, out_dir=tmp_path)
        written = json.loads((tmp_path / CONTAMINATION_FILENAME).read_text())
        assert written["contaminated"] == report.contaminated
        assert written["revision"]  # the pinned revision is recorded, not just "TTM"


def test_pinned_revision_is_specific_not_generic():
    from projekat.config import TTM_MODEL_ID, TTM_REVISION

    assert TTM_REVISION and TTM_REVISION != "main"
    assert "granite-timeseries-ttm" in TTM_MODEL_ID


def test_component_wise_correction_leaves_bias_but_combined_does_not():
    """The rule behind combined_forecast_correction: a correction estimated
    per component and then summed leaves the SUM with no controlled bias,
    while one correction estimated on the combined forecast removes it."""
    rng = np.random.default_rng(0)
    n = 20_000

    # two components with different multiplicative bias and different noise
    c_true = rng.lognormal(mean=-9.0, sigma=0.5, size=n)
    j_true = rng.lognormal(mean=-11.0, sigma=1.2, size=n)
    y_true = c_true + j_true

    # each component's raw forecast is biased in log space
    c_log_hat = np.log(c_true) - rng.normal(0.3, 0.5, n)
    j_log_hat = np.log(j_true) - rng.normal(0.8, 1.2, n)

    # WRONG: correct each component separately, then sum
    a_c = residual_correction(np.log(c_true), c_log_hat)
    a_j = residual_correction(np.log(j_true), j_log_hat)
    per_component_sum = np.exp(c_log_hat + a_c) + np.exp(j_log_hat + a_j)

    # RIGHT: sum first, then one correction on the combined forecast
    raw_combined = np.exp(c_log_hat) + np.exp(j_log_hat)
    m = combined_forecast_correction(y_true, raw_combined)
    combined_corrected = apply_combined_correction(raw_combined, m)

    bias_per_component = np.mean(y_true / per_component_sum) - 1.0
    bias_combined = np.mean(y_true / combined_corrected) - 1.0

    # the combined correction is unbiased in the ratio sense by construction;
    # the component-wise route is measurably not
    assert abs(bias_combined) < 1e-9
    assert abs(bias_per_component) > 1e-3
    assert abs(bias_combined) < abs(bias_per_component)


def test_j_is_forecast_in_levels_not_logs():
    """The documented exception: J = 0 on most days, so it must never be
    pushed through a log transform. ttm_c_plus_j asks _ttm_base for J with
    log_space=False -- assert that contract holds on a series containing
    real zeros, which log_space=True could not represent."""
    from projekat.model.foundation import _ttm_base

    calls = {}

    def fake_ttm_forecast_series(series, horizon):
        calls["series"] = series
        return float(np.mean(series))

    original = _ttm_base.ttm_forecast_series
    _ttm_base.ttm_forecast_series = fake_ttm_forecast_series
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=10),
                "symbol": ["AAA"] * 10,
                "J_d": [0.0, 0.0, 1e-5, 0.0, 0.0, 2e-5, 0.0, 0.0, 0.0, 3e-5],
            }
        )
        out = _ttm_base.forecast_per_symbol(df, "J_d", 1, log_space=False)
    finally:
        _ttm_base.ttm_forecast_series = original

    # zeros survive untouched -- proof no log/clip was applied
    assert (calls["series"] == 0.0).sum() == 7
    assert out["AAA"] == pytest.approx(df["J_d"].mean())


def test_hybrid_weights_are_fixed_half_and_never_optimized():
    from projekat.model.foundation import log_har_ttm as mod

    assert mod._W_HAR == 0.5
    assert mod._W_TTM == 0.5
