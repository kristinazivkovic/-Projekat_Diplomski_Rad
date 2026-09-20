"""TTM models: contamination check in trading days, the daily 512-day
context, J in levels, one ratio-of-sums correction on the combined
forecast, fixed hybrid weights. Never calls the real checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from projekat.model.foundation import _ttm_base
from projekat.model.foundation.contamination import CONTAMINATION_FILENAME, check_and_write, check_contamination


def test_contamination_is_counted_in_trading_days():
    report = check_contamination("2015-01-01", "2025-12-31")
    assert report.contaminated
    assert report.day_unit == "XNYS trading sessions"
    # 2015..2023 before the 2024-01-01 cutoff, 2024..2025 clean
    assert 2250 < report.overlap_days < 2280           # ~252 x 9, not 3,287 calendar days
    assert 2750 < report.test_days < 2780
    assert report.overlap_end == "2023-12-29"
    assert report.overlap_share_of_test == pytest.approx(report.overlap_days / report.test_days)
    assert "KEPT IN" in report.note


def test_no_contamination_after_the_cutoff_and_written_either_way(tmp_path):
    clean = check_contamination("2024-01-01", "2025-12-31")
    assert not clean.contaminated and clean.overlap_days == 0 and clean.test_days > 490
    for start, end in [("2015-01-01", "2016-12-31"), ("2024-01-01", "2024-12-31")]:
        report = check_and_write(start, end, out_dir=tmp_path)
        written = json.loads((tmp_path / CONTAMINATION_FILENAME).read_text())
        assert written["contaminated"] == report.contaminated
        assert written["revision"]


def test_pinned_revision_is_specific_not_generic():
    from projekat.config import TTM_CONTEXT_LENGTH, TTM_MODEL_ID, TTM_REVISION

    assert TTM_REVISION == "512-192-r2" and TTM_CONTEXT_LENGTH == 512
    assert "granite-timeseries-ttm" in TTM_MODEL_ID


def _history():
    dates = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame({
        "symbol": ["AAA"] * 600 + ["BBB"] * 600,
        "date": list(dates) * 2,
        "RV_d": np.r_[np.arange(1, 601) * 1e-6, np.arange(1, 601) * 2e-6],
        "J_d": np.r_[np.where(np.arange(600) % 10 == 0, 1e-6, 0.0), np.zeros(600)],
        "C_d": np.r_[np.arange(1, 601) * 1e-6, np.arange(1, 601) * 2e-6],
    })


def test_context_ends_at_the_origin_and_never_crosses_symbols():
    h = _history()
    origins = np.array([599, 600 + 10])            # AAA's last day, BBB's 11th day
    ctx, ok = _ttm_base.contexts_at(h, "RV_d", origins, log_space=False)
    assert ok.all() and ctx.shape == (2, 512)
    assert ctx[0, -1] == pytest.approx(600e-6)    # ends exactly at the origin day
    assert ctx[0, 0] == pytest.approx(89e-6)      # 512 days back: rows 88..599
    assert ctx[1, -1] == pytest.approx(11 * 2e-6)
    assert np.allclose(ctx[1, :-11], 2e-6)         # left-padded with BBB's own first value


def test_horizon_forecast_is_the_mean_of_the_first_h_steps(monkeypatch):
    monkeypatch.setattr(_ttm_base, "ttm_forecast_batch", lambda c: np.tile(np.arange(1.0, 193.0), (len(c), 1)))
    out = _ttm_base.forecast_series_at(_history(), "RV_d", np.array([500]), (1, 5, 22), log_space=True)
    assert out[1][0] == 1.0 and out[5][0] == 3.0 and out[22][0] == 11.5


def test_j_is_forecast_in_levels_and_summed_with_c_in_levels(monkeypatch):
    seen = []

    def fake(contexts):
        seen.append(contexts.copy())
        return np.tile(contexts[:, -1:], (1, 192))

    monkeypatch.setattr(_ttm_base, "ttm_forecast_batch", fake)
    from projekat.model.foundation.ttm_c_plus_j import _predict_c_plus_j

    h = _history()
    out = _predict_c_plus_j(h, np.array([590]), (1,))
    c_ctx, j_ctx = seen
    assert (j_ctx == 0.0).sum() > 400                # zeros survive: no log/clip applied to J
    assert out[1][0] == pytest.approx(np.exp(c_ctx[0, -1]) + max(j_ctx[0, -1], 0.0))


def test_hybrid_weights_are_fixed_half_and_never_optimized():
    from projekat.model.foundation import log_har_ttm as mod

    assert mod._W_HAR == 0.5 and mod._W_TTM == 0.5
    assert mod.log_har_ttm.correction == "ratio_of_sums"
