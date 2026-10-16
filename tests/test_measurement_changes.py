"""Measurement-layer changes: ln(1+J), per-estimator BNS theta, the pinned
significance levels, the vectorized LM local sigma and its cache, and the
panel diagnostics CSV."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from projekat.config import BNS_ALPHA, LM_CONFIDENCE, LM_FDR_Q
from projekat.measure import estimators, jumps  # noqa: F401  (registration)
from projekat.measure.design import MeasurementDesign
from projekat.measure.estimators.bpv import bpv
from projekat.measure.jumps._gumbel import critical_value, exceedance_p_value, gumbel_c_and_s, local_sigma_series
from projekat.measure.jumps.bns import bns, theta_for
from projekat.measure.jumps.lee_mykland import lee_mykland
from projekat.measure.panel import _lm_day_stats, build_symbol_design
from projekat.measure.realized import realized_variance
from projekat.measure.registry import ESTIMATORS, JUMP_TESTS
from projekat.model.features import build_sequences, har_j_features, union_features
from tests.conftest import make_day_returns

# ── A1: ln(1 + J) ────────────────────────────────────────────────────────


def _feature_frame(j):
    n = len(j)
    rv = np.full(n, 1e-4)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n), "symbol": "AAA",
        "RV_d": rv, "RV_w": rv, "RV_m": rv, "C_d": rv - j, "C_w": rv, "C_m": rv,
        "J_d": j, "RS_pos": rv / 2, "RS_neg": rv / 2, "RQ": rv**2, "vix_lag1": 18.0, "r_d": 0.001,
    })


def test_jump_regressor_is_log1p_and_zero_on_non_jump_days():
    """J = 0 must map to exactly 0, not ln(1e-12) = -27.6 -- otherwise the
    J coefficient mixes jump occurrence with jump size and depends on eps."""
    j = np.array([0.0, 0.0, 2e-5, 0.0, 5e-4])
    df = _feature_frame(j)
    for frame in (har_j_features(df), union_features(df)):
        np.testing.assert_allclose(frame["J_d_lag1"].to_numpy(), np.log1p(j))
        assert (frame.loc[j == 0, "J_d_lag1"] == 0.0).all()


def test_patchtst_jump_channel_is_log1p():
    j = np.where(np.arange(70) % 7 == 0, 3e-5, 0.0)
    df = _feature_frame(j)
    seqs, origin = build_sequences(df, seq_len=60)
    assert len(seqs) == 11
    # channel order: RV_d, C_d, J_d, RS_pos, RS_neg, r_d
    np.testing.assert_allclose(seqs[0][:, 2], np.log1p(j[:60]))
    assert seqs[:, :, 2].min() >= 0.0


# ── A2: BNS theta per estimator ──────────────────────────────────────────


def test_every_registered_estimator_declares_its_bns_theta():
    assert set(ESTIMATORS) == {"bpv", "medrv", "trbpv"}
    assert theta_for("bpv") == pytest.approx((math.pi / 2) ** 2 + math.pi - 5)
    assert theta_for("trbpv") == pytest.approx(0.6090, abs=1e-4)
    assert theta_for("medrv") == pytest.approx(0.96)


def test_unknown_estimator_is_an_error_not_a_silent_bpv_default():
    with pytest.raises(ValueError, match="bns_theta"):
        theta_for("no_such_estimator")


def test_bns_statistic_scales_with_the_estimator_theta():
    """Same returns and same robust value: only theta differs, so the
    statistic ratio is exactly sqrt(theta_bpv / theta_medrv)."""
    r = make_day_returns(390, seed=3, jump_at=100, jump_size=0.004)
    rv, robust = realized_variance(r), bpv(r)
    z_bpv = bns(r, rv, robust, estimator="bpv").statistic
    z_medrv = bns(r, rv, robust, estimator="medrv").statistic
    assert z_medrv / z_bpv == pytest.approx(math.sqrt(theta_for("bpv") / 0.96))


def test_panel_passes_the_design_estimator_to_the_jump_test(monkeypatch):
    """panel.py resolves the test by name and hands it the design's
    estimator name -- the only way bns can pick the right theta."""
    seen = []
    real = JUMP_TESTS["bns"]

    def spy(r, rv, robust, **kwargs):
        seen.append(kwargs.get("estimator"))
        return real(r, rv, robust, **kwargs)

    spy.name = "bns"
    monkeypatch.setitem(JUMP_TESTS, "bns", spy)
    returns = {f"2020-01-{d:02d}": make_day_returns(78, seed=d) for d in range(2, 30)}
    build_symbol_design("AAA", MeasurementDesign("5m", "medrv", "bns"), returns, {})
    assert seen and set(seen) == {"medrv"}


# ── A3: significance levels ──────────────────────────────────────────────


def test_pinned_significance_levels():
    assert BNS_ALPHA == 0.001
    assert LM_CONFIDENCE == 0.999
    assert LM_FDR_Q == 0.05


def test_bns_is_one_sided_at_alpha_0001():
    """p = 1 - Phi(Z) and significant iff p < 0.001: a day with p between
    0.001 and 0.05 (significant under the old 5% level) is NOT a jump."""
    r_base = make_day_returns(390, seed=0)
    found_between = False
    for size in np.linspace(0.0, 0.01, 60):
        r = r_base.copy()
        r[200] += size
        v = bns(r, realized_variance(r), bpv(r), estimator="bpv")
        assert v.p_value == pytest.approx(1 - __import__("scipy").stats.norm.cdf(v.statistic))
        assert v.significant == (v.p_value < 0.001)
        if 0.001 < v.p_value < 0.05:
            found_between = True
            assert not v.significant
    assert found_between, "no jump size landed between the old and new level"


def test_lm_critical_value_defaults_to_0999_and_matches_its_p_value():
    for n in (26, 78, 390):
        c_n, s_n = gumbel_c_and_s(n)
        expected = c_n + s_n * (-math.log(-math.log(0.999)))
        assert critical_value(n) == pytest.approx(expected)
        # the p-value and the critical value come from one distribution
        assert exceedance_p_value(critical_value(n), n) == pytest.approx(0.001, rel=1e-9)


def test_lee_mykland_significance_is_p_below_0001():
    for seed in range(5):
        r = make_day_returns(78, seed=seed, jump_at=40, jump_size=0.004 * seed)
        v = lee_mykland(r, realized_variance(r), bpv(r))
        assert v.significant == (v.p_value < 1 - LM_CONFIDENCE)


def test_lm_fdr_day_p_value_is_the_gumbel_formula():
    rng = np.random.default_rng(5)
    returns = {f"d{i:03d}": rng.standard_normal(78) * 1e-3 for i in range(10)}
    stats, _sigma = _lm_day_stats(sorted(returns), returns, "5m")
    assert stats
    for s in stats.values():
        c_n, s_n = gumbel_c_and_s(s.n)
        assert s.p_value == pytest.approx(1 - math.exp(-math.exp(-(s.max_l - c_n) / s_n)))


# ── vectorized local sigma and the LM cache ──────────────────────────────


def _local_sigma_loop(r, k):
    n = len(r)
    sigma = np.full(n, np.nan)
    abs_r = np.abs(r)
    for i in range(k, n):
        window = abs_r[i - k: i]
        sigma[i] = math.sqrt(np.sum(window[1:] * window[:-1]) / (k - 1))
    return sigma


def test_vectorized_local_sigma_equals_the_reference_loop():
    r = np.random.default_rng(0).standard_normal(3000) * 1e-3
    for k in (2, 26, 312):
        np.testing.assert_allclose(local_sigma_series(r, k), _local_sigma_loop(r, k), rtol=1e-9, equal_nan=True)


def test_lm_cache_gives_identical_panels():
    returns = {f"2020-{m:02d}-{d:02d}": make_day_returns(78, seed=m * 40 + d)
               for m in (1, 2, 3) for d in range(1, 28)}
    for test in ("lm", "lm_fdr"):
        design = MeasurementDesign("5m", "trbpv", test)
        plain = build_symbol_design("AAA", design, returns, {})
        cache: dict = {}
        build_symbol_design("AAA", MeasurementDesign("5m", "bpv", test), returns, {}, lm_cache=cache)
        cached = build_symbol_design("AAA", design, returns, {}, lm_cache=cache)
        pd.testing.assert_frame_equal(plain, cached)


# ── A5: panel diagnostics ────────────────────────────────────────────────


def test_panel_diagnostics_sections_and_values():
    from projekat.qa.panel_diagnostics import build_panel_diagnostics

    dates = pd.bdate_range("2020-01-01", periods=10)
    rows = []
    flags = {"bns": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], "lm": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    for grid in ("1m", "5m"):
        for test, f in flags.items():
            for sym in ("AAA", "BBB"):
                sym_dates = dates if not (sym == "BBB" and grid == "1m") else dates[:-1]
                for d, flag in zip(sym_dates, f):
                    rows.append({"design_id": f"{grid}__bpv__{test}", "symbol": sym, "date": d, "jump_flag": bool(flag)})
    panel = pd.DataFrame(rows)
    exclusions = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB"], "date": ["x", "y", "z"],
                               "reason": ["half_day", "half_day", "insufficient_1m_coverage"]})
    corporate = pd.DataFrame({"symbol": ["AAA", "BBB"], "status": ["checked", "skipped_fetch_failed"], "error": ["", "net"]})

    out = build_panel_diagnostics(panel, exclusions, corporate)
    assert set(out["section"]) == {"jump_day_share", "jaccard", "days_per_grid",
                                   "days_identical_across_grids", "corporate_checks_skipped", "excluded_days"}
    assert "RV_d" not in panel.columns   # the quantization sections need RV_d; see the test below

    share = out[(out["section"] == "jump_day_share") & (out["grid"] == "5m") & (out["jump_test"] == "bns")]
    assert share["value"].iloc[0] == pytest.approx(0.2)

    jac = out[(out["section"] == "jaccard") & (out["grid"] == "5m")]
    assert jac["value"].iloc[0] == pytest.approx(1 / 3)    # {d0,d3} vs {d0,d1}

    ident = out[out["section"] == "days_identical_across_grids"].set_index("symbol")["value"]
    assert ident["AAA"] == 1 and ident["BBB"] == 0 and ident["__all__"] == 0

    skipped = out[out["section"] == "corporate_checks_skipped"]
    assert skipped["value"].iloc[0] == 1 and skipped["detail"].iloc[0] == "BBB"

    excl = out[(out["section"] == "excluded_days") & (out["symbol"] == "__all__")].set_index("reason")["value"]
    assert excl["half_day"] == 2 and excl["insufficient_1m_coverage"] == 1


def test_panel_diagnostics_flag_price_quantization():
    """A symbol whose back-adjusted prices are quantized (NVDA 2014-2016 at
    ~0.46 USD, ~2.2% tick) shows zero-RV days and an RV(1m)/RV(5m) ratio far
    above the ordinary ~1.1-1.2. Both must be visible in the diagnostics --
    that symbol otherwise carries almost the whole QLIKE mass downstream."""
    from projekat.qa.panel_diagnostics import build_panel_diagnostics

    dates = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(0)
    rows = []
    for symbol, ratio, zeros in (("GOOD", 1.15, 0), ("QUANT", 2.4, 6)):
        rv5 = rng.lognormal(-9.5, 0.4, len(dates))
        for i, d in enumerate(dates):
            rv5_i = 0.0 if i < zeros and symbol == "QUANT" else rv5[i]
            for grid, value in (("5m", rv5_i), ("1m", rv5_i * ratio)):
                rows.append({"design_id": f"{grid}__bpv__bns", "symbol": symbol, "date": d,
                             "jump_flag": False, "RV_d": value})
    out = build_panel_diagnostics(pd.DataFrame(rows), None, None)

    zero_days = out[out["section"] == "zero_rv_days"].set_index(["symbol", "grid"])["value"]
    assert zero_days[("QUANT", "5m")] == 6 and ("GOOD", "5m") not in zero_days.index
    ratios = out[out["section"] == "rv_ratio_1m_over_5m"].set_index("symbol")["value"]
    assert ratios["GOOD"] == pytest.approx(1.15) and ratios["QUANT"] == pytest.approx(2.4)
