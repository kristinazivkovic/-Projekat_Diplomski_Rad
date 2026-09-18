"""WI-11 (levels 3 & 4) and WI-12 (diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.evaluation.diagnostics import (
    SIGNATURE_PLOT_MINUTES,
    h5_placebo_panel,
    signature_plot,
)
from projekat.evaluation.stratify import tag_jump_day, tag_regime
from projekat.evaluation.var import var_backtest_table
from projekat.qa.checks import (
    cohens_kappa,
    cohens_kappa_matrix,
    jaccard_index,
    jump_component_diagnostics,
)


# ── WI-11: VaR (level 3) ─────────────────────────────────────────────────


def _var_results(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model in ["log_har", "char"]:
        for window_id in range(3):
            for d in pd.bdate_range("2020-01-01", periods=n // 3):
                rv = float(rng.lognormal(-9.5, 0.4))
                rows.append(
                    {
                        "model": model,
                        "design_id": "5m__bpv__bns",
                        "horizon": 1,
                        "symbol": "AAA",
                        "window_id": window_id,
                        "date": d,
                        "y_hat": rv,
                        "r_d": float(rng.normal(0, np.sqrt(rv))),
                    }
                )
    return pd.DataFrame(rows)


def test_var_backtest_is_pooled_across_windows_not_per_window():
    """A 252-day window expects ~2.5 breaches at 1% and leaves Kupiec no
    power -- the table must pool every window into one test per
    (model, design, horizon)."""
    df = _var_results()
    out = var_backtest_table(df)

    assert "window_id" not in out.columns
    assert out["pooled_over_windows"].all()
    # one row per (model, design, horizon) -- not one per window
    assert len(out) == 2
    # each row's n_obs spans all three windows
    assert (out["n_obs"] > 500).all()


def test_var_backtest_reports_exact_breach_counts_next_to_p_values():
    out = var_backtest_table(_var_results())
    for col in ["n_breaches", "expected_breaches", "breach_rate", "kupiec_p", "christoffersen_p", "dq_p"]:
        assert col in out.columns
    assert out["n_breaches"].dtype.kind in "iu"          # exact counts, not rates only
    assert (out["n_breaches"] <= out["n_obs"]).all()


# ── WI-11: level 4 jump characterization ─────────────────────────────────


def test_cohens_kappa_corrects_for_chance_where_jaccard_does_not():
    """Two independent flaggers with the same marginal rate still earn a
    non-trivial Jaccard; kappa must sit near zero because they agree no
    more than chance predicts."""
    rng = np.random.default_rng(0)
    a = pd.Series(rng.random(4000) < 0.05)
    b = pd.Series(rng.random(4000) < 0.05)

    assert jaccard_index(a, b) > 0.0
    assert abs(cohens_kappa(a, b)) < 0.05      # chance-corrected ~ 0


def test_cohens_kappa_is_one_for_identical_non_degenerate_flags():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.random(500) < 0.2)
    assert cohens_kappa(a, a) == pytest.approx(1.0)


def test_cohens_kappa_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(2)
    variants = {name: pd.Series(rng.random(400) < 0.1) for name in ["naive", "bns", "lm"]}
    m = cohens_kappa_matrix(variants)
    assert list(m.index) == list(m.columns) == ["naive", "bns", "lm"]
    for name in variants:
        assert m.loc[name, name] == pytest.approx(1.0)
    assert m.loc["naive", "bns"] == pytest.approx(m.loc["bns", "naive"])


def test_jump_diagnostics_separate_occurrence_from_size():
    """A near-zero forecaster (the expected ttm_j behaviour) must score no
    skill against the zero reference and yield no jump-size rank
    correlation -- MSE alone would make it look fine."""
    rng = np.random.default_rng(0)
    j_true = np.where(rng.random(800) < 0.1, rng.lognormal(-8, 1, 800), 0.0)

    near_zero = jump_component_diagnostics(j_true, np.full(800, 1e-12))
    assert near_zero["skill_vs_zero"] == pytest.approx(0.0, abs=1e-6)
    assert near_zero["n_jump_days"] > 0

    # a forecaster that knows the size on jump days must score a real
    # rank correlation on exactly those days
    informed = jump_component_diagnostics(j_true, j_true * 0.8)
    assert informed["rank_corr_on_jump_days"] == pytest.approx(1.0)
    assert informed["skill_vs_zero"] > 0.5


def test_jump_diagnostics_report_clipping_activation_rate():
    j_true = np.array([0.0, 1e-8, 0.0, 2e-8])
    out = jump_component_diagnostics(j_true, np.array([-1.0, 1e-8, -2.0, 2e-8]))
    assert out["clipping_rate"] == pytest.approx(0.5)


def test_no_precision_recall_is_exposed():
    """There is no ground truth for a 'real' jump, so classification
    metrics must not exist in the Level-4 surface."""
    from projekat.qa import checks

    exported = dir(checks)
    for forbidden in ["precision", "recall", "f1_score", "roc_auc"]:
        assert not any(forbidden in name.lower() for name in exported)


# ── WI-11: stratification ────────────────────────────────────────────────


def test_regime_tagging_uses_an_explicit_fixed_boundary():
    df = pd.DataFrame({"RV_m": [0.1, 1.0, 5.0], "jump_flag": [True, False, True]})
    regimes = tag_regime(df, threshold=1.0)
    assert list(regimes) == ["calm", "calm", "turbulent"]
    assert list(tag_jump_day(df)) == ["jump", "no_jump", "jump"]


# ── WI-12: diagnostics ───────────────────────────────────────────────────


def test_signature_plot_covers_the_declared_frequencies():
    """Two A1 levels cannot show an interior optimum -- the signature plot
    must span the full declared minute grid."""
    rng = np.random.default_rng(0)
    by_grid = {
        f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 1e-4 for i in range(15)}
        for m in SIGNATURE_PLOT_MINUTES
    }
    out = signature_plot(by_grid)
    assert list(out["minutes"]) == list(SIGNATURE_PLOT_MINUTES)
    for col in ["mean_rv", "mean_bpv", "rv_minus_bpv"]:
        assert col in out.columns
    assert out["rv_minus_bpv"].notna().all()


def test_h5_placebo_preserves_marginal_jump_frequency_but_breaks_alignment():
    rng = np.random.default_rng(0)
    n = 300
    rows = []
    for sym in ["AAA", "BBB"]:
        rv = np.abs(rng.standard_normal(n)) * 1e-4 + 1e-5
        flags = rng.random(n) < 0.15
        for d, r, f in zip(pd.bdate_range("2020-01-01", periods=n), rv, flags):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "design_id": "5m__bpv__bns",
                    "RV_d": r,
                    "J_d": r * 0.1 if f else 0.0,
                    "C_d": r * 0.9 if f else r,
                    "jump_flag": bool(f),
                }
            )
    panel = pd.DataFrame(rows)
    placebo = h5_placebo_panel(panel, seed=0)

    for sym in ["AAA", "BBB"]:
        real = panel[panel["symbol"] == sym]
        fake = placebo[placebo["symbol"] == sym]
        # marginal frequency preserved exactly
        assert real["jump_flag"].sum() == fake["jump_flag"].sum()
        # alignment to specific days destroyed
        merged = real.merge(fake, on=["date", "symbol"], suffixes=("_r", "_f"))
        assert not merged["jump_flag_r"].equals(merged["jump_flag_f"])

    # C + J == RV must still hold after permutation, or the comparison
    # would measure decomposition inconsistency instead of lost jump info
    assert np.allclose(placebo["C_d"] + placebo["J_d"], placebo["RV_d"])
