"""Levels 3 & 4 and the WI-12 diagnostics: VaR against the next day's
return without crossing symbols, regime labels from one grid, the H5
placebo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.evaluation.diagnostics import SIGNATURE_PLOT_MINUTES, h5_placebo_panel, h5_placebo_test, signature_plot
from projekat.evaluation.stratify import compute_regime_labels, tag_jump_day, tag_regime
from projekat.evaluation.var import christoffersen_test, dynamic_quantile_test, var_backtest_table
from projekat.qa.checks import cohens_kappa, cohens_kappa_matrix, jaccard_index, jump_component_diagnostics
from tests.conftest import make_panel

# ── level 3: VaR ─────────────────────────────────────────────────────────


def _var_results(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model in ["log_har", "char"]:
        for symbol in ["AAA", "BBB"]:
            for horizon in (1, 5):
                for d in pd.bdate_range("2020-01-01", periods=n):
                    rv = float(rng.lognormal(-9.5, 0.4))
                    rows.append({"model": model, "design_id": "5m__bpv__bns", "horizon": horizon,
                                 "symbol": symbol, "date": d, "y_hat": rv,
                                 "r_d": 0.0, "r_d_next": float(rng.normal(0, np.sqrt(rv)))})
    return pd.DataFrame(rows)


def test_var_is_h1_only_pooled_and_uses_the_next_days_return():
    out = var_backtest_table(_var_results())
    assert set(out["horizon"]) == {1}
    assert len(out) == 2                                    # one row per (model, design)
    assert (out["n_obs"] == 800).all() and (out["n_symbols"] == 2).all()
    # r_d (the origin day's own return) is all zero here: a test against it
    # would show no breaches at all; r_d_next shows the ~1% it should
    assert (out["n_breaches"] > 0).all()


def test_var_backtest_reports_exact_breach_counts_next_to_p_values():
    out = var_backtest_table(_var_results())
    for col in ["n_breaches", "expected_breaches", "breach_rate", "kupiec_p", "christoffersen_p", "dq_p", "lopez_loss"]:
        assert col in out.columns
    assert out["n_breaches"].dtype.kind in "iu"


def test_christoffersen_never_counts_a_transition_across_symbols():
    """AAA ends with a breach and BBB starts with one: pooled naively that
    is a (1 -> 1) transition; within symbols it does not exist."""
    b_aaa = np.array([0, 0, 0, 0, 1])
    b_bbb = np.array([1, 0, 0, 0, 0])
    pooled = np.r_[b_aaa, b_bbb]
    groups = np.array(["AAA"] * 5 + ["BBB"] * 5)
    naive = christoffersen_test(pooled)
    within = christoffersen_test(pooled, groups=groups)
    assert naive.lr_statistic != pytest.approx(within.lr_statistic)
    separate = [christoffersen_test(b) for b in (b_aaa, b_bbb)]
    assert within.lr_statistic >= 0 and all(np.isfinite(s.lr_statistic) for s in separate)


def test_dq_builds_lags_within_each_symbol():
    rng = np.random.default_rng(3)
    b = (rng.uniform(size=600) < 0.05).astype(int)
    v = rng.uniform(0.01, 0.03, 600)
    groups = np.repeat(["AAA", "BBB", "CCC"], 200)
    res = dynamic_quantile_test(b, v, level=0.01, groups=groups)
    assert np.isfinite(res.f_statistic)
    # 3 symbols x (200 - 4 lags) rows, not 600 - 4
    single = dynamic_quantile_test(b, v, level=0.01)
    assert res.f_statistic != pytest.approx(single.f_statistic)


# ── strata: regime labels ────────────────────────────────────────────────


def test_regime_labels_come_from_5m_rv_m_and_are_identical_across_designs():
    panel = make_panel(design_ids=("1m__bpv__bns", "5m__bpv__bns", "5m__medrv__lm"), end="2014-12-31")
    # make 1m RV_m larger, as microstructure noise does
    one_m = panel["design_id"] == "1m__bpv__bns"
    panel.loc[one_m, "RV_m"] = panel.loc[one_m, "RV_m"] * 3.0
    labels, provenance = compute_regime_labels(panel)
    assert provenance["grid"] == "5m" and provenance["quantile"] == 0.5
    assert provenance["date_max"] < "2014-01-01"            # first training period only
    assert labels.duplicated(["symbol", "date"]).sum() == 0  # ONE label per (symbol, date)
    five = panel[panel["design_id"] == "5m__bpv__bns"].drop_duplicates(["symbol", "date"])
    first = five[five["date"] < pd.Timestamp("2014-01-01")]["RV_m"].dropna()
    assert provenance["threshold"] == pytest.approx(first.median())
    # the 1m inflation changes nothing: labels are not a function of the grid
    merged = labels.merge(five[["symbol", "date", "RV_m"]], on=["symbol", "date"])
    ok = merged["RV_m"].notna()
    expected = np.where(merged.loc[ok, "RV_m"] > provenance["threshold"], "turbulent", "calm")
    assert (merged.loc[ok, "regime"].to_numpy() == expected).all()


def test_regime_needs_the_5m_grid():
    with pytest.raises(ValueError, match="5m"):
        compute_regime_labels(make_panel(design_ids=("1m__bpv__bns",), end="2014-06-30"))


def test_regime_and_jump_day_tags():
    df = pd.DataFrame({"RV_m": [0.1, 1.0, 5.0], "jump_flag": [True, False, True]})
    assert list(tag_regime(df, threshold=1.0)) == ["calm", "calm", "turbulent"]
    assert list(tag_jump_day(df)) == ["jump", "no_jump", "jump"]


# ── level 4 ──────────────────────────────────────────────────────────────


def test_cohens_kappa_corrects_for_chance_where_jaccard_does_not():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.random(4000) < 0.05), pd.Series(rng.random(4000) < 0.05)
    assert jaccard_index(a, b) > 0.0
    assert abs(cohens_kappa(a, b)) < 0.05


def test_cohens_kappa_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(2)
    variants = {name: pd.Series(rng.random(400) < 0.1) for name in ["naive", "bns", "lm"]}
    m = cohens_kappa_matrix(variants)
    for name in variants:
        assert m.loc[name, name] == pytest.approx(1.0)
    assert m.loc["naive", "bns"] == pytest.approx(m.loc["bns", "naive"])


def test_jump_diagnostics_separate_occurrence_from_size():
    rng = np.random.default_rng(0)
    j_true = np.where(rng.random(800) < 0.1, rng.lognormal(-8, 1, 800), 0.0)
    near_zero = jump_component_diagnostics(j_true, np.full(800, 1e-12))
    assert near_zero["skill_vs_zero"] == pytest.approx(0.0, abs=1e-6)
    informed = jump_component_diagnostics(j_true, j_true * 0.8)
    assert informed["rank_corr_on_jump_days"] == pytest.approx(1.0)


def test_no_precision_recall_is_exposed():
    from projekat.qa import checks

    for forbidden in ["precision", "recall", "f1_score", "roc_auc"]:
        assert not any(forbidden in name.lower() for name in dir(checks))


# ── WI-12 ────────────────────────────────────────────────────────────────


def test_signature_plot_covers_the_declared_frequencies():
    rng = np.random.default_rng(0)
    by_grid = {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 1e-4 for i in range(15)}
               for m in SIGNATURE_PLOT_MINUTES}
    out = signature_plot(by_grid)
    assert list(out["minutes"]) == list(SIGNATURE_PLOT_MINUTES)
    assert out["rv_minus_bpv"].notna().all()


def test_h5_placebo_preserves_frequency_breaks_alignment_and_stays_consistent():
    panel = make_panel(design_ids=("5m__bpv__bns",), end="2013-12-31")
    placebo = h5_placebo_panel(panel, seed=0)
    for sym in ["AAA", "BBB"]:
        real = panel[panel["symbol"] == sym].sort_values("date")
        fake = placebo[placebo["symbol"] == sym].sort_values("date")
        assert real["jump_flag"].sum() == fake["jump_flag"].sum()
        assert not np.array_equal(real["jump_flag"].to_numpy(), fake["jump_flag"].to_numpy())
        # J_d moves with its own flag, and J/C aggregates are recomputed from it
        assert ((fake["J_d"] > 0) == fake["jump_flag"]).all()
        assert np.allclose(fake["C_d"] + fake["J_d"], fake["RV_d"])
        expected_jw = fake["J_d"].rolling(5, min_periods=5).mean()
        assert np.allclose(fake["J_w"], expected_jw, equal_nan=True)
        expected_cm = fake["C_d"].rolling(22, min_periods=22).mean()
        assert np.allclose(fake["C_m"], expected_cm, equal_nan=True)


def test_h5_placebo_test_compares_the_real_delta_with_the_permutation_distribution(monkeypatch):
    import projekat.evaluation.diagnostics as diag

    panel = make_panel(design_ids=("5m__bpv__bns", "5m__bpv__lm"), end="2015-03-31")
    real_run_one = __import__("projekat.model.runner", fromlist=["run_one"]).run_one

    def short(design_df, **kwargs):
        return real_run_one(design_df, first="2014-11-01", last="2015-03-31", **kwargs)

    monkeypatch.setattr("projekat.model.runner.run_one", short)
    summary, perms = h5_placebo_test(panel, n_permutations=4)
    assert set(summary["model"]) == set(diag.PLACEBO_MODELS)
    assert set(summary["jump_test"]) == {"bns", "lm"}
    assert (summary["n_permutations"] == 4).all()
    assert len(perms) == 4 * 2 * 2
    for _, row in summary.iterrows():
        dist = perms[(perms["design_id"] == row["design_id"]) & (perms["model"] == row["model"])]["delta_qlike"]
        assert row["p_value"] == pytest.approx((1 + (dist <= row["delta_qlike_real"]).sum()) / (1 + len(dist)))
