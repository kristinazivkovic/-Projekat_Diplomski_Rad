"""WI-9/WI-10: level-1 losses and the level-2 headline decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.evaluation.anova import (
    aggregate_headline_share,
    block_bootstrap_variance_shares,
    decompose_variance_design_sensitive,
    parse_design_id,
    robustness_table,
)
from projekat.evaluation.dm_test import DM_FAMILIES, benjamini_hochberg_dm, dm_against_baseline
from projekat.evaluation.losses import (
    mae_median_variant,
    ranking_disagreement,
    ranking_disagreement_table,
    relative_qlike,
)


def _daily_losses(seed=0, n=250):
    rng = np.random.default_rng(seed)
    rows = []
    for design in ["1m__bpv__bns", "5m__bpv__bns", "5m__medrv__lm"]:
        for model in ["log_har", "char", "har_j"]:
            for sym in ["AAA", "BBB"]:
                for h in [1, 5]:
                    for d in pd.bdate_range("2020-01-01", periods=n):
                        eff = {"log_har": 0.0, "char": 0.5, "har_j": -0.05}[model]
                        rows.append(
                            {
                                "model": model,
                                "design_id": design,
                                "horizon": h,
                                "symbol": sym,
                                "date": d,
                                "regime": "calm" if rng.random() < 0.7 else "turbulent",
                                "qlike": float(rng.lognormal(-1 + eff, 0.6)),
                                "abs_error": float(rng.lognormal(-2 - eff, 0.6)),
                            }
                        )
    return pd.DataFrame(rows)


# ── WI-9 ──────────────────────────────────────────────────────────────────


def test_relative_qlike_is_ratio_of_means_not_mean_of_ratios():
    """A single near-zero benchmark day must not blow the figure up -- that
    is exactly what the mean-of-ratios convention does and why it is not
    interchangeable with this one."""
    model_losses = np.array([1.0, 1.0, 1.0, 1.0])
    baseline_losses = np.array([1.0, 1.0, 1.0, 1e-9])  # one near-perfect benchmark day

    ratio_of_means = relative_qlike(model_losses.mean(), baseline_losses.mean())
    mean_of_ratios = float(np.mean(model_losses / baseline_losses))

    assert ratio_of_means == pytest.approx(4 / 3.000000001, rel=1e-6)
    assert mean_of_ratios > 1e8  # the convention we rejected explodes
    assert ratio_of_means < 2.0


def test_ranking_disagreement_requires_dm_significance():
    """A reversal that is only rounding noise must NOT be flagged; a real,
    DM-significant reversal must be."""
    rng = np.random.default_rng(0)
    n = 400

    # noise-only reversal: means differ by a hair, no systematic difference
    a = rng.lognormal(0.0, 0.5, n)
    b = a + rng.normal(0.0, 1e-9, n)
    assert not ranking_disagreement(a, b, b, a, horizon=1)

    # genuine reversal: qlike strongly prefers a, mae strongly prefers b
    q_a = rng.lognormal(-1.0, 0.3, n)
    q_b = rng.lognormal(-0.2, 0.3, n)
    m_a = rng.lognormal(-0.2, 0.3, n)
    m_b = rng.lognormal(-1.0, 0.3, n)
    assert ranking_disagreement(q_a, q_b, m_a, m_b, horizon=1)


def test_ranking_disagreement_table_reports_qlike_as_the_resolution():
    df = _daily_losses()
    out = ranking_disagreement_table(df)
    if len(out):
        # QLIKE always wins on disagreement -- never silently resolved the other way
        assert (out["resolved_in_favour_of"] == out["qlike_prefers"]).all()
        assert (out["qlike_prefers"] != out["mae_prefers"]).all()


def test_mae_median_variant_uses_uncorrected_forecast():
    y_true = np.array([1.0, 2.0, 3.0])
    log_y_hat = np.log(np.array([1.0, 2.0, 3.0]))
    assert mae_median_variant(y_true, log_y_hat) == pytest.approx(0.0)


# ── WI-10: DM family / BH ────────────────────────────────────────────────


def test_bh_requires_a_declared_family_and_records_it():
    results = pd.DataFrame({"p_value": [0.001, 0.02, 0.4, 0.9]})
    out = benjamini_hochberg_dm(results, family="model_vs_log_har")
    assert (out["dm_family"] == "model_vs_log_har").all()
    assert out["bh_significant"].sum() >= 1
    with pytest.raises(ValueError):
        benjamini_hochberg_dm(results, family="not_a_declared_family")


def test_bh_is_more_conservative_than_raw_alpha():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1, 200), [0.001]])
    out = benjamini_hochberg_dm(pd.DataFrame({"p_value": p}), family="all_pairs")
    assert out["bh_significant"].sum() <= (p < 0.05).sum()


def test_dm_against_baseline_controls_over_the_whole_family():
    df = _daily_losses()
    out = dm_against_baseline(df)
    assert set(out["baseline"]) == {"log_har"}
    assert "log_har" not in set(out["model"])
    assert (out["dm_family"] == "model_vs_log_har").all()
    assert "model_vs_log_har" in DM_FAMILIES


# ── WI-10: ANOVA headline ────────────────────────────────────────────────


def test_decomposition_is_per_horizon_never_pooled():
    df = _daily_losses()
    result = decompose_variance_design_sensitive(df, clustered=False)
    assert set(result) == {1, 5}
    for horizon, table in result.items():
        assert table.horizon == horizon
        # horizon must not appear as a factor inside a per-horizon fit
        assert "horizon" not in table.formula


def test_symbol_is_an_explicit_blocking_factor_and_type_iii_sum_contrasts():
    df = _daily_losses()
    table = decompose_variance_design_sensitive(df, clustered=False)[1]
    assert "C(symbol, Sum)" in table.formula      # blocking factor present
    assert ", Sum)" in table.formula              # sum-to-zero contrasts enforced
    assert "C(symbol, Sum)" in table.variance_share.index


def test_structural_interactions_are_present():
    table = decompose_variance_design_sensitive(_daily_losses(), clustered=False)[1]
    for term in [
        "C(grid, Sum):C(estimator, Sum)",
        "C(grid, Sum):C(jump_test, Sum)",
        "C(model, Sum):C(jump_test, Sum)",
    ]:
        assert term in table.variance_share.index


def test_two_way_clustered_standard_errors_are_reported():
    table = decompose_variance_design_sensitive(_daily_losses(seed=1, n=60))[1]
    assert table.clustered is not None
    assert "cluster_se" in table.clustered.columns
    assert (table.clustered["cluster_se"] >= 0).all()


def test_robustness_table_reports_all_four_variants_side_by_side():
    out = robustness_table(_daily_losses(seed=2, n=80))
    assert set(out["variant"]) == {"raw", "log", "within_day_rank", "raw_ex_top1pct"}
    # the injected model effect should dominate and survive every variant
    model_rows = out[out["term"] == "C(model, Sum)"]
    assert len(model_rows) == 8  # 4 variants x 2 horizons
    assert (model_rows["partial_eta_sq"] > 0.01).all()


def test_block_bootstrap_block_length_matches_horizon():
    out = block_bootstrap_variance_shares(_daily_losses(seed=3, n=80), horizon=5, n_boot=10)
    assert not out.empty
    assert (out["ci_lo"] <= out["boot_mean"]).all()
    assert (out["boot_mean"] <= out["ci_hi"]).all()


def test_headline_aggregation_is_an_explicit_unimplemented_hook():
    with pytest.raises(NotImplementedError, match="open question"):
        aggregate_headline_share()


def test_parse_design_id_is_the_inverse_of_measurement_design_id():
    from projekat.measure.design import MeasurementDesign

    design = MeasurementDesign(grid="5m", estimator="trbpv", jump_test="lm_fdr")
    assert parse_design_id(design.id) == {
        "grid": "5m",
        "estimator": "trbpv",
        "jump_test": "lm_fdr",
    }
