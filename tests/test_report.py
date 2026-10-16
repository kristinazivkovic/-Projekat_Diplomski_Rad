"""Results-layer wiring: every first-class evaluation result must actually
reach an artefact on disk (the failure mode changelog items 5-7 hit three
times: a correct function with no production caller)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from projekat.results.report import build_all_reports, jump_component_report

DESIGNS = ("5m__bpv__bns", "1m__bpv__bns", "5m__medrv__lm")


def _results(n_days=160, seed=0):
    """Forecast rows in the shape walkforward.py emits (all horizons, warm-up flag)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-12-01", periods=n_days)
    rows = []
    # grid-only models run on one design per grid; design-sensitive on all
    plan = [("log_har", "5m__bpv__bns"), ("log_har", "1m__bpv__bns"), ("har", "5m__bpv__bns"),
            ("har", "1m__bpv__bns")] + [(m, d) for m in ("char", "har_j") for d in DESIGNS]
    for model, design_id in plan:
        for symbol in ("AAA", "BBB"):
            for horizon in (1, 5):
                for d in dates:
                    y_true = float(rng.lognormal(-9.5, 0.4))
                    y_hat = y_true * float(rng.lognormal(0, 0.25))
                    rows.append({
                        "model": model, "design_id": design_id, "horizon": horizon, "symbol": symbol,
                        "date": d, "seed": np.nan, "is_warmup": d < pd.Timestamp("2015-01-01"),
                        "y_true": y_true, "y_hat": y_hat, "y_hat_recalibrated": y_hat * 1.02,
                        "jump_flag": bool(rng.random() < 0.08),
                        "r_d_next": float(rng.normal(0, np.sqrt(y_true))),
                    })
    return pd.DataFrame(rows)


def _labels():
    dates = pd.bdate_range("2014-12-01", periods=160)
    return pd.DataFrame([{"symbol": s, "date": d, "regime": "calm" if i % 3 else "turbulent"}
                         for s in ("AAA", "BBB") for i, d in enumerate(dates)])


def _panel(n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for design_id in ("5m__bpv__bns", "5m__bpv__naive", "5m__bpv__lm"):
        for symbol in ("AAA", "BBB"):
            for d in dates:
                rv = float(rng.lognormal(-9.5, 0.4))
                is_jump = bool(rng.random() < 0.1)
                j = rv * 0.3 if is_jump else 0.0
                rows.append({"design_id": design_id, "symbol": symbol, "date": d, "RV_d": rv,
                             "J_d": j, "C_d": rv - j, "jump_flag": is_jump})
    return pd.DataFrame(rows)


@pytest.fixture
def reports(tmp_path):
    return build_all_reports(_results(), regime_labels=_labels(), panel_df=_panel(), out_dir=tmp_path)


def test_every_evaluation_level_reaches_an_artefact(reports):
    expected = {
        "main_result_table", "ranking_disagreement", "mincer_zarnowitz",      # level 1
        "dm_vs_log_har", "mcs_membership", "anova_design_sensitive", "anova_grid_only",  # level 2
        "var_backtest", "h4_kendall_tau",                                      # level 3
        "jump_overlap_jaccard", "jump_overlap_kappa", "jump_characterization",  # level 4
        "stratified_losses",
    }
    missing = expected - set(reports)
    assert not missing, f"evaluation results with no artefact: {sorted(missing)}"
    for name, path in reports.items():
        assert path.exists(), f"{name} reported a path that was never written"


def test_warm_up_rows_never_reach_a_table(reports):
    daily = pd.read_parquet(reports["daily_losses"])
    assert daily["date"].min() >= pd.Timestamp("2015-01-01")


def test_regime_comes_from_the_run_labels(reports):
    daily = pd.read_parquet(reports["daily_losses"])
    assert set(daily["regime"]) == {"calm", "turbulent"}
    # one label per (symbol, date): identical across models and designs
    assert daily.groupby(["symbol", "date"])["regime"].nunique().max() == 1


def test_the_two_decompositions_are_separate_and_disjoint(reports):
    ds = pd.read_csv(reports["anova_design_sensitive"])
    go = pd.read_csv(reports["anova_grid_only"])
    assert any(":" in t for t in ds["term"])
    assert not any("estimator" in t for t in go["term"])
    assert not any("headline" in n for n in reports)
    daily = pd.read_parquet(reports["daily_losses"])
    n_ds = daily[daily["model"].isin(["char", "har_j"])].groupby("horizon").size()
    assert (ds.groupby("horizon")["n_obs"].first() == n_ds).all()   # no copied rows in (a)
    n_go = daily[daily["model"].isin(["log_har", "har"])].groupby("horizon").size()
    assert (go.groupby("horizon")["n_obs"].first() == n_go).all()   # no copied rows in (b)


def test_anova_is_per_horizon_with_partial_eta_squared(reports):
    table = pd.read_csv(reports["anova_design_sensitive"])
    assert set(table["horizon"].unique()) == {1, 5}
    shares = table.loc[table["term"] != "Residual", "partial_eta_sq"]
    assert ((shares >= 0) & (shares <= 1)).all()


def test_clustered_standard_errors_accompany_the_decomposition(reports):
    clustered = [n for n in reports if n.startswith("anova_clustered_")]
    assert clustered
    frame = pd.read_csv(reports[clustered[0]], index_col=0)
    assert {"coef", "cluster_se", "p_value"} <= set(frame.columns)


def test_robustness_table_has_all_four_variants(reports):
    table = pd.read_csv(reports["anova_robustness_design_sensitive"])
    assert set(table["variant"].unique()) == {"raw", "log", "within_day_rank", "raw_ex_top1pct"}


def test_var_is_h1_only_and_h4_ranks_models_per_design(reports):
    var = pd.read_csv(reports["var_backtest"])
    assert set(var["horizon"]) == {1}
    kendall = pd.read_csv(reports["h4_kendall_tau"])
    assert {"design_id", "kendall_tau", "p_value", "n_models"} <= set(kendall.columns)
    # grid-only models join every design of their grid: 2 grid-only + 2 own models
    per_design = kendall[kendall["design_id"] != "__mean_over_designs__"]
    assert (per_design["n_models"] == 4).all()
    assert "__mean_over_designs__" in set(kendall["design_id"])


def test_recalibrated_qlike_is_reported_beside_raw(reports):
    table = pd.read_csv(reports["main_result_table"])
    assert {"qlike", "qlike_recalibrated"} <= set(table.columns)


def test_block_bootstrap_is_opt_in_and_uses_block_length_max_h_5(tmp_path):
    default = build_all_reports(_results(), out_dir=tmp_path / "a")
    assert not any("bootstrap" in n for n in default)
    opted = build_all_reports(_results(), out_dir=tmp_path / "b", bootstrap=True, n_boot=2)
    boot = pd.read_csv(opted["anova_block_bootstrap_design_sensitive"])
    assert set(boot.groupby("horizon")["block_length"].first()) == {5}   # h=1 -> 5, h=5 -> 5


def test_jump_characterization_has_no_precision_or_recall(reports):
    payload = json.loads(reports["jump_characterization"].read_text(encoding="utf-8"))
    for entry in payload:
        assert {"jump_day_share", "mean_jump_size", "j_share_of_rv"} <= set(entry)
        assert not any(k in entry for k in ("precision", "recall", "f1"))


def test_jump_component_diagnostics_reach_an_artefact(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    j_true = np.where(rng.random(n) < 0.1, rng.lognormal(-10, 0.5, n), 0.0)
    rows = pd.DataFrame({"design_id": "5m__bpv__bns", "horizon": 1, "symbol": "AAA",
                         "J_true": j_true, "J_hat": j_true * 0.5 + rng.normal(0, 1e-6, n)})
    path = jump_component_report(rows, out_dir=tmp_path)
    out = pd.read_csv(path)
    for col in ("mse_vs_zero", "mse_vs_trailing_mean", "clipping_rate", "rank_corr_on_jump_days", "n_jump_days"):
        assert col in out.columns


def test_empty_results_write_nothing_rather_than_crashing(tmp_path):
    assert build_all_reports(pd.DataFrame(), out_dir=tmp_path) == {}


def test_unscoreable_rows_are_counted_and_then_dropped(tmp_path):
    """QLIKE is undefined at y_true = 0 (a zero-variance day) and at a
    missing/non-positive/infinite forecast (measured: 65 zero-RV days on
    NVDA; ARFIMA overflowed on 126 rows). Such rows must be COUNTED in
    their own artefact and dropped everywhere else -- never left to turn a
    mean into inf or NaN."""
    rows = _results(n_days=60)
    evaluated = rows.index[~rows["is_warmup"].to_numpy()]         # warm-up rows are dropped anyway
    rows.loc[evaluated[:5], "y_true"] = 0.0                       # zero-variance day
    rows.loc[evaluated[5:9], "y_hat"] = np.inf                    # overflowed forecast
    rows.loc[evaluated[9:12], "y_hat"] = 0.0                      # clipped to zero
    rows.loc[evaluated[12:15], "y_hat"] = np.nan                  # no forecast

    reports = build_all_reports(rows, regime_labels=_labels(), out_dir=tmp_path)
    counts = pd.read_csv(reports["unscoreable_rows"])
    by_reason = counts.groupby("reason")["n_rows"].sum()
    assert by_reason["zero_target"] == 5
    assert by_reason["non_positive_or_infinite_forecast"] == 7    # inf and 0 alike
    assert by_reason["missing_forecast"] == 3

    daily = pd.read_parquet(reports["daily_losses"])
    assert np.isfinite(daily["qlike"]).all()
    assert (daily["y_true"] > 0).all() and (daily["y_hat"] > 0).all()
    main = pd.read_csv(reports["main_result_table"])
    assert np.isfinite(main["qlike"]).all()


def test_wi12_diagnostics_are_not_part_of_the_main_report(reports):
    assert not any(n in reports for n in ("signature_plot", "reestimation_frequency", "h5_placebo"))


def test_signature_plot_aggregates_per_symbol_not_per_day():
    from projekat.evaluation.diagnostics import SIGNATURE_PLOT_MINUTES, signature_plot_for_symbols

    rng = np.random.default_rng(0)
    by_symbol = {
        "AAA": {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 1e-4 for i in range(40)}
                for m in SIGNATURE_PLOT_MINUTES},
        "BBB": {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 5e-4 for i in range(10)}
                for m in SIGNATURE_PLOT_MINUTES},
    }
    out = signature_plot_for_symbols(by_symbol)
    for minutes in SIGNATURE_PLOT_MINUTES:
        cell = out[out["minutes"] == minutes].set_index("symbol")
        assert cell.loc["__mean__", "mean_rv"] == pytest.approx((cell.loc["AAA", "mean_rv"] + cell.loc["BBB", "mean_rv"]) / 2)
