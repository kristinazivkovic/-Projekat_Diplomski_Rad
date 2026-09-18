"""Results-layer wiring: every first-class evaluation result must actually
reach an artefact on disk.

This is the regression guard for the failure mode that changelog items 5-7
already hit three times -- a correct function with no production caller,
whose result silently never appears in the output.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from projekat.results.report import build_all_reports, jump_component_report


def _test_results(n_days=120, seed=0):
    """Forecast rows in exactly the shape runner.py::run_all emits."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    # one design-sensitive model and one grid-only model, so both
    # decompositions have rows
    for model in ("char", "log_har"):
        for design_id in ("5m__bpv__bns", "1m__bpv__bns", "5m__medrv__lm"):
            for symbol in ("AAA", "BBB"):
                for horizon in (1, 5):
                    for d in dates:
                        y_true = float(rng.lognormal(-9.5, 0.4))
                        rows.append(
                            {
                                "model": model,
                                "design_id": design_id,
                                "horizon": horizon,
                                "symbol": symbol,
                                "date": d,
                                "window_id": 0,
                                "seed": np.nan,
                                "y_true": y_true,
                                "y_hat": y_true * float(rng.lognormal(0, 0.25)),
                                "RV_d": y_true,
                                "RV_m": y_true,
                                "jump_flag": bool(rng.random() < 0.08),
                                "r_d": float(rng.normal(0, np.sqrt(y_true))),
                            }
                        )
    return pd.DataFrame(rows)


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
                rows.append(
                    {
                        "design_id": design_id,
                        "symbol": symbol,
                        "date": d,
                        "RV_d": rv,
                        "J_d": j,
                        "C_d": rv - j,
                        "jump_flag": is_jump,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def reports(tmp_path):
    return build_all_reports(_test_results(), panel_df=_panel(), out_dir=tmp_path)


def test_every_evaluation_level_reaches_an_artefact(reports):
    """Levels 1-4 each write at least one file -- the guard against an
    evaluation function having no production caller."""
    expected = {
        "main_result_table",        # level 1
        "ranking_disagreement",     # level 1
        "dm_vs_log_har",            # level 2
        "mcs_membership",           # level 2
        "anova_design_sensitive",   # level 2 headline
        "anova_grid_only",          # level 2 headline
        "var_backtest",             # level 3
        "jump_overlap_jaccard",     # level 4
        "jump_overlap_kappa",       # level 4
        "jump_characterization",    # level 4
        "stratified_losses",
    }
    missing = expected - set(reports)
    assert not missing, f"evaluation results with no artefact: {sorted(missing)}"
    for name, path in reports.items():
        assert path.exists(), f"{name} reported a path that was never written"


def test_the_two_decompositions_are_written_separately_never_aggregated(reports):
    """Whether they may be combined is an open question with the advisor --
    until then they are two files and there is no headline aggregate."""
    assert reports["anova_design_sensitive"] != reports["anova_grid_only"]
    assert not any("headline" in name for name in reports)

    ds = pd.read_csv(reports["anova_design_sensitive"])
    go = pd.read_csv(reports["anova_grid_only"])
    # the design-sensitive decomposition carries the structural interactions;
    # the grid-only one has no estimator/jump_test factor at all
    assert any(":" in t for t in ds["term"])
    assert not any("estimator" in t for t in go["term"])


def test_anova_reports_partial_eta_squared_per_term_and_is_per_horizon(reports):
    table = pd.read_csv(reports["anova_design_sensitive"])
    assert "partial_eta_sq" in table.columns
    # fitted separately per horizon, never pooled
    assert set(table["horizon"].unique()) == {1, 5}
    shares = table.loc[table["term"] != "Residual", "partial_eta_sq"]
    assert ((shares >= 0) & (shares <= 1)).all()


def test_clustered_standard_errors_accompany_the_decomposition(reports):
    """Designs derive from the same raw 1m bars, so any reported F/p must be
    readable against two-way (day x symbol) clustered SEs."""
    clustered = [n for n in reports if n.startswith("anova_clustered_")]
    assert clustered, "no clustered-SE artefact written alongside the ANOVA"
    frame = pd.read_csv(reports[clustered[0]], index_col=0)
    assert {"coef", "cluster_se", "p_value"} <= set(frame.columns)


def test_robustness_table_is_written_not_buried(reports):
    """Skewness robustness is required, not optional -- instability must be
    visible before the thesis text is written."""
    key = "anova_robustness_design_sensitive"
    assert key in reports
    table = pd.read_csv(reports[key])
    assert set(table["variant"].unique()) == {
        "raw", "log", "within_day_rank", "raw_ex_top1pct"
    }


def test_block_bootstrap_is_opt_in(tmp_path):
    """The optional alternative inference path costs n_boot full
    decompositions per horizon, so it must not run by default."""
    default = build_all_reports(_test_results(), out_dir=tmp_path)
    assert not any("bootstrap" in n for n in default)

    opted_in = build_all_reports(
        _test_results(), out_dir=tmp_path, bootstrap=True, n_boot=3
    )
    assert any("bootstrap" in n for n in opted_in)


def test_jump_characterization_carries_no_precision_or_recall(reports):
    """There is no ground truth for a 'real' jump, so no variant may be
    treated as the label."""
    payload = json.loads(reports["jump_characterization"].read_text(encoding="utf-8"))
    assert payload
    for entry in payload:
        assert {"jump_day_share", "mean_jump_size", "j_share_of_rv"} <= set(entry)
        assert not any(k in entry for k in ("precision", "recall", "f1"))


def test_jump_component_diagnostics_reach_an_artefact(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    j_true = np.where(rng.random(n) < 0.1, rng.lognormal(-10, 0.5, n), 0.0)
    rows = pd.DataFrame(
        {
            "design_id": "5m__bpv__bns",
            "horizon": 1,
            "symbol": "AAA",
            "J_true": j_true,
            "J_hat": j_true * 0.5 + rng.normal(0, 1e-6, n),
        }
    )
    path = jump_component_report(rows, out_dir=tmp_path)
    assert path is not None and path.exists()

    out = pd.read_csv(path)
    # both references, the clipping rate, and the size-vs-occurrence split
    for col in (
        "mse_vs_zero", "mse_vs_trailing_mean", "clipping_rate",
        "rank_corr_on_jump_days", "n_jump_days",
    ):
        assert col in out.columns


def test_empty_results_write_nothing_rather_than_crashing(tmp_path):
    assert build_all_reports(pd.DataFrame(), out_dir=tmp_path) == {}


# ── WI-12: diagnostics are wired, but deliberately kept separate ─────────


def test_wi12_diagnostics_are_not_part_of_the_main_report(reports):
    """WI-12 is the lowest priority and is cut first if time runs short --
    it must not ride along with the headline results."""
    assert not any(
        n in reports for n in ("signature_plot", "reestimation_frequency", "h5_placebo")
    )


def test_signature_plot_aggregates_per_symbol_not_per_day(tmp_path):
    """The mean row must be a mean of per-symbol means -- pooling every
    day of every symbol would let the symbol with more days pull the
    curve, making the RV - BPV gap partly measure sample composition."""
    from projekat.evaluation.diagnostics import (
        SIGNATURE_PLOT_MINUTES,
        signature_plot_for_symbols,
    )

    rng = np.random.default_rng(0)
    # AAA deliberately has 4x the days of BBB
    by_symbol = {
        "AAA": {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 1e-4
                          for i in range(40)} for m in SIGNATURE_PLOT_MINUTES},
        "BBB": {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 5e-4
                          for i in range(10)} for m in SIGNATURE_PLOT_MINUTES},
    }
    out = signature_plot_for_symbols(by_symbol)
    assert set(out["symbol"]) == {"AAA", "BBB", "__mean__"}

    for minutes in SIGNATURE_PLOT_MINUTES:
        cell = out[out["minutes"] == minutes].set_index("symbol")
        expected = (cell.loc["AAA", "mean_rv"] + cell.loc["BBB", "mean_rv"]) / 2
        assert cell.loc["__mean__", "mean_rv"] == pytest.approx(expected)


def test_signature_plot_spans_the_full_minute_grid(tmp_path):
    """Three A1 levels cannot show an interior optimum; the diagnostic
    must span all seven declared frequencies."""
    from projekat.evaluation.diagnostics import (
        SIGNATURE_PLOT_MINUTES,
        signature_plot_for_symbols,
    )
    from projekat.results.report import build_diagnostic_reports

    rng = np.random.default_rng(0)
    by_symbol = {
        "AAA": {f"{m}m": {f"d{i}": rng.standard_normal(max(390 // m, 2)) * 1e-4
                          for i in range(10)} for m in SIGNATURE_PLOT_MINUTES}
    }
    written = build_diagnostic_reports(
        _panel(), returns_by_symbol=by_symbol, out_dir=tmp_path
    )
    assert "signature_plot" in written

    frame = pd.read_csv(written["signature_plot"])
    assert sorted(frame["minutes"].unique()) == sorted(SIGNATURE_PLOT_MINUTES)
    # the gap between the curves is the whole point of the diagnostic
    assert "rv_minus_bpv" in frame.columns
