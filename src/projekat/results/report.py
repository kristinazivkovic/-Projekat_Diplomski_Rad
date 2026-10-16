"""Reporting orchestration: the one place that CALLS every evaluation
function and WRITES its output.

Rule: EVERY first-class result from levels 1-4 has exactly one call here
and exactly one written artefact -- a correct evaluation function with no
caller here is dead code (changelog items 5-7 hit this three times).

WHAT GETS WRITTEN
=================
Level 1 main_result_table.csv          mean QLIKE/RMSE/MAE (+ recalibrated QLIKE) per cell
        ranking_disagreement.csv       QLIKE-vs-MAE reversals (DM-significant on both)
        mincer_zarnowitz.csv           MZ intercept/slope over the evaluation period (HAC)
Level 2 dm_vs_log_har.csv              DM against the baseline, BH-controlled
        mcs_membership.csv             MCS membership (NOT a ranking)
        anova_design_sensitive.csv     decomposition (a), per horizon
        anova_grid_only.csv            decomposition (b), per horizon
        anova_clustered_*.csv          two-way clustered SEs beside any F/p
        anova_robustness_*.csv         four variants side by side
        anova_block_bootstrap_*.csv    optional alternative inference
Level 3 var_backtest.csv               Kupiec/Christoffersen/DQ/Lopez, h = 1 only
        h4_kendall_tau.csv             QLIKE ranking vs Lopez ranking (H4)
Level 4 jump_overlap_jaccard.csv       A3 variant overlap
        jump_overlap_kappa.csv         the same, chance-corrected
        jump_characterization.json     day share, size, J-share of RV
Strata  stratified_losses.csv          regime x jump_day x horizon
WI-12   (build_diagnostic_reports)     signature plot, re-estimation frequency, H5 placebo

Only EVALUATION rows (is_warmup == False, 2015-2025) enter any table: the
2014 warm-up year exists solely to supply the first trailing
out-of-sample errors for the correction.

THE TWO DECOMPOSITIONS ARE NEVER MERGED -- whether they may be combined is
an open question with the advisor (evaluation/anova.py::
aggregate_headline_share raises NotImplementedError). Neither duplicates
rows: grid-only models are copied onto the other designs of their grid
ONLY for the within-design comparisons (DM, MCS, ranking, H4), never for
the ANOVA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projekat.config import RESULTS_DIR
from projekat.evaluation.anova import (
    block_bootstrap_variance_shares,
    decompose_variance_design_sensitive,
    decompose_variance_grid_only,
    parse_design_id,
    robustness_table,
)
from projekat.evaluation.dm_test import benjamini_hochberg_dm, dm_against_baseline
from projekat.evaluation.losses import ranking_disagreement_table
from projekat.evaluation.stratify import stratified_table
from projekat.evaluation.var import var_backtest_table
from projekat.qa.checks import cohens_kappa_matrix, jaccard_matrix, jump_component_diagnostics, level4_summary
from projekat.results.tables import (
    attach_losses,
    attach_strata,
    collapse_seed_losses,
    main_result_table,
    mcs_membership_table,
    mincer_zarnowitz_table,
    unscoreable_rows_table,
)

# Which models' losses feed which decomposition. The split is by
# depends_on (model/protocols.py), never by name -- see _split_by_dependence.
_GRID_ONLY_DEPENDS = frozenset({"grid"})

#: the forecast-table columns the report reads (run_report.py loads only these)
#: `log_y_hat` is the UNCORRECTED log-space forecast; it is read only so that
#: main_result_table can report losses.mae_median_variant beside MAE (NaN for
#: the level models, which have no log-space forecast).
REPORT_COLUMNS = [
    "date", "symbol", "design_id", "model", "horizon", "seed", "is_warmup",
    "y_true", "y_hat", "log_y_hat", "y_hat_recalibrated", "jump_flag", "r_d_next",
]


def _write(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def _grid_only_models() -> set[str]:
    from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401
    from projekat.model.registry import MODELS

    return {name for name, model in MODELS.items() if frozenset(model.depends_on) == _GRID_ONLY_DEPENDS}


def _split_by_dependence(daily_losses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two DISJOINT row sets for the two decompositions, by the model's own
    `depends_on` from the registry -- never by a hard-coded name list."""
    is_grid_only = daily_losses["model"].isin(_grid_only_models())
    return daily_losses[~is_grid_only], daily_losses[is_grid_only]


def _iter_design_cells(daily_losses: pd.DataFrame, columns: list[str]):
    """For the within-design COMPARISONS only (DM, MCS, ranking): yield one
    frame per design_id holding its own rows plus every grid-only model's
    rows of the same grid, relabelled to that design.

    A grid-only model runs once per grid, on one representative design, so
    its rows carry only that design_id; its forecast and its target
    (RV-based, identical across designs of a grid) are the same for every
    estimator x jump_test at that grid. Built one design at a time, so the
    12x copy never exists in memory at once. NEVER used for the ANOVA."""
    cols = [c for c in columns if c in daily_losses.columns]
    frame = daily_losses[cols]
    grid_only = frame["model"].isin(_grid_only_models())
    grid_of = {d: parse_design_id(d)["grid"] for d in frame["design_id"].unique()}
    go_rows = frame[grid_only]
    go_grid = go_rows["design_id"].map(grid_of)
    own = frame[~grid_only]
    designs = sorted(grid_of)
    by_design = dict(tuple(own.groupby("design_id", observed=True))) if not own.empty else {}
    for design_id in designs:
        parts = []
        if design_id in by_design:
            parts.append(by_design[design_id])
        shared = go_rows[go_grid == grid_of[design_id]]
        if not shared.empty:
            parts.append(shared.assign(design_id=design_id))
        if parts:
            yield design_id, pd.concat(parts, ignore_index=True)


def _within_design_comparisons(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dm, mcs, ranking = [], [], []
    for _design_id, cell in _iter_design_cells(
        daily, ["date", "symbol", "model", "design_id", "horizon", "qlike", "abs_error"]
    ):
        dm.append(dm_against_baseline(cell))
        mcs.append(mcs_membership_table(cell))
        ranking.append(ranking_disagreement_table(cell))
    dm_all = pd.concat([f for f in dm if not f.empty], ignore_index=True) if any(not f.empty for f in dm) else pd.DataFrame()
    if not dm_all.empty:
        # BH over the WHOLE family, not per design cell
        dm_all = benjamini_hochberg_dm(
            dm_all.drop(columns=["dm_family", "bh_significant", "bh_threshold"], errors="ignore"),
            family="model_vs_log_har",
        )
    cat = lambda frames: pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()  # noqa: E731
    return {"dm": dm_all, "mcs": cat(mcs), "ranking": cat(ranking)}


def h4_kendall_table(daily: pd.DataFrame, var_table: pd.DataFrame) -> pd.DataFrame:
    """H4: does the model ranking change from statistical to economic
    evaluation? Per design_id at h = 1: Kendall's tau between the ranking
    of models by mean QLIKE and by Lopez loss (grid-only models enter every
    design of their grid, as in DM/MCS), plus a pooled row over the mean of
    each model across designs. Low or negative tau = the ranking changes."""
    from scipy.stats import kendalltau

    if var_table.empty:
        return pd.DataFrame()
    q = daily[daily["horizon"] == 1].groupby(["model", "design_id"], observed=True)["qlike"].mean().reset_index()
    both = q.merge(var_table[["model", "design_id", "lopez_loss"]], on=["model", "design_id"])
    if both.empty:
        return pd.DataFrame()
    grid_only = both["model"].isin(_grid_only_models())
    all_designs = sorted(daily["design_id"].unique())
    grid_of = {d: parse_design_id(d)["grid"] for d in set(all_designs) | set(both["design_id"])}
    parts = [both[~grid_only]]
    for _, row in both[grid_only].iterrows():
        for d in all_designs:
            if grid_of[d] == grid_of[row["design_id"]]:
                parts.append(pd.DataFrame([{**row.to_dict(), "design_id": d}]))
    full = pd.concat(parts, ignore_index=True).drop_duplicates(["model", "design_id"])

    rows = []
    for design_id, cell in full.groupby("design_id"):
        if len(cell) < 3:
            continue
        tau, p = kendalltau(cell["qlike"], cell["lopez_loss"])
        rows.append({"design_id": design_id, "horizon": 1, "n_models": len(cell), "kendall_tau": float(tau), "p_value": float(p)})
    pooled = full.groupby("model")[["qlike", "lopez_loss"]].mean()
    if len(pooled) >= 3:
        tau, p = kendalltau(pooled["qlike"], pooled["lopez_loss"])
        rows.append({"design_id": "__mean_over_designs__", "horizon": 1, "n_models": len(pooled),
                     "kendall_tau": float(tau), "p_value": float(p)})
    return pd.DataFrame(rows)


def _write_anova(per_horizon: dict, out_dir: Path, stem: str) -> list[Path]:
    """One row per (horizon, term) with its partial eta-squared, plus a
    separate file of two-way clustered SEs per horizon."""
    written: list[Path] = []
    rows = []
    for horizon, table in per_horizon.items():
        for term, share in table.variance_share.items():
            rows.append({
                "horizon": horizon, "term": term,
                "sum_sq": float(table.sum_sq.get(term, float("nan"))),
                "df": float(table.df.get(term, float("nan"))),
                "partial_eta_sq": float(share), "n_obs": table.n_obs,
                "formula": table.formula, "note": table.note,
            })
        if table.clustered is not None and not table.clustered.empty:
            written.append(_write(table.clustered, out_dir / f"anova_clustered_{stem}_h{horizon}.csv", index=True))
    written.insert(0, _write(pd.DataFrame(rows), out_dir / f"anova_{stem}.csv"))
    return written


def build_all_reports(
    results: pd.DataFrame,
    *,
    regime_labels: pd.DataFrame | None = None,
    panel_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    bootstrap: bool = False,
    n_boot: int = 200,
) -> dict[str, Path]:
    """Run the ENTIRE evaluation layer over walk-forward forecast rows and
    write every result into `out_dir` (results/ by default). `panel_df` is
    needed only for level 4 (computed over the panel, not the forecasts)."""
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if results.empty:
        return written

    if "is_warmup" in results.columns:
        results = results[~results["is_warmup"].astype(bool)]

    # -- preparation: daily losses + strata ------------------------------
    # what cannot be scored, and why, BEFORE it is dropped
    written["unscoreable_rows"] = _write(unscoreable_rows_table(results), out / "unscoreable_rows.csv")
    daily = attach_losses(results)
    # one row per forecast, a stochastic model's loss being the mean of its
    # per-seed losses -- before ANY aggregation, test or table
    daily = collapse_seed_losses(daily)
    daily = attach_strata(daily, regime_labels)
    daily_path = out / "daily_losses.parquet"
    daily.to_parquet(daily_path, index=False)
    written["daily_losses"] = daily_path

    # -- level 1 ---------------------------------------------------------
    written["main_result_table"] = _write(main_result_table(daily), out / "main_result_table.csv")
    written["mincer_zarnowitz"] = _write(mincer_zarnowitz_table(daily), out / "mincer_zarnowitz.csv")

    # -- level 1/2: within-design comparisons ----------------------------
    comparisons = _within_design_comparisons(daily)
    written["ranking_disagreement"] = _write(comparisons["ranking"], out / "ranking_disagreement.csv")
    written["dm_vs_log_har"] = _write(comparisons["dm"], out / "dm_vs_log_har.csv")
    written["mcs_membership"] = _write(comparisons["mcs"], out / "mcs_membership.csv")

    # -- level 2: decomposition (the thesis's headline result) -----------
    design_sensitive, grid_only = _split_by_dependence(daily)
    for frame, decompose, stem in (
        (design_sensitive, decompose_variance_design_sensitive, "design_sensitive"),
        (grid_only, decompose_variance_grid_only, "grid_only"),
    ):
        if frame.empty:
            continue
        try:
            per_horizon = decompose(frame, loss_col="qlike")
        except Exception as exc:  # pragma: no cover - degenerate subset
            written[f"anova_{stem}_error"] = _write(pd.DataFrame([{"error": str(exc)}]), out / f"anova_{stem}_ERROR.csv")
            continue
        paths = _write_anova(per_horizon, out, stem)
        written[f"anova_{stem}"] = paths[0]
        for p in paths[1:]:
            written[p.stem] = p
        # skewness robustness is REQUIRED -- visible before the text is written
        written[f"anova_robustness_{stem}"] = _write(
            robustness_table(frame, decomposition=stem, loss_col="qlike"), out / f"anova_robustness_{stem}.csv")
        if bootstrap:
            boot_rows = []
            for horizon in sorted(frame["horizon"].unique()):
                draw = block_bootstrap_variance_shares(frame, horizon=int(horizon), decomposition=stem,
                                                       loss_col="qlike", n_boot=n_boot)
                draw["horizon"] = horizon
                boot_rows.append(draw)
            if boot_rows:
                written[f"anova_block_bootstrap_{stem}"] = _write(
                    pd.concat(boot_rows, ignore_index=True), out / f"anova_block_bootstrap_{stem}.csv")
    del design_sensitive, grid_only

    # -- level 3: VaR (h = 1) and H4 --------------------------------------
    if "r_d_next" in daily.columns:
        var_table = var_backtest_table(daily)
        written["var_backtest"] = _write(var_table, out / "var_backtest.csv")
        written["h4_kendall_tau"] = _write(h4_kendall_table(daily, var_table), out / "h4_kendall_tau.csv")

    # -- stratification --------------------------------------------------
    cols = [c for c in ("horizon", "regime", "jump_day", "qlike", "abs_error", "sq_error") if c in daily.columns]
    written["stratified_losses"] = _write(stratified_table(daily[cols]), out / "stratified_losses.csv")

    # -- level 4: jump-detection characterization ------------------------
    if panel_df is not None and not panel_df.empty:
        written.update(build_level4_reports(panel_df, out))
    return written


def build_level4_reports(panel_df: pd.DataFrame, out: Path) -> dict[str, Path]:
    """Jump-detection characterization over the PANEL (not the forecasts).
    No precision/recall: there is no ground truth for a "real" jump.
    Overlap is Jaccard AND Cohen's kappa side by side."""
    written: dict[str, Path] = {}
    by_variant: dict[str, pd.Series] = {}
    for design_id, cell in panel_df.groupby("design_id", observed=True):
        if "jump_flag" not in cell.columns:
            continue
        f = parse_design_id(design_id)
        key = f"{f['grid']}__{f['estimator']}__{f['jump_test']}"
        by_variant[key] = cell.set_index(["symbol", "date"])["jump_flag"].astype(bool)

    if len(by_variant) >= 2:
        common = None
        for series in by_variant.values():
            common = series.index if common is None else common.intersection(series.index)
        if common is not None and len(common):
            aligned = {k: v.loc[common] for k, v in by_variant.items()}
            written["jump_overlap_jaccard"] = _write(jaccard_matrix(aligned), out / "jump_overlap_jaccard.csv", index=True)
            written["jump_overlap_kappa"] = _write(cohens_kappa_matrix(aligned), out / "jump_overlap_kappa.csv", index=True)

    summary_rows = []
    for design_id, cell in panel_df.groupby("design_id", observed=True):
        try:
            summary_rows.append({"design_id": design_id, **level4_summary(cell)})
        except (KeyError, ZeroDivisionError):  # pragma: no cover
            continue
    if summary_rows:
        path = out / "jump_characterization.json"
        path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        written["jump_characterization"] = path
    return written


def build_diagnostic_reports(
    panel_df: pd.DataFrame,
    *,
    returns_by_symbol: dict[str, dict[str, dict]] | None = None,
    design_id: str = "5m__bpv__bns",
    grid: str = "5m",
    estimator: str = "bpv",
    horizon: int = 1,
    out_dir: Path | None = None,
    n_permutations: int | None = None,
    workers: int = 1,
    parts: tuple[str, ...] = ("signature_plot", "reestimation_frequency", "h5_placebo"),
) -> dict[str, Path]:
    """WI-12: three diagnostics, DELIBERATELY outside `build_all_reports`
    (they re-run walk-forwards -- an order of magnitude more expensive --
    and they SUPPORT the results rather than being results). None of them
    is a fourth measurement factor."""
    from projekat.evaluation.diagnostics import (
        N_PERMUTATIONS,
        h5_placebo_test,
        reestimation_frequency_comparison,
        signature_plot_for_symbols,
    )

    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "signature_plot" in parts and returns_by_symbol:
        frame = signature_plot_for_symbols(returns_by_symbol)
        if not frame.empty:
            written["signature_plot"] = _write(frame, out / "signature_plot.csv")

    if "reestimation_frequency" in parts:
        try:
            frame = reestimation_frequency_comparison(panel_df, design_id=design_id, horizon=horizon)
            if not frame.empty:
                written["reestimation_frequency"] = _write(frame, out / "reestimation_frequency.csv")
        except (ValueError, KeyError) as exc:
            written["reestimation_frequency_error"] = _write(
                pd.DataFrame([{"error": str(exc)}]), out / "reestimation_frequency_ERROR.csv")

    if "h5_placebo" in parts:
        summary, perms = h5_placebo_test(
            panel_df, grid=grid, estimator=estimator, horizon=horizon,
            n_permutations=N_PERMUTATIONS if n_permutations is None else n_permutations, workers=workers,
        )
        if not summary.empty:
            written["h5_placebo"] = _write(summary, out / "h5_placebo.csv")
            written["h5_placebo_permutations"] = _write(perms, out / "h5_placebo_permutations.csv")
    return written


def jump_component_report(ttm_rows: pd.DataFrame, *, out_dir: Path | None = None) -> Path | None:
    """J-component diagnostics (MSE against two references, clipping rate,
    rank correlation on jump days only) for rows carrying J_true/J_hat.
    Expect J_hat near zero -- a MEASUREMENT supporting CHAR, not a bug."""
    if ttm_rows.empty or not {"J_true", "J_hat"} <= set(ttm_rows.columns):
        return None
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    rows = []
    group_cols = [c for c in ("design_id", "horizon", "symbol") if c in ttm_rows.columns]
    for keys, cell in ttm_rows.groupby(group_cols, observed=True) if group_cols else [((), ttm_rows)]:
        diagnostics = jump_component_diagnostics(cell["J_true"], cell["J_hat"])
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rows.append({**key_map, **diagnostics})
    return _write(pd.DataFrame(rows), out / "jump_component_diagnostics.csv")
