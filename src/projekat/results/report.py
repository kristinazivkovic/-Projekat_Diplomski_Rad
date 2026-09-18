"""Reporting orchestration: the one place that CALLS every evaluation
function and WRITES its output into results/.

WHY THIS MODULE EXISTS
======================
The evaluation layer (evaluation/, qa/) was correct but had NO CALLER
outside the tests. That is exactly the class of defect already caught by
changelog items 5-7 (`ranking_disagreement` defined but never called;
`verify_split_adjustment` only in tests; the hard checks in qa/checks.py
never run against a real panel): the function is correct, the test
passes, and the result NEVER REACHES THE OUTPUT. For the thesis that is
the same as not existing -- a finding written nowhere can neither be
reported nor challenged.

Hence this module's rule: EVERY first-class result from levels 1-4 has
exactly one call here and exactly one written artefact. Adding a new
evaluation function without an entry in `build_all_reports` makes that
function dead code again.

WHAT GETS WRITTEN (results/)
============================
Level 1 main_result_table.csv          mean QLIKE/RMSE/MAE per cell
        ranking_disagreement.csv       QLIKE-vs-MAE reversals (DM-significant)
Level 2 dm_vs_log_har.csv              DM against the baseline, BH-controlled
        mcs_membership.csv             MCS membership (NOT a ranking)
        anova_design_sensitive.csv     decomposition, per horizon
        anova_grid_only.csv            decomposition, per horizon
        anova_clustered_*.csv          two-way clustered SEs beside any F/p
        anova_robustness.csv           four variants side by side
        anova_block_bootstrap.csv      optional alternative inference
Level 3 var_backtest.csv               Kupiec/Christoffersen/DQ/Lopez
Level 4 jump_overlap_jaccard.csv       A3 variant overlap
        jump_overlap_kappa.csv         the same, chance-corrected
        jump_characterization.json     day share, size, J-share of RV
        jump_component_diagnostics.csv J-component diagnostics
Strata  stratified_losses.csv          regime x jump_day x horizon
WI-12   signature_plot.csv             RV and BPV across 7 frequencies
        reestimation_frequency.csv     annual/monthly/daily refitting
        h5_placebo.csv                 real versus permuted panel

The WI-12 diagnostics are the LOWEST PRIORITY and live behind a separate
switch (`build_diagnostic_reports`), not inside `build_all_reports`: two
of the three (refitting, placebo) re-run the entire walk-forward per
model, costing an order of magnitude more than all of levels 1-4
together. If time runs short these three are dropped and the headline
results are untouched.

THE TWO DECOMPOSITIONS ARE NEVER MERGED. `anova_design_sensitive.csv` and
`anova_grid_only.csv` are separate files deliberately -- combining them
into one "headline" number is an open question with the advisor, and the
hook for that decision is
evaluation/anova.py::aggregate_headline_share. This module never calls it.

The block bootstrap does NOT run by default (`bootstrap=False`): it is an
optional alternative inference path costing n_boot full decompositions
per horizon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projekat.config import RESULTS_DIR
from projekat.evaluation.anova import (
    block_bootstrap_variance_shares,
    decompose_variance_design_sensitive,
    decompose_variance_grid_only,
    robustness_table,
)
from projekat.evaluation.dm_test import dm_against_baseline
from projekat.evaluation.losses import ranking_disagreement_table
from projekat.evaluation.stratify import stratified_table
from projekat.evaluation.var import var_backtest_table
from projekat.qa.checks import (
    cohens_kappa_matrix,
    jaccard_matrix,
    jump_component_diagnostics,
    level4_summary,
)
from projekat.results.tables import (
    attach_losses,
    attach_mz_recalibration,
    attach_strata,
    main_result_table,
    mcs_membership_table,
)

# Which models' losses feed which decomposition. The split is by
# depends_on (model/protocols.py), never by name -- see _split_by_dependence.
_GRID_ONLY_DEPENDS = frozenset({"grid"})


def _write(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def _split_by_dependence(daily_losses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into the two decomposition sets by the model's own
    `depends_on` from the registry -- never by a hard-coded name list,
    which would go stale the moment a model is added (the same reasoning
    as the registry pattern in measure/ and model/)."""
    # the same import-for-registration as runner.py:20 -- MODELS is empty
    # until the submodules are imported (registry pattern, see model/registry.py)
    from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401
    from projekat.model.registry import MODELS

    grid_only_names = {
        name for name, model in MODELS.items()
        if frozenset(model.depends_on) == _GRID_ONLY_DEPENDS
    }
    is_grid_only = daily_losses["model"].isin(grid_only_names)
    return daily_losses[~is_grid_only], daily_losses[is_grid_only]


def _write_anova(
    per_horizon: dict[int, "object"], out_dir: Path, stem: str
) -> list[Path]:
    """One row per (horizon, term) with its partial eta-squared, plus a
    separate file of two-way clustered SEs per horizon.

    Partial eta-squared is a DESCRIPTIVE variance share and needs no
    independence assumption; any F/p must be read against the clustered
    SEs (see the evaluation/anova.py docstring)."""
    written: list[Path] = []
    rows = []
    for horizon, table in per_horizon.items():
        for term, share in table.variance_share.items():
            rows.append(
                {
                    "horizon": horizon,
                    "term": term,
                    "sum_sq": float(table.sum_sq.get(term, float("nan"))),
                    "df": float(table.df.get(term, float("nan"))),
                    "partial_eta_sq": float(share),
                    "n_obs": table.n_obs,
                    "formula": table.formula,
                    "note": table.note,
                }
            )
        if table.clustered is not None and not table.clustered.empty:
            written.append(
                _write(
                    table.clustered,
                    out_dir / f"anova_clustered_{stem}_h{horizon}.csv",
                    index=True,
                )
            )
    written.insert(0, _write(pd.DataFrame(rows), out_dir / f"anova_{stem}.csv"))
    return written


def build_all_reports(
    test_results: pd.DataFrame,
    val_results: pd.DataFrame | None = None,
    *,
    panel_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    bootstrap: bool = False,
    n_boot: int = 200,
) -> dict[str, Path]:
    """Run the ENTIRE evaluation layer over runner.py::run_all's output and
    write every result into `out_dir` (results/ by default).

    `test_results`/`val_results` are exactly what `run_all` returns.
    `panel_df` is needed only for the jump-detection characterization
    (level 4), which is computed over the panel rather than the forecasts
    -- if it is not passed, those artefacts are skipped and everything
    else is still written.

    Returns {artefact_name: path} so the caller (and the tests) can see
    exactly what was written."""
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if test_results.empty:
        return written

    # -- preparation: daily losses + strata ------------------------------
    daily = attach_losses(test_results)
    if val_results is not None and not val_results.empty:
        # MZ recalibration from the VALIDATION fold only -- recalibrating
        # before the primary metric would be lookahead (see tables.py)
        daily = attach_mz_recalibration(daily, attach_losses(val_results))
    try:
        daily = attach_strata(daily)
    except FileNotFoundError:
        # the regime boundary was never computed (stratify.py::compute_and_
        # persist_regime_boundary over window 1) -- the stratified artefacts
        # degrade, everything else is still written
        daily["regime"] = "unknown"
        daily["jump_day"] = "unknown"

    written["daily_losses"] = _write(daily, out / "daily_losses.csv")

    # -- level 1 ---------------------------------------------------------
    written["main_result_table"] = _write(main_result_table(daily), out / "main_result_table.csv")
    written["ranking_disagreement"] = _write(
        ranking_disagreement_table(daily), out / "ranking_disagreement.csv"
    )

    # -- level 2: significance -------------------------------------------
    written["dm_vs_log_har"] = _write(dm_against_baseline(daily), out / "dm_vs_log_har.csv")
    written["mcs_membership"] = _write(mcs_membership_table(daily), out / "mcs_membership.csv")

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
            written[f"anova_{stem}_error"] = _write(
                pd.DataFrame([{"error": str(exc)}]), out / f"anova_{stem}_ERROR.csv"
            )
            continue
        paths = _write_anova(per_horizon, out, stem)
        written[f"anova_{stem}"] = paths[0]
        for p in paths[1:]:
            written[p.stem] = p

        # skewness robustness is REQUIRED, not optional -- it must be
        # visible before the thesis text is written
        written[f"anova_robustness_{stem}"] = _write(
            robustness_table(frame, decomposition=stem, loss_col="qlike"),
            out / f"anova_robustness_{stem}.csv",
        )

        if bootstrap:
            boot_rows = []
            for horizon in sorted(frame["horizon"].unique()):
                draw = block_bootstrap_variance_shares(
                    frame, horizon=int(horizon), decomposition=stem,
                    loss_col="qlike", n_boot=n_boot,
                )
                draw["horizon"] = horizon
                boot_rows.append(draw)
            if boot_rows:
                written[f"anova_block_bootstrap_{stem}"] = _write(
                    pd.concat(boot_rows, ignore_index=True),
                    out / f"anova_block_bootstrap_{stem}.csv",
                )

    # NOTE: aggregate_headline_share is DELIBERATELY not called -- the two
    # decompositions stay separate files until the advisor decides.

    # -- level 3: VaR ----------------------------------------------------
    if "r_d" in daily.columns:
        written["var_backtest"] = _write(var_backtest_table(daily), out / "var_backtest.csv")

    # -- stratification --------------------------------------------------
    written["stratified_losses"] = _write(
        stratified_table(daily[["horizon", "regime", "jump_day", "qlike", "abs_error", "sq_error"]]),
        out / "stratified_losses.csv",
    )

    # -- level 4: jump-detection characterization ------------------------
    if panel_df is not None and not panel_df.empty:
        written.update(_build_level4_reports(panel_df, out))

    return written


def _build_level4_reports(panel_df: pd.DataFrame, out: Path) -> dict[str, Path]:
    """Jump-detection characterization over the PANEL (not the forecasts).

    No precision/recall: there is no ground-truth label for a "real" jump,
    so no variant may be treated as the reference truth. Overlap is
    measured with Jaccard AND Cohen's kappa side by side -- Jaccard alone
    overstates agreement because it ignores the (large) set of days both
    variants agree are non-jumps."""
    from projekat.evaluation.anova import parse_design_id

    written: dict[str, Path] = {}

    # A3 variants at a fixed grid and estimator, aligned on (symbol, day)
    designs = panel_df["design_id"].drop_duplicates()
    parsed = {d: parse_design_id(d) for d in designs}
    by_variant: dict[str, pd.Series] = {}
    for design_id, fields in parsed.items():
        cell = panel_df[panel_df["design_id"] == design_id]
        if cell.empty or "jump_flag" not in cell.columns:
            continue
        key = f"{fields['grid']}__{fields['estimator']}__{fields['jump_test']}"
        by_variant[key] = cell.set_index(["symbol", "date"])["jump_flag"].astype(bool)

    if len(by_variant) >= 2:
        common = None
        for series in by_variant.values():
            common = series.index if common is None else common.intersection(series.index)
        aligned = {k: v.loc[common] for k, v in by_variant.items()} if len(common) else {}
        if aligned:
            written["jump_overlap_jaccard"] = _write(
                jaccard_matrix(aligned), out / "jump_overlap_jaccard.csv", index=True
            )
            written["jump_overlap_kappa"] = _write(
                cohens_kappa_matrix(aligned), out / "jump_overlap_kappa.csv", index=True
            )

    # jump-day share, mean jump size, J-share of RV -- per design
    summary_rows = []
    for design_id, cell in panel_df.groupby("design_id", observed=True):
        try:
            summary = level4_summary(cell)
        except (KeyError, ZeroDivisionError):  # pragma: no cover
            continue
        summary_rows.append({"design_id": design_id, **summary})
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
    seed: int = 0,
) -> dict[str, Path]:
    """WI-12: three diagnostics, DELIBERATELY outside `build_all_reports`.

    They are separated for two reasons. First, cost:
    `reestimation_frequency_comparison` and `h5_placebo_comparison` re-run
    the entire anchored walk-forward, so they take an order of magnitude
    longer than all of levels 1-4 together -- and WI-12 is explicitly the
    lowest priority, the first thing cut if time runs short. Second,
    meaning: these are DIAGNOSTICS that SUPPORT the main results rather
    than results themselves, and must not sit in the same artefact set as
    the reported variance shares.

    `returns_by_symbol` ({symbol: {grid: {day: returns}}}) is needed only
    for the signature plot; without it that one artefact is skipped. The
    other two work directly from the panel.

    None of the three is a fourth measurement factor -- see the
    evaluation/diagnostics.py docstring."""
    from projekat.evaluation.diagnostics import (
        h5_placebo_comparison,
        reestimation_frequency_comparison,
        signature_plot_for_symbols,
    )

    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # 1. signature plot -- recovers the interior optimum that three A1
    #    levels cannot show; the gap between the curves is the evidence
    #    that noise bias does not cancel in RV - BPV
    if returns_by_symbol:
        frame = signature_plot_for_symbols(returns_by_symbol)
        if not frame.empty:
            written["signature_plot"] = _write(frame, out / "signature_plot.csv")

    # 2. re-estimation frequency -- econometric models ONLY, ONE fixed
    #    design, and explicitly NOT a measurement factor: it changes no RV
    #    value at all, so folding it into Factor A would put something
    #    that isn't measurement inside the headline measurement share
    try:
        frame = reestimation_frequency_comparison(
            panel_df, design_id=design_id, horizon=horizon
        )
        if not frame.empty:
            written["reestimation_frequency"] = _write(
                frame, out / "reestimation_frequency.csv"
            )
    except (ValueError, KeyError) as exc:
        written["reestimation_frequency_error"] = _write(
            pd.DataFrame([{"error": str(exc)}]), out / "reestimation_frequency_ERROR.csv"
        )

    # 3. H5 placebo -- jump_flag and J_* permuted WITHIN SYMBOL: preserves
    #    the marginal jump frequency while destroying the link to actual
    #    jump days. A small QLIKE difference = the advantage did not come
    #    from jump information. One grid, one estimator, all four tests.
    frame = h5_placebo_comparison(
        panel_df, grid=grid, estimator=estimator, horizon=horizon, seed=seed
    )
    if not frame.empty:
        written["h5_placebo"] = _write(frame, out / "h5_placebo.csv")

    return written


def jump_component_report(
    ttm_rows: pd.DataFrame, *, out_dir: Path | None = None
) -> Path | None:
    """J-component diagnostics for ttm_c_plus_j (level 4).

    A separate function because it needs the `J_true`/`J_hat` columns that
    only that one model emits -- MSE against TWO references (J_hat = 0 and
    a trailing mean), the clipping activation rate, and a rank correlation
    over days with J > 0 ONLY. That last one separates predicting jump
    SIZE from predicting jump OCCURRENCE, which MSE conflates into a
    single number.

    Expect ttm_j to sit near zero throughout -- that is a MEASUREMENT
    supporting the CHAR rationale, not a bug to fix."""
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
