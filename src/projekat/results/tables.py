"""The main h x regime x jump_day result table, MCS membership, and the
Mincer-Zarnowitz summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.evaluation.dm_test import collapse_to_daily
from projekat.evaluation.losses import qlike
from projekat.evaluation.mcs import model_confidence_set
from projekat.evaluation.mincer_zarnowitz import mincer_zarnowitz
from projekat.evaluation.stratify import tag_jump_day


#: why a forecast row cannot be scored (QLIKE needs y_true > 0 and y_hat > 0)
EXCLUSION_REASONS = ("no_realized_target", "zero_target", "missing_forecast", "non_positive_or_infinite_forecast")


def _scoreable(results_df: pd.DataFrame) -> pd.Series:
    y = results_df["y_true"].to_numpy(dtype=float)
    f = results_df["y_hat"].to_numpy(dtype=float)
    return pd.Series(np.isfinite(y) & (y > 0) & np.isfinite(f) & (f > 0), index=results_df.index)


def unscoreable_rows_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, design_id, horizon, reason) with the count of
    forecast rows that CANNOT be scored, written as its own artefact.

    QLIKE = y/y_hat - ln(y/y_hat) - 1 is undefined at y_true = 0 (a day whose
    realized variance is exactly zero -- coarse price quantization, see the
    panel diagnostics) and at y_hat <= 0 or non-finite (a model whose
    forecast under/overflowed). Such rows are dropped from every table, so
    the count must be visible: it says how much of a model's sample is
    missing and why, instead of the loss silently becoming inf or NaN."""
    y = results_df["y_true"].to_numpy(dtype=float)
    f = results_df["y_hat"].to_numpy(dtype=float)
    reason = pd.Series(pd.NA, index=results_df.index, dtype="object")
    reason[np.isfinite(y) & (y <= 0)] = "zero_target"
    reason[~np.isfinite(y)] = "no_realized_target"
    still_ok = reason.isna()
    # NaN means the model produced no forecast for that day (a symbol it
    # skipped, too little history); +-inf or <= 0 means it produced one that
    # QLIKE cannot score (an overflow, or a clip to zero) -- different
    # failures, counted apart
    reason[still_ok & np.isnan(f)] = "missing_forecast"
    reason[still_ok & ~np.isnan(f) & (~np.isfinite(f) | (f <= 0))] = "non_positive_or_infinite_forecast"
    flagged = results_df.loc[reason.notna(), ["model", "design_id", "horizon"]].assign(reason=reason[reason.notna()])
    if flagged.empty:
        return pd.DataFrame(columns=["model", "design_id", "horizon", "reason", "n_rows"])
    return (flagged.groupby(["model", "design_id", "horizon", "reason"], observed=True)
            .size().reset_index(name="n_rows").sort_values("n_rows", ascending=False))


def attach_losses(results_df: pd.DataFrame) -> pd.DataFrame:
    """Daily losses per forecast row, over the SCOREABLE rows only (see
    `unscoreable_rows_table`: no realized target, a zero target, or a
    missing/non-positive forecast). Left in, they would turn a whole cell's
    RMSE/MAE into NaN and a mean QLIKE into inf.

    `qlike_recalibrated` scores the Mincer-Zarnowitz-recalibrated forecast
    (coefficients from the trailing out-of-sample window strictly before t,
    model/backtransform.py) -- a PARALLEL secondary figure, never a
    replacement for the primary `qlike`. A recalibrated forecast <= 0 has
    no QLIKE and is NaN."""
    df = results_df[_scoreable(results_df)].copy()
    y, f = df["y_true"].to_numpy(), df["y_hat"].to_numpy()
    df["qlike"] = qlike(y, f)
    df["abs_error"] = np.abs(y - f)
    df["sq_error"] = (y - f) ** 2
    if "log_y_hat" in df.columns:
        # MAE partner za uslovnu MEDIJANU (losses.mae_median_variant):
        # |y - exp(log_y_hat)|, dakle nad NEKORIGOVANOM prognozom. Računa se
        # OVDE, pre collapse_seed_losses, da bi se preko semena uprosečio
        # GUBITAK, a ne log-prognoza (prosek log-prognoza pa exp() bio bi
        # prosek prognoza -- drugo pravilo od onog koje rad prijavljuje).
        # NaN za modele u nivoima (har, mem, ttm_c_plus_j): oni nemaju
        # log-prognozu, pa ni Jensenovu korekciju koju bi trebalo poništiti.
        with np.errstate(over="ignore", invalid="ignore"):
            df["abs_error_median_variant"] = np.abs(y - np.exp(df["log_y_hat"].to_numpy(dtype=float)))
    if "y_hat_recalibrated" in df.columns:
        rec = df["y_hat_recalibrated"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["qlike_recalibrated"] = np.where(rec > 0, qlike(y, np.where(rec > 0, rec, 1.0)), np.nan)
    return df


#: one forecast of one model for one symbol-day; everything else is an attribute
FORECAST_KEYS = ["model", "design_id", "horizon", "symbol", "date"]


def collapse_seed_losses(daily: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY figure for a stochastic model: a day's loss is the MEAN OF
    ITS PER-SEED LOSSES -- one row per forecast key, never one row per
    seed (otherwise each day would reach the ANOVA, VaR and every n_obs as
    several non-independent copies). `y_hat` becomes the seed-mean forecast,
    used only where one forecast is unavoidable (VaR). Deterministic rows
    pass through unchanged."""
    if "seed" not in daily.columns:
        return daily
    stochastic = daily["seed"].notna()
    if not stochastic.any():
        return daily
    st = daily[stochastic]
    numeric = [c for c in st.select_dtypes("number").columns if c not in FORECAST_KEYS and c != "seed"]
    other = [c for c in st.columns if c not in numeric and c not in FORECAST_KEYS and c != "seed"]
    agg = {c: "mean" for c in numeric} | {c: "first" for c in other}
    collapsed = st.groupby(FORECAST_KEYS, observed=True, sort=False).agg(agg).reset_index()
    collapsed["seed"] = np.nan
    return pd.concat([daily[~stochastic], collapsed[daily.columns]], ignore_index=True)


def attach_strata(results_df: pd.DataFrame, regime_labels: pd.DataFrame | None) -> pd.DataFrame:
    """Regime from the run's labels -- ONE label per (symbol, date), from
    5m RV_m, identical for every design (evaluation/stratify.py) -- and the
    row's own jump flag."""
    df = results_df.copy()
    if regime_labels is None or regime_labels.empty:
        df["regime"] = "unknown"
    else:
        labels = regime_labels[["symbol", "date", "regime"]]
        df = df.merge(labels, on=["symbol", "date"], how="left")
        df["regime"] = df["regime"].fillna("unknown")
    df["jump_day"] = tag_jump_day(df)
    return df


#: regime / jump_day value of the row that pools over every stratum
POOLED = "__all__"


def _cell_means(df: pd.DataFrame, keys: list[str], *, has_mz: bool, has_log: bool) -> pd.DataFrame:
    agg = {"qlike": ("qlike", "mean"), "sq": ("sq_error", "mean"), "mae": ("abs_error", "mean"),
           "n_obs": ("qlike", "size")}
    if has_mz:
        agg["qlike_recalibrated"] = ("qlike_recalibrated", "mean")
    if has_log:
        agg["mae_median"] = ("abs_error_median_variant", "mean")
    out = df.groupby(keys, observed=True).agg(**agg).reset_index()
    out["rmse"] = np.sqrt(out.pop("sq"))
    return out


def main_result_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE/RMSE/MAE per cell, with a POOLED row beside the
    regime x jump_day cells, plus relative QLIKE and the median-variant MAE.

    THE POOLED ROW IS NOT OPTIONAL. Without it the only way to get the
    headline "QLIKE by model and horizon" figure the thesis reports is to
    aggregate the four (regime x jump_day) cells by hand -- and a plain
    mean of those four cells is WRONG, because the cells hold very
    different numbers of days. Measured on this panel: a plain mean differs
    from the n_obs-weighted one by 5% at the median and 64% at the 95th
    percentile, and it reorders 8 of 16 models at h = 1 on the reference
    design. So the pooled row is computed HERE, directly from the daily
    rows (exact, not a re-weighting of cell means), and carries
    regime = jump_day = POOLED.

    `relative_qlike` is the RATIO OF MEANS against log_har at the SAME grid
    (losses.relative_qlike; log_har is a grid-only model, so its cell is
    matched by grid, horizon, regime and jump_day -- never by design_id,
    which it does not share with design-sensitive models).

    `mae_median` is losses.mae_median_variant: MAE against the UNCORRECTED
    forecast exp(log_y_hat). MAE is minimised by the conditional median and
    exp(log_y_hat) estimates it, while QLIKE/RMSE are minimised by the
    conditional mean that `y_hat` estimates -- comparing models by MAE on
    the mean-corrected forecast penalises a model for doing what QLIKE
    asks. It is NaN for level models (`har`, `mem`, `ttm_c_plus_j`), which
    have no log-space forecast and therefore no Jensen correction to undo.

    QLIKE-vs-MAE ranking disagreement is NOT flagged here: it needs
    Diebold-Mariano significance on the daily loss series
    (losses.ranking_disagreement_table)."""
    from projekat.evaluation.anova import parse_design_id
    from projekat.evaluation.losses import RELATIVE_QLIKE_BASELINE

    df = results_df
    has_mz = "qlike_recalibrated" in df.columns
    has_log = "abs_error_median_variant" in df.columns

    strata = ["model", "design_id", "horizon", "regime", "jump_day"]
    by_cell = _cell_means(df, strata, has_mz=has_mz, has_log=has_log)
    pooled = _cell_means(df, ["model", "design_id", "horizon"], has_mz=has_mz, has_log=has_log)
    pooled["regime"] = POOLED
    pooled["jump_day"] = POOLED
    out = pd.concat([by_cell, pooled[by_cell.columns]], ignore_index=True)

    # relative QLIKE: ratio of MEANS against log_har at the same grid
    out["grid"] = out["design_id"].map(lambda d: parse_design_id(d)["grid"])
    baseline = (out[out["model"] == RELATIVE_QLIKE_BASELINE]
                .set_index(["grid", "horizon", "regime", "jump_day"])["qlike"])
    keys = pd.MultiIndex.from_arrays(
        [out["grid"], out["horizon"], out["regime"], out["jump_day"]])
    base_vals = baseline.reindex(keys).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["relative_qlike"] = np.where(base_vals > 0, out["qlike"].to_numpy() / base_vals, np.nan)
    out = out.drop(columns=["grid"])

    cols = ["model", "design_id", "horizon", "regime", "jump_day",
            "qlike", "relative_qlike", "rmse", "mae", "n_obs"]
    if has_log:
        cols.insert(cols.index("mae") + 1, "mae_median")
    if has_mz:
        cols.append("qlike_recalibrated")
    return out[cols].sort_values(["model", "design_id", "horizon", "regime", "jump_day"]).reset_index(drop=True)


def mcs_membership_table(results_df: pd.DataFrame, *, alpha: float = 0.10) -> pd.DataFrame:
    """Per (design_id, horizon) cell: which models are in the MCS. Loss
    series are ALIGNED on (date, symbol), restricted to the days every model
    in the cell has, and then COLLAPSED TO A DAILY SERIES (the cross-sectional
    mean over symbols) before any test.

    The collapse is required, not cosmetic: the elimination step is a
    Diebold-Mariano test, and on a (date, symbol)-stacked series its HAC lags
    span symbols instead of days while n is inflated by the cross-section --
    every elimination test then over-rejects and the surviving set comes out
    far too narrow. See evaluation/dm_test.py's module docstring."""
    rows = []
    for (design_id, horizon), group in results_df.groupby(["design_id", "horizon"], observed=True):
        wide = group.pivot_table(index=["date", "symbol"], columns="model", values="qlike",
                                 observed=True).dropna()
        if len(wide) < 3 or wide.shape[1] < 2:
            continue
        wide = collapse_to_daily(wide)
        if len(wide) < 3:
            continue
        losses = {m: wide[m].to_numpy() for m in wide.columns}
        mcs = model_confidence_set(losses, alpha=alpha, horizon=int(horizon))
        for model_name in losses:
            rows.append({"design_id": design_id, "horizon": horizon, "model": model_name,
                         "in_mcs": model_name in mcs})
    return pd.DataFrame(rows)


def mincer_zarnowitz_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mincer-Zarnowitz over the evaluation period per (model, design,
    horizon): intercept, slope and HAC (h-1 lags) p-values for alpha = 0
    and beta = 1 -- the bias QLIKE does not isolate. Descriptive: the
    recalibration itself uses only trailing coefficients (backtransform.py)."""
    rows = []
    for (model, design_id, horizon), cell in results_df.groupby(["model", "design_id", "horizon"], observed=True):
        ok = np.isfinite(cell["y_true"].to_numpy(dtype=float)) & np.isfinite(cell["y_hat"].to_numpy(dtype=float))
        if ok.sum() < 10:
            continue
        cell = cell[ok].sort_values(["symbol", "date"])
        res = mincer_zarnowitz(cell["y_true"].to_numpy(), cell["y_hat"].to_numpy(), horizon=int(horizon))
        rows.append({"model": model, "design_id": design_id, "horizon": horizon, "n_obs": int(ok.sum()),
                     "mz_intercept": res.intercept, "mz_slope": res.slope,
                     "intercept_p": res.intercept_pvalue, "slope_eq_1_p": res.slope_pvalue})
    return pd.DataFrame(rows)
