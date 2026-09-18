"""The main h x regime x jump_day result table, plus MCS membership."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.evaluation.losses import mae, qlike, rmse
from projekat.evaluation.mcs import model_confidence_set
from projekat.evaluation.mincer_zarnowitz import mincer_zarnowitz
from projekat.evaluation.stratify import tag_jump_day, tag_regime


def attach_losses(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["qlike"] = qlike(df["y_true"].to_numpy(), df["y_hat"].to_numpy())
    df["abs_error"] = (df["y_true"] - df["y_hat"]).abs()
    df["sq_error"] = (df["y_true"] - df["y_hat"]) ** 2
    return df


def attach_mz_recalibration(test_results: pd.DataFrame, val_results: pd.DataFrame) -> pd.DataFrame:
    """PARALELNA sekundarna figura (nikad zamena za primarni QLIKE):
    Mincer-Zarnowitz koeficijenti (intercept, slope) ocenjeni ISKLJUČIVO iz
    van-uzorka predikcija na validacionom skupu svakog prozora
    (val_results, iz runner.py::run_all), po (model, design_id, horizon,
    window_id) ćeliji -- nikad iz test skupa, što bi bilo unapredno
    gledanje (rekalibracija testa koristeći koeficijente ocenjene na tom
    istom test skupu). Test predikcije se rekalibrišu primenom te
    (intercept, slope) na test y_hat: y_hat_recalibrated = intercept +
    slope * y_hat. Vraća test_results sa dodatim kolonama
    `y_hat_recalibrated`, `qlike_recalibrated`, `mz_intercept`, `mz_slope`
    -- sirov i rekalibrisan QLIKE se izveštavaju JEDAN PORED DRUGOG, nikad
    se rekalibrisan ne prijavljuje kao da je zamenio primarni."""
    if test_results.empty or val_results.empty:
        out = test_results.copy()
        out["y_hat_recalibrated"] = np.nan
        out["qlike_recalibrated"] = np.nan
        out["mz_intercept"] = np.nan
        out["mz_slope"] = np.nan
        return out

    cell_cols = ["model", "design_id", "horizon", "window_id"]
    mz_rows = []
    for keys, group in val_results.groupby(cell_cols, observed=True):
        model, design_id, horizon, window_id = keys
        if len(group) < 3:
            continue
        result = mincer_zarnowitz(group["y_true"].to_numpy(), group["y_hat"].to_numpy(), horizon=int(horizon))
        mz_rows.append(
            {
                "model": model,
                "design_id": design_id,
                "horizon": horizon,
                "window_id": window_id,
                "mz_intercept": result.intercept,
                "mz_slope": result.slope,
            }
        )
    mz_df = pd.DataFrame(mz_rows)

    out = test_results.merge(mz_df, on=cell_cols, how="left")
    out["y_hat_recalibrated"] = out["mz_intercept"] + out["mz_slope"] * out["y_hat"]
    out["qlike_recalibrated"] = qlike(out["y_true"].to_numpy(), out["y_hat_recalibrated"].to_numpy())
    return out


def attach_strata(results_df: pd.DataFrame) -> pd.DataFrame:
    """Tags against the FIXED regime boundary (Decision C6) -- never a
    per-call training-window distribution."""
    df = results_df.copy()
    df["regime"] = tag_regime(df)
    df["jump_day"] = tag_jump_day(df)
    return df


def main_result_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, design_id, horizon, regime, jump_day) cell,
    with mean QLIKE/RMSE/MAE.

    QLIKE-vs-MAE ranking disagreement is NOT flagged here: it is defined
    per model PAIR and requires Diebold-Mariano significance on the daily
    loss series (see losses.py::ranking_disagreement), which this already-
    aggregated table no longer has. Call
    `losses.ranking_disagreement_table(daily_losses)` on the un-aggregated
    per-day frame instead -- flagging every mean-level reversal here would
    report rounding noise as disagreement, which is exactly what the DM
    requirement exists to prevent.

    If `results_df` carries a `qlike_recalibrated` column (from
    attach_mz_recalibration), the output also reports mean
    `qlike_recalibrated` alongside raw `qlike` -- a PARALLEL secondary
    figure, never a replacement for the primary QLIKE column."""
    df = results_df.copy()
    has_mz = "qlike_recalibrated" in df.columns
    grouped = df.groupby(["model", "design_id", "horizon", "regime", "jump_day"], observed=True)

    rows = []
    for keys, group in grouped:
        model, design_id, horizon, regime, jump_day = keys
        row = {
            "model": model,
            "design_id": design_id,
            "horizon": horizon,
            "regime": regime,
            "jump_day": jump_day,
            "qlike": group["qlike"].mean(),
            "rmse": rmse(group["y_true"].to_numpy(), group["y_hat"].to_numpy()),
            "mae": mae(group["y_true"].to_numpy(), group["y_hat"].to_numpy()),
            "n_obs": len(group),
        }
        if has_mz:
            row["qlike_recalibrated"] = group["qlike_recalibrated"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def mcs_membership_table(results_df: pd.DataFrame, *, alpha: float = 0.10) -> pd.DataFrame:
    """Per (design_id, horizon) cell: which models are in the MCS."""
    rows = []
    for (design_id, horizon), group in results_df.groupby(["design_id", "horizon"], observed=True):
        losses = {m: g["qlike"].to_numpy() for m, g in group.groupby("model", observed=True)}
        mcs = model_confidence_set(losses, alpha=alpha, horizon=horizon)
        for model_name in losses:
            rows.append(
                {
                    "design_id": design_id,
                    "horizon": horizon,
                    "model": model_name,
                    "in_mcs": model_name in mcs,
                }
            )
    return pd.DataFrame(rows)
