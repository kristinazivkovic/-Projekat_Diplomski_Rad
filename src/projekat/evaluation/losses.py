"""Nivo 1: QLIKE (primarna metrika), RMSE, MAE (sekundarne, radi
uporedivosti sa literaturom), relativni QLIKE. QLIKE ima prednost pri
neslaganju, prema sopstvenom uputstvu iz rada -- eksplicitno se prijavljuje,
nikad tiho ne rešava.

MAE i QLIKE/RMSE ne ciljaju istu veličinu: QLIKE i RMSE su minimizovani
USLOVNOM SREDINOM, a MAE USLOVNOM MEDIJANOM. Zato mae_median_variant()
postoji kao provera robusnosti -- prognoza bez Jensenove korekcije
(a = 0) je bliža uslovnoj medijani nego uslovnoj sredini, pa je to
ispravna prognoza za MAE, dok je korigovana ispravna za QLIKE/RMSE.
Poređenje modela po MAE nad sredinski-korigovanim prognozama kažnjava
model zbog toga što radi tačno ono što QLIKE traži.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.evaluation.dm_test import diebold_mariano

RELATIVE_QLIKE_BASELINE = "log_har"


def qlike(y_true: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    ratio = y_true / y_hat
    return ratio - np.log(ratio) - 1.0


def rmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


def mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_hat)))


def mae_median_variant(y_true: np.ndarray, log_y_hat: np.ndarray) -> float:
    """MAE nad NEKORIGOVANOM prognozom (Jensenova korekcija a = 0), kao
    provera robusnosti. MAE je minimizovan uslovnom MEDIJANOM, a
    exp(log_y_hat) bez korekcije je upravo ocena uslovne medijane pod
    log-normalnošću, dok je exp(log_y_hat + a) ocena uslovne SREDINE (koju
    traže QLIKE i RMSE). Merenje MAE nad sredinski-korigovanom prognozom
    meša dva različita ciljna funkcionala -- ovo je ispravan MAE partner."""
    return float(np.mean(np.abs(y_true - np.exp(log_y_hat))))


def relative_qlike(model_qlike_mean: float, baseline_qlike_mean: float) -> float:
    """ODNOS SREDINA (ratio of means), NE sredina odnosa:

        relative_qlike = mean(QLIKE_model) / mean(QLIKE_log_har)

    obe sredine uzete preko SPOJENOG (pooled) test perioda. Ovo NIJE
    zamenljivo sa mean(QLIKE_model / QLIKE_log_har): sredina odnosa
    eksplodira čim gubitak referentnog modela na jednom jedinom danu priđe
    nuli (QLIKE -> 0 kad je prognoza tog dana skoro savršena, što se
    dešava), pa jedan dan može da preuzme ceo izveštani broj. Odnos
    sredina nema taj problem jer imenilac agregira ceo uzorak.

    Referentni model je `log_har` (RELATIVE_QLIKE_BASELINE) -- standardan
    log-HAR iz literature. `har` (bez prefiksa) je poseban NNLS-u-nivoima
    model sa drugačijim mehanizmom pozitivnosti i NIJE referentni model.
    0.95 znači 5% bolje od log_har."""
    if not baseline_qlike_mean:
        return float("nan")
    return model_qlike_mean / baseline_qlike_mean


def ranking_disagreement(
    qlike_a: np.ndarray,
    qlike_b: np.ndarray,
    mae_a: np.ndarray,
    mae_b: np.ndarray,
    *,
    horizon: int = 1,
    alpha: float = 0.05,
) -> bool:
    """True ako QLIKE i MAE poređaju dva modela (a, b) OBRNUTO **i** je ta
    razlika Diebold-Mariano značajna pri `alpha` po OBE metrike.

    Nizovi su dnevni gubici, poravnati po danu, unutar jedne (horizont,
    dizajn) ćelije. Zahtev za DM značajnošću je suštinski: obeležavanje
    svakog obrtanja poretka zatrpalo bi izlaz šumom -- poenta zastavice je
    da se QLIKE i MAE ZAISTA ne slažu, a ne da se razlikuju na nivou
    zaokruživanja. Kada se ne slažu, QLIKE ima prednost, ali neslaganje
    mora biti eksplicitno prijavljeno, nikad tiho rešeno."""
    qlike_prefers_a = np.mean(qlike_a) < np.mean(qlike_b)
    mae_prefers_a = np.mean(mae_a) < np.mean(mae_b)
    if qlike_prefers_a == mae_prefers_a:
        return False  # slažu se oko poretka -- nema šta da se prijavi

    qlike_dm = diebold_mariano(qlike_a, qlike_b, h=horizon)
    mae_dm = diebold_mariano(mae_a, mae_b, h=horizon)
    return bool(qlike_dm.p_value < alpha and mae_dm.p_value < alpha)


def ranking_disagreement_table(
    daily_losses: pd.DataFrame,
    *,
    alpha: float = 0.05,
    qlike_col: str = "qlike",
    mae_col: str = "abs_error",
) -> pd.DataFrame:
    """Po (horizont, design_id) ćeliji: svi parovi modela kod kojih se
    QLIKE i MAE poredak obrću uz DM značajnost po obe metrike.

    `daily_losses` nosi jedan red po (model, design_id, horizon, symbol,
    date) sa dnevnim gubicima. Vraća jedan red po obeleženom paru --
    prazan okvir znači da nigde nema pravog neslaganja."""
    rows = []
    for (horizon, design_id), cell in daily_losses.groupby(["horizon", "design_id"], observed=True):
        wide_q = cell.pivot_table(index=["date", "symbol"], columns="model", values=qlike_col)
        wide_m = cell.pivot_table(index=["date", "symbol"], columns="model", values=mae_col)
        common = wide_q.dropna().index.intersection(wide_m.dropna().index)
        if len(common) < 3:
            continue
        wide_q, wide_m = wide_q.loc[common], wide_m.loc[common]

        models = sorted(wide_q.columns)
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                flagged = ranking_disagreement(
                    wide_q[a].to_numpy(), wide_q[b].to_numpy(),
                    wide_m[a].to_numpy(), wide_m[b].to_numpy(),
                    horizon=int(horizon), alpha=alpha,
                )
                if flagged:
                    rows.append(
                        {
                            "horizon": horizon,
                            "design_id": design_id,
                            "model_a": a,
                            "model_b": b,
                            "qlike_prefers": a if wide_q[a].mean() < wide_q[b].mean() else b,
                            "mae_prefers": a if wide_m[a].mean() < wide_m[b].mean() else b,
                            "resolved_in_favour_of": a if wide_q[a].mean() < wide_q[b].mean() else b,
                        }
                    )
    return pd.DataFrame(rows)
