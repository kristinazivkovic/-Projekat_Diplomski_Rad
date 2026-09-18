"""Orkestrira: za svaku RAZLIČITU kombinaciju (model, dizajn) implicirаnu
sa depends_on, za svaki od 11 prozora, svaki horizont, svako seme (seed)
ako je stohastičko -- treniraj, predvidi, transformiši unazad, sačuvaj
jedan red po prognozi. Razrešava modele iz registra po imenu -- nikad ne
uvozi (import) konkretan modul modela.

Nabrajanje zasnovano na depends_on (Odluka C2): model koji zavisi samo od
"mreže" (grid) se trenira jednom po mreži (3 dizajna), a ne jednom po
punom design_id (do 36) -- treniranje 12 puta po mreži na bajt-identičnim
ulazima bi i trošilo računske resurse i kvarilo udele varijanse
ocenjivač/test_skoka u ANOVA analizi dupliranim, međusobno zavisnim
redovima.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401  (registracija)
from projekat.model.backtransform import backtransform
from projekat.model.registry import MODELS
from projekat.model.targets import TARGET_COL, add_target
from projekat.model.windows import Window, anchored_windows, split_with_validation

HORIZONS = (1, 5, 22)
STOCHASTIC_SEEDS = (0, 1, 2, 3, 4)


def _representative_designs(panel_df: pd.DataFrame, depends_on: frozenset[str]) -> pd.DataFrame:
    """Jedan predstavnički design_id po različitoj kombinaciji faktora u
    depends_on. Npr. depends_on={"grid"} -> jedan design_id po mreži
    (3 reda); depends_on={"grid","estimator","jump_test"} -> svaki
    design_id (do 36 redova)."""
    from projekat.evaluation.anova import parse_design_id

    designs = panel_df[["design_id"]].drop_duplicates().copy()
    parsed = designs["design_id"].apply(parse_design_id).apply(pd.Series)
    designs = pd.concat([designs, parsed], axis=1)

    if not depends_on:
        # model ne zavisi ni od čega specifičnog za dizajn: uzmi samo jedan design_id
        return designs.iloc[[0]]

    key_cols = sorted(depends_on)
    return designs.drop_duplicates(subset=key_cols)


def run_one(
    design_df: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    window: Window,
    seed: int | None = None,
    window_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treniraj jedan model na podacima za treniranje jednog prozora
    (isključujući validacioni skup), predvidi na test prozoru,
    transformiši unazad, vrati (test_rows, val_rows) -- test_rows je
    primarni rezultat (jedan red po (datum, simbol) prognozi), val_rows su
    van-uzorka predikcije na validacionom skupu istog prozora, u istom
    obliku, korišćene ISKLJUČIVO za Mincer-Zarnowitz koeficijente
    (evaluation/mincer_zarnowitz.py) -- nikad za podešavanje
    hiperparametara niti kao dodatna evaluacija."""
    model = MODELS[model_name]
    df = add_target(design_df, horizon)
    fit_train, val_fold, test_df = split_with_validation(df, window)

    if model.is_sequence_model:
        return _run_sequence_model(model, fit_train, val_fold, test_df, horizon=horizon, seed=seed, window_id=window_id)

    fitted = model.fit(fit_train, val_fold, horizon, seed=seed)

    def _predict_rows(frame: pd.DataFrame) -> pd.DataFrame:
        X = model.features(frame)
        valid = X.notna().all(axis=1)
        pred = fitted.predict(X.loc[valid])

        rows = frame.loc[valid, ["date", "symbol", "design_id", "RV_d", "jump_flag", "RV_m", "r_d"]].copy()
        rows["model"] = model_name
        rows["horizon"] = horizon
        rows["seed"] = seed
        rows["window_id"] = window_id
        if model.predicts_levels:
            # positivity is structural (e.g. NNLS non-negative coefficients),
            # so `pred` is already y_{t,h} in levels -- backtransform() would
            # be wrong here (it assumes a log-space prediction).
            rows["log_y_hat"] = np.nan
            rows["backtransform_correction"] = np.nan
            rows["y_hat"] = pred
        else:
            rows["log_y_hat"] = pred
            rows["backtransform_correction"] = fitted.backtransform_correction
            rows["y_hat"] = backtransform(pred, fitted.backtransform_correction)
        rows["y_true"] = np.exp(frame.loc[valid, TARGET_COL].to_numpy())
        return rows

    return _predict_rows(test_df), _predict_rows(val_fold)


def _run_sequence_model(
    model, fit_train: pd.DataFrame, val_fold: pd.DataFrame, test_df: pd.DataFrame, *, horizon: int, seed: int | None, window_id: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fitted = model.fit(fit_train, val_fold, horizon, seed=seed)

    def _predict_rows(frame: pd.DataFrame) -> pd.DataFrame:
        sequences, origin = model.build_sequences(frame)
        if len(sequences) == 0:
            return pd.DataFrame()

        pred = fitted.predict(sequences)

        targets_log = model.targets_for_sequences(frame, origin, horizon)
        valid = ~np.isnan(targets_log)

        frame_sorted = frame.sort_values("date").reset_index(drop=True)
        rows = frame_sorted.iloc[origin[valid]][["date", "symbol", "design_id", "RV_d", "jump_flag", "RV_m", "r_d"]].copy()
        rows["model"] = model.name
        rows["horizon"] = horizon
        rows["seed"] = seed
        rows["window_id"] = window_id
        if model.predicts_levels:
            rows["log_y_hat"] = np.nan
            rows["backtransform_correction"] = np.nan
            rows["y_hat"] = pred[valid]
        else:
            rows["log_y_hat"] = pred[valid]
            rows["backtransform_correction"] = fitted.backtransform_correction
            rows["y_hat"] = backtransform(pred[valid], fitted.backtransform_correction)
        rows["y_true"] = np.exp(targets_log[valid])
        return rows

    return _predict_rows(test_df), _predict_rows(val_fold)


def run_all(
    panel_df: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    horizons=HORIZONS,
    windows: list[Window] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Puna mreža: svaki model x horizont x prozor x (različit dizajn po
    depends_on) x (5 semena za stohastičke modele). Vraća (test_results,
    val_results) -- val_results je van-uzorka validacioni skup svakog
    prozora, u istom obliku kao test_results, korišćen isključivo za
    Mincer-Zarnowitz koeficijente nizvodno (evaluation/mincer_zarnowitz.py),
    nikad kao dodatna evaluacija."""
    names = model_names or list(MODELS.keys())
    wins = windows or anchored_windows()

    # Provera kontaminacije predtreniranjem je USLOV za svaki ttm_* broj:
    # zapiši artefakt u results/ PRE nego što ijedna ttm_* prognoza nastane,
    # bez obzira na ishod (vidi model/foundation/contamination.py).
    if any(n.startswith("ttm_") or n == "log_har_ttm" for n in names) and wins:
        from projekat.model.foundation.contamination import check_and_write

        check_and_write(wins[0].test_start, wins[-1].test_end)

    test_frames = []
    val_frames = []
    for model_name in names:
        model = MODELS[model_name]
        rep_designs = _representative_designs(panel_df, model.depends_on)

        for _, design_row in rep_designs.iterrows():
            design_df = panel_df[panel_df["design_id"] == design_row["design_id"]]

            for horizon in horizons:
                for window_id, window in enumerate(wins):
                    if model.stochastic:
                        for seed in STOCHASTIC_SEEDS:
                            test_rows, val_rows = run_one(
                                design_df, model_name=model_name, horizon=horizon, window=window, seed=seed, window_id=window_id
                            )
                            test_frames.append(test_rows)
                            val_frames.append(val_rows)
                    else:
                        test_rows, val_rows = run_one(
                            design_df, model_name=model_name, horizon=horizon, window=window, window_id=window_id
                        )
                        test_frames.append(test_rows)
                        val_frames.append(val_rows)

    test_results = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    val_results = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
    return test_results, val_results


def collapse_seeds(results_df: pd.DataFrame) -> pd.DataFrame:
    """Odluka C3: PRIMARNA vrednost po (model, design_id, horizont, datum,
    simbol) je prosek QLIKE-relevantnih veličina po semenu (seed) --
    konkretno, za stohastičke modele ovo i dalje vraća jedan red po semenu
    (tako da QLIKE/funkcije gubitka primenjene nizvodno, pa zatim
    usrednjene preko semena, daju ispravan "prosek gubitka po semenu", a ne
    "gubitak prosečne prognoze"). Ova funkcija je bez efekta (no-op) za
    determinističke modele (seed je None, već jedan red) i ovde je
    dokumentovana da tačka usrednjavanja bude nedvosmislena: usrednji
    GUBITKE preko semena, a ne PROGNOZE, za primarnu vrednost."""
    return results_df


def ensemble_forecast(results_df: pd.DataFrame) -> pd.DataFrame:
    """SEKUNDARNA, eksplicitno obeležena ensemble vrednost iz Odluke C3:
    prognoza usrednjena preko semena NA LOG SKALI PRE transformacije
    unazad -- usrednjava se `log_y_hat` (sirova log-predikcija modela,
    pre backtransform() korekcije), nikad `log(y_hat)` (već
    transformisana unazad prognoza). Ovo poslednje bi bilo pogrešno: svako
    seme nosi sopstvenu backtransform_correction, pa bi log(y_hat) =
    log_y_hat + correction već ugradio korekciju po semenu pre usrednjavanja,
    duplirajući je. Umesto toga, backtransform() se poziva TAČNO JEDNOM na
    kraju, na usrednjenom log_y_hat sa usrednjenom backtransform_correction
    preko semena -- ne po semenu. Ovo je zaista drugačija,
    bolje-performirajuća veličina od bilo kog pojedinačnog semena i nikad
    ne sme da zameni primarnu vrednost proseka gubitka po semenu."""
    stochastic_rows = results_df[results_df["seed"].notna()]
    if stochastic_rows.empty:
        return pd.DataFrame()

    group_cols = ["model", "design_id", "horizon", "date", "symbol"]
    grouped = stochastic_rows.groupby(group_cols, observed=True)

    out = grouped.agg(
        y_true=("y_true", "first"),
        log_y_hat_mean=("log_y_hat", "mean"),
        correction_mean=("backtransform_correction", "mean"),
    ).reset_index()
    out["y_hat_ensemble"] = backtransform(out["log_y_hat_mean"].to_numpy(), out["correction_mean"].to_numpy())
    return out
