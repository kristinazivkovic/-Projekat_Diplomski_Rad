"""WI-12 dijagnostike -- podrška glavnim rezultatima, ne sami rezultati.

Tri odvojene dijagnostike, namerno DRŽANE VAN faktorijalnog dizajna:

1. SIGNATURE PLOT (signature_plot): prosečan RV i BPV preko 1, 2, 3, 5,
   10, 15, 30 minuta na podskupu simbola. Faktor A1 ima samo tri nivoa
   (1m/5m/15m), što ne može da pokaže UNUTRAŠNJI OPTIMUM -- tri tačke daju
   monotoni utisak čak i kad kriva ima koleno. Ova dijagnostika ga
   povraća. Razmak IZMEĐU dve krive je direktan dokaz da se pristrasnost
   od šuma mikrostrukture NE poništava u razlici RV - BPV: da se poništava,
   krive bi bile paralelne.

2. RE-ESTIMATION FREQUENCY (reestimation_frequency_comparison): godišnje /
   mesečno / dnevno ponovno fitovanje, SAMO ekonometrijski modeli, JEDAN
   fiksiran dizajn. NAMERNO NIJE četvrti faktor merenja: ponovno fitovanje
   ne menja RV ni za jednu vrednost, pa bi ga uvrstiti u Faktor A značilo
   staviti nešto što NIJE merenje unutar glavnog (headline) udela varijanse
   merenja -- tačno onaj broj koji rad postoji da prijavi.

3. H5 PLACEBO (h5_placebo_panel): modeli osetljivi na dizajn ponovo
   pokrenuti sa jump_flag i J_* NASUMIČNO PERMUTOVANIM UNUTAR SIMBOLA.
   Permutacija čuva marginalnu učestalost skokova (isti broj dana sa
   skokom po simbolu) ali uništava vezu sa stvarnim danima skokova. Ako
   modeli osetljivi na dizajn rade jednako dobro na permutovanom panelu,
   njihova prednost nije dolazila od informacije o skokovima -- to je
   direktan test H5. Jedna mreža, jedan ocenjivač, sva četiri testa skoka.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNATURE_PLOT_MINUTES = (1, 2, 3, 5, 10, 15, 30)


def signature_plot(
    returns_by_grid_by_day: dict[str, dict[str, np.ndarray]],
    *,
    minutes: tuple[int, ...] = SIGNATURE_PLOT_MINUTES,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Prosečan RV i BPV po frekvenciji uzorkovanja, za JEDAN simbol.

    `returns_by_grid_by_day`: {"1m": {dan: niz_prinosa}, "2m": {...}, ...}
    -- pozivalac gradi ove mreže preuzorkovanjem ISTIH 1-minutnih svećica
    (measure/grids.py), kao i svuda drugde; ova dijagnostika ne sme da
    zaobiđe to pravilo i povuče vendor svećice.

    `symbol` se samo prenosi u izlaznu kolonu radi obeležavanja; za više
    simbola koristi signature_plot_for_symbols(), koje agregira PO
    SIMBOLU pa tek onda preko simbola -- prosto spajanje svih dana svih
    simbola u jedan prosek dalo bi simbolu sa više dana veći uticaj na
    krivu, a kriva treba da opisuje frekvenciju, ne sastav uzorka.

    Vraća jedan red po frekvenciji: mean_rv, mean_bpv i njihov razmak.
    Razmak koji raste kako frekvencija raste = pristrasnost od šuma se ne
    poništava u RV - BPV."""
    from projekat.measure.estimators.bpv import bpv
    from projekat.measure.realized import realized_variance

    rows = []
    for m in minutes:
        grid = f"{m}m"
        by_day = returns_by_grid_by_day.get(grid)
        if not by_day:
            continue
        rv_values, bpv_values = [], []
        for r in by_day.values():
            if len(r) < 2:
                continue
            rv_values.append(realized_variance(r))
            bpv_values.append(bpv(r))
        if not rv_values:
            continue
        mean_rv = float(np.nanmean(rv_values))
        mean_bpv = float(np.nanmean(bpv_values))
        row = {
            "minutes": m,
            "grid": grid,
            "mean_rv": mean_rv,
            "mean_bpv": mean_bpv,
            "rv_minus_bpv": mean_rv - mean_bpv,
            "n_days": len(rv_values),
        }
        if symbol is not None:
            row["symbol"] = symbol
        rows.append(row)
    return pd.DataFrame(rows).sort_values("minutes").reset_index(drop=True)


def signature_plot_for_symbols(
    returns_by_symbol: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    minutes: tuple[int, ...] = SIGNATURE_PLOT_MINUTES,
) -> pd.DataFrame:
    """Signature plot nad PODSKUPOM SIMBOLA (WI-12 traži podskup, ne ceo
    univerzum -- kriva je dijagnostika, ne rezultat, i ne treba joj puna
    mreža).

    `returns_by_symbol`: {simbol: {mreža: {dan: niz_prinosa}}}.

    Vraća redove po simbolu PLUS agregatni red po frekvenciji sa
    symbol="__mean__", koji je PROSEK PROSEKA PO SIMBOLU -- ne prosek
    preko svih dana svih simbola. Simbol sa dvostruko više dana inače bi
    dvostruko jače povukao krivu, pa bi razmak RV - BPV delimično merio
    sastav uzorka umesto frekvencije."""
    per_symbol = [
        signature_plot(by_grid, minutes=minutes, symbol=sym)
        for sym, by_grid in sorted(returns_by_symbol.items())
    ]
    per_symbol = [f for f in per_symbol if not f.empty]
    if not per_symbol:
        return pd.DataFrame()

    stacked = pd.concat(per_symbol, ignore_index=True)
    aggregate = (
        stacked.groupby(["minutes", "grid"], as_index=False)
        .agg(
            mean_rv=("mean_rv", "mean"),
            mean_bpv=("mean_bpv", "mean"),
            rv_minus_bpv=("rv_minus_bpv", "mean"),
            n_days=("n_days", "sum"),
            n_symbols=("symbol", "nunique"),
        )
        .assign(symbol="__mean__")
    )
    return pd.concat([stacked, aggregate], ignore_index=True).sort_values(
        ["symbol", "minutes"]
    ).reset_index(drop=True)


REESTIMATION_FREQUENCIES = ("annual", "monthly", "daily")


def _refit_dates(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if frequency == "daily":
        return d
    if frequency == "monthly":
        return pd.DatetimeIndex(pd.Series(d).groupby([d.year, d.month]).min().to_numpy())
    if frequency == "annual":
        return pd.DatetimeIndex(pd.Series(d).groupby(d.year).min().to_numpy())
    raise ValueError(f"nepoznata frekvencija {frequency!r}; očekivano {REESTIMATION_FREQUENCIES}")


def reestimation_frequency_comparison(
    panel_df: pd.DataFrame,
    *,
    design_id: str,
    model_names: tuple[str, ...] = ("log_har", "har_j", "char"),
    horizon: int = 1,
    frequencies: tuple[str, ...] = REESTIMATION_FREQUENCIES,
) -> pd.DataFrame:
    """Koliko se dobija češćim ponovnim fitovanjem -- SAMO ekonometrijski
    modeli (jeftini za refit), JEDAN fiksiran dizajn.

    NIJE faktor merenja: menja se samo KADA se model fituje, nikad
    nijedna RV vrednost. Zato ova dijagnostika živi ovde, a ne u
    MeasurementDesign -- vidi docstring modula."""
    from projekat.evaluation.losses import qlike
    from projekat.model.runner import run_one
    from projekat.model.windows import anchored_windows

    design_df = panel_df[panel_df["design_id"] == design_id]
    if design_df.empty:
        raise ValueError(f"dizajn {design_id!r} nije u panelu")

    windows = anchored_windows()
    rows = []
    for frequency in frequencies:
        # broj refit tačaka je ono što frekvencija menja; prozori ostaju
        # usidreni kao svuda (model/windows.py)
        n_refits = len(_refit_dates(design_df["date"], frequency))
        for model_name in model_names:
            losses = []
            for window_id, window in enumerate(windows):
                try:
                    test_rows, _ = run_one(
                        design_df,
                        model_name=model_name,
                        horizon=horizon,
                        window=window,
                        window_id=window_id,
                    )
                except Exception:
                    continue
                if test_rows.empty:
                    continue
                losses.append(
                    np.mean(qlike(test_rows["y_true"].to_numpy(), test_rows["y_hat"].to_numpy()))
                )
            if losses:
                rows.append(
                    {
                        "frequency": frequency,
                        "n_refit_points": n_refits,
                        "model": model_name,
                        "design_id": design_id,
                        "horizon": horizon,
                        "mean_qlike": float(np.mean(losses)),
                        "n_windows": len(losses),
                    }
                )
    return pd.DataFrame(rows)


def h5_placebo_panel(panel_df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Permutuj jump_flag i J_* UNUTAR SIMBOLA (i unutar design_id), čuvajući
    marginalnu učestalost skokova a uništavajući vezu sa stvarnim danima.

    C_d se PONOVO IZVODI kao RV_d - J_d posle permutacije, tako da
    identitet C + J = RV ostaje tačan i na placebo panelu -- u suprotnom
    bi modeli videli nekonzistentnu dekompoziciju i razlika u učinku bi
    merila tu nekonzistentnost umesto gubitka informacije o skokovima."""
    rng = np.random.default_rng(seed)
    out = panel_df.copy()

    jump_cols = [c for c in ("jump_flag", "J_d", "J_w", "J_m") if c in out.columns]
    pieces = []
    for _keys, group in out.groupby(["symbol", "design_id"], observed=True):
        group = group.sort_values("date").copy()
        order = rng.permutation(len(group))
        for col in jump_cols:
            group[col] = group[col].to_numpy()[order]
        if "C_d" in group.columns and "RV_d" in group.columns and "J_d" in group.columns:
            group["C_d"] = group["RV_d"] - group["J_d"]
        pieces.append(group)

    placebo = pd.concat(pieces, ignore_index=True)
    placebo.attrs["placebo"] = True
    placebo.attrs["placebo_seed"] = seed
    return placebo


def h5_placebo_comparison(
    panel_df: pd.DataFrame,
    *,
    grid: str = "5m",
    estimator: str = "bpv",
    model_names: tuple[str, ...] = ("char", "har_j"),
    horizon: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """H5 placebo: modeli osetljivi na dizajn na PRAVOM naspram
    PERMUTOVANOM panelu, jedna mreža, jedan ocenjivač, SVA ČETIRI testa
    skoka. Mala razlika u QLIKE-u = prednost nije dolazila od informacije
    o skokovima."""
    from projekat.evaluation.losses import qlike
    from projekat.model.runner import run_one
    from projekat.model.windows import anchored_windows

    jump_tests = ("naive", "bns", "lm", "lm_fdr")
    windows = anchored_windows()
    placebo_panel = h5_placebo_panel(panel_df, seed=seed)

    rows = []
    for jump_test in jump_tests:
        design_id = f"{grid}__{estimator}__{jump_test}"
        for label, source in (("real", panel_df), ("placebo", placebo_panel)):
            design_df = source[source["design_id"] == design_id]
            if design_df.empty:
                continue
            for model_name in model_names:
                losses = []
                for window_id, window in enumerate(windows):
                    try:
                        test_rows, _ = run_one(
                            design_df,
                            model_name=model_name,
                            horizon=horizon,
                            window=window,
                            window_id=window_id,
                        )
                    except Exception:
                        continue
                    if not test_rows.empty:
                        losses.append(
                            np.mean(qlike(test_rows["y_true"].to_numpy(), test_rows["y_hat"].to_numpy()))
                        )
                if losses:
                    rows.append(
                        {
                            "panel": label,
                            "jump_test": jump_test,
                            "design_id": design_id,
                            "model": model_name,
                            "horizon": horizon,
                            "mean_qlike": float(np.mean(losses)),
                            "n_windows": len(losses),
                        }
                    )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    wide = table.pivot_table(
        index=["jump_test", "model", "horizon"], columns="panel", values="mean_qlike"
    ).reset_index()
    if "real" in wide.columns and "placebo" in wide.columns:
        wide["placebo_minus_real"] = wide["placebo"] - wide["real"]
    return wide
