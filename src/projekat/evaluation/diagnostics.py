"""WI-12 dijagnostike -- podrška glavnim rezultatima, ne sami rezultati.

Tri odvojene dijagnostike, namerno DRŽANE VAN faktorijalnog dizajna:

1. SIGNATURE PLOT (signature_plot): prosečan RV i BPV preko 1, 2, 3, 5,
   10, 15, 30 minuta na podskupu simbola. Faktor A1 ima samo tri nivoa,
   što ne može da pokaže UNUTRAŠNJI OPTIMUM. Razmak IZMEĐU dve krive je
   direktan dokaz da se pristrasnost od šuma mikrostrukture NE poništava u
   razlici RV - BPV.

2. UČESTALOST PONOVNE OCENE (reestimation_frequency_comparison): ista
   walk-forward šema sa ponovnom ocenom svakih 252 / 21 / 1 dan trgovanja,
   SAMO ekonometrijski modeli, JEDAN dizajn. NIJE četvrti faktor merenja:
   ne menja nijednu RV vrednost.

3. H5 PLACEBO (h5_placebo_test): HAR-J i CHAR ponovo pokrenuti na panelu
   na kom su indikator skoka i J_d NASUMIČNO PERMUTOVANI UNUTAR SIMBOLA
   (ista permutacija za oba), N_PERMUTATIONS = 100 puta. Permutacija čuva
   marginalnu učestalost skokova, a uništava vezu sa stvarnim danima.
   C_d = RV_d - J_d i HAR agregati C_w/C_m/J_w/J_m se PONOVO RAČUNAJU iz
   permutovanog J_d (istim pravilom kliznog prozora kao u sloju merenja),
   da bi dekompozicija ostala konzistentna.
   Statistika: dQLIKE = prosečni QLIKE modela - prosečni QLIKE log_har,
   na istim (dan, simbol) redovima perioda evaluacije, h = 1. Stvarna
   vrednost se poredi sa raspodelom 100 permutovanih; p = (1 + #{dQLIKE_perm
   <= dQLIKE_stvarno}) / (1 + 100). Ako je stvarna prednost uobičajena
   među permutacijama, ona nije dolazila od informacije o skokovima.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

SIGNATURE_PLOT_MINUTES = (1, 2, 3, 5, 10, 15, 30)
N_PERMUTATIONS = 100
PLACEBO_MODELS = ("har_j", "char")
PLACEBO_BASELINE = "log_har"


def signature_plot(
    returns_by_grid_by_day: dict[str, dict[str, np.ndarray]],
    *,
    minutes: tuple[int, ...] = SIGNATURE_PLOT_MINUTES,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Prosečan RV i BPV po frekvenciji uzorkovanja, za JEDAN simbol.
    `returns_by_grid_by_day`: {"1m": {dan: niz_prinosa}, "2m": {...}, ...},
    izgrađeno preuzorkovanjem ISTIH 1-minutnih svećica (measure/grids.py)."""
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
        row = {"minutes": m, "grid": grid, "mean_rv": mean_rv, "mean_bpv": mean_bpv,
               "rv_minus_bpv": mean_rv - mean_bpv, "n_days": len(rv_values)}
        if symbol is not None:
            row["symbol"] = symbol
        rows.append(row)
    return pd.DataFrame(rows).sort_values("minutes").reset_index(drop=True)


def signature_plot_for_symbols(
    returns_by_symbol: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    minutes: tuple[int, ...] = SIGNATURE_PLOT_MINUTES,
) -> pd.DataFrame:
    """Signature plot nad podskupom simbola, plus agregatni red po
    frekvenciji (symbol="__mean__") koji je PROSEK PROSEKA PO SIMBOLU -- ne
    prosek preko svih dana svih simbola."""
    per_symbol = [signature_plot(by_grid, minutes=minutes, symbol=sym)
                  for sym, by_grid in sorted(returns_by_symbol.items())]
    per_symbol = [f for f in per_symbol if not f.empty]
    if not per_symbol:
        return pd.DataFrame()
    stacked = pd.concat(per_symbol, ignore_index=True)
    aggregate = (
        stacked.groupby(["minutes", "grid"], as_index=False)
        .agg(mean_rv=("mean_rv", "mean"), mean_bpv=("mean_bpv", "mean"),
             rv_minus_bpv=("rv_minus_bpv", "mean"), n_days=("n_days", "sum"),
             n_symbols=("symbol", "nunique"))
        .assign(symbol="__mean__")
    )
    return pd.concat([stacked, aggregate], ignore_index=True).sort_values(["symbol", "minutes"]).reset_index(drop=True)


# ── učestalost ponovne ocene ─────────────────────────────────────────────

REESTIMATION_EVERY = {"annual": 252, "monthly": 21, "daily": 1}


def _eval_qlike(frame: pd.DataFrame) -> pd.Series:
    """QLIKE po (date, symbol) na periodu evaluacije, samo nad redovima koji
    se mogu oceniti (results/tables.py::_scoreable): QLIKE je nedefinisan pri
    y_true = 0 ili pri prognozi koja nije konačna i pozitivna, a jedan takav
    red bi ceo prosek pretvorio u inf/NaN."""
    from projekat.evaluation.losses import qlike
    from projekat.results.tables import _scoreable

    ev = frame[~frame["is_warmup"]]
    ev = ev[_scoreable(ev)]
    return pd.Series(qlike(ev["y_true"].to_numpy(), ev["y_hat"].to_numpy()),
                     index=pd.MultiIndex.from_frame(ev[["date", "symbol"]]))


def reestimation_frequency_comparison(
    panel_df: pd.DataFrame,
    *,
    design_id: str,
    model_names: tuple[str, ...] = ("log_har", "har_j", "char"),
    horizon: int = 1,
    frequencies: tuple[str, ...] = tuple(REESTIMATION_EVERY),
) -> pd.DataFrame:
    """Prosečni QLIKE iste walk-forward šeme pri ponovnoj oceni svakih
    252 / 21 / 1 dan trgovanja -- SAMO ekonometrijski modeli, JEDAN dizajn."""
    from projekat.model.runner import run_one

    design_df = panel_df[panel_df["design_id"] == design_id]
    if design_df.empty:
        raise ValueError(f"dizajn {design_id!r} nije u panelu")
    rows = []
    for frequency in frequencies:
        every = REESTIMATION_EVERY[frequency]
        for model_name in model_names:
            frame = run_one(design_df, model_name=model_name, horizons=(horizon,),
                            refit_every=every, refit_rule="every_21")
            losses = _eval_qlike(frame)
            rows.append({"frequency": frequency, "refit_every_sessions": every, "model": model_name,
                         "design_id": design_id, "horizon": horizon,
                         "mean_qlike": float(losses.mean()), "n_obs": int(len(losses))})
    return pd.DataFrame(rows)


# ── H5 placebo ───────────────────────────────────────────────────────────


def h5_placebo_panel(panel_df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Permutuj SAMO INDIKATOR skoka UNUTAR SIMBOLA (i dizajna), pa iz njega
    iznova izvedi celu dekompoziciju tog dana:

        J_d = 1{permutovan indikator} * max(RV_d - IV_d, 0)
        C_d = RV_d - J_d

    uz ponovno računanje HAR agregata J_w/J_m/C_w/C_m (measure/rolling.py).
    To je tačno ista formula kao u sloju merenja (measure/decompose.py), pa
    permutovani panel ostaje interno konzistentan: C_d je i dalje
    nenegativan i i dalje <= RV_d.

    ZAŠTO SE NE PERMUTUJE J_d. Ranija verzija je premeštala J_d zajedno sa
    indikatorom, čime je J jednog dana sletao na RV nekog drugog dana. Pošto
    je J izvedeno IZ RV tog dana, tako premešteno J ume da premaši RV, pa je
    C_d = RV_d - J_d postajalo NEGATIVNO -- izmereno na 1.3% redova, dok na
    pravom panelu to ne biva nijednom. Karakteristike koriste _log() sa
    donjim ograničenjem 1e-12, pa su ti redovi ulazili u regresiju kao
    ln(C_d) = -27.6, naspram minimuma -11.6 na pravom panelu. Placebo modeli
    su tako dobijali ekstremne regresore kakve stvarno pokretanje nikad ne
    vidi, što je pogoršavalo permutacionu raspodelu i PRISTRASNO išlo u
    korist zaključka da informacija o skokovima pomaže. Permutovanjem samo
    indikatora menja se tačno ono što hipoteza H5 i tvrdi da je važno --
    KOJI su dani proglašeni danima sa skokom -- i ništa drugo."""
    from projekat.measure.rolling import add_har_aggregates

    if "IV_d" not in panel_df.columns:
        raise ValueError(
            "H5 placebo zahteva kolonu IV_d (ocena integrisane varijanse po dizajnu) da bi "
            "J i C mogli da se izvedu iz permutovanog indikatora. Panel je izgrađen pre nego "
            "što je IV_d dodat u measure/schema.py -- ponovo izgradi panel."
        )

    rng = np.random.default_rng(seed)
    pieces = []
    for _keys, group in panel_df.groupby(["symbol", "design_id"], observed=True, sort=True):
        group = group.sort_values("date").copy()
        order = rng.permutation(len(group))
        flag = group["jump_flag"].to_numpy()[order]
        group["jump_flag"] = flag
        rv = group["RV_d"].to_numpy(dtype=float)
        iv = group["IV_d"].to_numpy(dtype=float)
        # ista formula kao measure/decompose.py: bez značajnog testa J je 0
        # bez obzira na IV; sa značajnim testom J je nedefinisano ako IV nije
        # izračunat (NaN se propagira, nikad se tiho ne ograničava)
        group["J_d"] = np.where(np.asarray(flag, dtype=bool), np.maximum(rv - iv, 0.0), 0.0)
        group["C_d"] = rv - group["J_d"].to_numpy(dtype=float)
        for col in ("C", "J"):
            if f"{col}_w" in group.columns or f"{col}_m" in group.columns:
                group = add_har_aggregates(group, col)
        pieces.append(group)
    placebo = pd.concat(pieces, ignore_index=True)
    placebo.attrs["placebo"] = True
    placebo.attrs["placebo_seed"] = seed
    return placebo


def _placebo_deltas(panel_df: pd.DataFrame, design_ids: list[str], models, horizon: int,
                    baseline: dict[str, pd.Series], seed: int | None) -> list[dict]:
    """dQLIKE = QLIKE(model) - QLIKE(log_har) za svaki dizajn/model, na
    pravom panelu (seed=None) ili na permutaciji `seed`."""
    from projekat.model.runner import run_one

    source = panel_df if seed is None else h5_placebo_panel(panel_df, seed=seed)
    rows = []
    for design_id in design_ids:
        design_df = source[source["design_id"] == design_id]
        base = baseline[design_id]
        for model_name in models:
            losses = _eval_qlike(run_one(design_df, model_name=model_name, horizons=(horizon,)))
            common = losses.index.intersection(base.index)
            rows.append({"design_id": design_id, "model": model_name, "permutation": seed,
                         "delta_qlike": float(losses.loc[common].mean() - base.loc[common].mean()),
                         "n_obs": int(len(common))})
    return rows


def h5_placebo_test(
    panel_df: pd.DataFrame,
    *,
    grid: str = "5m",
    estimator: str = "bpv",
    model_names: tuple[str, ...] = PLACEBO_MODELS,
    horizon: int = 1,
    n_permutations: int = N_PERMUTATIONS,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H5: stvarni dQLIKE naspram raspodele n_permutations permutovanih,
    jedna mreža, jedan ocenjivač, sva četiri testa skoka.

    Vraća (sažetak po (dizajn, model), sve permutacione vrednosti)."""
    from projekat.evaluation.anova import parse_design_id
    from projekat.model.runner import run_one

    design_ids = [d for d in (f"{grid}__{estimator}__{t}" for t in ("naive", "bns", "lm", "lm_fdr"))
                  if d in set(panel_df["design_id"])]
    panel = panel_df[panel_df["design_id"].isin(design_ids)]

    # log_har ne vidi skokove, pa ga permutacija ne menja -- računa se jednom
    baseline = {}
    for design_id in design_ids:
        baseline[design_id] = _eval_qlike(
            run_one(panel[panel["design_id"] == design_id], model_name=PLACEBO_BASELINE, horizons=(horizon,)))

    real = pd.DataFrame(_placebo_deltas(panel, design_ids, model_names, horizon, baseline, None))
    seeds = list(range(n_permutations))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_placebo_deltas, [panel] * len(seeds), [design_ids] * len(seeds),
                               [model_names] * len(seeds), [horizon] * len(seeds),
                               [baseline] * len(seeds), seeds)
            perm_rows = [row for chunk in results for row in chunk]
    else:
        perm_rows = [row for s in seeds for row in _placebo_deltas(panel, design_ids, model_names, horizon, baseline, s)]
    perms = pd.DataFrame(perm_rows)

    summary = []
    for (design_id, model_name), group in perms.groupby(["design_id", "model"]):
        actual = real[(real["design_id"] == design_id) & (real["model"] == model_name)]["delta_qlike"].iloc[0]
        dist = group["delta_qlike"].to_numpy()
        summary.append({
            "design_id": design_id, "jump_test": parse_design_id(design_id)["jump_test"], "model": model_name,
            "horizon": horizon, "baseline": PLACEBO_BASELINE,
            "delta_qlike_real": actual,
            "perm_mean": float(dist.mean()), "perm_sd": float(dist.std(ddof=1)) if len(dist) > 1 else float("nan"),
            "perm_q05": float(np.quantile(dist, 0.05)), "perm_q95": float(np.quantile(dist, 0.95)),
            "p_value": float((1 + np.sum(dist <= actual)) / (1 + len(dist))),
            "n_permutations": int(len(dist)),
        })
    return pd.DataFrame(summary), perms
