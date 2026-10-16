"""Orkestracija: za svaku RAZLIČITU kombinaciju (model, dizajn) implicirаnu
sa depends_on, za svako seme ako je model stohastički -- jedan prolaz
walk-forward šeme (model/walkforward.py) koji daje dnevne prognoze za sve
horizonte. Razrešava modele iz registra po imenu -- nikad ne uvozi
konkretan modul modela.

Nabrajanje zasnovano na depends_on: model koji zavisi samo od "mreže"
(grid) se pokreće jednom po mreži (3 dizajna), a ne jednom po punom
design_id (36) -- ocenjivanje 12 puta po mreži na bajt-identičnim ulazima
bi i trošilo računske resurse i unelo duplirane, međusobno zavisne redove.

`run_all` drži sve u memoriji i služi testovima na malom panelu; pun
eksperiment ide kroz model/sharded.py, koji svaku kombinaciju odmah
zapisuje kao zasebnu tabelu.
"""

from __future__ import annotations

import pandas as pd

from projekat.config import EVAL_END, FIRST_FORECAST_DATE, HORIZONS, PATCHTST_SEEDS
from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401  (registracija)
from projekat.model.backtransform import backtransform
from projekat.model.registry import MODELS
from projekat.model.walkforward import run_walk_forward

STOCHASTIC_SEEDS = PATCHTST_SEEDS


def _representative_designs(panel_df: pd.DataFrame, depends_on: frozenset[str]) -> pd.DataFrame:
    """Jedan predstavnički design_id po različitoj kombinaciji faktora u
    depends_on. Npr. depends_on={"grid"} -> jedan design_id po mreži;
    depends_on={"grid","estimator","jump_test"} -> svaki design_id."""
    from projekat.evaluation.anova import parse_design_id

    designs = panel_df[["design_id"]].drop_duplicates().sort_values("design_id").copy()
    parsed = designs["design_id"].apply(parse_design_id).apply(pd.Series)
    designs = pd.concat([designs, parsed], axis=1)

    if not depends_on:
        return designs.iloc[[0]]
    return designs.drop_duplicates(subset=sorted(depends_on))


def seeds_for(model_name: str) -> tuple:
    return STOCHASTIC_SEEDS if MODELS[model_name].stochastic else (None,)


def run_one(design_df: pd.DataFrame, *, model_name: str, horizons=HORIZONS, seed: int | None = None, **kwargs) -> pd.DataFrame:
    """Jedna kombinacija (model x dizajn x seme): dnevne prognoze za sve
    horizonte, sa trailing korekcijom i MZ rekalibracijom."""
    return run_walk_forward(design_df, MODELS[model_name], horizons=horizons, seed=seed, **kwargs)


def run_all(
    panel_df: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    horizons=HORIZONS,
    first: str = FIRST_FORECAST_DATE,
    last: str = EVAL_END,
    **kwargs,
) -> pd.DataFrame:
    """Puna mreža u memoriji: svaki model x (različit dizajn po depends_on)
    x seme. Vraća jedan okvir -- nema zasebnog validacionog okvira."""
    names = model_names or list(MODELS.keys())

    # Provera kontaminacije predtreniranjem je USLOV za svaki ttm_* broj.
    if any(n.startswith("ttm_") or n == "log_har_ttm" for n in names):
        from projekat.config import EVAL_START
        from projekat.model.foundation.contamination import check_and_write

        eval_start, eval_end = max(EVAL_START, first), last
        # Na skraćenom panelu (testovi, probna pokretanja) period evaluacije
        # ume da bude prazan: `last` padne pre EVAL_START. Tada nema nijednog
        # reda evaluacije, pa ni ttm_* broja koji bi trebalo označiti -- a
        # ranije se u tom slučaju upisivao nalaz "CLEAN" sa nula test dana,
        # artefakt koji tvrdi suprotno od stvarnog nalaza na punom periodu.
        # Zato se provera preskače umesto da se upisuje besmislen nalaz.
        if eval_end >= eval_start:
            check_and_write(eval_start, eval_end)

    frames = []
    for model_name in names:
        model = MODELS[model_name]
        for design_id in _representative_designs(panel_df, model.depends_on)["design_id"]:
            design_df = panel_df[panel_df["design_id"] == design_id]
            for seed in seeds_for(model_name):
                frames.append(run_one(design_df, model_name=model_name, horizons=horizons, seed=seed,
                                      first=first, last=last, **kwargs))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collapse_seeds(results_df: pd.DataFrame) -> pd.DataFrame:
    """PRIMARNA vrednost stohastičkog modela je prosek GUBITAKA po semenu,
    ne gubitak prosečne prognoze. Ova funkcija je namerno bez efekta: vraća
    jedan red po semenu, da bi funkcije gubitka nizvodno radile po semenu i
    tek onda bile usrednjene (results/tables.py::collapse_seed_losses)."""
    return results_df


def ensemble_forecast(results_df: pd.DataFrame) -> pd.DataFrame:
    """SEKUNDARNA, eksplicitno obeležena ensemble vrednost: prognoza
    usrednjena preko semena NA LOG SKALI PRE povratne transformacije --
    usrednjavaju se `log_y_hat` i `backtransform_correction` odvojeno, pa
    se backtransform() poziva TAČNO JEDNOM. log(y_hat) bi već ugradio
    korekciju svakog semena i duplirao je. Budući da je svaka korekcija
    ocenjena na grešci SVOG semena, prosek korekcija je aproksimacija
    korekcije samog ansambla -- zato je ovo sekundarna figura. Nikad ne
    zamenjuje primarnu vrednost proseka gubitka po semenu."""
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
