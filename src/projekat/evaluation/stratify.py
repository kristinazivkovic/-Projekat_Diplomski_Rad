"""Naknadna (post-hoc) stratifikacija: modeli se ne treniraju po stratumu;
već dobijene prognoze se obeležavaju režimom i jump_day oznakom, pa se tek
onda grupišu.

REŽIM: JEDNA OZNAKA PO (simbol, dan), ISTA ZA SVE DIZAJNE
=========================================================
Oznaka se računa iz RV_m na mreži config.REGIME_GRID (5m), naspram granice
= config.REGIME_QUANTILE (medijana) 5m RV_m u PRVOM periodu obuke (svi
redovi pre config.FIRST_FORECAST_DATE, objedinjeno preko simbola).

  * Iz JEDNE mreže, ne iz RV_m samog reda: RV_m na 1m je naduvan šumom
    mikrostrukture u odnosu na 15m, pa bi isti dan češće bio "turbulentan"
    na 1m -- režim bi tada nosio deo varijanse Faktora A1, a u ANOVA je
    režim blokirajući faktor.
  * Granica se računa JEDNOM, iz prvog perioda obuke, i ne pomera se: dan
    mora imati istu oznaku bez obzira na to koja je ponovna ocena napravila
    prognozu.

Oznake se pišu u run dir (store.write_regime_labels) i spajaju sa
prognozama po (simbol, dan) u results/tables.py::attach_strata.
"""

from __future__ import annotations

import pandas as pd

from projekat.config import FIRST_FORECAST_DATE, REGIME_GRID, REGIME_QUANTILE


def compute_regime_labels(
    panel_df: pd.DataFrame,
    *,
    grid: str = REGIME_GRID,
    quantile: float = REGIME_QUANTILE,
    first_forecast: str = FIRST_FORECAST_DATE,
) -> tuple[pd.DataFrame, dict]:
    """(oznake po (symbol, date), provenijencija granice)."""
    from projekat.evaluation.anova import parse_design_id

    grids = {d: parse_design_id(d)["grid"] for d in panel_df["design_id"].unique()}
    on_grid = panel_df[panel_df["design_id"].map(grids) == grid]
    if on_grid.empty:
        raise ValueError(f"panel nema nijedan dizajn na mreži {grid!r} -- režim se računa iz {grid} RV_m")
    # RV_m zavisi samo od mreže, pa su svi dizajni jedne mreže isti
    rv_m = on_grid[["symbol", "date", "RV_m"]].drop_duplicates(["symbol", "date"])
    train = rv_m[(rv_m["date"] < pd.Timestamp(first_forecast)) & rv_m["RV_m"].notna()]
    if train.empty:
        raise ValueError(f"nema {grid} RV_m pre {first_forecast} za granicu režima")
    threshold = float(train["RV_m"].quantile(quantile))
    labels = rv_m.assign(regime=tag_regime(rv_m, threshold=threshold))
    labels.loc[rv_m["RV_m"].isna().to_numpy(), "regime"] = "unknown"
    provenance = {
        "grid": grid,
        "quantile": quantile,
        "threshold": threshold,
        "first_training_period_end": first_forecast,
        "n_rows": int(len(train)),
        "n_symbols": int(train["symbol"].nunique()),
        "date_min": str(train["date"].min().date()),
        "date_max": str(train["date"].max().date()),
    }
    return labels[["symbol", "date", "regime"]].reset_index(drop=True), provenance


def tag_regime(df: pd.DataFrame, *, threshold: float) -> pd.Series:
    """Mirno naspram turbulentno u odnosu na FIKSNU granicu."""
    return (df["RV_m"] > threshold).map({True: "turbulent", False: "calm"})


def tag_jump_day(df: pd.DataFrame) -> pd.Series:
    return df["jump_flag"].astype(bool).map({True: "jump", False: "no_jump"})


def stratified_table(results_df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Prosečan gubitak grupisan po stratumima (podrazumevano: horizont x
    režim x jump_day)."""
    cols = group_cols or ["horizon", "regime", "jump_day"]
    return results_df.groupby(cols, observed=True).mean(numeric_only=True).reset_index()
