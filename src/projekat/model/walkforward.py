"""Validaciona šema: prošireni prozor sa DNEVNIM prognozama.

Zamenjuje raniju šemu od 11 godišnjih prozora sa zasebnim validacionim
preklopom. Jedna kombinacija (model x dizajn x seme) je jedan prolaz kroz
vreme:

  * PROGNOZA SVAKOG DANA t od config.FIRST_FORECAST_DATE do EVAL_END, za
    svaki horizont h iz config.HORIZONS.
  * PONOVNA OCENA: svakih config.REFIT_EVERY_SESSIONS dana trgovanja
    (Model.refit = "every_21"), jednom godišnje (PatchTST, "yearly"), ili
    nikad (modeli bez treniranja: naive, ttm_*; "never"). Na datumu ponovne
    ocene R model se ocenjuje na SVIM redovima strogo pre R (prozor počinje
    od config.START_DATE i samo se proširuje), a red ulazi u obuku za h samo
    ako mu je cilj realizovan pre R (target_end_h < R) -- inače bi cilj
    reda s = R - 1 na h = 22 nosio RV iz narednih 22 dana.
  * DNEVNI ULAZI/STANJE: između dve ponovne ocene parametri su fiksni, ali
    predict() za dan t vidi sve redove <= t (HAR regresori dana t, MEM
    rekurzija provučena kroz svaki dan, ARFIMA filtriranje, TTM kontekst od
    512 dana koji se završava u t, PatchTST sekvenca od 60 dana).
  * KOREKCIJA i MINCER-ZARNOWITZ: iz sopstvenih van-uzoračkih grešaka
    modela realizovanih u poslednjih 252 dana trgovanja strogo pre t
    (backtransform.py). Nema validacionog preklopa.
  * 2014. JE GODINA ZAGREVANJA: prognoze postoje (daju prve greške za
    korekciju), ali nose is_warmup = True i ne ulaze u evaluaciju.

Kauzalnost se ne podrazumeva nego proverava: tests/test_walkforward.py
proverava da svaka ocena vidi samo redove i ciljeve pre R, da korekcija za
t ne zavisi ni od jedne greške realizovane na t ili kasnije, i da izmena
bilo kog reda posle t ne menja prognozu za t ni kod jednog modela.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from projekat.config import (
    CORRECTION_WINDOW_SESSIONS,
    EVAL_END,
    EVAL_START,
    FIRST_FORECAST_DATE,
    HORIZONS,
    REFIT_EVERY_SESSIONS,
)
from projekat.model.backtransform import trailing_corrections, trailing_mincer_zarnowitz
from projekat.model.targets import add_targets, target_columns

#: kolone reda prognoze koje se prenose iz panela (za evaluaciju/stratifikaciju)
CARRIED_COLUMNS = ("date", "symbol", "design_id", "RV_d", "RV_m", "jump_flag", "r_d")


@lru_cache(maxsize=1)
def trading_sessions() -> pd.DatetimeIndex:
    """XNYS sesije sa marginom oko uzorka -- osnova za "dan trgovanja"."""
    from projekat.data.calendar import sessions_between

    # exchange_calendars zna sesije ~1 godinu unapred; uzorak se završava 2025.
    return _naive(sessions_between("2008-01-01", "2026-12-31"))


def _naive(values) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    return index.tz_localize(None) if index.tz is not None else index


def session_positions(dates) -> np.ndarray:
    """Indeks dana trgovanja za svaki datum (datum koji nije sesija --
    samo u sintetičkim testovima -- dobija indeks sledeće sesije)."""
    return trading_sessions().searchsorted(_naive(dates), side="left").astype(np.int64)


def refit_dates(rule: str, *, first: str = FIRST_FORECAST_DATE, last: str = EVAL_END,
                every: int = REFIT_EVERY_SESSIONS) -> list[pd.Timestamp]:
    """Datumi ponovne ocene = početci blokova prognoze.

    "every_21" i "never": svaki `every`-ti dan trgovanja od `first` ("never"
    ocenjuje samo na prvom, a ostali datumi samo dele predviđanje u
    blokove). "yearly": prvi dan trgovanja svake kalendarske godine."""
    sessions = trading_sessions()
    span = sessions[(sessions >= pd.Timestamp(first)) & (sessions <= pd.Timestamp(last))]
    if len(span) == 0:
        return []
    if rule == "yearly":
        return list(pd.Series(span).groupby(span.year).min())
    if rule in ("every_21", "never"):
        return list(span[::every])
    raise ValueError(f"nepoznato pravilo ponovne ocene {rule!r}")


def training_frame(df: pd.DataFrame, refit_date: pd.Timestamp, horizons) -> pd.DataFrame:
    """Redovi strogo pre R; cilj horizonta h je NaN gde nije realizovan pre R."""
    train = df[df["date"] < refit_date].copy()
    for h in horizons:
        log_col, level_col, end_col = target_columns(h)
        not_realized = ~(train[end_col] < refit_date)
        train.loc[not_realized, [log_col, level_col]] = np.nan
        train.loc[not_realized, end_col] = pd.NaT
    return train


def next_session_return(df: pd.DataFrame) -> pd.Series:
    """r_d sledećeg dana trgovanja istog simbola (za VaR: prognoza sa dana
    t naspram prinosa dana t+1). NaN ako sledeći red simbola nije baš
    sledeća sesija (izbačen dan) ili ga nema."""
    pos = pd.Series(session_positions(df["date"]), index=df.index)
    nxt_pos = pos.groupby(df["symbol"], sort=False).shift(-1)
    nxt_r = df.groupby("symbol", sort=False)["r_d"].shift(-1)
    return nxt_r.where(nxt_pos == pos + 1)


def run_walk_forward(
    design_df: pd.DataFrame,
    model,
    *,
    horizons=HORIZONS,
    seed: int | None = None,
    first: str = FIRST_FORECAST_DATE,
    last: str = EVAL_END,
    refit_every: int = REFIT_EVERY_SESSIONS,
    refit_rule: str | None = None,
    blocks: list[int] | None = None,
    timings: list | None = None,
    finalize: bool = True,
) -> pd.DataFrame:
    """Jedna kombinacija: dnevne prognoze za sve horizonte.

    `refit_rule` zamenjuje model.refit (samo za dijagnostiku učestalosti
    ponovne ocene); `blocks` ograničava prolaz na izabrane blokove (samo za
    merenje vremena, --benchmark); `timings` (lista) prima merenja po bloku.
    Sa finalize=False vraća sirove prognoze bez korekcije."""
    horizons = tuple(int(h) for h in horizons)
    rule = refit_rule or model.refit
    panel_cols = [c for c in design_df.columns]
    df = add_targets(design_df, horizons)
    df["r_d_next"] = next_session_return(df)

    origin_mask = (df["date"] >= pd.Timestamp(first)) & (df["date"] <= pd.Timestamp(last))
    starts = refit_dates(rule, first=first, last=last, every=refit_every)
    selected = set(range(len(starts))) if blocks is None else {b % len(starts) for b in blocks}

    parts: list[pd.DataFrame] = []
    fitted = None
    dates = df["date"]
    for k, start in enumerate(starts):
        if k not in selected:
            continue
        end = starts[k + 1] if k + 1 < len(starts) else None
        in_block = origin_mask & (dates >= start)
        if end is not None:
            in_block &= dates < end
        if not in_block.any():
            continue

        t0 = time.perf_counter()
        refit = fitted is None or rule != "never"
        n_train = 0
        if refit:
            train = training_frame(df, start, horizons)
            n_train = len(train)
            fitted = model.fit(train, horizons, seed=seed)
            del train
        t1 = time.perf_counter()

        block_end = dates[in_block].max()
        hist_mask = (dates <= block_end).to_numpy()
        history = df.loc[hist_mask, panel_cols].reset_index(drop=True)
        origin_idx = np.flatnonzero(in_block.to_numpy()[hist_mask])
        preds = fitted.predict(history, origin_idx, horizons)
        t2 = time.perf_counter()

        block = df.loc[in_block]
        for h in horizons:
            log_col, level_col, end_col = target_columns(h)
            part = block[[c for c in CARRIED_COLUMNS if c in block.columns]].copy()
            part["r_d_next"] = block["r_d_next"].to_numpy()
            part["model"] = model.name
            part["horizon"] = h
            part["seed"] = np.nan if seed is None else float(seed)
            part["refit_date"] = start if rule != "never" else starts[min(selected)]
            part["y_true"] = block[level_col].to_numpy()
            part["target_end"] = block[end_col].to_numpy()
            part["raw_pred"] = np.asarray(preds[h], dtype=float)
            parts.append(part)

        if timings is not None:
            timings.append({"block": k, "start": start, "refit": refit, "n_train": n_train,
                            "n_origins": int(in_block.sum()), "fit_seconds": t1 - t0, "predict_seconds": t2 - t1})

    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    return finalize_forecasts(raw, model) if finalize else raw


def finalize_forecasts(raw: pd.DataFrame, model, *, window: int = CORRECTION_WINDOW_SESSIONS) -> pd.DataFrame:
    """Trailing korekcija, povratna transformacija i MZ rekalibracija po
    horizontu, plus oznaka godine zagrevanja. Serija za korekciju je cela
    serija modela x dizajna x horizonta x semena, objedinjena preko
    simbola."""
    out = []
    for _h, part in raw.groupby("horizon", sort=True):
        part = part.sort_values(["date", "symbol"]).reset_index(drop=True)
        origin_idx = session_positions(part["date"])
        realized = part["target_end"]
        realized_idx = np.where(realized.notna(), session_positions(realized.fillna(part["date"])), -1)
        corr = trailing_corrections(
            origin_idx, realized_idx, part["raw_pred"].to_numpy(), part["y_true"].to_numpy(),
            kind=model.correction, predicts_levels=model.predicts_levels, window=window,
        )
        for col, values in corr.items():
            part[col] = values
        part["correction_kind"] = model.correction
        intercept, slope = trailing_mincer_zarnowitz(
            origin_idx, realized_idx, part["y_hat"].to_numpy(), part["y_true"].to_numpy(), window=window
        )
        part["mz_intercept"] = intercept
        part["mz_slope"] = slope
        part["y_hat_recalibrated"] = intercept + slope * part["y_hat"].to_numpy()
        part["is_warmup"] = part["date"] < pd.Timestamp(EVAL_START)
        out.append(part)
    return pd.concat(out, ignore_index=True)


def rows_by_symbol(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """{simbol: pozicije njegovih redova u history} -- history je sortiran po
    (symbol, date), pa su pozicije jednog simbola uzastopne i hronološke."""
    groups = history.groupby("symbol", sort=False).indices
    return {s: np.sort(idx) for s, idx in groups.items()}
