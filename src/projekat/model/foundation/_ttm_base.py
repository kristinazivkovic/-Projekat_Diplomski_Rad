"""Deljena TTM (IBM Granite TinyTimeMixer) mašinerija za sve ttm_* modele.

ZERO-SHOT, DETERMINISTIČKI, NIKAD TRENIRAN: checkpoint se učitava prikovan
(config.TTM_MODEL_ID + TTM_REVISION) i koristi se isključivo u inference
režimu (`torch.no_grad()`, `net.eval()`). Nema `fit`-a u smislu ostalih
modela -- `fit()` postoji samo da bi se zadovoljio Model protokol i da
izračuna back-transform korekciju na validacionom skupu; sami težinski
parametri se NIKAD ne menjaju. Zato `stochastic = False` i nema semena.

JEDNA SERIJA NA ULAZU: TTM se poziva po simbolu, nad jednom univarijantnom
serijom (num_input_channels=1 u prikovanom checkpoint-u), nikad nad
panelom više simbola odjednom -- isti razlog kao kod arfima/mem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projekat.config import (
    TTM_CONTEXT_LENGTH,
    TTM_MODEL_ID,
    TTM_PREDICTION_LENGTH,
    TTM_REVISION,
)

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_ttm():
    """Učitaj PRIKOVAN checkpoint (keširan po (model_id, revizija) da se ne
    preuzima i ne instancira iznova za svaki simbol/prozor)."""
    key = (TTM_MODEL_ID, TTM_REVISION)
    if key not in _MODEL_CACHE:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        net = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_ID, revision=TTM_REVISION)
        net.eval()
        _MODEL_CACHE[key] = net
    return _MODEL_CACHE[key]


def ttm_forecast_series(series: np.ndarray, horizon: int) -> float:
    """Zero-shot prognoza h-dnevnog PROSEKA iz jedne univarijantne serije.

    Uzima poslednjih TTM_CONTEXT_LENGTH tačaka kao kontekst, vraća prosek
    prvih `horizon` koraka predikcije -- što odgovara konstrukciji cilja u
    targets.py (h-dnevni prosek), ne pojedinačnoj tački na koraku h.
    Serija kraća od konteksta se dopunjava (left-pad) svojom prvom
    vrednošću; to je jedina razumna opcija za zero-shot model sa fiksnim
    kontekstom i pogađa samo najranije prozore.
    """
    net = load_ttm()
    if len(series) == 0:
        return float("nan")

    context = series[-TTM_CONTEXT_LENGTH:]
    if len(context) < TTM_CONTEXT_LENGTH:
        pad = np.full(TTM_CONTEXT_LENGTH - len(context), context[0])
        context = np.concatenate([pad, context])

    x = torch.tensor(context, dtype=torch.float32).reshape(1, TTM_CONTEXT_LENGTH, 1)
    with torch.no_grad():
        out = net(past_values=x)
    pred = out.prediction_outputs.numpy().reshape(-1)

    steps = min(horizon, TTM_PREDICTION_LENGTH, len(pred))
    return float(np.mean(pred[:steps]))


def symbol_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Degenerisan feature okvir sa samo `symbol` -- kao kod arfima/mem,
    stvarni ulaz TTM-a je istorijska serija sačuvana u fitovanom objektu,
    a ne red lagovanih karakteristika."""
    return df[["symbol"]].copy()


def forecast_per_symbol(
    fit_train: pd.DataFrame, column: str, horizon: int, *, log_space: bool
) -> dict[str, float]:
    """Jedna zero-shot TTM prognoza po simbolu za dati stubac panela.

    log_space=True: serija se log-transformiše pre TTM-a i prognoza se
    vraća U LOG PROSTORU (normalan put, kao za svaki drugi log-model).
    log_space=False: serija ide u TTM u NIVOIMA i prognoza se vraća u
    nivoima (izuzetak za J -- vidi ttm_c_plus_j.py)."""
    out: dict[str, float] = {}
    for symbol, group in fit_train.sort_values("date").groupby("symbol"):
        values = group[column].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        series = np.log(np.clip(values, 1e-12, None)) if log_space else values
        out[symbol] = ttm_forecast_series(series, horizon)
    return out


def predict_from_map(X: pd.DataFrame, per_symbol: dict[str, float]) -> np.ndarray:
    out = np.empty(len(X))
    for i, symbol in enumerate(X["symbol"].to_numpy()):
        out[i] = per_symbol.get(symbol, np.nan)
    return out
