"""Deljena TTM (IBM Granite TinyTimeMixer) mašinerija za sve ttm_* modele.

ZERO-SHOT, DETERMINISTIČKI, NIKAD TRENIRAN: checkpoint se učitava prikovan
(config.TTM_MODEL_ID + TTM_REVISION) i koristi se isključivo za inferenciju
(`torch.no_grad()`, `net.eval()`); težine se nikad ne menjaju. Zato
`stochastic = False`, nema semena, a Model.refit = "never" (fit() ne radi
ništa -- korekciju računa walkforward.py iz van-uzoračkih grešaka).

DNEVNI KONTEKST: prognoza za dan t koristi poslednjih TTM_CONTEXT_LENGTH
(512) vrednosti serije TOG SIMBOLA zaključno sa danom t -- nikad stanje sa
datuma ocene. Serija kraća od konteksta se dopunjava sleva svojom prvom
vrednošću (pogađa samo simbole sa kratkom istorijom). Izlaz ima 192 koraka;
prognoza za horizont h je PROSEK PRVIH h KORAKA (odgovara cilju: h-dnevni
prosek). Jedna inferencija po (simbol, dan) služi sva tri horizonta.

JEDNA SERIJA NA ULAZU: num_input_channels=1 u prikovanom checkpoint-u.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from projekat.config import TTM_CONTEXT_LENGTH, TTM_MODEL_ID, TTM_PREDICTION_LENGTH, TTM_REVISION

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_BATCH = 1024


def _device():
    import torch

    if torch.cuda.is_available() and os.getenv("PROJEKAT_DEVICE", "").lower() != "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def load_ttm():
    """Učitaj PRIKOVAN checkpoint (keširan po (model_id, revizija))."""
    key = (TTM_MODEL_ID, TTM_REVISION)
    if key not in _MODEL_CACHE:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        net = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_ID, revision=TTM_REVISION)
        net.eval()
        net.to(_device())
        _MODEL_CACHE[key] = net
    return _MODEL_CACHE[key]


def ttm_forecast_batch(contexts: np.ndarray) -> np.ndarray:
    """(B, TTM_CONTEXT_LENGTH) konteksti -> (B, TTM_PREDICTION_LENGTH)
    zero-shot prognoze."""
    import torch

    net = load_ttm()
    device = _device()
    out = []
    for lo in range(0, len(contexts), _BATCH):
        chunk = torch.tensor(contexts[lo: lo + _BATCH, :, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = net(past_values=chunk).prediction_outputs
        out.append(pred[..., 0].float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, TTM_PREDICTION_LENGTH))


def contexts_at(history: pd.DataFrame, column: str, origin_idx: np.ndarray, *, log_space: bool) -> tuple[np.ndarray, np.ndarray]:
    """Kontekst od TTM_CONTEXT_LENGTH vrednosti `column` koji se završava u
    svakom redu history.iloc[origin_idx], iz reda istog simbola. Vraća
    (konteksti[B, L], ok[B]); NaN vrednosti se popunjavaju unapred unutar
    simbola."""
    from projekat.model.walkforward import rows_by_symbol

    L = TTM_CONTEXT_LENGTH
    values = history[column].to_numpy(dtype=float)
    if log_space:
        values = np.log(np.clip(values, 1e-12, None))
    contexts = np.zeros((len(origin_idx), L), dtype=np.float32)
    ok = np.zeros(len(origin_idx), dtype=bool)
    position_in_origin = {p: i for i, p in enumerate(origin_idx)}
    for _symbol, rows in rows_by_symbol(history).items():
        wanted = [p for p in rows if p in position_in_origin]
        if not wanted:
            continue
        series = pd.Series(values[rows]).ffill().to_numpy()
        for p in wanted:
            local = int(np.searchsorted(rows, p))
            window = series[max(0, local - L + 1): local + 1]
            if len(window) == 0 or not np.isfinite(window).all():
                continue
            if len(window) < L:
                window = np.concatenate([np.full(L - len(window), window[0]), window])
            i = position_in_origin[p]
            contexts[i] = window
            ok[i] = True
    return contexts, ok


def horizon_means(pred: np.ndarray, horizons) -> dict[int, np.ndarray]:
    """Prosek prvih h koraka izlaza za svaki horizont."""
    out = {}
    for h in horizons:
        steps = min(h, pred.shape[1]) if pred.ndim == 2 and pred.shape[1] else 0
        out[h] = pred[:, :steps].mean(axis=1) if steps else np.full(len(pred), np.nan)
    return out


def forecast_series_at(history: pd.DataFrame, column: str, origin_idx: np.ndarray, horizons, *, log_space: bool) -> dict[int, np.ndarray]:
    """TTM prognoza serije `column` za svako poreklo i svaki horizont."""
    contexts, ok = contexts_at(history, column, origin_idx, log_space=log_space)
    out = {h: np.full(len(origin_idx), np.nan) for h in horizons}
    if ok.any():
        means = horizon_means(ttm_forecast_batch(contexts[ok]), horizons)
        for h in horizons:
            out[h][ok] = means[h]
    return out


class NoFit:
    """Zero-shot: fit() ne menja ništa, predict() je sama inferencija."""

    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        return self._predict_fn(history, origin_idx, horizons)
