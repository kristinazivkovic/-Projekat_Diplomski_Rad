"""log_har_ttm: hibrid log-HAR + ttm_rv, kombinacija JEDNAKIH TEGOVA.

TEGOVI SU FIKSIRANI NA 0.5/0.5 I NIKAD SE NE OPTIMIZUJU -- ni na jednom
podskupu podataka. Optimizovana kombinacija bi učinila da doprinos hibrida
varijansi odražava uloženi trud pretrage umesto arhitekture.

KOMBINOVANJE U LOG PROSTORU, PRE KOREKCIJE:

    log_y_raw = 0.5 * log_har_pred + 0.5 * ttm_rv_pred     (obe u logu)

Ponderisanje u NIVOIMA bi bio drugačiji ocenjivač (aritmetička naspram
geometrijske sredine) i ne sme se tiho zameniti. HAR deo se ponovo ocenjuje
svakih 21 dan kao log_har; TTM deo je zero-shot sa kontekstom od 512 dana
koji se završava na dan prognoze (ttm_rv komponenta).

JEDNA KOREKCIJA NA KOMBINOVANOJ PROGNOZI (correction = "ratio_of_sums"):
m* = sum(y) / sum(exp(log_y_raw)) iz van-uzoračkih grešaka poslednjih 252
dana strogo pre t -- nikad po komponenti. Vidi backtransform.py.

Zavisi samo od mreže (grid) -- obe komponente zavise samo od RV_d."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import har_features
from projekat.model.foundation._ttm_base import forecast_series_at
from projekat.model.registry import register_model

_W_HAR = 0.5   # fiksirano, nikad optimizovano
_W_TTM = 0.5   # fiksirano, nikad optimizovano


class _FittedLogHARTTM:
    def __init__(self, fitted_ols):
        self._fitted_ols = fitted_ols

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        har_log = self._fitted_ols.predict(history, origin_idx, horizons)
        ttm_log = forecast_series_at(history, "RV_d", origin_idx, horizons, log_space=True)
        return {h: _W_HAR * har_log[h] + _W_TTM * ttm_log[h] for h in horizons}


class _LogHARTTM:
    name = "log_har_ttm"
    stochastic = False
    predicts_levels = False          # kombinacija je u logu
    refit = "every_21"               # HAR deo
    correction = "ratio_of_sums"     # jedna korekcija na kombinovanoj prognozi
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedLogHARTTM:
        return _FittedLogHARTTM(fit_ols(har_features, train, horizons))


log_har_ttm = _LogHARTTM()
register_model(log_har_ttm)
