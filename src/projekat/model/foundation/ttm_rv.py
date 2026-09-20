"""ttm_rv: zero-shot TTM prognoza log(RV_d), prikovan checkpoint, kontekst
od 512 dana koji se završava na dan prognoze.

Zavisi samo od mreže (grid) -- RV_d je identičan pod svakom A2/A3
kombinacijom na datoj mreži. Normalan log put: prognoza u log prostoru, pa
deljena trailing korekcija (walkforward.py / backtransform.py)."""

from __future__ import annotations

import pandas as pd

from projekat.model.foundation._ttm_base import NoFit, forecast_series_at
from projekat.model.registry import register_model


class _TTMRV:
    name = "ttm_rv"
    stochastic = False          # zero-shot, deterministički -- nema semena
    predicts_levels = False     # log put
    refit = "never"
    correction = "log"
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> NoFit:
        return NoFit(lambda history, idx, hs: forecast_series_at(history, "RV_d", idx, hs, log_space=True))


ttm_rv = _TTMRV()
register_model(ttm_rv)
