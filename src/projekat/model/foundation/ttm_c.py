"""ttm_c: zero-shot TTM prognoza log(C_d) -- samo kontinuirana komponenta,
foundation analogon CHAR-a; kontekst od 512 dana koji se završava na dan
prognoze.

Zavisi od grid, estimator I jump_test: C je izlaz uslovne dekompozicije
(measure/decompose.py). Normalan log put -- C_d je strogo pozitivan, za
razliku od J (vidi ttm_c_plus_j.py)."""

from __future__ import annotations

import pandas as pd

from projekat.model.foundation._ttm_base import NoFit, forecast_series_at
from projekat.model.registry import register_model


class _TTMC:
    name = "ttm_c"
    stochastic = False
    predicts_levels = False
    refit = "never"
    correction = "log"
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> NoFit:
        return NoFit(lambda history, idx, hs: forecast_series_at(history, "C_d", idx, hs, log_space=True))


ttm_c = _TTMC()
register_model(ttm_c)
