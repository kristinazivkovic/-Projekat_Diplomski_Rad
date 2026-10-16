"""ttm_c_plus_j: foundation analogon HAR-J -- odvojene zero-shot TTM
prognoze kontinuirane (C) i skokovite (J) komponente, svaka odsečena na
nuli, pa SABRANE U NIVOIMA.

=====================================================================
J SE PROGNOZIRA U NIVOIMA
=====================================================================
J = 0 na VEĆINI dana (uslovna dekompozicija), pa se J ne gura kroz log.
C prati normalan log put; zbir je u nivoima:

    y_raw = max(exp(C_hat_log), 0) + max(J_hat, 0)

Sabiranje u logovima bi bilo množenje komponenti, što nije dekompozicija
RV-a. Odsecanje na nuli je sopstveni mehanizam pozitivnosti ovog modela.

=====================================================================
JEDNA KOREKCIJA NA KOMBINOVANOJ PROGNOZI (correction = "ratio_of_sums")
=====================================================================
m* = sum(y) / sum(y_raw) iz van-uzoračkih grešaka poslednjih 252 dana
strogo pre t, primenjena jednom na zbir -- nikad po komponenti (Jensenova
nejednakost nije aditivna), i kao ODNOS ZBIROVA, ne prosek odnosa (y_raw
može biti blizu nule posle odsecanja). Vidi backtransform.py.

=====================================================================
OČEKIVANO PONAŠANJE: prognoza J ~ 0
=====================================================================
To je MERENJE koje podupire CHAR obrazloženje (skokovi su gotovo
nepredvidivi iz sopstvene istorije), a NE greška -- ne "popravljati" ga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.foundation._ttm_base import NoFit, forecast_series_at
from projekat.model.registry import register_model


def _predict_c_plus_j(history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
    c_log = forecast_series_at(history, "C_d", origin_idx, horizons, log_space=True)
    j_level = forecast_series_at(history, "J_d", origin_idx, horizons, log_space=False)
    return {h: np.clip(np.exp(c_log[h]), 0.0, None) + np.clip(j_level[h], 0.0, None) for h in horizons}


class _TTMCPlusJ:
    name = "ttm_c_plus_j"
    stochastic = False
    predicts_levels = True        # zbir je u nivoima
    refit = "never"
    correction = "ratio_of_sums"  # jedna multiplikativna korekcija na zbiru
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> NoFit:
        return NoFit(_predict_c_plus_j)


ttm_c_plus_j = _TTMCPlusJ()
register_model(ttm_c_plus_j)
