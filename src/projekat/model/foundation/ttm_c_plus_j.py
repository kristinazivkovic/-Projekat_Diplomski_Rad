"""ttm_c_plus_j: foundation analogon HAR-J -- odvojene zero-shot TTM
prognoze kontinuirane (C) i skokovite (J) komponente, svaka odsečena na
nuli, pa SABRANE U NIVOIMA (ne u logovima).

=====================================================================
IZUZETAK OD PRAVILA "CILJEVI SE MODELUJU U LOGU, UNIFORMNO"
=====================================================================
CLAUDE.md navodi da se ciljevi modeluju u log prostoru uniformno. OVAJ
MODUL ODSTUPA OD TOGA ZA KOMPONENTU J -- svesno i eksplicitno, ne tiho:

  * J = 0 na VEĆINI dana (uslovna dekompozicija postavlja J = 0 na svaki
    dan na kom test skoka nije značajan -- vidi measure/decompose.py), pa
    je log(J) nedefinisan i log put bi pucao ili tiho klipovao nulu na
    1e-12, čineći "prognozu" logaritmom proizvoljne epsilon konstante.
  * Zato se J prognozira U NIVOIMA. C prati normalan log put.
  * Zbir se radi U NIVOIMA: y_hat = max(C_hat, 0) + max(J_hat, 0).
    Sabiranje u logovima bi bilo množenje komponenti, što nije
    dekompozicija RV-a.

OVO TREBA POMIRITI U CLAUDE.md -- tamošnja rečenica o uniformnom log
cilju je sada nepotpuna: tačan invariant je "log uniformno, OSIM
skokovite komponente, koja je u nivoima jer je J = 0 legitimna i česta
vrednost". Ne popravljati ovde tako što bi se J gurnuo natrag u log.

=====================================================================
JEDNA KOREKCIJA NA KOMBINOVANOJ PROGNOZI
=====================================================================
`a = ln(mean(exp(rezidual)))` je QLIKE-optimalna SAMO za seriju na kojoj
je ocenjena. Korigovati C i J odvojeno pa ih sabrati ostavlja prognozu
RV-a bez ijedne kontrolisane korekcije. Zato se ovde koristi
backtransform.combined_forecast_correction(): TAČNO JEDNA korekcija,
ocenjena na validacionom skupu iz VEĆ SABRANE prognoze naspram UKUPNOG
RV-a, primenjena jednom na zbir. Ovo je razlog zašto model nosi
predicts_levels = True -- zbir je već u nivoima i ne sme proći kroz
log-prostorni backtransform().

=====================================================================
OČEKIVANO PONAŠANJE: ttm_j ≈ 0
=====================================================================
Očekuje se da TTM prognozira J blizu nule kroz ceo uzorak. To je MERENJE
koje podupire CHAR obrazloženje (skokovi su gotovo nepredvidivi iz
sopstvene istorije), a NE greška -- ne "popravljati" ga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.backtransform import (
    apply_combined_correction,
    combined_forecast_correction,
)
from projekat.model.foundation._ttm_base import (
    forecast_per_symbol,
    predict_from_map,
    symbol_only_features,
)
from projekat.model.registry import register_model
from projekat.model.targets import LEVEL_TARGET_COL


class _FittedTTMCPlusJ:
    def __init__(self, c_per_symbol: dict[str, float], j_per_symbol: dict[str, float], correction: float):
        self._c_per_symbol = c_per_symbol
        self._j_per_symbol = j_per_symbol
        # jedna multiplikativna korekcija na KOMBINOVANOJ prognozi
        self.combined_correction = correction
        # log-prostorna korekcija se ne koristi (predicts_levels = True)
        self.backtransform_correction = 0.0

    def raw_combined(self, X: pd.DataFrame) -> np.ndarray:
        """Nekorigovan zbir u nivoima: max(C_hat,0) + max(J_hat,0),
        gde C_hat dolazi iz log prostora (exp), a J_hat je već u nivoima."""
        c_log = predict_from_map(X, self._c_per_symbol)
        j_level = predict_from_map(X, self._j_per_symbol)
        c_level = np.exp(c_log)
        return np.clip(c_level, 0.0, None) + np.clip(j_level, 0.0, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return apply_combined_correction(self.raw_combined(X), self.combined_correction)


class _TTMCPlusJ:
    name = "ttm_c_plus_j"
    stochastic = False
    is_sequence_model = False
    predicts_levels = True  # zbir je u nivoima; korekcija je multiplikativna, ne log-aditivna
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        # C: normalan log put. J: NIVOI (vidi docstring modula).
        c_per_symbol = forecast_per_symbol(fit_train, "C_d", horizon, log_space=True)
        j_per_symbol = forecast_per_symbol(fit_train, "J_d", horizon, log_space=False)

        fitted = _FittedTTMCPlusJ(c_per_symbol, j_per_symbol, 1.0)

        X_val = symbol_only_features(val_fold)
        y_val_level = val_fold[LEVEL_TARGET_COL]
        valid = X_val["symbol"].isin(c_per_symbol) & y_val_level.notna()
        if valid.any():
            combined = fitted.raw_combined(X_val.loc[valid])
            ok = ~np.isnan(combined) & (combined > 0)
            if ok.any():
                fitted.combined_correction = combined_forecast_correction(
                    y_val_level.loc[valid].to_numpy()[ok], combined[ok]
                )
        return fitted

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return symbol_only_features(df)


ttm_c_plus_j = _TTMCPlusJ()
register_model(ttm_c_plus_j)
