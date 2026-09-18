"""log_har_ttm (WI-8): hibrid log-HAR + ttm_rv, kombinacija JEDNAKIH TEGOVA.

TEGOVI SU FIKSIRANI NA 0.5/0.5 I NIKAD SE NE OPTIMIZUJU -- ni na
validacionom skupu, nigde. Isto obrazloženje kao prikovani LightGBM
hiperparametri (WI-5): optimizovana kombinacija bi učinila da doprinos
hibrida varijansi odražava uloženi trud pretrage umesto arhitekture, a
validacioni skup je već posvećen back-transform korekciji i
Mincer-Zarnowitz koeficijentima.

KOMBINOVANJE SE RADI U LOG PROSTORU, PRE back-transform korekcije:

    log_y_hat = 0.5 * log_har_pred + 0.5 * ttm_rv_pred     (obe u logu)

Ponderisanje u NIVOIMA bi bio drugačiji ocenjivač (aritmetička naspram
geometrijske sredine) i ne sme se tiho zameniti.

JEDNA KOREKCIJA NA KOMBINOVANOJ PROGNOZI: korekcija se ocenjuje iz
reziduala VEĆ KOMBINOVANE log-prognoze naspram log-cilja na validacionom
skupu -- nikad po komponenti (vidi backtransform.py i ttm_c_plus_j.py).
Pošto je kombinacija u log prostoru, ovde je to obična
residual_correction() nad kombinovanom log-prognozom, što je tačno
"jedna korekcija na kombinovanoj prognozi" za log-aditivni slučaj.

Zavisi samo od mreže (grid) -- obe komponente zavise samo od RV_d."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.model.backtransform import residual_correction
from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import har_features
from projekat.model.foundation._ttm_base import (
    forecast_per_symbol,
    predict_from_map,
    symbol_only_features,
)
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL

_W_HAR = 0.5   # fiksirano, nikad optimizovano
_W_TTM = 0.5   # fiksirano, nikad optimizovano


def _hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """HAR karakteristike plus `symbol` (potreban za TTM komponentu)."""
    out = har_features(df)
    out["symbol"] = df["symbol"]
    return out


class _FittedLogHARTTM:
    def __init__(self, fitted_ols, ttm_per_symbol: dict[str, float], backtransform_correction: float):
        self._fitted_ols = fitted_ols
        self._ttm_per_symbol = ttm_per_symbol
        self.backtransform_correction = backtransform_correction

    def combined_log(self, X: pd.DataFrame) -> np.ndarray:
        """Kombinacija u LOG prostoru, pre bilo kakve korekcije."""
        har_cols = [c for c in X.columns if c != "symbol"]
        har_log = self._fitted_ols.predict(X[har_cols])
        ttm_log = predict_from_map(X, self._ttm_per_symbol)
        return _W_HAR * har_log + _W_TTM * ttm_log

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.combined_log(X)


class _LogHARTTM:
    name = "log_har_ttm"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False   # kombinacija je u logu; runner primenjuje backtransform jednom
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        # log-HAR komponenta: OLS se fituje na log cilju kao i inače.
        # fit_ols sam računa svoju korekciju, ali je OVDE NE KORISTIMO --
        # hibrid dobija jednu korekciju na kombinovanoj prognozi.
        fitted_ols = fit_ols(har_features, fit_train, val_fold)
        ttm_per_symbol = forecast_per_symbol(fit_train, "RV_d", horizon, log_space=True)

        fitted = _FittedLogHARTTM(fitted_ols, ttm_per_symbol, 0.0)

        X_val = _hybrid_features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid = X_val.notna().all(axis=1) & y_val.notna() & X_val["symbol"].isin(ttm_per_symbol)
        if valid.any():
            combined = fitted.combined_log(X_val.loc[valid])
            ok = ~np.isnan(combined)
            if ok.any():
                fitted.backtransform_correction = residual_correction(
                    y_val.loc[valid].to_numpy()[ok], combined[ok]
                )
        return fitted

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return _hybrid_features(df)


log_har_ttm = _LogHARTTM()
register_model(log_har_ttm)
