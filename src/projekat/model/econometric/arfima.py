"""ARFIMA(p,d,q) on log(RV_d), fit independently PER SYMBOL (a univariate
long-memory model cannot be pooled across symbols the way the cross-
sectional HAR family can).

d is estimated per symbol via the local-Whittle estimator (Robinson 1995)
on log(RV_d)'s periodogram -- the standard semiparametric estimator for the
fractional-differencing parameter in long-memory volatility series,
avoiding a parametric assumption about the full spectral shape. The series
is then fractionally differenced by d, and (p, q) are chosen by BIC over
{0, 1, 2}^2 on the differenced series (statsmodels ARIMA with d=0, since
fractional differencing already happened here).

Depends only on grid -- RV_d exists identically under every
estimator/jump_test, and ARFIMA (unlike CHAR/HAR-J) never touches the A2/A3
conditional decomposition."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln
from statsmodels.tsa.arima.model import ARIMA

from projekat.model.backtransform import residual_correction
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL

_D_ORDERS = (0, 1, 2)
_MIN_OBS_FOR_FIT = 60  # below this, long-memory/ARMA order selection is unreliable


def _local_whittle_d(x: np.ndarray, *, bandwidth_fraction: float = 0.5) -> float:
    """Local-Whittle estimate of the fractional-differencing parameter d
    (Robinson 1995): minimizes the local-Whittle objective over the low-
    frequency periodogram ordinates of the (mean-removed) series. Uses a
    coarse grid search over d in (-0.5, 1.0) rather than a full numerical
    optimizer -- adequate for a semiparametric plug-in estimate feeding a
    forecasting pipeline, not a standalone inferential result."""
    n = len(x)
    x = x - np.mean(x)
    fft = np.fft.fft(x)
    periodogram = (np.abs(fft) ** 2) / (2 * np.pi * n)
    freqs = 2 * np.pi * np.arange(1, n) / n

    m = max(int(n * bandwidth_fraction * 0.5), 2)  # number of low frequencies used
    m = min(m, n - 1)
    lam = freqs[:m]
    ip = periodogram[1 : m + 1]

    def objective(d: float) -> float:
        scaled = ip * (lam ** (2 * d))
        mean_scaled = np.mean(scaled)
        if mean_scaled <= 0:
            return np.inf
        return float(np.log(mean_scaled) - 2 * d * np.mean(np.log(lam)))

    grid = np.linspace(-0.45, 0.95, 141)
    values = [objective(d) for d in grid]
    return float(grid[int(np.argmin(values))])


def _fractional_diff(x: np.ndarray, d: float, *, max_lag: int | None = None) -> np.ndarray:
    """Fractionally difference x by d using the binomial-series expansion
    of (1-L)^d, with an EXPANDING window truncated to at most max_lag terms
    (default: full series length). out[t] uses min(t+1, max_lag) terms of
    history, so every index has a defined value (using whatever history is
    available) rather than only the tail index once max_lag == n -- the
    earlier version filled almost the entire output with NaN in that
    default case, since a fixed max_lag window before index max_lag-1 has
    no defined value."""
    n = len(x)
    max_lag = n if max_lag is None else min(max_lag, n)
    weights = np.empty(max_lag)
    weights[0] = 1.0
    for k in range(1, max_lag):
        weights[k] = weights[k - 1] * (k - 1 - d) / k
    out = np.empty(n)
    for t in range(n):
        window_len = min(t + 1, max_lag)
        window = x[t - window_len + 1 : t + 1][::-1]
        out[t] = float(np.dot(weights[:window_len], window))
    return out


def _fit_arma_by_bic(series: np.ndarray) -> tuple[int, int, object]:
    """Select (p, q) in {0,1,2}^2 by BIC on the (already fractionally
    differenced) series, fitting ARIMA(p, 0, q) for each candidate."""
    best = None
    for p in _D_ORDERS:
        for q in _D_ORDERS:
            if p == 0 and q == 0:
                continue  # a pure white-noise "model" carries no forecast information
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(series, order=(p, 0, q)).fit()
            except Exception:
                continue
            if best is None or result.bic < best[2].bic:
                best = (p, q, result)
    if best is None:
        # fallback: AR(1), the least that can fail
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ARIMA(series, order=(1, 0, 0)).fit()
        return 1, 0, result
    return best


_INV_INTEGRATION_LAG_CAP = 500  # bound the (1-L)^-d convolution window for speed on long histories


class _FittedARFIMASymbol:
    """One symbol's fitted (d, p, q, ARMA result), plus the tail of its
    fractionally-differenced training series needed to forecast forward.

    NOTE on re-integration: y = (1-L)^-d y_diff reconstructs the level
    series from the FULL differenced history, not from "last level + a
    local delta" -- (1-L)^-d is a long-memory (slowly-decaying-weight)
    operator, so truncating its convolution to a short window changes the
    reconstructed level itself, not just its precision. We therefore keep
    a bounded but reasonably long tail of the differenced training series
    (_INV_INTEGRATION_LAG_CAP terms) and integrate the forecast path
    forward from there, accepting the small truncation error inherent to
    any finite-order approximation of a long-memory filter -- the same
    trade-off _fractional_diff itself already makes when differencing."""

    def __init__(self, d: float, arma_result, tail_diff: np.ndarray, horizon: int):
        self.d = d
        self._arma_result = arma_result
        self._tail_diff = tail_diff
        self._horizon = horizon

    def forecast_h(self) -> float:
        """Average of the 1..h-step-ahead forecasts of log(RV_d)
        (matching this codebase's h-day-average target construction),
        re-integrated from the fractionally-differenced ARMA forecast back
        to levels via the inverse binomial-series weights of (1-L)^-d,
        convolved against [training tail ++ forecast path]."""
        diff_forecast = self._arma_result.forecast(steps=self._horizon)
        d = self.d

        path = np.concatenate([self._tail_diff, diff_forecast])
        n_tail = len(self._tail_diff)
        max_lag = len(path)
        inv_weights = np.empty(max_lag)
        inv_weights[0] = 1.0
        for k in range(1, max_lag):
            inv_weights[k] = inv_weights[k - 1] * (k - 1 + d) / k

        # reintegrated[t] = sum_{j=0}^{t} inv_weights[j] * path[t-j], i.e.
        # (1-L)^-d applied to the full [tail, forecast] path. Only the
        # forecast-horizon tail of this convolution is needed.
        level_forecasts = np.empty(self._horizon)
        for h in range(1, self._horizon + 1):
            t = n_tail + h - 1
            window = path[: t + 1][::-1]
            level_forecasts[h - 1] = float(np.dot(inv_weights[: len(window)], window))
        return float(np.mean(level_forecasts))


class _FittedARFIMA:
    def __init__(self, per_symbol: dict[str, _FittedARFIMASymbol], backtransform_correction: float):
        self._per_symbol = per_symbol
        self.backtransform_correction = backtransform_correction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """X has one row per (date, symbol) forecast origin, carrying only
        `symbol` -- ARFIMA's forecast for a symbol is the same h-step-ahead
        value for every origin row of that symbol within one fit/window
        (the model isn't re-estimated per day within a window), consistent
        with every other model here being refit once per anchored window,
        not per day."""
        out = np.empty(len(X))
        for i, symbol in enumerate(X["symbol"].to_numpy()):
            fitted_symbol = self._per_symbol.get(symbol)
            out[i] = fitted_symbol.forecast_h() if fitted_symbol is not None else np.nan
        return out


def _log_rv_features(df: pd.DataFrame) -> pd.DataFrame:
    """ARFIMA's `features()` is degenerate by HAR standards: it only needs
    `symbol` to know which per-symbol fitted model to forecast from (the
    actual autoregressive information lives inside the fitted ARMA
    state, not in a lagged-feature row)."""
    return df[["symbol"]].copy()


class _ARFIMA:
    name = "arfima"
    stochastic = False
    is_sequence_model = False
    predicts_levels = False
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None) -> _FittedARFIMA:
        per_symbol: dict[str, _FittedARFIMASymbol] = {}
        for symbol, group in fit_train.sort_values("date").groupby("symbol"):
            series = np.log(group["RV_d"].to_numpy().clip(min=1e-12))
            if len(series) < _MIN_OBS_FOR_FIT:
                continue
            d = _local_whittle_d(series)
            diff = _fractional_diff(series, d)
            if len(diff) < _MIN_OBS_FOR_FIT:
                continue
            _p, _q, arma_result = _fit_arma_by_bic(diff)
            tail_len = min(len(diff), _INV_INTEGRATION_LAG_CAP)
            per_symbol[symbol] = _FittedARFIMASymbol(
                d=d,
                arma_result=arma_result,
                tail_diff=diff[-tail_len:],
                horizon=horizon,
            )

        fitted = _FittedARFIMA(per_symbol, backtransform_correction=0.0)

        X_val = self.features(val_fold)
        y_val = val_fold[TARGET_COL]
        valid_val = X_val["symbol"].isin(per_symbol) & y_val.notna()
        if valid_val.any():
            val_pred = fitted.predict(X_val.loc[valid_val])
            ok = ~np.isnan(val_pred)
            if ok.any():
                fitted.backtransform_correction = residual_correction(
                    y_val.loc[valid_val].to_numpy()[ok], val_pred[ok]
                )
        return fitted

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return _log_rv_features(df)


arfima = _ARFIMA()
register_model(arfima)
