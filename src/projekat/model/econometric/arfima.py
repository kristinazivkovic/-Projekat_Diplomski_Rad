"""ARFIMA(p,d,q) on log(RV_d), fit independently PER SYMBOL (a univariate
long-memory model cannot be pooled across symbols the way the cross-
sectional HAR family can).

AT EACH REFIT (every 21 trading days, on the log-RV history strictly before
the refit date):
  * mu = training mean of x = ln RV_d;
  * d by the local-Whittle estimator (Robinson 1995) on x - mu, using the
    lowest m = floor(n^0.65) Fourier frequencies (the usual n^alpha
    bandwidth, alpha in (0.5, 0.8)); coarse grid search over d;
  * u = (1-L)^d (x - mu) via the binomial expansion over the FULL expanding
    history (weights w_k = w_{k-1} (k-1-d)/k);
  * (p, q) in {0,1,2}^2 \\ {(0,0)} by BIC, ARIMA(p, 0, q) on u
    (statsmodels; 'const' is the process mean of u).

EVERY DAY BETWEEN REFITS (mu, d, the ARMA order and coefficients FIXED):
the forecast for origin t filters the history up to t -- u from the
fixed-d expansion, ARMA innovations by the exact recursion
eps_t = phi(L)/theta(L) (u_t - c) (zero pre-sample values, the conditional
form) -- then forecasts u forward k = 1..h steps and re-integrates with
(1-L)^-d (weights v_k = v_{k-1} (k-1+d)/k) over the FULL history:

    x_hat_{t+k} - mu = sum_{j<k} v_j u_hat_{t+k-j} + sum_{j>=k} v_j u_{t+k-j}

which is the exact inverse of the expanding-window differencing (no
truncation). The h-day forecast is the average of x_hat_{t+1..t+h}; the
log-target Jensen gap is left to the trailing correction.

Depends only on grid -- RV_d exists identically under every
estimator/jump_test."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, lfilter
from statsmodels.tsa.arima.model import ARIMA

from projekat.model.registry import register_model
from projekat.model.walkforward import rows_by_symbol

_ORDERS = (0, 1, 2)
_MIN_OBS_FOR_FIT = 60   # below this, long-memory/ARMA order selection is unreliable
LW_BANDWIDTH_EXPONENT = 0.65
#: a log forecast further than this outside the observed range is a
#: numerical failure (see _SymbolARFIMA.forecast), not a forecast
_MAX_LOG_DEVIATION = 10.0


def lw_bandwidth(n: int, *, exponent: float = LW_BANDWIDTH_EXPONENT) -> int:
    """m = floor(n^0.65) lowest Fourier frequencies (at least 2, at most n/2)."""
    return min(max(int(n ** exponent), 2), n // 2)


def _local_whittle_d(x: np.ndarray, *, exponent: float = LW_BANDWIDTH_EXPONENT) -> float:
    """Local-Whittle d (Robinson 1995) over the lowest m = floor(n^exponent)
    periodogram ordinates of the (mean-removed) series."""
    n = len(x)
    x = x - np.mean(x)
    periodogram = (np.abs(np.fft.fft(x)) ** 2) / (2 * np.pi * n)
    m = lw_bandwidth(n, exponent=exponent)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    ip = periodogram[1: m + 1]
    mean_log_lam = np.mean(np.log(lam))

    grid = np.linspace(-0.45, 0.95, 141)
    values = []
    for d in grid:
        scaled = np.mean(ip * lam ** (2 * d))
        values.append(np.log(scaled) - 2 * d * mean_log_lam if scaled > 0 else np.inf)
    return float(grid[int(np.argmin(values))])


def _binomial_weights(d: float, n: int, *, inverse: bool = False) -> np.ndarray:
    """(1-L)^d weights (w) or (1-L)^-d weights (v), n terms."""
    k = np.arange(1, n)
    ratios = (k - 1 + d) / k if inverse else (k - 1 - d) / k
    return np.concatenate([[1.0], np.cumprod(ratios)])


def _fractional_diff(x: np.ndarray, d: float) -> np.ndarray:
    """u_t = sum_{i=0}^{t} w_i x_{t-i}, the full expanding window."""
    n = len(x)
    return fftconvolve(x, _binomial_weights(d, n))[:n]


def _fit_arma_by_bic(series: np.ndarray):
    best = None
    for p in _ORDERS:
        for q in _ORDERS:
            if p == 0 and q == 0:
                continue  # a pure white-noise "model" carries no forecast information
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = ARIMA(series, order=(p, 0, q)).fit()
            except Exception:
                continue
            if best is None or result.bic < best.bic:
                best = result
    if best is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = ARIMA(series, order=(1, 0, 0)).fit()
    return best


def _arma_coefficients(result) -> tuple[float, np.ndarray, np.ndarray]:
    names = list(result.model.param_names)
    params = np.asarray(result.params)
    const = float(params[names.index("const")]) if "const" in names else 0.0
    ar = np.array([params[i] for i, nm in enumerate(names) if nm.startswith("ar.L")])
    ma = np.array([params[i] for i, nm in enumerate(names) if nm.startswith("ma.L")])
    return const, ar, ma


class _SymbolARFIMA:
    def __init__(self, mu: float, d: float, const: float, ar: np.ndarray, ma: np.ndarray):
        self.mu, self.d, self.const, self.ar, self.ma = mu, d, const, ar, ma

    def forecast(self, x: np.ndarray, local_origins: np.ndarray, horizons) -> dict[int, np.ndarray]:
        """x: the symbol's full log-RV history (row 0 = its first panel day);
        local_origins: positions in x to forecast from. Uses only x[:t+1]
        for origin t (the filters are causal)."""
        h_max = max(horizons)
        n = len(x)
        u = _fractional_diff(x - self.mu, self.d)
        eta = u - self.const
        p, q = len(self.ar), len(self.ma)
        ar_poly = np.concatenate([[1.0], -self.ar])
        ma_poly = np.concatenate([[1.0], self.ma])
        eps = lfilter(ar_poly, ma_poly, eta)
        v = _binomial_weights(self.d, n + h_max, inverse=True)

        out = {h: np.empty(len(local_origins)) for h in horizons}
        for i, t in enumerate(local_origins):
            # forecast eta_{t+1..t+h_max}
            eta_path = list(eta[max(0, t - p + 1): t + 1]) if p else []
            eps_hist = eps[max(0, t - q + 1): t + 1] if q else np.array([])
            eta_hat = np.empty(h_max)
            for k in range(1, h_max + 1):
                val = 0.0
                for lag in range(1, p + 1):
                    idx = len(eta_path) - lag
                    if idx >= 0:
                        val += self.ar[lag - 1] * eta_path[idx]
                for lag in range(k, q + 1):       # only past innovations remain
                    idx = len(eps_hist) - (lag - k) - 1
                    if idx >= 0:
                        val += self.ma[lag - 1] * eps_hist[idx]
                eta_hat[k - 1] = val
                if p:
                    eta_path.append(val)
            u_hat = eta_hat + self.const

            # re-integration: known part sum_{j>=k} v_j u_{t+k-j} over the full history
            past = u[: t + 1][::-1]                # u_t, u_{t-1}, ..., u_0
            level = np.empty(h_max)
            for k in range(1, h_max + 1):
                known = float(np.dot(v[k: k + t + 1], past))
                future = float(np.dot(v[:k], u_hat[:k][::-1]))
                level[k - 1] = self.mu + known + future
            cumulative = np.cumsum(level)
            for h in horizons:
                out[h][i] = cumulative[h - 1] / h

        # Numerical guard. The re-integration sums thousands of terms and the
        # ARMA innovations are obtained by inverting the MA polynomial; with a
        # near-unit MA root that inversion can amplify over a long history and
        # produce a log forecast tens of orders of magnitude away from the
        # data (measured: up to +55,848, i.e. exp() overflow to inf). A
        # forecast outside the observed range by more than
        # _MAX_LOG_DEVIATION is a numerical failure, not a forecast, so it
        # becomes NaN -- recorded downstream as a missing forecast
        # (results/tables.py::unscoreable_rows_table) instead of an inf that
        # would silently own every mean it enters.
        lo, hi = float(np.min(x)) - _MAX_LOG_DEVIATION, float(np.max(x)) + _MAX_LOG_DEVIATION
        for h in horizons:
            out[h][(out[h] < lo) | (out[h] > hi) | ~np.isfinite(out[h])] = np.nan
        return out


class _FittedARFIMA:
    def __init__(self, per_symbol: dict[str, _SymbolARFIMA]):
        self._per_symbol = per_symbol

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        out = {h: np.full(len(origin_idx), np.nan) for h in horizons}
        x_all = np.log(history["RV_d"].to_numpy(dtype=float).clip(min=1e-12))
        position_in_origin = {p: i for i, p in enumerate(origin_idx)}
        for symbol, rows in rows_by_symbol(history).items():
            fitted = self._per_symbol.get(symbol)
            wanted = [p for p in rows if p in position_in_origin] if fitted is not None else []
            if not wanted:
                continue
            local = np.searchsorted(rows, wanted)
            forecasts = fitted.forecast(x_all[rows], local, horizons)
            target = [position_in_origin[p] for p in wanted]
            for h in horizons:
                out[h][target] = forecasts[h]
        return out


class _ARFIMA:
    name = "arfima"
    stochastic = False
    predicts_levels = False
    refit = "every_21"
    correction = "log"
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedARFIMA:
        per_symbol = {}
        for symbol, group in train.groupby("symbol", sort=False):
            x = np.log(group.sort_values("date")["RV_d"].to_numpy(dtype=float).clip(min=1e-12))
            if len(x) < _MIN_OBS_FOR_FIT:
                continue
            mu = float(np.mean(x))
            d = _local_whittle_d(x - mu)
            u = _fractional_diff(x - mu, d)
            const, ar, ma = _arma_coefficients(_fit_arma_by_bic(u))
            per_symbol[symbol] = _SymbolARFIMA(mu, d, const, ar, ma)
        return _FittedARFIMA(per_symbol)


arfima = _ARFIMA()
register_model(arfima)
