"""MEM (Multiplicative Error Model, Engle 2002) on RV_d, fit by exponential
QML, independently PER SYMBOL.

RV_t = mu_t * eps_t, eps_t > 0, E[eps_t] = 1, with a GARCH(1,1)-style
recursion for the conditional mean:

    mu_t = omega + alpha * RV_{t-1} + beta * mu_{t-1}

Fit by (quasi-)maximizing the exponential log-likelihood
sum_t [-log(mu_t) - RV_t/mu_t] over (omega, alpha, beta) with omega > 0,
alpha >= 0, beta >= 0, alpha + beta < 1 -- exponential QML because the
estimator stays consistent for any true positive-support error
distribution with E[eps_t] = 1 (Engle 2002; the multiplicative analogue of
the Bollerslev-Wooldridge 1992 QML result for GARCH).

Positivity is STRUCTURAL: mu_t is a linear combination of non-negative
terms with non-negative coefficients and omega > 0, so mu_t > 0 by
construction -- no log transform and no correction (correction = "none").

DAILY STATE: parameters are refit every 21 trading days on the RV_d history
strictly before the refit date; between refits the recursion is rolled
through EVERY day with the fixed parameters, so the forecast made on day t
uses mu_t and RV_t of that very day (never the state at the refit date):

    E[mu_{t+1} | F_t] = omega + alpha RV_t + beta mu_t
    E[mu_{t+k} | F_t] = omega + (alpha + beta) E[mu_{t+k-1} | F_t],  k > 1

and the h-day forecast is the average over k = 1..h (the target is the
h-day average). The recursion always starts at the symbol's first row with
mu_0 = the training-sample mean of RV (frozen at the refit), exactly as in
the likelihood.

Depends only on grid -- RV_d exists identically under every
estimator/jump_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from projekat.model.registry import register_model
from projekat.model.walkforward import rows_by_symbol

_MIN_OBS_FOR_FIT = 30
_EPS = 1e-12


def _mu_path(rv: np.ndarray, omega: float, alpha: float, beta: float, mu0: float) -> np.ndarray:
    """mu_0 = mu0; mu_t = omega + alpha RV_{t-1} + beta mu_{t-1} (vectorized
    through lfilter, identical to the explicit loop)."""
    n = len(rv)
    mu = np.empty(n)
    mu[0] = max(mu0, _EPS)
    if n > 1:
        drive = omega + alpha * rv[:-1]
        mu[1:], _ = lfilter([1.0], [1.0, -beta], drive, zi=[beta * mu[0]])
    return np.maximum(mu, _EPS)


def _neg_log_likelihood(params: np.ndarray, rv: np.ndarray, mu0: float) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e12
    mu = _mu_path(rv, omega, alpha, beta, mu0)
    return float(np.sum(np.log(mu) + rv / mu))


def _fit_mem_qml(rv: np.ndarray, mu0: float) -> tuple[float, float, float]:
    x0 = np.array([mu0 * 0.05, 0.1, 0.8])
    bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
    result = minimize(_neg_log_likelihood, x0, args=(rv, mu0), method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = result.x
    if alpha + beta >= 1:
        # bounds constrain each parameter, not their sum
        scale = 0.999 / (alpha + beta)
        alpha *= scale
        beta *= scale
    return float(omega), float(alpha), float(beta)


def _forecast_means(omega, alpha, beta, rv_t: np.ndarray, mu_t: np.ndarray, horizons) -> dict[int, np.ndarray]:
    """Average of E[mu_{t+k}|F_t] over k = 1..h, for every horizon."""
    h_max = max(horizons)
    step = omega + alpha * rv_t + beta * mu_t
    cumulative = np.zeros_like(step)
    out = {}
    persistence = alpha + beta
    for k in range(1, h_max + 1):
        if k > 1:
            step = omega + persistence * step
        cumulative = cumulative + step
        if k in horizons:
            out[k] = cumulative / k
    return out


class _FittedMEM:
    def __init__(self, params: dict[str, tuple[float, float, float, float]]):
        self._params = params  # symbol -> (omega, alpha, beta, mu0)

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        out = {h: np.full(len(origin_idx), np.nan) for h in horizons}
        rv_all = history["RV_d"].to_numpy(dtype=float)
        position_in_origin = {p: i for i, p in enumerate(origin_idx)}
        for symbol, rows in rows_by_symbol(history).items():
            params = self._params.get(symbol)
            wanted = [p for p in rows if p in position_in_origin] if params is not None else []
            if not wanted:
                continue
            omega, alpha, beta, mu0 = params
            rv = np.nan_to_num(rv_all[rows], nan=0.0)
            mu = _mu_path(rv, omega, alpha, beta, mu0)
            local = np.searchsorted(rows, wanted)
            means = _forecast_means(omega, alpha, beta, rv[local], mu[local], horizons)
            target = [position_in_origin[p] for p in wanted]
            for h in horizons:
                out[h][target] = means[h]
        return out


class _MEM:
    name = "mem"
    stochastic = False
    predicts_levels = True
    refit = "every_21"
    correction = "none"
    depends_on = frozenset({"grid"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedMEM:
        params = {}
        for symbol, group in train.groupby("symbol", sort=False):
            rv = group.sort_values("date")["RV_d"].to_numpy(dtype=float)
            rv = np.nan_to_num(rv, nan=0.0)
            if len(rv) < _MIN_OBS_FOR_FIT:
                continue
            mu0 = max(float(np.mean(rv)), _EPS)
            omega, alpha, beta = _fit_mem_qml(rv, mu0)
            params[symbol] = (omega, alpha, beta, mu0)
        return _FittedMEM(params)


mem = _MEM()
register_model(mem)
