"""MEM (Multiplicative Error Model, Engle 2002) on RV_d, fit by exponential
QML, independently PER SYMBOL.

RV_t = mu_t * eps_t, eps_t > 0, E[eps_t] = 1, with a GARCH(1,1)-style
recursion for the conditional mean:

    mu_t = omega + alpha * RV_{t-1} + beta * mu_{t-1}

Fit by (quasi-)maximizing the exponential log-likelihood
sum_t [-log(mu_t) - RV_t/mu_t] over (omega, alpha, beta) with omega > 0,
alpha >= 0, beta >= 0, alpha + beta < 1 (stationarity) -- exponential QML
because the QML estimator of (omega, alpha, beta) under a
misspecified-but-mean-correct exponential likelihood remains consistent
for any true positive-support error distribution with E[eps_t] = 1
(Engle 2002's own robustness argument for MEM, the multiplicative analogue
of Bollerslev's QML result for GARCH).

Positivity is STRUCTURAL, not achieved by a log transform: mu_t is a
convex combination of non-negative terms (omega, alpha*RV_{t-1},
beta*mu_{t-1}) with non-negative coefficients, so mu_t >= 0 for every t by
construction whenever the parameters satisfy the box constraints above.
This model therefore takes NO back-transform correction (predicts_levels =
True) -- the third and last of this codebase's three deliberately distinct
positivity mechanisms (NNLS-in-levels in har.py, log+correction in
log_har.py/char.py/etc., and this structural multiplicative form). Never
unify these.

Depends only on grid -- RV_d exists identically under every
estimator/jump_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from projekat.model.registry import register_model
from projekat.model.targets import LEVEL_TARGET_COL

_MIN_OBS_FOR_FIT = 30
_EPS = 1e-12


def _mu_path(rv: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = len(rv)
    mu = np.empty(n)
    mu[0] = max(np.mean(rv), _EPS)  # unconditional-mean initialization
    for t in range(1, n):
        mu[t] = omega + alpha * rv[t - 1] + beta * mu[t - 1]
        mu[t] = max(mu[t], _EPS)
    return mu


def _neg_log_likelihood(params: np.ndarray, rv: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e12
    mu = _mu_path(rv, omega, alpha, beta)
    # exponential log-likelihood: -log(mu_t) - rv_t/mu_t
    ll = -np.log(mu) - rv / mu
    return float(-np.sum(ll))


def _fit_mem_qml(rv: np.ndarray) -> tuple[float, float, float]:
    unconditional_mean = max(np.mean(rv), _EPS)
    x0 = np.array([unconditional_mean * 0.05, 0.1, 0.8])
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    result = minimize(_neg_log_likelihood, x0, args=(rv,), method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = result.x
    if alpha + beta >= 1:
        # project back onto the stationarity region if the optimizer landed
        # just outside it (bounds constrain each parameter individually,
        # not their sum)
        scale = 0.999 / (alpha + beta)
        alpha *= scale
        beta *= scale
    return float(omega), float(alpha), float(beta)


class _FittedMEMSymbol:
    def __init__(self, omega: float, alpha: float, beta: float, last_rv: float, last_mu: float, horizon: int):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self._last_rv = last_rv
        self._last_mu = last_mu
        self._horizon = horizon

    def forecast_h(self) -> float:
        """h-step-ahead forecast of the average future RV, matching this
        codebase's target construction (Corsi's h-day average). Under the
        MEM recursion, E[RV_{t+k}|F_t] = E[mu_{t+k}|F_t] since E[eps]=1, and
        E[mu_{t+k}|F_t] follows the same GARCH-style recursion in
        expectation: E[mu_{t+1}|F_t] = omega + alpha*RV_t + beta*mu_t, then
        E[mu_{t+k}|F_t] = omega + (alpha+beta)*E[mu_{t+k-1}|F_t] for k>1.
        Average those expected values over k=1..h."""
        persistence = self.alpha + self.beta
        forecasts = np.empty(self._horizon)
        mu_k = self.omega + self.alpha * self._last_rv + self.beta * self._last_mu
        forecasts[0] = mu_k
        for k in range(1, self._horizon):
            mu_k = self.omega + persistence * mu_k
            forecasts[k] = mu_k
        return float(np.mean(forecasts))


class _FittedMEM:
    def __init__(self, per_symbol: dict[str, _FittedMEMSymbol]):
        self._per_symbol = per_symbol
        # structural positivity (see module docstring) -- no correction needed
        self.backtransform_correction = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X))
        for i, symbol in enumerate(X["symbol"].to_numpy()):
            fitted_symbol = self._per_symbol.get(symbol)
            out[i] = fitted_symbol.forecast_h() if fitted_symbol is not None else np.nan
        return out


def _symbol_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Degenerate feature frame carrying only `symbol` -- like arfima.py,
    MEM's actual state (mu_t recursion) lives inside the fitted per-symbol
    object, not in a lagged-feature row."""
    return df[["symbol"]].copy()


class _MEM:
    name = "mem"
    stochastic = False
    is_sequence_model = False
    predicts_levels = True
    depends_on = frozenset({"grid"})

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None) -> _FittedMEM:
        per_symbol: dict[str, _FittedMEMSymbol] = {}
        for symbol, group in fit_train.sort_values("date").groupby("symbol"):
            rv = group["RV_d"].to_numpy()
            rv = rv[~np.isnan(rv)]
            if len(rv) < _MIN_OBS_FOR_FIT:
                continue
            omega, alpha, beta = _fit_mem_qml(rv)
            mu_path = _mu_path(rv, omega, alpha, beta)
            per_symbol[symbol] = _FittedMEMSymbol(
                omega=omega, alpha=alpha, beta=beta,
                last_rv=float(rv[-1]), last_mu=float(mu_path[-1]), horizon=horizon,
            )
        return _FittedMEM(per_symbol)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return _symbol_only_features(df)


mem = _MEM()
register_model(mem)
