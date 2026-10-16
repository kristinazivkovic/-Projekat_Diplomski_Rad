"""HARQ (Bollerslev, Patton & Quaedvlieg 2016), log-space form: log-HAR
plus the interaction z * ln(RV_d), where z is sqrt(realized quarticity)
STANDARDIZED on the training data of each refit (features.harq_features).
The daily coefficient is then (beta_d + beta_Q * z): it varies with that
day's measurement precision, and falls on noisy days only if beta_Q < 0 --
an estimated sign, not an assumption. Depends only on grid -- RQ is
computed from returns identically regardless of estimator/jump_test."""

from __future__ import annotations

from projekat.model.econometric._ols_base import fit_ols
from projekat.model.features import harq_features, harq_standardization
from projekat.model.registry import register_model


class _HARQ:
    name = "harq"
    stochastic = False
    predicts_levels = False
    refit = "every_21"
    correction = "log"
    depends_on = frozenset({"grid"})

    def fit(self, train, horizons, *, seed: int | None = None):
        # (mean, sd) of sqrt(RQ) come from the training rows only and are
        # frozen inside the fitted object until the next refit
        return fit_ols(harq_features, train, horizons, prepare=harq_standardization)


harq = _HARQ()
register_model(harq)
