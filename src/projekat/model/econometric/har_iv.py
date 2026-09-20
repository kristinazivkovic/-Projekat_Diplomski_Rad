"""HAR-IV: HAR extended with VIX(t-1) -- tests whether the options market
carries information not already in the price history. Depends only on
grid -- vix_lag1 is identical across estimator/jump_test."""

from __future__ import annotations

from projekat.model.econometric._ols_base import OLSModel
from projekat.model.features import har_iv_features
from projekat.model.registry import register_model


class _HARIV(OLSModel):
    name = "har_iv"
    depends_on = frozenset({"grid"})
    features_fn = staticmethod(har_iv_features)


har_iv = _HARIV()
register_model(har_iv)
