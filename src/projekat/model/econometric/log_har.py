"""Log-HAR (Corsi 2009): log y_{t,h} regressed on log daily/weekly/monthly
RV, by OLS, back-transformed via the shared non-parametric trailing
correction (backtransform.py). This is the HEADLINE BENCHMARK for relative
QLIKE (evaluation/losses.py::relative_qlike) -- the standard form of HAR
reported throughout the literature. Depends only on grid -- RV_d/w/m exist
identically under every estimator/jump_test.

Deliberately kept distinct from `har.py` (NNLS-in-levels): the two use
different positivity mechanisms (log+correction here vs. non-negative
coefficients there) and must never be unified into one implementation --
see har.py's docstring for why."""

from __future__ import annotations

from projekat.model.econometric._ols_base import OLSModel
from projekat.model.features import har_features
from projekat.model.registry import register_model


class _LogHAR(OLSModel):
    name = "log_har"
    depends_on = frozenset({"grid"})
    features_fn = staticmethod(har_features)


log_har = _LogHAR()
register_model(log_har)
