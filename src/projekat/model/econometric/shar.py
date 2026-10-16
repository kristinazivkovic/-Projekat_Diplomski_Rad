"""SHAR: positive/negative semivariance decomposition (Patton & Sheppard
2015) plus weekly/monthly components -- captures asymmetry between
negative and positive shocks. Depends only on grid -- RS_pos/RS_neg are
computed directly from returns, not from the A2/A3 conditional
decomposition."""

from __future__ import annotations

from projekat.model.econometric._ols_base import OLSModel
from projekat.model.features import shar_features
from projekat.model.registry import register_model


class _SHAR(OLSModel):
    name = "shar"
    depends_on = frozenset({"grid"})
    features_fn = staticmethod(shar_features)


shar = _SHAR()
register_model(shar)
