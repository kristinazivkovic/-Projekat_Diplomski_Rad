"""CHAR: HAR structure but using only the continuous component C (jumps
excluded). Depends on grid, estimator, AND jump_test -- C is the
conditional decomposition's output, which depends on both A2 and A3."""

from __future__ import annotations

from projekat.model.econometric._ols_base import OLSModel
from projekat.model.features import char_features
from projekat.model.registry import register_model


class _CHAR(OLSModel):
    name = "char"
    depends_on = frozenset({"grid", "estimator", "jump_test"})
    features_fn = staticmethod(char_features)


char = _CHAR()
register_model(char)
