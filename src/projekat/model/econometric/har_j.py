"""HAR-J: RV decomposed into continuous (C) and jump (J) components,
entering as separate regressors -- tests whether jumps have different
persistence than the continuous part. J enters as ln(1 + J_d), which is 0
on every day without a significant jump (features.log1p_jump), never as
ln(max(J, eps)). Depends on grid, estimator, and jump_test (both C and J
come from the conditional decomposition)."""

from __future__ import annotations

from projekat.model.econometric._ols_base import OLSModel
from projekat.model.features import har_j_features
from projekat.model.registry import register_model


class _HARJ(OLSModel):
    name = "har_j"
    depends_on = frozenset({"grid", "estimator", "jump_test"})
    features_fn = staticmethod(har_j_features)


har_j = _HARJ()
register_model(har_j)
