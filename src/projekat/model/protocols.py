"""Ugovori (contracts) koje svaki model u Faktoru C mora da zadovolji.
Prati isti obrazac kao Estimator/JumpTest u measure/protocols.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class FittedModel(Protocol):
    backtransform_correction: float  # a = ln(mean(exp(rezidual))) iz validacionog skupa (NE Gaussova sigma^2/2, vidi backtransform.py). Ignorisano ako model.predicts_levels je True.

    def predict(self, X) -> np.ndarray:
        """Vrati predviđeni log(y_{t,h}) za svaki red/sekvencu iz X -- OSIM
        ako model.predicts_levels je True, u kom slučaju vraća predviđeni
        y_{t,h} DIREKTNO U NIVOIMA (levels), bez logaritma."""
        ...


class Model(Protocol):
    name: str
    stochastic: bool  # True za PatchTST -- pokreće ponavljanje sa 5 semena (seeds)
    depends_on: frozenset[str]  # podskup {"grid", "estimator", "jump_test"}
    is_sequence_model: bool  # True za modele koji koriste build_sequences/targets_for_sequences umesto features()
    predicts_levels: bool  # True za modele koji predviđaju y_{t,h} direktno u nivoima (npr. NNLS HAR, MEM -- pozitivnost je strukturna) -- runner.py PRESKAČE backtransform() za takve modele umesto da ga pogrešno primeni na već-transformisanu prognozu

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None) -> FittedModel:
        """fit_train isključuje validacioni skup (poslednja godina
        treniranja); val_fold se koristi samo za ocenu backtransform_correction
        van skupa za treniranje (out-of-fold) -- ignorisano kad je
        predicts_levels True."""
        ...
