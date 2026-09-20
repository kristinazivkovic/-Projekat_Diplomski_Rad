"""Ugovori (contracts) koje svaki model u Faktoru C mora da zadovolji.
Prati isti obrazac kao Estimator/JumpTest u measure/protocols.py.

TOK JEDNE KOMBINACIJE (model/walkforward.py): prognoza se pravi svakog
dana t; na svakom datumu ponovne ocene R model se ocenjuje (`fit`) na
redovima strogo pre R, pa isti ocenjen objekat prognozira (`predict`) za
sve dane do sledeće ponovne ocene, sa ulazima/stanjem ažuriranim dnevno.
Korekcija povratne transformacije NIJE deo modela: računa je
walkforward.py iz van-uzoračkih grešaka samog modela (backtransform.py),
pa nijedan model ne može da je "sam smisli".
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

#: dozvoljene vrednosti Model.refit
REFIT_RULES = ("every_21", "yearly", "never")
#: dozvoljene vrednosti Model.correction
CORRECTION_KINDS = ("log", "ratio_of_sums", "none")


class FittedModel(Protocol):
    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
        """Sirova prognoza za svaki red `history.iloc[origin_idx]` i svaki
        horizont: {h: niz dužine len(origin_idx)}, NaN gde prognoza ne
        postoji.

        `history` je jedan dizajn (svi simboli), sortiran po (symbol, date),
        sa kolonama panela i BEZ ciljnih kolona, i sadrži samo redove sa
        datumom <= poslednjeg dana u bloku. Prognoza za red t sme da koristi
        samo redove istog simbola sa datumom <= t (kauzalnost se proverava
        testom tests/test_walkforward.py).

        Vrednost je log(y_{t,h}) za modele sa predicts_levels = False, a
        y_{t,h} u nivoima za predicts_levels = True."""
        ...


class Model(Protocol):
    name: str
    stochastic: bool          # True za PatchTST -- walkforward se ponavlja za config.PATCHTST_SEEDS
    depends_on: frozenset[str]  # podskup {"grid", "estimator", "jump_test"}
    predicts_levels: bool     # True: predict() vraća nivoe (NNLS HAR, MEM, ttm_c_plus_j)
    refit: str                # "every_21" | "yearly" | "never" (REFIT_RULES)
    correction: str           # "log" | "ratio_of_sums" | "none" (CORRECTION_KINDS)

    def fit(self, train: pd.DataFrame, horizons: tuple[int, ...], *, seed: int | None = None) -> FittedModel:
        """`train`: redovi dizajna sa datumom STROGO pre datuma ponovne
        ocene R, sortirani po (symbol, date), sa kolonama panela i, za svaki
        h iz `horizons`, ciljnim kolonama targets.target_columns(h). Cilj je
        NaN gde nije realizovan pre R -- model ga ne sme koristiti. Jedan
        poziv ocenjuje model za SVE horizonte (modeli čija ocena zavisi od
        horizonta, npr. OLS, ocenjuju po horizontu unutar poziva)."""
        ...
