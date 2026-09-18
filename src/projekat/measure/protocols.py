"""Ugovori (contracts) koje svaki A2 ocenjivač i A3 test skoka moraju da
zadovolje.

JumpVerdict.intraday_times popunjava Lee-Mykland (vremenske oznake
pojedinačno obeleženih intradnevnih prinosa), ali je to samo za
dijagnostiku/Nivo 4 -- nikad se ne koristi u J/C dekompoziciji
(measure/decompose.py). Korišćenje tamo bi Lee-Mykland testu dalo drugačiju
J formulu od svake druge A3 varijante, mešajući "efekat testa" sa "efektom
formule" u ANOVA faktoru koji pripisuje varijansu gubitka Faktoru A3. Vidi
rešenje ove tačke u planu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    name: str

    def __call__(self, r: np.ndarray, **kwargs) -> float:
        """Ocena kontinuirane varijanse otporna na skokove, iz intradnevnih
        log-prinosa `r` jednog dana. kwargs nosi parametre specifične za
        ocenjivač (npr. TrBPV-ov `threshold`); svaki registrovan ocenjivač
        mora da prihvati i ignoriše kwargs koji ne koristi."""
        ...


@dataclass(frozen=True)
class JumpVerdict:
    significant: bool
    statistic: float
    p_value: float
    intraday_times: tuple[pd.Timestamp, ...] = field(default_factory=tuple)


class JumpTest(Protocol):
    name: str

    def __call__(self, r: np.ndarray, rv: float, robust: float, **kwargs) -> JumpVerdict:
        """Testiraj da li dnevni RV sadrži značajan skok, na osnovu
        intradnevnih prinosa dana `r`, njegovog RV-a, i ocene kontinuirane
        varijanse otporne na skokove `robust` (dobijene iz Estimator-a)."""
        ...
