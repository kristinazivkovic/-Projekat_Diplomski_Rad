"""Uslovna (conditional) C/J dekompozicija -- pojedinačno najvažnija
ispravka u ovom planu (Problem 1 u istoriji revizija plana).

BEZUSLOVNA (unconditional) dekompozicija (C = min(RV, ocena), J = max(RV -
ocena, 0)) zavisi samo od A2 ocenjivača, nikad od A3 testa skoka --
zamenom BNS-a Lee-Mykland testom J i C bi ostali bajt-identični, čineći
Faktor A3 bez ikakvog efekta za svaki ulaz modela (HAR-J, CHAR) osim
stratifikacije. To bi tiho poništilo suštinu rada, čija je centralna teza
da izbor procedure merenja menja ulaze modela.

USLOVNA dekompozicija ispod ovo ispravlja (Andersen, Bollerslev & Diebold,
2007):

    J_t = 1{test značajan na dan t} * max(RV_t - ocena_t, 0)
    C_t = RV_t - J_t

J sada zavisi i od A2 (koji ocenjivač) i od A3 (koji test obeležava dan
kao skok). Na dane bez značajnog testa, J = 0 i sva varijacija se pripisuje
kontinuiranom delu -- što je takođe teorijski ispravan tretman: empirijski
je provereno da je prosečan jaz (RV-BPV)/RV svega 3.5%, što znači da je
sirovi jaz na danima bez skoka šum ocenjivača, a ne skok.

Ista jedinstvena formula se koristi za svaku A3 varijantu
(naive/bns/lm/lm_fdr) umesto da se Lee-Mykland testu da drugačija formula
zasnovana na intradnevnim vremenskim oznakama (npr. J = suma r_i^2 preko
obeleženih prinosa): korišćenje drugačije formule za jednu varijantu bi
pomešalo "efekat testa" sa "efektom formule" u ANOVA faktoru koji pripisuje
varijansu gubitka Faktoru A3, što je tačno onaj broj radi čijeg postojanja
je H3 zamišljena. Vidi docstring za protocols.JumpVerdict.

Ovaj modul je jedino mesto gde se A2 i A3 sreću.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from projekat.measure.protocols import JumpVerdict


@dataclass(frozen=True)
class Decomposition:
    C: float
    J: float


def decompose(rv: float, robust_estimate: float, verdict: JumpVerdict) -> Decomposition:
    """robust_estimate može biti NaN na dane ispod MIN_INTRADAY_OBS (vidi
    panel.py) -- NaN se ovde eksplicitno prosleđuje dalje umesto da se
    oslanjamo na to kako Python poredi max()/NaN, što ovde slučajno radi ali
    nije dokumentovan ugovor (contract). Na dan bez značajnog testa J je 0
    bez obzira na robust_estimate (formula ga u tom slučaju uopšte ne
    koristi), pa tanak dan i dalje daje dobro definisano C=RV, J=0, osim ako
    test JESTE značajan -- u tom slučaju nedostajući ocenjivač čini J zaista
    nedefinisanim i on mora biti prosleđen kao NaN, a ne tiho ograničen na
    neku vrednost."""
    if verdict.significant:
        if math.isnan(robust_estimate):
            j = float("nan")
        else:
            j = max(rv - robust_estimate, 0.0)
    else:
        j = 0.0
    c = rv - j
    return Decomposition(C=c, J=j)
