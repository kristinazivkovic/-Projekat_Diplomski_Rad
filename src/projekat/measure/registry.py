"""Samoregistracija za A2 ocenjivače i A3 testove skoka.

Kriterijum prihvatanja (iz plana): dodavanje šestog ocenjivača mora
zahtevati tačno jedan novi fajl sa dekoratorom @register_estimator i
nijednu izmenu bilo kog postojećeg fajla. panel.py razrešava implementacije
po imenu iz ovih rečnika i nikad direktno ne uvozi (import) konkretan modul
ocenjivača/testa.
"""

from __future__ import annotations

from projekat.measure.protocols import Estimator, JumpTest

ESTIMATORS: dict[str, Estimator] = {}
JUMP_TESTS: dict[str, JumpTest] = {}


def register_estimator(obj: Estimator) -> Estimator:
    ESTIMATORS[obj.name] = obj
    return obj


def register_jump_test(obj: JumpTest) -> JumpTest:
    JUMP_TESTS[obj.name] = obj
    return obj
