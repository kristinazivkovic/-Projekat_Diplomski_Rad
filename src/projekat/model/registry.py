"""Samoregistracija za modele. Prati isti obrazac kao measure/registry.py.
runner.py razrešava modele po imenu i nikad ne uvozi (import) konkretan
modul modela."""

from __future__ import annotations

from projekat.model.protocols import Model

MODELS: dict[str, Model] = {}


def register_model(obj: Model) -> Model:
    MODELS[obj.name] = obj
    return obj
