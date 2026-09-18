"""MeasurementDesign: prostor dizajna rada kao objekat prvog reda
(first-class object), umesto da bude implicitno kodiran u imenima fajlova
ili logici raspoređivanja (dispatch).

Dodavanje petog testa skoka ili četvrtog ocenjivača nikad ne bi trebalo da
zahteva izmenu ovog fajla -- trebalo bi samo da zahteva novi fajl unutar
estimators/ ili jumps/ sa dekoratorom @register_estimator /
@register_jump_test. Vidi measure/registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from projekat.config import TRBPV_THRESHOLD


@dataclass(frozen=True)
class MeasurementDesign:
    grid: str            # "1m" | "5m" | "15m"
    estimator: str       # "bpv" | "medrv" | "trbpv"
    jump_test: str       # "naive" | "bns" | "lm" | "lm_fdr"
    threshold: float = TRBPV_THRESHOLD

    @property
    def id(self) -> str:
        base = f"{self.grid}__{self.estimator}__{self.jump_test}"
        # Prag (threshold) ulazi u id samo tamo gde ima efekat (koristi ga
        # jedino TrBPV) -- inače bi svaki BPV/MedRV dizajn nosio besmislen
        # sufiks "__t3.0", čineći design_id nepouzdanim ključem ANOVA faktora.
        if self.estimator == "trbpv":
            return f"{base}__t{self.threshold:g}"
        return base
