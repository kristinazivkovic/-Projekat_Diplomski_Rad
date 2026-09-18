"""Bipower varijacija sa pragom (threshold).

Isključuje svaki prinos čija apsolutna vrednost prelazi `threshold *
local_sigma` pre sabiranja bipower-stila unakrsnih proizvoda, gde je
local_sigma robusna lokalna ocena volatilnosti (ovde: medijana apsolutnog
prinosa tog dana, skalirana na jedinicu volatilnosti, u istom duhu kao
Lee-Mykland lokalni ocenjivač, ali namerno jednostavnija jer je uloga ovog
ocenjivača samo da bude treća, pragovski zasnovana A2 varijanta).

Prag (podrazumevano 3.0, tj. 3 lokalne standardne devijacije) je FIKSIRAN
kroz ceo prostor dizajna umesto da bude uveden kao 4. dimenzija
MeasurementDesign -- vidi design.py i obrazloženu odluku iz plana:
uvođenje bi povećalo mrežu sa 36 na 108 dizajna, a pošto se parametar
odnosi samo na jedan od tri ocenjivača, dizajn bi postao neuravnotežen,
čineći udele varijanse po faktoru u ANOVA analizi (suštinu H3)
neinterpretabilnim. Vrednost je ipak polje na MeasurementDesign sa
navedenom podrazumevanom vrednošću, tako da bi njeno kasnije uvođenje bila
ograničena izmena.
"""

from __future__ import annotations

import math

import numpy as np

from projekat.measure.registry import register_estimator

_MU1_INV_SQ = math.pi / 2


def _local_sigma(r: np.ndarray) -> float:
    # Robusna lokalna skala: medijana apsolutnog prinosa, pretvorena u
    # jedinicu ekvivalentnu standardnoj devijaciji pod normalnošću
    # (0.6745 = Phi^-1(0.75)).
    mad = np.median(np.abs(r))
    return mad / 0.6745 if mad > 0 else float(np.std(r))


def trbpv(r: np.ndarray, threshold: float = 3.0, **kwargs) -> float:
    n = len(r)
    if n < 2:
        return float("nan")
    sigma = _local_sigma(r)
    cutoff = threshold * sigma
    r_thresholded = np.where(np.abs(r) > cutoff, 0.0, r)
    abs_r = np.abs(r_thresholded)
    cross = np.sum(abs_r[1:] * abs_r[:-1])
    finite_sample_correction = n / (n - 1)
    return _MU1_INV_SQ * finite_sample_correction * float(cross)


trbpv.name = "trbpv"
register_estimator(trbpv)
