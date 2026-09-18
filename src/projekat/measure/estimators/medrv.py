"""Medijalna realizovana varijansa (Andersen, Dobrev & Schaumburg, 2012).

Koristi medijanu svake trojke uzastopnih apsolutnih prinosa, čime se
ignoriše najveći od tri -- otporniji je od BPV-a kada se skokovi dese u
susednim prinosima (slučaj u kom je BPV-ov proizvod dva prinosa potpuno
kontaminiran).
"""

from __future__ import annotations

import math

import numpy as np

from projekat.measure.registry import register_estimator

_MEDRV_SCALE = math.pi / (6 - 4 * math.sqrt(3) + math.pi)


def medrv(r: np.ndarray, **kwargs) -> float:
    n = len(r)
    if n < 3:
        return float("nan")
    abs_r = np.abs(r)
    triplets = np.stack([abs_r[:-2], abs_r[1:-1], abs_r[2:]], axis=1)
    medians = np.median(triplets, axis=1)
    finite_sample_correction = n / (n - 2)  # po danu; doprinosi n-2 trojke
    return _MEDRV_SCALE * finite_sample_correction * float(np.sum(medians**2))


medrv.name = "medrv"
register_estimator(medrv)
