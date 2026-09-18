"""Zajednička korekcija pristrasnosti na log-skali. Modeli predviđaju
log(y_{t,h}); naivna exp() transformacija unazad sistematski potcenjuje
nivo (Jensenova nejednakost), pa svaki model koristi ovu jedinu
implementaciju umesto da rizikuje razmimoilaženje formula po modelu.

Korekcija je NEPARAMETARSKA: a = ln(mean(exp(rezidual))), NE Gaussova
sigma^2/2. Gaussova formula pretpostavlja da su reziduali log(y_true) -
log(y_hat) tačno normalno raspodeljeni; ako raspodela ima debele repove ili
je asimetrična (očekivano za RV reziduale), sigma^2/2 sistematski
pogrešno oceni pravu korekciju E[exp(rezidual)]. Neparametarska ocena
umesto toga direktno ocenjuje taj očekivani exp(rezidual) iz uzorka, bez
pretpostavke o obliku raspodele.

`a` MORA doći iz rezidualâ VAN skupa za treniranje (out-of-fold) na
validacionom skupu (poslednja godina treniranja,
windows.split_with_validation), nikad iz reziduala unutar uzorka
(in-sample) -- rezidualni podaci modela zasnovanih na stablima unutar
uzorka su blizu nule, što bi tiho nedovoljno korigovalo prognoze modela sa
stablima u odnosu na OLS modele.

KOMBINOVANE (SUM-OF-COMPONENTS) PROGNOZE -- zaseban slučaj, obavezan:
residual_correction() ocenjuje korekciju koja čini y_hat = exp(log_y_hat +
a) QLIKE-optimalnim ZA SERIJU NA KOJOJ JE OCENJENA. Ako model predviđa
zbir komponenti (npr. C_hat + J_hat, gde je svaka komponenta možda
uklopljena/transformisana odvojeno -- vidi model/foundation/ttm_c_plus_j.py
kada bude implementiran), primena zasebne korekcije po komponenti i
sabiranje POSLE korekcije ostavlja ZBIR bez ikakve kontrolisane korekcije:
ne postoji garancija da zbir dve odvojeno-QLIKE-optimalne korekcije daje
QLIKE-optimalnu korekciju za njihov zbir (Jensenova nejednakost se ne
sabira linearno preko komponenti). combined_forecast_correction() ispod
je namerno odvojena funkcija koja ocenjuje TAČNO JEDNU multiplikativnu
korekciju na već sabranu (levels-prostor) prognozu naspram ukupnog cilja,
tako da nijedan model ne može lokalno "sam smisliti" korekciju po
komponenti umesto da prođe kroz ovaj deljeni API."""

from __future__ import annotations

import numpy as np


def backtransform(log_y_hat: np.ndarray, correction: float) -> np.ndarray:
    """y_hat = exp(log_y_hat + a), gde je `a` neparametarska korekcija iz
    residual_correction() (NE 0.5*sigma^2)."""
    return np.exp(log_y_hat + correction)


def residual_correction(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """a = ln(mean(exp(rezidual))) za neparametarsku Jensenovu korekciju.
    Ovo pozivati ISKLJUČIVO na predikcijama van skupa za treniranje
    (validacioni skup) -- vidi docstring modula. Ime zadržano konceptualno
    blisko staroj residual_variance() (sada uklonjenoj) da pozivaoci znaju
    da je ovo njena zamena, ne dodatna veličina.

    NE koristiti ovo po komponenti za model koji predviđa zbir komponenti
    (C_hat + J_hat i sl.) -- vidi combined_forecast_correction() i docstring
    modula."""
    resid = y_true_log - y_pred_log
    return float(np.log(np.mean(np.exp(resid))))


def combined_forecast_correction(y_true_level: np.ndarray, combined_y_hat_level: np.ndarray) -> float:
    """Multiplikativni faktor `m` za korekciju VEĆ SABRANE (levels-prostor)
    prognoze zbira komponenti, ocenjen na van-uzorka (validacionom) skupu:
    korigovana_prognoza = m * combined_y_hat_level, gde je
    m = mean(y_true_level / combined_y_hat_level) na validacionom skupu --
    neparametarski analogon exp(residual_correction(...)) primenjen
    direktno u levels-prostoru na već sabranu prognozu, umesto na log
    rezidual jedne komponente.

    OBAVEZNO za svaki model čija je prognoza zbir komponenti fitovanih
    (potencijalno) odvojeno -- primena residual_correction() po komponenti
    pre sabiranja ostavlja zbir bez ikakve kontrolisane korekcije (vidi
    docstring modula). Pozvati TAČNO JEDNOM po modelu, na kombinovanoj
    prognozi, nikad po komponenti."""
    valid = combined_y_hat_level > 0
    if not np.any(valid):
        return 1.0
    ratio = y_true_level[valid] / combined_y_hat_level[valid]
    return float(np.mean(ratio))


def apply_combined_correction(combined_y_hat_level: np.ndarray, correction: float) -> np.ndarray:
    """Primeni multiplikativnu korekciju iz combined_forecast_correction()
    na kombinovanu (levels-prostor) prognozu: korigovana = m *
    combined_y_hat_level."""
    return correction * combined_y_hat_level
