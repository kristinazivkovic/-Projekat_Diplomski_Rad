"""Zajednička mašinerija za porodicu Lee-Mykland testova: ocena lokalne
volatilnosti i zatvorena (closed-form) kritična vrednost po n za
statistiku max-|L|.

REVIZIJA 9 -- Revizija 8 je dijagnostikovala pravi problem (kritična
vrednost LM statistike zavisi od n, a fiksni prag poput |L|>4.0 bi pomešao
Faktor A1 sa Faktorom A3), ali je posegnula za pogrešnom ispravkom.
"Simulirala" je tabelu kritičnih vrednosti i utvrdila da je zatvorena
Gumbel formula 60-500 puta previše konzervativna. To merenje je uporedilo
pogrešne dve stvari: simulirana tabela je (do 3 decimale) tačan zatvoren
kvantil max|Z| preko n nezavisnih standardnih normalnih promenljivih,

    c(n) = Phi^-1( (1 + 0.95^(1/n)) / 2 )

-- direktno provereno naspram brojeva koje je zabeležila Revizija 8 (14:
2.906 naspram 2.906; 78: 3.406 naspram 3.407; 390: 3.827 naspram 3.826).
Ali LM statistika NIJE max|Z|. Sopstveni ocenjivač lokalne volatilnosti
Lee-a i Myklanda izostavlja mu_1^-2 bipower korekciju:

    sigma_hat_i^2 = (1/(K-1)) * suma |r_j| |r_{j-1}|     (K-1 članova)

što konvergira ka sigma^2 * mu_1^2 (mu_1 = sqrt(2/pi) ~ 0.7979), A NE ka
sigma^2 -- pa sigma_hat -> sigma * mu_1, i

    L_i = r_i / sigma_hat_i  ->  Z / mu_1  =  Z * 1.2533

Statistika je 25% VEĆA od |Z|, i faktor 1/c u zatvorenoj Gumbel formuli
(c = sqrt(2/pi) = mu_1) je tačno to preskaliranje -- nije ukras. (Delilac
K-1, broj sabiraka, je varijanta; Lee i Mykland koriste 1/(K-2).)

Provereno od početka do kraja (ne samo kvantil-naspram-kvantila):
simulacijom CELE procedure -- nezavisne Gausove putanje, ovaj tačan kod
local_sigma_series, K=270, n=78 -- naspram zatvorenog praga daje
empirijsku veličinu blizu nominalne. Vidi tests/test_gumbel.py.

NIVO POVERENJA je config.LM_CONFIDENCE = 0.999 (verovatnoća lažne uzbune
po danu 0.1%, isto kao BNS_ALPHA). critical_value() i exceedance_p_value()
koriste istu Gumbelovu raspodelu, pa je p < 1 - LM_CONFIDENCE tačno
ekvivalentno max|L| > critical_value(n). Na 0.999 asimptotska formula je
ravna (~6.02) za n ~ 24-34, sa minimumom na n = 29 (S_n * beta* opada brže
nego što C_n raste); na punim danima tri mreže (n = 26 / 78 / 389) vrednosti
su 6.02 / 6.08 / 6.30, dakle i dalje rastu sa frekvencijom.

Ovo je takođe razlog zašto lm_fdr NE SME ponovo da primeni
Benjamini-Hochberg korekciju unutar statistike istog dana: zatvorena
kritična vrednost već predstavlja kontrolu na nivou porodice
(family-wise), po danu. lm_fdr umesto toga primenjuje BH korekciju PREKO
dana. Vidi lm_fdr.py.
"""

from __future__ import annotations

import math

import numpy as np

from projekat.config import LM_CONFIDENCE

_C = math.sqrt(2.0 / math.pi)  # = mu_1


def gumbel_c_and_s(n: int) -> tuple[float, float]:
    """(C_n, S_n): pozicija i skala Gumbel granice za max|L| preko n
    testova, gde je L_i = Z_i / mu_1 pod nultom hipotezom (vidi docstring
    modula zašto preskaliranje 1/mu_1 nije opciono)."""
    if n < 2:
        return 0.0, 1.0
    log_n = math.log(n)
    l = math.sqrt(2 * log_n)
    c_n = l / _C - (math.log(math.pi) + math.log(log_n)) / (2 * _C * l)
    s_n = 1.0 / (_C * l)
    return c_n, s_n


def critical_value(n: int, confidence: float = LM_CONFIDENCE) -> float:
    """Zatvorena kritična vrednost po danu za max|L|, izračunata iz n --
    nikad fiksirana konstanta: q_n = C_n + S_n * beta*, gde je
    beta* = -ln(-ln(confidence)). Vezana za konvenciju sigma u
    local_sigma_series (bez mu_1^-1 korekcije) -- ako se ta konvencija ikad
    promeni, ova formula mora da se promeni zajedno sa njom."""
    beta_star = -math.log(-math.log(confidence))
    c_n, s_n = gumbel_c_and_s(n)
    return c_n + s_n * beta_star


def exceedance_p_value(max_l: float, n: int) -> float:
    """P(max|L| > opaženo) pod nultom hipotezom, preko Gumbel aproksimacije:
    p = 1 - exp(-exp(-(max|L| - C_n)/S_n))."""
    c_n, s_n = gumbel_c_and_s(n)
    x = (max_l - c_n) / s_n
    return float(1 - math.exp(-math.exp(-x)))


def local_sigma_series(r: np.ndarray, k: int) -> np.ndarray:
    """Klizna (trailing) ocena lokalne volatilnosti na svakom indeksu i >= k,
    koristeći bipower-stil ocenjivača Lee-a i Myklanda (2008) preko
    prethodnih k prinosa -- BEZ mu_1^-2 korekcije (Konvencija A; vidi
    docstring modula). Ovo je namerno i ključno: zatvorena formula u
    critical_value() je izvedena pod pretpostavkom baš ove konvencije.
    Dodavanje faktora mu_1^-1 ovde bez odgovarajuće izmene formule u
    critical_value() (ili obrnuto) tiho dekalibriše test.

    Vraća niz usklađen sa r, sa NaN za i < k (nedovoljna klizna istorija --
    period zagrevanja/warm-up, isključen iz testiranja umesto da bude
    izračunat na kratkom, neuporedivom prozoru).

    Vektorizovano preko kumulativne sume unakrsnih proizvoda: prozor za
    indeks i su prinosi r[i-k:i], čijih k-1 susednih proizvoda
    |r_j||r_{j-1}| su cross[i-k .. i-2] -- isto što i ranija petlja, bez
    Python petlje od ~1.5 miliona iteracija po simbolu na 1m mreži."""
    n = len(r)
    sigma = np.full(n, np.nan)
    if n <= k or k < 2:
        return sigma
    abs_r = np.abs(np.asarray(r, dtype=float))
    cross = abs_r[1:] * abs_r[:-1]                   # cross[j] = |r_{j+1}||r_j|
    csum = np.concatenate([[0.0], np.cumsum(cross)])
    i = np.arange(k, n)
    window_sum = csum[i - 1] - csum[i - k]           # sum cross[i-k .. i-2]
    sigma[k:] = np.sqrt(np.clip(window_sum, 0.0, None) / (k - 1))
    return sigma
