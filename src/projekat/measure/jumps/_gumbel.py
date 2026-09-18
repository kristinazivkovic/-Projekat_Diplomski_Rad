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

    sigma_hat_i^2 = (1/(K-2)) * suma |r_j| |r_{j-1}|

što konvergira ka sigma^2 * mu_1^2 (mu_1 = sqrt(2/pi) ~ 0.7979), A NE ka
sigma^2 -- pa sigma_hat -> sigma * mu_1, i

    L_i = r_i / sigma_hat_i  ->  Z / mu_1  =  Z * 1.2533

Statistika je 25% VEĆA od |Z|, i faktor 1/c u zatvorenoj Gumbel formuli
(c = sqrt(2/pi) = mu_1) je tačno to preskaliranje -- nije ukras. Poređenje
neskaliranog max|Z| praga sa statistikom koja je zapravo Z/mu_1 čini da
prag izgleda daleko previše konzervativan iz razloga koji nema nikakve veze
sa sporom konvergencijom Gumbel granice.

ISPRAVKA (ova revizija): local_sigma_series ispod više ne primenjuje
mu_1^-1 -- odgovara sopstvenoj konvenciji Lee-a i Myklanda (bez bipower
korekcije), tako da je statistika koju prosleđuje u critical_value()
zaista Z/mu_1 pod nultom hipotezom, tačno ono za šta je izvedena zatvorena
formula (gumbel_c_and_s). Provereno od početka do kraja (ne samo
kvantil-naspram-kvantila, što je tautologija kada obe strane koriste istu
konvenciju): simulacijom CELE procedure -- nezavisne Gausove putanje, ovaj
tačan kod local_sigma_series, K=270, n=78, 3000 pokušaja -- naspram
zatvorenog praga daje empirijsku veličinu od ~4%, blizu nominalnih 5%
(preostali jaz je šum ocene konačnog K u sigma_hat plus uobičajena greška
Gumbel-asimptotske aproksimacije, a ne greška konvencije skaliranja). Vidi
tests/test_gumbel.py za ovo kao stalan regresioni test.

max|Z| kvantili (ono što je Revizija 8 tabelirala) sami po sebi imaju tačnu
zatvorenu formu (c(n) iznad) -- nikad nije postojao razlog da se simuliraju,
i to je proizvelo tačno izračunat odgovor na pitanje koje pipeline ne
postavlja.

Ovo je takođe razlog zašto lm_fdr NE SME ponovo da primeni
Benjamini-Hochberg korekciju unutar statistike istog dana: zatvorena
kritična vrednost je konstruisana tako da P(max|L| > crit) odgovara
nominalnom nivou preko n testova tog dana -- tj. već predstavlja kontrolu
na nivou porodice (family-wise), po danu. lm_fdr umesto toga primenjuje BH
korekciju PREKO dana. Vidi lm_fdr.py.
"""

from __future__ import annotations

import math

import numpy as np

_C = math.sqrt(2.0 / math.pi)  # = mu_1
_BETA_STAR_95 = -math.log(-math.log(0.95))


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


def critical_value(n: int, confidence: float = 0.95) -> float:
    """Zatvorena kritična vrednost po danu za max|L|, izračunata iz n --
    nikad fiksirana konstanta. Vezana za konvenciju sigma u
    local_sigma_series (bez mu_1^-1 korekcije) -- ako se ta konvencija ikad
    promeni, ova formula mora da se promeni zajedno sa njom (vidi docstring
    modula)."""
    beta_star = _BETA_STAR_95 if confidence == 0.95 else -math.log(-math.log(confidence))
    c_n, s_n = gumbel_c_and_s(n)
    return c_n + s_n * beta_star


def exceedance_p_value(max_l: float, n: int) -> float:
    """P(max|L| > opaženo) pod nultom hipotezom, preko Gumbel aproksimacije."""
    c_n, s_n = gumbel_c_and_s(n)
    x = (max_l - c_n) / s_n
    return float(1 - math.exp(-math.exp(-x)))


def local_sigma_series(r: np.ndarray, k: int) -> np.ndarray:
    """Klizna (trailing) ocena lokalne volatilnosti na svakom indeksu i >= k,
    koristeći sopstveni bipower-stil ocenjivača Lee-a i Myklanda (2008) preko
    prethodnih k prinosa -- BEZ mu_1^-2 korekcije (Konvencija A; vidi
    docstring modula). Ovo je namerno i ključno: zatvorena formula u
    critical_value() je izvedena pod pretpostavkom baš ove konvencije.
    Dodavanje faktora mu_1^-1 ovde bez odgovarajuće izmene formule u
    critical_value() (ili obrnuto) tiho dekalibriše test -- to je tačno
    greška koju je Revizija 9 ispravila, pa se ova dva mesta nikad ne smeju
    menjati nezavisno jedno od drugog.

    Vraća niz usklađen sa r, sa NaN za i < k (nedovoljna klizna istorija --
    period zagrevanja/warm-up, isključen iz testiranja umesto da bude
    izračunat na kratkom, neuporedivom prozoru)."""
    n = len(r)
    sigma = np.full(n, np.nan)
    abs_r = np.abs(r)
    for i in range(k, n):
        window = abs_r[i - k : i]
        cross = np.sum(window[1:] * window[:-1])
        sigma[i] = math.sqrt(cross / (k - 1))
    return sigma
