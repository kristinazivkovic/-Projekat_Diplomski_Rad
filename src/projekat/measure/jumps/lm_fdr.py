"""Lee-Mykland sa Benjamini-Hochberg FDR kontrolom -- PREKO dana, ne unutar
jednog dana.

Ovaj modul postoji zbog konkretne, potvrđene greške u ranijoj verziji ovog
plana: Gumbel kritična vrednost za lm je već kontrola greške na nivou
porodice po danu (FWER) preko ~n intradnevnih testova tog dana (vidi
docstring u _gumbel.py i tamošnju Monte Karlo proveru). Primena BH
korekcije nad statistikom ISTOG dana je druga korekcija iste porodice, a
pošto je FDR slabiji kriterijum od FWER, test time postaje LIBERALNIJI, ne
strožiji -- direktno izmereno: BH unutar dana je dao udeo dana sa skokom od
60.0% naspram 24.0% za lm na istom uzorku, pogrešan pravac.

Ono što navedena motivacija (Bajgrowicz, Scaillet & Treccani; briga o
višestrukom testiranju iz rada iz februara 2026) zapravo ispravlja je
akumulacija PREKO DANA: čak i kada je svaki dan pojedinačno kontrolisan na
nivou porodice, hiljade dana zajedno generišu lažna otkrića. To je prirodno
objašnjenje zašto udeo dana sa skokom koji je po danu ispravan za lm
(~21-32% preko izmerenih uzoraka) i dalje ostaje iznad opsega od 5-15% iz
literature.

Bezbednost od unapred gledanja (lookahead): BH se primenjuje preko KLIZNOG
prozora od W prethodnih dana plus tekućeg dana
(config.LM_FDR_TRAILING_WINDOW, W=250, ~jedna trgovinska godina) -- nikad
preko celog uzorka. Primena preko celog uzorka bi klasifikovala dan
koristeći p-vrednosti iz godina koje mu tek slede, što je unapredno
gledanje (lookahead bias) u sloju koji je inače vrlo pažljiv da baš to
izbegne (VIX pomeraj, vremenske podele, isključivo klizni K prozor).

Zagrevanje (warm-up): dani pre prvih W dana simbola nemaju pun klizni
prozor i koriste običnu lm značajnost (fdr_warmup=True u panelu).

Provereno (AAPL, 5m, 2012-2013, 488 upotrebljivih dana, 235 sa punim
250-dnevnim kliznim prozorom): lm 31.9% -> lm_fdr 17.9%, neslaganje na
14.0% dana. Pravac je potvrđen; tačna veličina će se razlikovati na punom
panelu i ne tvrdi se da je ovo konačno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projekat.config import LM_FDR_TRAILING_WINDOW
from projekat.measure.jumps.lee_mykland import lee_mykland
from projekat.measure.protocols import JumpVerdict
from projekat.measure.registry import register_jump_test


@dataclass
class DayStat:
    day: str
    max_l: float
    p_value: float
    n: int


def benjamini_hochberg_significant(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Standardna BH procedura: vraća masku značajnosti po elementu."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not np.any(below):
        return [False] * m
    max_k = np.max(np.where(below)[0])
    cutoff = sorted_p[max_k]
    return [p <= cutoff for p in p_values]


def lm_fdr_significance_series(day_stats: list[DayStat], *, window: int = LM_FDR_TRAILING_WINDOW, q: float = 0.05) -> dict[str, tuple[bool, bool]]:
    """Za date LM statistike po danu, u HRONOLOŠKOM REDOSLEDU, izračunaj
    lm_fdr značajnost za svaki dan koristeći klizni prozor od `window`
    PRETHODNIH dana plus tekućeg dana (kauzalno -- nikad ne gleda unapred).
    Vraća {dan: (značajno, is_warmup)}.
    """
    out: dict[str, tuple[bool, bool]] = {}
    for i, stat in enumerate(day_stats):
        if i < window:
            # nema dovoljno klizne istorije: koristi običan lm test
            from projekat.measure.jumps._gumbel import critical_value

            crit = critical_value(stat.n)
            out[stat.day] = (stat.max_l > crit, True)
            continue
        window_stats = day_stats[i - window : i + 1]
        p_values = [s.p_value for s in window_stats]
        sig_mask = benjamini_hochberg_significant(p_values, q=q)
        out[stat.day] = (sig_mask[-1], False)
    return out


def lm_fdr(r: np.ndarray, rv: float, robust: float, *, precomputed: tuple[bool, bool] | None = None, **kwargs) -> JumpVerdict:
    """Ulazna tačka za jedan dan koja odgovara JumpTest protokolu. U praksi
    lm_fdr zahteva kontekst preko više dana (klizni prozor), pa panel.py
    poziva lm_fdr_significance_series jednom po simbolu i prosleđuje
    rezultat nazad kroz `precomputed=(značajno, is_warmup)`. Ova funkcija i
    dalje računa osnovnu LM statistiku za taj dan tako da su `statistic` i
    `p_value` uvek popunjeni čak i kada je precomputed prosleđen."""
    base_verdict = lee_mykland(r, rv, robust, **kwargs)
    if precomputed is None:
        # izolovana/test upotreba bez konteksta preko više dana: svedi na običan lm
        return base_verdict
    significant, _is_warmup = precomputed
    return JumpVerdict(
        significant=significant,
        statistic=base_verdict.statistic,
        p_value=base_verdict.p_value,
        intraday_times=base_verdict.intraday_times,
    )


lm_fdr.name = "lm_fdr"
register_jump_test(lm_fdr)
