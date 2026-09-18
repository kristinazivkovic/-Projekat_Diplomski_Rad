"""Intradnevni test skoka po Lee & Mykland (2008).

Lokalizuje skokove na konkretne intradnevne prinose koristeći klizni
(trailing) prozor lokalne volatilnosti od K opservacija, zatim poredi
maksimalnu standardizovanu statistiku dana sa kritičnom vrednošću za taj
dan, SIMULIRANOM (measure/jumps/_gumbel.py) -- NIJE fiksna konstanta, a
nije ni udžbenička zatvorena Gumbel asimptotska formula (provereno da je
60-500 puta previše konzervativna na svakom n koji ovaj dizajn koristi --
vidi docstring modula _gumbel.py). Dan se obeležava kao značajan ako bilo
koji intradnevni |L_i| prelazi kritičnu vrednost tog dana.

K je fiksiran na 4 trgovinske sesije za SVAKU mrežu (1560/312/104 na
1m/5m/15m) -- namerna odluka, a ne preuzeto poklapanje sa literaturom.
Lee i Mykland (2008) sami tabeliraju K=156 na 15-minutnom uzorkovanju (6
sesija), ali korišćenje njihovog broja ovde bi omogućilo da dužina K-a u
sesijama varira sa Faktorom A1, mešajući "promenu frekvencije
uzorkovanja" sa "promenom dužine lokalnog prozora" u ANOVA varijansi
pripisanoj A1. Držanje K-a na konstantne 4 sesije svuda čini da A1 menja
isključivo frekvenciju uzorkovanja. Takođe nije "jedan trgovinski mesec" --
ranija računska greška (1560/390 = 4.0, a ne 22) je uočena i ispravljena.
Vidi config.LM_LOCAL_WINDOW_K.

Zagrevanje (warm-up): prvih K prinosa istorije simbola nema pun klizni
prozor. Ocena lokalne volatilnosti je tamo NaN (obrađuje
_gumbel.local_sigma_series), test skoka se ne ocenjuje, i ti dani se
isključuju iz statistike o danima sa skokom -- u skladu sa tim kako svaka
druga klizna veličina u ovom pipeline-u tretira nedovoljnu istoriju.

Bezbednost od noćnog jaza: ova funkcija mora biti pozvana na već izolovanom
nizu intradnevnih prinosa JEDNOG dana, koristeći klizni prozor izvučen iz
SEKVENCE prinosa spojene po danima (vidi grids.py) -- nikad iz sirove
razlike spljoštene serije cena preko više dana. Onaj ko poziva funkciju
(panel.py) je odgovoran za ispravnu izgradnju te sekvence; ovaj modul to
pretpostavlja.
"""

from __future__ import annotations

import numpy as np

from projekat.measure.jumps._gumbel import critical_value, exceedance_p_value
from projekat.measure.protocols import JumpVerdict
from projekat.measure.registry import register_jump_test


def lee_mykland(
    r: np.ndarray,
    rv: float,
    robust: float,
    *,
    local_sigma: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    **kwargs,
) -> JumpVerdict:
    """Oceni LM test za jedan dan, na osnovu niza prinosa `r` tog dana i
    unapred izračunatog kliznog niza local_sigma ISTE DUŽINE kao r
    (koji onaj ko poziva funkciju proizvodi iz cele istorijske sekvence
    prinosa, tako da se K-prozor može protezati pre sopstvenog početka
    ovog dana). Ako je local_sigma None, koristi rezervnu ocenu unutar
    dana (upotrebljivo samo za izolovane/test scenarije, ne za pravi
    pipeline, koji uvek obezbeđuje istoriju)."""
    n = len(r)
    if n < 2:
        return JumpVerdict(significant=False, statistic=0.0, p_value=1.0)

    if local_sigma is None:
        # rezervna opcija za izolovane jedinične testove: koristi sopstvenu bipower skalu ovog dana
        from projekat.measure.estimators.bpv import bpv as bpv_fn

        sigma_scalar = np.sqrt(max(bpv_fn(r), 1e-18) / max(n, 1))
        sigma = np.full(n, sigma_scalar)
    else:
        sigma = local_sigma

    valid = ~np.isnan(sigma) & (sigma > 0)
    if not np.any(valid):
        return JumpVerdict(significant=False, statistic=0.0, p_value=1.0)

    l_stats = np.full(n, np.nan)
    l_stats[valid] = np.abs(r[valid]) / sigma[valid]

    max_idx = int(np.nanargmax(l_stats))
    max_l = float(l_stats[max_idx])
    crit = critical_value(n)
    p_value = exceedance_p_value(max_l, n)
    significant = max_l > crit

    flagged_idx = np.where(valid & (l_stats > crit))[0]
    if timestamps is not None and len(flagged_idx) > 0:
        intraday_times = tuple(timestamps[i] for i in flagged_idx)
    else:
        intraday_times = ()

    return JumpVerdict(
        significant=significant,
        statistic=max_l,
        p_value=p_value,
        intraday_times=intraday_times,
    )


lee_mykland.name = "lm"
register_jump_test(lee_mykland)
