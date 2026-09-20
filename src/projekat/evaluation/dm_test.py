"""Diebold-Mariano test sa Harvey-Leybourne-Newbold korekcijom za male uzorke
i HAC (Newey-West, minimalno h-1 pomeraja/lag) varijansom -- HAC varijansa
već obrađuje autokorelaciju izazvanu preklapajućim ciljnim vrednostima na
h=5/h=22 (targets.py), na seriji razlike gubitaka.

PANEL SE PRVO SVODI NA JEDNU VREDNOST PO DANU
=============================================
Gubici u ovom radu su panel: jedan red po (dan, simbol). Ako se takav panel
prosledi DM-u kao jedan niz složen po (dan, simbol), dešavaju se DVE greške
odjednom, i obe naduvavaju značajnost:

  1. HAC pomeraji prestaju da znače ono što treba. Sa ~28 simbola po danu,
     susedni elementi niza su ISTI DAN a različit simbol, pa max(h-1,1)
     pomeraja pokriva samo poprečni presek unutar dana. Prava
     autokorelacija od preklapajućih ciljeva na h=22 leži ~28*21 pozicija
     dalje i HAC je uopšte ne vidi.
  2. n je naduvan brojem simbola (npr. 78.838 umesto 2.720 dana), pa je
     standardna greška ~sqrt(28) puta premala.

Izmereno na panelu ovog rada (h=22, referentni dizajn): p-vrednosti se
razlikuju 5-7 redova veličine, a 4 od 15 modela menja zaključak na
alpha = 0.05 (npr. ARFIMA 4.2e-08 naspram 0.21).

Zato se razlika gubitaka PRVO uprosečava preko simbola unutar dana
(`collapse_to_daily`), pa se tek na toj dnevnoj seriji primenjuje HAC sa
h-1 pomeraja. Tada jedan pomeraj zaista znači jedan trgovinski dan, n je
broj dana, a poprečna korelacija među simbolima je apsorbovana u
uprosečavanju umesto da se lažno računa kao nezavisna informacija.

VIŠESTRUKO TESTIRANJE: sa 24 dizajna i ~14 modela, iscrpno poređenje svih
parova daje hiljade DM testova, pa bi se pri alpha=0.05 stotine "značajnih"
rezultata pojavilo čisto slučajno. Zato se svaka prijavljena PORODICA
(family) DM testova provlači kroz Benjamini-Hochberg FDR kontrolu --
`benjamini_hochberg_dm()` ispod. BH je već konvencija ovog repozitorijuma
(measure/jumps/lm_fdr.py primenjuje istu proceduru preko dana), pa se
koristi i ovde radi doslednosti umesto Bonferroni-ja ili unapred
proglašenih kontrasta.

PORODICA se mora navesti eksplicitno pri svakom pozivu (argument `family`)
i zapisati uz rezultat: FDR kontrola nema smisla bez izjave nad čime se
kontroliše. Uobičajene porodice u ovom radu:
  - "model_vs_log_har": svaki model naspram log_har, unutar (horizont, dizajn)
  - "design_vs_5m_bpv_bns": svaki dizajn naspram referentnog 5m__bpv__bns
  - "all_pairs": sva poređenja parova unutar ćelije (najveća porodica)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float


#: ime nivoa indeksa koji nosi trgovinski dan u širokim (wide) okvirima gubitaka
DATE_LEVEL = "date"


def collapse_to_daily(wide: "pd.DataFrame", *, level: str = DATE_LEVEL) -> "pd.DataFrame":
    """Svedi okvir gubitaka indeksiran po (dan, simbol) na JEDAN red po danu,
    uzimajući poprečni prosek preko simbola.

    Ovo je korak koji mora da prethodi svakom DM testu nad panelom (vidi
    docstring modula): posle njega jedan HAC pomeraj znači jedan trgovinski
    dan, a n je broj dana umesto broja (dan, simbol) parova.

    Prosek razlika jednak je razlici proseka jer su svi modeli poravnati na
    isti skup (dan, simbol) redova pre poziva (pivot + dropna), pa je
    svejedno da li se prvo oduzima pa uprosečava ili obrnuto.

    Okvir koji je već indeksiran samo po danu vraća se nepromenjen."""
    if not isinstance(wide.index, pd.MultiIndex):
        return wide.sort_index()
    if level not in (wide.index.names or []):
        raise ValueError(f"okvir nema nivo indeksa {level!r}; ima {list(wide.index.names)}")
    return wide.groupby(level=level, observed=True).mean().sort_index()


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, h: int = 1) -> DMResult:
    d = loss_a - loss_b
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC dugoročna varijansa sa max(h-1, 1) pomeraja (lags) --
    # h-1 je teorijski implicirani pomeraj za razliku gubitaka prognoze
    # unapred za h koraka, ograničen odozdo na minimum 1 radi numeričke
    # stabilnosti.
    lags = max(h - 1, 1)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / (lags + 1)) * cov
    var_d = max(var_d, 1e-12)

    dm_stat = d_bar / np.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold korekcija za male uzorke
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    return DMResult(statistic=float(dm_stat_corrected), p_value=float(p_value))


DM_FAMILIES = (
    "model_vs_log_har",       # svaki model naspram log_har referentnog
    "design_vs_5m_bpv_bns",   # svaki dizajn naspram referentnog dizajna
    "all_pairs",              # sva poređenja parova (najveća porodica)
)

REFERENCE_DESIGN_ID = "5m__bpv__bns"


#: kolone koje, zajedno sa statistikom i p-vrednošću, identifikuju JEDAN
#: test. design_id je namerno izostavljen: modeli koji zavise samo od mreže
#: se za poređenja unutar dizajna dele na svih 12 dizajna svoje mreže
#: (results/report.py::_iter_design_cells), pa se isti test pojavljuje pod
#: 12 različitih design_id vrednosti sa bit-identičnim rezultatom.
_TEST_IDENTITY = ("horizon", "model", "baseline", "dm_statistic", "p_value")


def benjamini_hochberg_dm(dm_results: pd.DataFrame, *, family: str, q: float = 0.05) -> pd.DataFrame:
    """BH FDR kontrola preko JEDNE prijavljene porodice DM testova.

    `dm_results` mora imati kolonu `p_value`; vraća kopiju sa dodatim
    `bh_significant` (bool), `bh_threshold` (odsečna p-vrednost porodice),
    `bh_n_tests` (veličina porodice na kojoj je prag računat) i `dm_family`
    (zapisano ime porodice, da se u izlazu uvek vidi nad čime je
    kontrolisano). Isti BH postupak kao measure/jumps/lm_fdr.py, samo preko
    poređenja modela/dizajna umesto preko dana.

    PORODICA SE PRVO DEDUPLIKUJE. Modeli koji zavise samo od mreže ulaze u
    izveštaj 12 puta (jednom po dizajnu svoje mreže) sa bit-identičnim
    testom, jer je to isti test nad istim podacima. BH-ova FDR garancija je
    definisana nad skupom RAZLIČITIH hipoteza, pa ulazak istog testa 12
    puta naduvava m i pomera prag. Izmereno na ovom radu: 1.156 od 1.620
    redova su duplikati, prag je 0.0277 naspram 0.0184 na 551 različitom
    testu, i 44 zaključka se menja (upravo kod char i har_j, dakle kod H2).
    Zato se prag računa nad RAZLIČITIM testovima, pa se zatim primenjuje na
    SVE redove -- duplikati dobijaju isti zaključak kao njihov original,
    kako i treba."""
    if family not in DM_FAMILIES:
        raise ValueError(f"nepoznata DM porodica {family!r}; očekivano jedno od {DM_FAMILIES}")

    out = dm_results.copy()
    out["dm_family"] = family
    if len(out) == 0:
        out["bh_significant"] = pd.Series(dtype=bool)
        out["bh_threshold"] = pd.Series(dtype=float)
        out["bh_n_tests"] = pd.Series(dtype=int)
        return out

    identity = [c for c in _TEST_IDENTITY if c in out.columns]
    distinct = out.drop_duplicates(identity) if identity else out
    p_family = distinct["p_value"].to_numpy(dtype=float)
    m = len(p_family)

    sorted_p = np.sort(p_family)
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds

    p = out["p_value"].to_numpy(dtype=float)
    out["bh_n_tests"] = m
    if not np.any(below):
        out["bh_significant"] = False
        out["bh_threshold"] = 0.0
        return out

    cutoff = float(sorted_p[np.max(np.where(below)[0])])
    out["bh_significant"] = p <= cutoff
    out["bh_threshold"] = cutoff
    return out


def dm_against_baseline(
    daily_losses: pd.DataFrame,
    *,
    baseline_model: str = "log_har",
    loss_col: str = "qlike",
    q: float = 0.05,
) -> pd.DataFrame:
    """Unapred proglašena porodica kontrasta: svaki model naspram
    `baseline_model`, po (horizont, design_id) ćeliji, sa BH FDR kontrolom
    preko cele porodice (ne po ćeliji -- porodica je skup SVIH prijavljenih
    poređenja).

    Gubici se poravnaju na zajedničke (dan, simbol) redove, pa se SVEDU NA
    DNEVNU SERIJU (collapse_to_daily) pre DM testa -- vidi docstring
    modula."""
    rows = []
    for (horizon, design_id), cell in daily_losses.groupby(["horizon", "design_id"], observed=True):
        wide = cell.pivot_table(index=["date", "symbol"], columns="model", values=loss_col).dropna()
        if baseline_model not in wide.columns or len(wide) < 3:
            continue
        n_pairs = len(wide)
        wide = collapse_to_daily(wide)
        if len(wide) < 3:
            continue
        for model_name in wide.columns:
            if model_name == baseline_model:
                continue
            result = diebold_mariano(
                wide[model_name].to_numpy(), wide[baseline_model].to_numpy(), h=int(horizon)
            )
            rows.append(
                {
                    "horizon": horizon,
                    "design_id": design_id,
                    "model": model_name,
                    "baseline": baseline_model,
                    "dm_statistic": result.statistic,
                    "p_value": result.p_value,
                    "n_days": int(len(wide)),
                    "n_symbol_days": int(n_pairs),
                }
            )
    return benjamini_hochberg_dm(pd.DataFrame(rows), family="model_vs_log_har", q=q)
