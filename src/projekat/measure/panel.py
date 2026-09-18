"""Sastavljanje panela. Razrešava A2/A3 implementacije iz registra SAMO PO
IMENU -- ovaj modul nikad ne sme da uveze (import) konkretan modul
ocenjivača ili testa skoka. Dodavanje šestog ocenjivača ili petog testa
skoka mora zahtevati tačno jedan novi fajl na drugom mestu sa dekoratorom
@register_estimator / @register_jump_test, i nijednu izmenu ovde.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.config import LM_LOCAL_WINDOW_K
from projekat.measure import estimators, jumps  # noqa: F401  (sporedni efekat: registracija)
from projekat.measure.decompose import decompose
from projekat.measure.design import MeasurementDesign
from projekat.measure.jumps._gumbel import local_sigma_series
from projekat.measure.jumps.lm_fdr import DayStat, lm_fdr_significance_series
from projekat.measure.realized import (
    daily_return,
    realized_quarticity,
    realized_variance,
    signed_jump_variation,
    signed_semivariances,
)
from projekat.measure.registry import ESTIMATORS, JUMP_TESTS
from projekat.measure.rolling import add_har_aggregates


def _concat_returns_in_order(returns_by_day: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Spoji nizove prinosa po danu (već izgrađene nezavisno po danu -- vidi
    grids.py) u jedan spljošten niz HRONOLOŠKIM REDOM, zajedno sa nizom
    indeksa dana koji označava kom danu pripada svaki element. Ovo spajanje
    nikad ne uvodi noćni prinos: niz svakog dana je izračunat isključivo iz
    sopstvenih cena tog dana."""
    days = sorted(returns_by_day)
    all_r = np.concatenate([returns_by_day[d] for d in days]) if days else np.array([])
    day_of_index = np.concatenate(
        [np.full(len(returns_by_day[d]), i) for i, d in enumerate(days)]
    ) if days else np.array([], dtype=int)
    return days, all_r, day_of_index


def _lm_day_stats(
    days: list[str], returns_by_day: dict[str, np.ndarray], grid: str
) -> tuple[dict[str, DayStat], dict[str, np.ndarray]]:
    """Izračunaj LM max-statistiku i Gumbel p-vrednost za svaki dan,
    koristeći klizni (trailing) K-prozor izvučen iz cele istorijske sekvence
    prinosa (tako da prozor može da se proteže i pre sopstvenog početka
    dana). Takođe vraća per-dan isečak local_sigma niza (usklađen sa
    returns_by_day[day]) tako da pozivalac može da prosledi JUMP_TESTS['lm']
    / JUMP_TESTS['lm_fdr'] isti local_sigma koji je ovde već izračunat,
    umesto da duplira Gumbel logiku ovde."""
    from projekat.measure.jumps._gumbel import exceedance_p_value

    k = LM_LOCAL_WINDOW_K[grid]
    _, all_r, day_of_index = _concat_returns_in_order(returns_by_day)
    sigma = local_sigma_series(all_r, k)

    out: dict[str, DayStat] = {}
    sigma_by_day: dict[str, np.ndarray] = {}
    for i, day in enumerate(days):
        mask = day_of_index == i
        r_day = all_r[mask]
        sigma_day = sigma[mask]
        sigma_by_day[day] = sigma_day
        valid = ~np.isnan(sigma_day) & (sigma_day > 0)
        n = len(r_day)
        if not np.any(valid) or n < 2:
            continue
        l_stats = np.full(n, np.nan)
        l_stats[valid] = np.abs(r_day[valid]) / sigma_day[valid]
        max_l = float(np.nanmax(l_stats))
        out[day] = DayStat(day=day, max_l=max_l, p_value=exceedance_p_value(max_l, n), n=n)
    return out, sigma_by_day


def build_symbol_design(
    symbol: str,
    design: MeasurementDesign,
    returns_by_day: dict[str, np.ndarray],
    half_day_flags: dict[str, bool],
    vix_lag1_by_day: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Sastavi dnevni panel za jedan par (simbol, dizajn). Svaka A2/A3
    implementacija se traži u registru po design.estimator /
    design.jump_test -- ova funkcija ne imenuje nijednu konkretnu
    implementaciju."""
    estimator_fn = ESTIMATORS[design.estimator]
    jump_test_name = design.jump_test
    days = sorted(returns_by_day)

    # lm i lm_fdr unapred zahtevaju statistiku po danu izračunatu preko cele
    # istorijske sekvence (i klizni K-prozor i klizni W-dnevni FDR prozor
    # zahtevaju hronološki kontekst širi od jednog dana).
    lm_stats_by_day: dict[str, DayStat] = {}
    lm_sigma_by_day: dict[str, np.ndarray] = {}
    lm_fdr_sig_by_day: dict[str, tuple[bool, bool]] = {}
    if jump_test_name in ("lm", "lm_fdr"):
        lm_stats_by_day, lm_sigma_by_day = _lm_day_stats(days, returns_by_day, design.grid)
        if jump_test_name == "lm_fdr":
            ordered_stats = [lm_stats_by_day[d] for d in days if d in lm_stats_by_day]
            lm_fdr_sig_by_day = lm_fdr_significance_series(ordered_stats)

    rows = []
    for day in days:
        r = returns_by_day[day]
        n = len(r)
        if n < 2:
            continue

        rv = realized_variance(r)
        # MIN_INTRADAY_OBS se primenjuje uzvodno, za celu mrežu, u
        # has_sufficient_coverage iz data/clean.py -- dan koji je pretanak za
        # RQ/MedRV/TrBPV uopšte ne stiže do ove funkcije, ni za JEDAN
        # ocenjivač na toj mreži (Revizija 9: ranija verzija je ovaj prag
        # primenjivala po ocenjivaču ovde, što je ostavljalo npr. bpv
        # dizajne sa više upotrebljivih dana nego medrv/trbpv dizajne na
        # istoj mreži -- neuravnotežen faktorijalni dizajn koji kvari udele
        # varijanse u ANOVA analizi). Tako da do trenutka kad r stigne
        # ovde, n već zadovoljava prag i robust/rq se uvek mogu izračunati.
        robust = estimator_fn(r, threshold=design.threshold)
        rq = realized_quarticity(r)
        rs_pos, rs_neg = signed_semivariances(r)
        sj = signed_jump_variation(rs_pos, rs_neg)
        r_d = daily_return(r)

        fdr_warmup = False
        if jump_test_name == "naive":
            verdict = JUMP_TESTS["naive"](r, rv, robust)
        elif jump_test_name == "bns":
            verdict = JUMP_TESTS["bns"](r, rv, robust)
        elif jump_test_name == "lm":
            stat = lm_stats_by_day.get(day)
            if stat is None:
                verdict = JUMP_TESTS["naive"](r, rv, robust)  # rezervni scenario kad K-istorija nije dovoljna
            else:
                verdict = JUMP_TESTS["lm"](r, rv, robust, local_sigma=lm_sigma_by_day[day])
        elif jump_test_name == "lm_fdr":
            precomputed = lm_fdr_sig_by_day.get(day)
            if precomputed is None:
                verdict = JUMP_TESTS["naive"](r, rv, robust)
                fdr_warmup = True
            else:
                _, fdr_warmup = precomputed
                verdict = JUMP_TESTS["lm_fdr"](
                    r, rv, robust, local_sigma=lm_sigma_by_day[day], precomputed=precomputed
                )
        else:
            raise ValueError(f"unknown jump_test {jump_test_name!r}")

        decomposition = decompose(rv, robust, verdict)

        rows.append(
            {
                "date": pd.Timestamp(day),
                "symbol": symbol,
                "design_id": design.id,
                "RV_d": rv,
                "C_d": decomposition.C,
                "J_d": decomposition.J,
                "jump_flag": verdict.significant,
                "RS_pos": rs_pos,
                "RS_neg": rs_neg,
                "SJ": sj,
                "RQ": rq,
                "r_d": r_d,
                "n_obs": n,
                "is_half_day": half_day_flags.get(day, False),
                "fdr_warmup": fdr_warmup,
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for col in ("RV", "C", "J"):
        df = add_har_aggregates(df, col)

    if vix_lag1_by_day is not None:
        df["vix_lag1"] = df["date"].dt.strftime("%Y-%m-%d").map(vix_lag1_by_day)
    else:
        df["vix_lag1"] = np.nan

    return df
