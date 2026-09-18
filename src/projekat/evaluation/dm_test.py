"""Diebold-Mariano test sa Harvey-Leybourne-Newbold korekcijom za male uzorke
i HAC (Newey-West, minimalno h-1 pomeraja/lag) varijansom -- HAC varijansa
već obrađuje autokorelaciju izazvanu preklapajućim ciljnim vrednostima na
h=5/h=22 (targets.py), na seriji razlike gubitaka.

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


def benjamini_hochberg_dm(dm_results: pd.DataFrame, *, family: str, q: float = 0.05) -> pd.DataFrame:
    """BH FDR kontrola preko JEDNE prijavljene porodice DM testova.

    `dm_results` mora imati kolonu `p_value`; vraća kopiju sa dodatim
    `bh_significant` (bool), `bh_threshold` (odsečna p-vrednost porodice) i
    `dm_family` (zapisano ime porodice, da se u izlazu uvek vidi nad čime
    je kontrolisano). Isti BH postupak kao measure/jumps/lm_fdr.py, samo
    preko poređenja modela/dizajna umesto preko dana."""
    if family not in DM_FAMILIES:
        raise ValueError(f"nepoznata DM porodica {family!r}; očekivano jedno od {DM_FAMILIES}")

    out = dm_results.copy()
    out["dm_family"] = family
    m = len(out)
    if m == 0:
        out["bh_significant"] = pd.Series(dtype=bool)
        out["bh_threshold"] = pd.Series(dtype=float)
        return out

    p = out["p_value"].to_numpy()
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not np.any(below):
        out["bh_significant"] = False
        out["bh_threshold"] = 0.0
        return out

    cutoff = sorted_p[np.max(np.where(below)[0])]
    out["bh_significant"] = p <= cutoff
    out["bh_threshold"] = float(cutoff)
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
    poređenja)."""
    rows = []
    for (horizon, design_id), cell in daily_losses.groupby(["horizon", "design_id"], observed=True):
        wide = cell.pivot_table(index=["date", "symbol"], columns="model", values=loss_col).dropna()
        if baseline_model not in wide.columns or len(wide) < 3:
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
                }
            )
    return benjamini_hochberg_dm(pd.DataFrame(rows), family="model_vs_log_har", q=q)
