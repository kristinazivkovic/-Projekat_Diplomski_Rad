"""Nivo 3: konstrukcija i bektestiranje VaR-a (Kupiec, Christoffersen,
Dynamic Quantile, Lopez gubitak).

SAMO h = 1, PROGNOZA SA DANA t NASPRAM PRINOSA DANA t+1. Prognoza
napravljena na kraju dana t je prognoza varijanse dana t+1, pa se
probijanje meri prema r_d(t+1) ISTOG simbola (kolona r_d_next, koju
walkforward.py upisuje samo kad je sledeći red simbola baš sledeći dan
trgovanja). Poređenje sa r_d(t) bi merilo prognozu prema prinosu koji je
već bio poznat i čiji je RV_t među ulazima modela -- probijanja bi bila
sistematski premalo. Na h = 5/22 y_hat nije varijansa sledećeg dana, pa se
ti horizonti ne bektestiraju.

Odluka C4: RV se računa ISKLJUČIVO iz intradnevnih prinosa (noćni prinos
je isključen), pa je sqrt(RV_hat) standardna devijacija SAMO intradnevnog
kretanja; r_d je zbir intradnevnih prinosa dana (poslednja cena dana na
mreži minus prva), nikad prinos od zatvaranja do zatvaranja.

OBJEDINJENO PREKO PROZORA I SIMBOLA, ALI NIKAD PREKO GRANICE SIMBOLA.
Kupiec broji probijanja nad svim danima i simbolima (jedan prozor od 252
dana očekuje ~2.5 probijanja na nivou od 1%, što testu ne daje snagu).
Christoffersen i DQ zavise od redosleda, pa se prelazi i pomeraji računaju
UNUTAR svakog simbola i tek onda sabiraju -- poslednji dan jednog simbola
nikad nije "juče" za prvi dan sledećeg.

Nula probijanja je ODBACIVANJE (previše konzervativan VaR), ne prolaz:
Kupiec koristi xlogy (0 * log 0 = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


def value_at_risk(rv_hat: np.ndarray, *, level: float = 0.01) -> np.ndarray:
    """VaR = kvantil * sqrt(RV_hat) -- standardna devijacija, ne varijansa."""
    z = stats.norm.ppf(1 - level)
    return z * np.sqrt(rv_hat)


def breaches_against_r_d(r_d: np.ndarray, var_estimate: np.ndarray) -> np.ndarray:
    """Probijanje: gubitak -r_d veći od VaR praga."""
    loss = -r_d
    return (loss > var_estimate).astype(int)


def _group_bounds(groups) -> list[tuple[int, int]]:
    """Granice uzastopnih blokova iste grupe (niz mora biti sortiran po grupi)."""
    if groups is None:
        return []
    g = np.asarray(groups)
    if len(g) == 0:
        return []
    change = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(g)]])
    return list(zip(starts, ends))


@dataclass(frozen=True)
class KupiecResult:
    n_breaches: int
    n_obs: int
    lr_statistic: float
    p_value: float


def kupiec_test(breach_series: np.ndarray, *, level: float) -> KupiecResult:
    n = len(breach_series)
    x = int(np.sum(breach_series))
    if n == 0 or level in (0.0, 1.0):
        return KupiecResult(n_breaches=x, n_obs=n, lr_statistic=float("nan"), p_value=float("nan"))
    pi_hat = x / n
    # xlogy: 0*log(0) = 0, so zero breaches gives the proper (large) LR --
    # an over-conservative VaR is rejected, not passed
    ll_null = special.xlogy(x, level) + special.xlogy(n - x, 1 - level)
    ll_alt = special.xlogy(x, pi_hat) + special.xlogy(n - x, 1 - pi_hat)
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return KupiecResult(n_breaches=x, n_obs=n, lr_statistic=float(lr), p_value=float(p_value))


@dataclass(frozen=True)
class ChristoffersenResult:
    lr_statistic: float
    p_value: float


def christoffersen_test(breach_series: np.ndarray, *, groups=None) -> ChristoffersenResult:
    """Test nezavisnosti: verovatnoća probijanja ne sme zavisiti od toga da
    li je juče bilo probijanje. Sa `groups` (npr. simbol, niz sortiran po
    grupi pa datumu) prelazi se broje unutar grupe i sabiraju."""
    b = np.asarray(breach_series)
    bounds = _group_bounds(groups) or [(0, len(b))]
    n00 = n01 = n10 = n11 = 0
    for lo, hi in bounds:
        prev, cur = b[lo:hi - 1], b[lo + 1:hi]
        n00 += int(np.sum((prev == 0) & (cur == 0)))
        n01 += int(np.sum((prev == 0) & (cur == 1)))
        n10 += int(np.sum((prev == 1) & (cur == 0)))
        n11 += int(np.sum((prev == 1) & (cur == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(p, n_event, n_total):
        if p in (0.0, 1.0) or n_total == 0:
            return 0.0
        return n_event * np.log(p) + (n_total - n_event) * np.log(1 - p)

    ll_null = _ll(pi, n01 + n11, n00 + n01 + n10 + n11)
    ll_alt = _ll(pi01, n01, n00 + n01) + _ll(pi11, n11, n10 + n11)
    lr = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return ChristoffersenResult(lr_statistic=float(lr), p_value=float(p_value))


@dataclass(frozen=True)
class DynamicQuantileResult:
    f_statistic: float
    p_value: float


def dynamic_quantile_test(
    breach_series: np.ndarray, var_series: np.ndarray, *, level: float, n_lags: int = 4, groups=None
) -> DynamicQuantileResult:
    """Pogoci (probijanje - nivo) regresirani na sopstvene pomeraje i VaR
    nivo; testira da li su svi koeficijenti == 0. Sa `groups` se pomeraji
    grade unutar svake grupe (prvih n_lags redova grupe otpada), pa se
    regresija radi nad naslaganim redovima."""
    import statsmodels.api as sm

    hit_all = np.asarray(breach_series, dtype=float) - level
    var_all = np.asarray(var_series, dtype=float)
    bounds = _group_bounds(groups) or [(0, len(hit_all))]
    ys, xs = [], []
    for lo, hi in bounds:
        hit, var = hit_all[lo:hi], var_all[lo:hi]
        n = len(hit)
        if n <= n_lags:
            continue
        ys.append(hit[n_lags:])
        cols = [hit[n_lags - lag: n - lag] for lag in range(1, n_lags + 1)]
        cols.append(var[n_lags:])
        xs.append(np.column_stack(cols))
    if not ys or sum(len(y) for y in ys) <= n_lags + 2:
        return DynamicQuantileResult(f_statistic=float("nan"), p_value=float("nan"))

    y = np.concatenate(ys)
    X = sm.add_constant(np.vstack(xs), has_constant="add")
    # With no breach (or only breaches) the lagged hits are constant and
    # collinear with the intercept: the test is undefined there -- Kupiec
    # already carries that cell's verdict.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return DynamicQuantileResult(f_statistic=float("nan"), p_value=float("nan"))
    model = sm.OLS(y, X).fit()
    return DynamicQuantileResult(f_statistic=float(model.fvalue), p_value=float(model.f_pvalue))


def lopez_loss(r_d: np.ndarray, var_estimate: np.ndarray) -> float:
    """Probijanja kažnjena srazmerno veličini premašaja, ne samo prebrojana."""
    loss = -r_d
    breach = loss > var_estimate
    penalty = np.where(breach, 1 + (loss - var_estimate) ** 2, 0.0)
    return float(np.mean(penalty))


def var_backtest_table(
    results_df: "pd.DataFrame",
    *,
    level: float = 0.01,
    y_hat_col: str = "y_hat",
    return_col: str = "r_d_next",
) -> "pd.DataFrame":
    """VaR bektest po (model, design_id), SAMO h = 1, prognoza sa dana t
    naspram r_d(t+1) istog simbola; objedinjeno preko prozora i simbola,
    ali Christoffersen/DQ nikad preko granice simbola. Tačni brojevi
    probijanja se prijavljuju pored p-vrednosti."""
    import pandas as pd

    df = results_df[results_df["horizon"] == 1]
    rows = []
    for (model, design_id), cell in df.groupby(["model", "design_id"], observed=True):
        cell = cell.sort_values(["symbol", "date"])
        rv_hat = cell[y_hat_col].to_numpy(dtype=float)
        r_next = cell[return_col].to_numpy(dtype=float)
        ok = np.isfinite(rv_hat) & np.isfinite(r_next) & (rv_hat > 0)
        if ok.sum() < 10:
            continue
        rv_hat, r_next = rv_hat[ok], r_next[ok]
        symbols = cell["symbol"].to_numpy()[ok]

        var_estimate = value_at_risk(rv_hat, level=level)
        breaches = breaches_against_r_d(r_next, var_estimate)
        kupiec = kupiec_test(breaches, level=level)
        christoffersen = christoffersen_test(breaches, groups=symbols)
        dq = dynamic_quantile_test(breaches, var_estimate, level=level, groups=symbols)
        rows.append(
            {
                "model": model,
                "design_id": design_id,
                "horizon": 1,
                "level": level,
                "n_obs": kupiec.n_obs,
                "n_symbols": int(len(np.unique(symbols))),
                "n_breaches": kupiec.n_breaches,
                "expected_breaches": level * kupiec.n_obs,
                "breach_rate": kupiec.n_breaches / kupiec.n_obs if kupiec.n_obs else float("nan"),
                "kupiec_lr": kupiec.lr_statistic,
                "kupiec_p": kupiec.p_value,
                "christoffersen_lr": christoffersen.lr_statistic,
                "christoffersen_p": christoffersen.p_value,
                "dq_f": dq.f_statistic,
                "dq_p": dq.p_value,
                "lopez_loss": lopez_loss(r_next, var_estimate),
                "pooled_over_windows_and_symbols": True,
            }
        )
    return pd.DataFrame(rows)
