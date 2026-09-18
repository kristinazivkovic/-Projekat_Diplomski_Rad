"""Nivo 3: konstrukcija i bekbekstestiranje VaR-a (Kupiec, Christoffersen,
Dynamic Quantile, Lopez gubitak).

Odluka C4: RV se računa ISKLJUČIVO iz intradnevnih prinosa -- noćni prinos
je namerno isključen od strane sloja za merenje -- pa je sqrt(RV_hat)
standardna devijacija SAMO intradnevnog kretanja. Probijanja (breaches)
zato MORAJU biti merena u odnosu na `r_d` (od otvaranja do zatvaranja, već
u panelu), nikad u odnosu na seriju prinosa od zatvaranja do zatvaranja
koja uključuje noćni jaz koji RV ne pokriva. Korišćenje prinosa od
zatvaranja do zatvaranja bi sistematski potcenilo VaR rizik i svaki Kupiec
test bi odbacio hipotezu kao artefakt neusklađenosti jedinica, a ne kao
pravi nalaz. Ovo je navedeno ograničenje VaR analize: odnosi se samo na
intradnevni rizik.

VaR bektestovi se računaju na SPOJENOJ (pooled) uniji svih test prozora
(~2.772 dana kroz 11 prozora), a ne po pojedinačnom prozoru -- jedan prozor
od 252 dana očekuje samo ~2.5 probijanja na nivou od 1%, što Kupiec testu
daje suštinski nikakvu snagu (power).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def value_at_risk(rv_hat: np.ndarray, *, level: float = 0.01) -> np.ndarray:
    """VaR = kvantil * sqrt(RV_hat) -- standardna devijacija, ne varijansa."""
    z = stats.norm.ppf(1 - level)
    return z * np.sqrt(rv_hat)


def breaches_against_r_d(r_d: np.ndarray, var_estimate: np.ndarray) -> np.ndarray:
    """Serija probijanja (breach) usklađena sa konstrukcijom RV-a koja
    obuhvata samo intradnevni deo: probijanje je gubitak (negativan r_d iznad
    -VaR) koji prelazi VaR prag. r_d ima znak; VaR je pozitivan prag na
    strani gubitka, pa se poredi -r_d (gubitak) sa var_estimate."""
    loss = -r_d
    return (loss > var_estimate).astype(int)


@dataclass(frozen=True)
class KupiecResult:
    n_breaches: int
    n_obs: int
    lr_statistic: float
    p_value: float


def kupiec_test(breach_series: np.ndarray, *, level: float) -> KupiecResult:
    n = len(breach_series)
    x = int(np.sum(breach_series))
    pi_hat = x / n if n > 0 else 0.0
    if pi_hat in (0.0, 1.0) or level in (0.0, 1.0):
        lr = 0.0
    else:
        ll_null = x * np.log(level) + (n - x) * np.log(1 - level)
        ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
        lr = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return KupiecResult(n_breaches=x, n_obs=n, lr_statistic=float(lr), p_value=float(p_value))


@dataclass(frozen=True)
class ChristoffersenResult:
    lr_statistic: float
    p_value: float


def christoffersen_test(breach_series: np.ndarray) -> ChristoffersenResult:
    """Test nezavisnosti: verovatnoća probijanja ne bi trebalo da zavisi od
    toga da li je juče bilo probijanje."""
    b = breach_series
    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

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


def dynamic_quantile_test(breach_series: np.ndarray, var_series: np.ndarray, *, level: float, n_lags: int = 4) -> DynamicQuantileResult:
    """Test nezavisnosti zasnovan na regresiji: pogoci (hits = probijanje -
    nivo) regresirani na pomerene (lagged) pogotke i sam VaR nivo; testira
    da li su svi koeficijenti == 0."""
    import statsmodels.api as sm

    hit = breach_series.astype(float) - level
    n = len(hit)
    if n <= n_lags + 2:
        return DynamicQuantileResult(f_statistic=float("nan"), p_value=float("nan"))

    y = hit[n_lags:]
    X_cols = [hit[n_lags - lag : n - lag] for lag in range(1, n_lags + 1)]
    X_cols.append(var_series[n_lags:])
    X = np.column_stack(X_cols)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    f_stat = model.fvalue
    p_value = model.f_pvalue
    return DynamicQuantileResult(f_statistic=float(f_stat), p_value=float(p_value))


def lopez_loss(r_d: np.ndarray, var_estimate: np.ndarray) -> float:
    """Probijanja (u odnosu na gubitak zasnovan na r_d, usklađeno sa
    konstrukcijom RV-a koja obuhvata samo intradnevni deo) kažnjena
    srazmerno svojoj veličini, a ne samo prebrojana."""
    loss = -r_d
    breach = loss > var_estimate
    penalty = np.where(breach, 1 + (loss - var_estimate) ** 2, 0.0)
    return float(np.mean(penalty))


def var_backtest_table(
    results_df: "pd.DataFrame",
    *,
    level: float = 0.01,
    y_hat_col: str = "y_hat",
    r_d_col: str = "r_d",
) -> "pd.DataFrame":
    """SPOJEN (pooled) VaR bektest po (model, design_id, horizont) -- nikad
    po prozoru.

    Jedan prozor od 252 dana očekuje ~2.5 probijanja na nivou od 1%, pa
    Kupiec na njemu nema praktično nikakvu snagu; spajanjem svih 11 test
    prozora (~2.772 dana) test postaje informativan. Zato se `window_id`
    NE nalazi među grupišućim kolonama -- to je namerno, ne propust.

    TAČNI BROJEVI PROBIJANJA se prijavljuju pored p-vrednosti: p-vrednost
    bez broja probijanja skriva razliku između "3 probijanja od očekivanih
    28" i "28 od 28", a upravo taj odnos je ono što se u radu tumači.
    """
    import pandas as pd

    rows = []
    for (model, design_id, horizon), cell in results_df.groupby(
        ["model", "design_id", "horizon"], observed=True
    ):
        cell = cell.sort_values(["symbol", "date"])
        rv_hat = cell[y_hat_col].to_numpy(dtype=float)
        r_d = cell[r_d_col].to_numpy(dtype=float)
        ok = ~np.isnan(rv_hat) & ~np.isnan(r_d) & (rv_hat > 0)
        if ok.sum() < 10:
            continue
        rv_hat, r_d = rv_hat[ok], r_d[ok]

        var_estimate = value_at_risk(rv_hat, level=level)
        breaches = breaches_against_r_d(r_d, var_estimate)

        kupiec = kupiec_test(breaches, level=level)
        christoffersen = christoffersen_test(breaches)
        dq = dynamic_quantile_test(breaches, var_estimate, level=level)

        rows.append(
            {
                "model": model,
                "design_id": design_id,
                "horizon": horizon,
                "level": level,
                "n_obs": kupiec.n_obs,
                "n_breaches": kupiec.n_breaches,          # tačan broj, uz p-vrednosti
                "expected_breaches": level * kupiec.n_obs,
                "breach_rate": kupiec.n_breaches / kupiec.n_obs if kupiec.n_obs else float("nan"),
                "kupiec_lr": kupiec.lr_statistic,
                "kupiec_p": kupiec.p_value,
                "christoffersen_lr": christoffersen.lr_statistic,
                "christoffersen_p": christoffersen.p_value,
                "dq_f": dq.f_statistic,
                "dq_p": dq.p_value,
                "lopez_loss": lopez_loss(r_d, var_estimate),
                "pooled_over_windows": True,
            }
        )
    return pd.DataFrame(rows)
