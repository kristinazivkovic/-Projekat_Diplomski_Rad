"""Задатак 1: декомпозиција (а) БЕЗ TTM-(C+J), сви хоризонти, све четири варијанте.

Изведено из постојећих daily_losses_h*.parquet. Ништа се не тренира поново.
Вредности за пун скуп од шест модела читају се из већ израчунатог
anova_robustness_design_sensitive.csv, па се овде рачунају само прилагодбе
за подскуп од пет модела.

    uv run python scripts/analiza_1_anova_bez_ttmcj.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from projekat.evaluation.anova import (
    _drop_top_loss_days,
    _within_day_ranks,
    decompose_variance_design_sensitive,
)
from projekat.results.report import _split_by_dependence

REP = ROOT / "results" / "run_v3" / "report"
OUT = ROOT / "results" / "run_v3" / "dopunske_analize"

HORIZONTI = (1, 5, 22)
VARIJANTE = ("raw", "log", "within_day_rank", "raw_ex_top1pct")
COLS = ["date", "symbol", "design_id", "model", "horizon", "qlike", "regime"]

MERENJE = ["C(grid, Sum)", "C(estimator, Sum)", "C(jump_test, Sum)",
           "C(grid, Sum):C(estimator, Sum)", "C(grid, Sum):C(jump_test, Sum)"]
BEZ_MREZE = ["C(estimator, Sum)", "C(jump_test, Sum)",
             "C(grid, Sum):C(estimator, Sum)", "C(grid, Sum):C(jump_test, Sum)"]
MODEL = "C(model, Sum)"


def primeni_varijantu(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Исте трансформације као anova.robustness_table."""
    f = frame.copy()
    if variant == "log":
        f["qlike"] = np.log(f["qlike"].clip(lower=1e-12))
    elif variant == "within_day_rank":
        f["qlike"] = _within_day_ranks(f, "qlike")
    elif variant == "raw_ex_top1pct":
        f = _drop_top_loss_days(f, "qlike", share=0.01)
    return f


def sazmi(share: pd.Series, n_obs: int, n_models: int, horizon: int,
          variant: str, subset: str) -> dict:
    mer = float(sum(share.get(t, 0.0) for t in MERENJE))
    bez = float(sum(share.get(t, 0.0) for t in BEZ_MREZE))
    mod = float(share.get(MODEL, np.nan))
    return {
        "horizon": horizon, "variant": variant, "subset": subset,
        "merenje_ukupno": mer, "merenje_bez_mreze": bez, "model": mod,
        "odnos_merenje": mer / mod if mod else np.nan,
        "odnos_bez_mreze": bez / mod if mod else np.nan,
        "n_obs": n_obs, "n_models": n_models,
    }


def pun_skup() -> list[dict]:
    """Вредности за свих шест модела, из већ израчунатог артефакта."""
    r = pd.read_csv(REP / "anova_robustness_design_sensitive.csv")
    redovi = []
    for h in HORIZONTI:
        for v in VARIJANTE:
            s = r[(r["horizon"] == h) & (r["variant"] == v)].set_index("term")["partial_eta_sq"]
            n = int(r[(r["horizon"] == h) & (r["variant"] == v)]["n_obs"].iloc[0])
            redovi.append(sazmi(s, n, 6, h, v, "sa_ttm_c_plus_j"))
    return redovi


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sazetak = pun_skup()
    clanovi = []

    for h in HORIZONTI:
        daily = pd.read_parquet(REP / f"daily_losses_h{h}.parquet", columns=COLS)
        ds, _ = _split_by_dependence(daily)
        del daily
        ds = ds[ds["model"] != "ttm_c_plus_j"]
        print(f"h={h}: {len(ds):,} редова, {ds['model'].nunique()} модела", flush=True)

        for v in VARIJANTE:
            frame = primeni_varijantu(ds, v)
            t = decompose_variance_design_sensitive(frame, loss_col="qlike", clustered=False)[h]
            share = t.variance_share.drop(index=["Residual"], errors="ignore")
            sazetak.append(sazmi(share, t.n_obs, int(frame["model"].nunique()),
                                 h, v, "bez_ttm_c_plus_j"))
            for term, val in share.items():
                clanovi.append({"horizon": h, "variant": v, "subset": "bez_ttm_c_plus_j",
                                "term": term, "partial_eta_sq": float(val)})
            red = sazetak[-1]
            print(f"   {v:16s} мерење={red['merenje_ukupno']:.5f} "
                  f"без мреже={red['merenje_bez_mreze']:.5f} "
                  f"модел={red['model']:.5f} "
                  f"односи {red['odnos_merenje']:.3f} / {red['odnos_bez_mreze']:.3f}", flush=True)
            del frame, t
            pd.DataFrame(sazetak).to_csv(OUT / "01_anova_sazetak.csv", index=False)
            pd.DataFrame(clanovi).to_csv(OUT / "01_anova_clanovi.csv", index=False)
        del ds

    print(f"\nуписано у {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
