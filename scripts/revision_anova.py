"""Декомпозиција (а) без `ttm_c_plus_j`, h = 1, варијанте `log` и `within_day_rank`.

Изведено из постојећих daily_losses_h1.parquet; ништа се не тренира поново.
Рангови се рачунају ИЗНОВА над преосталих пет модела, никад филтрирањем
рангова израчунатих над шест.

    uv run python scripts/revision_anova.py --check          # само провера, без прилагодбе
    uv run python scripts/revision_anova.py --variant log
    uv run python scripts/revision_anova.py --variant within_day_rank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

REP = ROOT / "results" / "run_v3" / "report"
OUT = REP / "revision"

MERENJE = ["C(grid, Sum)", "C(estimator, Sum)", "C(jump_test, Sum)",
           "C(grid, Sum):C(estimator, Sum)", "C(grid, Sum):C(jump_test, Sum)"]
BEZ_MREZE = MERENJE[1:]
MODEL = "C(model, Sum)"
COLS = ["date", "symbol", "design_id", "model", "horizon", "qlike", "regime"]


def sazmi(share: pd.Series) -> dict:
    mer = float(sum(share.get(t, 0.0) for t in MERENJE))
    bez = float(sum(share.get(t, 0.0) for t in BEZ_MREZE))
    mod = float(share.get(MODEL, np.nan))
    return {"model": mod, "merenje_ukupno": mer, "merenje_bez_mreze": bez,
            "odnos_merenje": mer / mod if mod else np.nan,
            "odnos_bez_mreze": bez / mod if mod else np.nan}


def sest_modela(variant: str) -> dict:
    """Постојеће вредности за свих шест модела, из већ израчунатог артефакта."""
    r = pd.read_csv(REP / "anova_robustness_design_sensitive.csv")
    sub = r[(r["horizon"] == 1) & (r["variant"] == variant)]
    out = sazmi(sub.set_index("term")["partial_eta_sq"])
    out["n_obs"] = int(sub["n_obs"].iloc[0])
    out["n_models"] = 6
    return out


def provera() -> int:
    ocekivano = {"log": (0.058, 0.021, 0.001), "within_day_rank": (0.110, 0.064, 0.003)}
    print("ПРОВЕРА: репродукција постојећих вредности за шест модела, h = 1\n")
    print(f"{'варијанта':18s} {'величина':22s} {'израчунато':>12s} {'у раду':>9s}  ")
    ok = True
    for v, (e_mod, e_mer, e_bez) in ocekivano.items():
        s = sest_modela(v)
        for ime, dobijeno, ocek in (("модел", s["model"], e_mod),
                                    ("мерење укупно", s["merenje_ukupno"], e_mer),
                                    ("мерење без мреже", s["merenje_bez_mreze"], e_bez)):
            slaze = abs(round(dobijeno, 3) - ocek) < 1e-9
            ok &= slaze
            print(f"{v:18s} {ime:22s} {dobijeno:12.5f} {ocek:9.3f}  {'OK' if slaze else 'НЕ СЛАЖЕ СЕ'}")
    print("\n" + ("све се слаже" if ok else "НЕСЛАГАЊЕ — стани"))
    return 0 if ok else 1


def pripremi(variant: str) -> Path:
    """Фаза 1: подскуп + трансформација зависне променљиве, на диск.

    Одвојена је од прилагодбе јер `within_day_rank` при рангирању прави велику
    међукопију; ако тај врх падне на врх OLS-а, процес бива убијен без изузетка.
    """
    from projekat.evaluation.anova import _within_day_ranks
    from projekat.results.report import _split_by_dependence

    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(REP / "daily_losses_h1.parquet", columns=COLS)
    ds, _ = _split_by_dependence(daily)
    del daily
    ds = ds[ds["model"] != "ttm_c_plus_j"]
    print(f"подскуп: {len(ds):,} редова, {ds['model'].nunique()} модела "
          f"({', '.join(sorted(ds['model'].unique()))})", flush=True)

    if variant == "log":
        ds["qlike"] = np.log(ds["qlike"].clip(lower=1e-12))
    elif variant == "within_day_rank":
        # рангови се рачунају ИЗНОВА, над преосталих пет модела
        ds["qlike"] = _within_day_ranks(ds, "qlike").astype("float32")
    else:
        raise SystemExit(f"непозната варијанта {variant!r}")

    # само оно што формула чита; категорије штеде меморију при учитавању
    ds = ds[["design_id", "model", "symbol", "regime", "horizon", "qlike"]]
    for c in ("design_id", "model", "symbol", "regime"):
        ds[c] = ds[c].astype("category")
    put = OUT / f"_priprema_{variant}.parquet"
    ds.to_parquet(put, index=False)
    print(f"фаза 1 готова -> {put}", flush=True)
    return put


def fit(variant: str) -> int:
    from projekat.evaluation.anova import decompose_variance_design_sensitive

    put = OUT / f"_priprema_{variant}.parquet"
    if not put.exists():
        raise SystemExit(f"нема {put}; прво покрени --prepare {variant}")
    ds = pd.read_parquet(put)
    ds["qlike"] = ds["qlike"].astype("float64")
    # design_id мора бити стринг: anova._prepare ради .apply(parse_design_id),
    # а над категоријском колоном pandas мапира по КАТЕГОРИЈАМА и покушава да
    # речник који та функција враћа употреби као нову категорију.
    ds["design_id"] = ds["design_id"].astype(str)
    print(f"фаза 2: {len(ds):,} редова, {ds['model'].nunique()} модела", flush=True)

    t = decompose_variance_design_sensitive(ds, loss_col="qlike", clustered=False)[1]
    share = t.variance_share.drop(index=["Residual"], errors="ignore")

    rezultat = sazmi(share)
    rezultat.update({"n_obs": t.n_obs, "n_models": int(ds["model"].nunique()),
                     "variant": variant, "subset": "bez_ttm_c_plus_j", "horizon": 1})
    (OUT / f"anova_bez_ttmcj_{variant}.json").write_text(
        json.dumps(rezultat, indent=2, ensure_ascii=False), encoding="utf-8")
    share.rename("partial_eta_sq").to_frame().to_csv(OUT / f"anova_bez_ttmcj_{variant}_clanovi.csv")

    print(f"\n{variant}, пет модела, h = 1:")
    for k in ("model", "merenje_ukupno", "merenje_bez_mreze", "odnos_merenje", "odnos_bez_mreze"):
        print(f"  {k:18s} {rezultat[k]:.5f}")
    print(f"\nуписано у {OUT}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--prepare", choices=["log", "within_day_rank"])
    p.add_argument("--variant", choices=["log", "within_day_rank"])
    a = p.parse_args()
    if a.check:
        return provera()
    if a.prepare:
        pripremi(a.prepare)
        return 0
    if not a.variant:
        raise SystemExit("наведи --check, --prepare или --variant")
    return fit(a.variant)


if __name__ == "__main__":
    raise SystemExit(main())
