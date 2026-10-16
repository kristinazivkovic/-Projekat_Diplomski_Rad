"""Dijagnostika izgrađenog panela -- brojevi koje rad navodi o merenju.

Jedna "uredna" (tidy) tabela, jedan red po broju, sa kolonom `section`:

  jump_day_share           udeo dana sa skokom po (mreža, ocenjivač, test),
                           na celom panelu (svi simboli, svi dani)
  jaccard                  Jaccard indeks između dva testa skoka na istoj
                           mreži i istom ocenjivaču, nad (simbol, dan)
                           poravnatim danima
  days_per_grid            broj dana po (simbol, mreža)
  days_identical_across_grids
                           1 ako simbol ima ISTI skup dana na svakoj mreži,
                           inače 0 (red sa symbol="__all__" = 1 samo ako
                           važi za sve simbole)
  zero_rv_days             broj dana sa RV_d = 0 po (simbol, mreža) -- znak
                           da je korak cene prevelik (unazad prilagođene
                           cene na mali nivo); QLIKE na takvom danu ne
                           postoji
  rv_ratio_1m_over_5m      medijana odnosa RV(1m)/RV(5m) po simbolu; ~1.1-1.2
                           je uobičajen šum mikrostrukture, vrednosti preko 2
                           znače da 1m meri zaokruživanje cene
  iv_nonpositive_share     udeo redova sa IV_d <= 0 po dizajnu -- na čistom
                           panelu 0 (jedini legitiman način je dan bez ijedne
                           promene cene, tj. pregrub korak cene)
  iv_nan_share             udeo redova sa IV_d = NaN po dizajnu -- treba 0
  iv_above_rv_share        udeo redova sa IV_d > RV_d po dizajnu. NIJE GREŠKA:
                           ocenjivač otporan na skokove je šumovita ocena iste
                           veličine koju ocenjuje i RV, pa na danu bez skoka
                           pada iznad RV u otprilike pola slučajeva (zato
                           decompose() uzima max(RV - IV, 0)). Zanimljivo je
                           odstupanje od ~0.5, a ne to što nije nula.
  corporate_checks_skipped broj simbola kojima je provera korporativnih
                           akcija preskočena jer preuzimanje nije uspelo
                           (spisak simbola u `detail`)
  excluded_days            broj izbačenih dana po (simbol, razlog), plus
                           zbir po razlogu (symbol="__all__")

Nema preciznosti/odziva: ne postoji osnovna istina o tome šta je "pravi"
skok, pa nijedan test nije referentna oznaka.

Poziva se iz build_panel.py posle izgradnje; može i samostalno nad već
sačuvanim fajlovima:
    uv run python -m projekat.qa.panel_diagnostics --panel data/processed/panel.parquet
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from projekat.qa.checks import diagnostic_iv_sanity, jaccard_index

COLUMNS = ["section", "symbol", "grid", "estimator", "jump_test", "jump_test_b", "reason", "value", "detail"]


def side_file_paths(panel_path: Path) -> dict[str, Path]:
    """Putanje pratećih fajlova panela (ista osnova imena)."""
    panel_path = Path(panel_path)
    stem = panel_path.with_suffix("")
    return {
        "exclusions": Path(f"{stem}_exclusions.parquet"),
        "corporate": Path(f"{stem}_corporate.csv"),
        "diagnostics": Path(f"{stem}_diagnostics.csv"),
    }


def _row(section: str, value, **fields) -> dict:
    row = {c: "" for c in COLUMNS}
    row.update(fields)
    row["section"] = section
    row["value"] = value
    return row


def build_panel_diagnostics(
    panel: pd.DataFrame, exclusions: pd.DataFrame | None, corporate: pd.DataFrame | None
) -> pd.DataFrame:
    from projekat.evaluation.anova import parse_design_id

    rows: list[dict] = []
    designs = panel[["design_id"]].drop_duplicates()
    parsed = {d: parse_design_id(d) for d in designs["design_id"]}
    df = panel[["design_id", "symbol", "date", "jump_flag"]].copy()
    df["grid"] = df["design_id"].map(lambda d: parsed[d]["grid"])
    df["estimator"] = df["design_id"].map(lambda d: parsed[d]["estimator"])
    df["jump_test"] = df["design_id"].map(lambda d: parsed[d]["jump_test"])
    df["jump_flag"] = df["jump_flag"].astype(bool)

    # -- udeo dana sa skokom po (mreža, ocenjivač, test) -------------------
    share = df.groupby(["grid", "estimator", "jump_test"], observed=True)["jump_flag"].agg(["mean", "size"])
    for (grid, est, test), rec in share.iterrows():
        rows.append(_row("jump_day_share", float(rec["mean"]), grid=grid, estimator=est,
                         jump_test=test, detail=f"n_days={int(rec['size'])}"))

    # -- Jaccard između testova, ista mreža i ocenjivač ---------------------
    for (grid, est), cell in df.groupby(["grid", "estimator"], observed=True):
        by_test = {t: g.set_index(["symbol", "date"])["jump_flag"] for t, g in cell.groupby("jump_test")}
        for a, b in itertools.combinations(sorted(by_test), 2):
            common = by_test[a].index.intersection(by_test[b].index)
            value = jaccard_index(by_test[a].loc[common], by_test[b].loc[common]) if len(common) else float("nan")
            rows.append(_row("jaccard", value, grid=grid, estimator=est, jump_test=a, jump_test_b=b,
                             detail=f"n_days={len(common)}"))

    # -- dani po (simbol, mreža) i da li su skupovi identični --------------
    days = df.drop_duplicates(["symbol", "grid", "date"])
    all_identical = True
    for symbol, sym_days in days.groupby("symbol"):
        sets = {g: frozenset(x["date"]) for g, x in sym_days.groupby("grid")}
        for grid, s in sorted(sets.items()):
            rows.append(_row("days_per_grid", len(s), symbol=symbol, grid=grid))
        identical = len(set(sets.values())) == 1
        all_identical &= identical
        rows.append(_row("days_identical_across_grids", int(identical), symbol=symbol,
                         detail=",".join(sorted(sets))))
    rows.append(_row("days_identical_across_grids", int(all_identical), symbol="__all__"))

    # -- kvantizacija cena: dani sa RV = 0 i odnos RV(1m)/RV(5m) -----------
    # Simbol čije su cene unazad prilagođene na mali nivo (npr. NVDA na ~0.46
    # USD 2014-2016 posle podela 4:1 i 10:1) ima korak cene od procenta i
    # više: RV tada meri zaokruživanje, a ne volatilnost. Znaci su dani sa
    # RV_d = 0 (na grubljim mrežama cena se ne promeni ceo dan) i odnos
    # RV(1m)/RV(5m) daleko iznad tipičnih ~1.1-1.2. Takav simbol ne sme tiho
    # da uđe u evaluaciju: na njemu QLIKE nije definisan, a i kad jeste,
    # jedan simbol može nositi gotovo celu masu gubitka.
    if "RV_d" in panel.columns:
        rv = panel[["symbol", "date", "design_id", "RV_d"]].copy()
        rv["grid"] = rv["design_id"].map(lambda d: parsed[d]["grid"])
        rv = rv.drop_duplicates(["symbol", "date", "grid"])
        zero = rv[rv["RV_d"] <= 0]
        for (symbol, grid), n in zero.groupby(["symbol", "grid"]).size().items():
            rows.append(_row("zero_rv_days", int(n), symbol=symbol, grid=grid))
        rows.append(_row("zero_rv_days", int(len(zero)), symbol="__all__"))
        wide = rv.pivot_table(index=["symbol", "date"], columns="grid", values="RV_d")
        if {"1m", "5m"} <= set(wide.columns):
            ratio = (wide["1m"] / wide["5m"].replace(0, float("nan"))).groupby(level="symbol").median()
            for symbol, value in ratio.items():
                rows.append(_row("rv_ratio_1m_over_5m", float(value), symbol=symbol,
                                 detail="median over days; ~1.1-1.2 is ordinary microstructure noise"))
            rows.append(_row("rv_ratio_1m_over_5m", float(ratio.median()), symbol="__all__"))

    # -- ispravnost ocenjivača IV_d ----------------------------------------
    # IV_d je vrednost iz koje se izvodi cela C/J dekompozicija, pa se meri
    # po dizajnu. PAŽNJA pri čitanju: iv_above_rv_share NIJE greška -- BPV i
    # srodni ocenjivači su šumovita ocena iste integrisane varijanse koju
    # ocenjuje i RV, pa na danu bez skoka padaju iznad RV otprilike u pola
    # slučajeva (zato decompose() i uzima max(RV - IV, 0)). Crvena zastavica
    # su iv_nonpositive_share i iv_nan_share, koji na čistom panelu treba da
    # budu nula.
    if "IV_d" in panel.columns:
        for design_id, cell in panel.groupby("design_id", observed=True):
            for name, value in diagnostic_iv_sanity(cell).items():
                f = parsed[design_id]
                rows.append(_row(name, float(value), grid=f["grid"], estimator=f["estimator"],
                                 jump_test=f["jump_test"]))
        for name, value in diagnostic_iv_sanity(panel).items():
            rows.append(_row(name, float(value), symbol="__all__",
                             detail="iv_above_rv_share ~0.5 is ordinary estimator noise, not a defect"))

    # -- preskočene provere korporativnih akcija ---------------------------
    if corporate is not None and not corporate.empty:
        skipped = corporate[corporate["status"] == "skipped_fetch_failed"]
        rows.append(_row("corporate_checks_skipped", int(len(skipped)), symbol="__all__",
                         detail=",".join(sorted(skipped["symbol"]))))
    else:
        rows.append(_row("corporate_checks_skipped", float("nan"), symbol="__all__", detail="no corporate log"))

    # -- izbačeni dani po razlogu -------------------------------------------
    if exclusions is not None and not exclusions.empty:
        counts = exclusions.groupby(["symbol", "reason"]).size()
        for (symbol, reason), n in counts.items():
            rows.append(_row("excluded_days", int(n), symbol=symbol, reason=reason))
        for reason, n in exclusions.groupby("reason").size().items():
            rows.append(_row("excluded_days", int(n), symbol="__all__", reason=reason))

    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True)
    args = parser.parse_args()
    panel_path = Path(args.panel)
    paths = side_file_paths(panel_path)
    panel = pd.read_parquet(panel_path, columns=["design_id", "symbol", "date", "jump_flag"])
    exclusions = pd.read_parquet(paths["exclusions"]) if paths["exclusions"].exists() else None
    corporate = pd.read_csv(paths["corporate"]) if paths["corporate"].exists() else None
    out = build_panel_diagnostics(panel, exclusions, corporate)
    out.to_csv(paths["diagnostics"], index=False)
    print(f"wrote {paths['diagnostics']} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
