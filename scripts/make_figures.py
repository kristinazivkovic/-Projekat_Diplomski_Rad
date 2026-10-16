"""Slike za rad, iz POSTOJEĆIH artefakata u results/run_v3/report/.

    uv run python scripts/make_figures.py

Ništa se ne pokreće ponovo i nijedan broj se ne računa iznova osim sabiranja
članova koji su već u anova_robustness_design_sensitive.csv.

Izlaz (vektorski PDF, u rad/slike/):
  udeli_varijanse.pdf  dekompozicija (a), log varijanta, po horizontu:
                       udeo modela, udeo mreže i udeo merenja BEZ mreže
  udeo_skokova.pdf     udeo dana sa skokom po mreži za svaka četiri testa,
                       estimator BPV

Obe slike su čitljive crno-belo: bez boje, razlika nosi šrafura i ivica.
Oznake su ćirilične, decimalni zarez.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "results" / "run_v3" / "report"
OUT = ROOT / "rad" / "slike"

HORIZONTI = (1, 5, 22)
#: udeo merenja BEZ glavnog efekta mreže -- estimator i test skoka, uz njihove
#: interakcije sa mrežom. Svi ti članovi porede dizajne UNUTAR iste mreže, pa
#: se ocenjuju u odnosu na isti cilj; glavni efekat mreže je izostavljen jer
#: mreža menja i proksi u odnosu na koji se gubitak računa.
BEZ_MREZE = ["C(estimator, Sum)", "C(jump_test, Sum)",
             "C(grid, Sum):C(estimator, Sum)", "C(grid, Sum):C(jump_test, Sum)"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})
zarez = FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ","))
zarez2 = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def udeli_varijanse() -> Path:
    r = pd.read_csv(REP / "anova_robustness_design_sensitive.csv")
    r = r[r["variant"] == "log"]
    model, mreza, bez = [], [], []
    for h in HORIZONTI:
        s = r[r["horizon"] == h].set_index("term")["partial_eta_sq"]
        model.append(float(s["C(model, Sum)"]))
        mreza.append(float(s["C(grid, Sum)"]))
        bez.append(float(sum(s[t] for t in BEZ_MREZE)))

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = range(len(HORIZONTI))
    w = 0.26
    serije = [
        ("модел", model, "white", "///"),
        ("мрежа (горња граница)", mreza, "0.75", ""),
        ("мерење без мреже", bez, "0.35", ".."),
    ]
    for i, (ime, vals, boja, sraf) in enumerate(serije):
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=ime,
               facecolor=boja, edgecolor="black", linewidth=0.7, hatch=sraf)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"$h = {h}$" for h in HORIZONTI])
    ax.set_xlabel("хоризонт предвиђања")
    ax.set_ylabel("парцијални ета-квадрат")
    ax.yaxis.set_major_formatter(zarez)
    # Legenda iznad crtane povrsine, da ni u jednom slucaju ne prekrije stubic.
    ax.legend(frameon=False, fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", linewidth=0.4, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "udeli_varijanse.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def udeo_skokova() -> Path:
    jc = pd.DataFrame(json.load(open(REP / "jump_characterization.json", encoding="utf-8")))
    f = jc["design_id"].str.split("__", expand=True)
    jc["grid"], jc["est"], jc["jt"] = f[0], f[1], f[2]
    jc = jc[jc["est"] == "bpv"]
    mreze = ["1m", "5m", "15m"]
    #: oznake na x-osi: obican uspravan tekst, ne mathtext -- u $1m$ se "m"
    #: crta kao matematicka promenljiva u kurzivu, sto ovde nije promenljiva
    #: nego jedinica.
    oznake_mreza = {"1m": "1 мин", "5m": "5 мин", "15m": "15 мин"}
    testovi = [("naive", "наивни", "white", "///"),
               ("bns", "BNS", "0.8", ""),
               ("lm", "LM", "0.5", "xx"),
               ("lm_fdr", "LM-FDR", "0.25", "..")]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = range(len(mreze))
    w = 0.2
    for i, (key, ime, boja, sraf) in enumerate(testovi):
        vals = [float(jc[(jc["grid"] == g) & (jc["jt"] == key)]["jump_day_share"].iloc[0])
                for g in mreze]
        ax.bar([xi + (i - 1.5) * w for xi in x], vals, w, label=ime,
               facecolor=boja, edgecolor="black", linewidth=0.7, hatch=sraf)
    ax.set_xticks(list(x))
    ax.set_xticklabels([oznake_mreza[g] for g in mreze])
    ax.set_xlabel("учесталост узорковања")
    ax.set_ylabel("удео дана са скоком")
    ax.yaxis.set_major_formatter(zarez2)
    ax.set_ylim(0, 0.85)
    # Legenda iznad crtane povrsine: ranije je „upper center“ presecala
    # najvisi stubic (LM-FDR na 1 мин).
    ax.legend(frameon=False, fontsize=8, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", linewidth=0.4, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "udeo_skokova.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (udeli_varijanse, udeo_skokova):
        print(f"  {fn():}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
