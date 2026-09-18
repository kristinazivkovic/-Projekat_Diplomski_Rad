"""ANOVA dekompozicija varijanse -- broj radi čijeg dobijanja postoji H3.

Odluka C7: zavisna promenljiva je DNEVNI QLIKE gubitak, jedan red po
(model, dizajn, horizont, simbol, dan) -- ovo daje pravu replikaciju unutar
svake ćelije, tako da je greška (error term) procenjiva bez da se
interakcije spoje (pool) i izgube. Sredine ćelija ne bi ostavile nikakav
član greške.

PO HORIZONTU, NIKAD SPOJENO (pooled). Horizont je u ključu reda, ali NIJE
faktor u formuli: h=1, h=5 i h=22 imaju različite skale ciljne veličine
(h-dnevni prosek je to glatkiji što je h veći), pa spajanje baca
objašnjivu varijansu u član greške, što DEFLATIRA svaki parcijalni
eta-kvadrat -- uključujući udeo merenja radi kog rad postoji. Zato
`decompose_*` vraća po jednu dekompoziciju PO HORIZONTU
({horizont: ANOVATable}), i te se dekompozicije ne spajaju.

SIMBOL JE EKSPLICITAN BLOKIRAJUĆI FAKTOR, iz istog razloga: ostavljen u
rezidualu, naduvava član greške i smanjuje SVE prijavljene udele.

TIP III SUMA KVADRATA, SA SUM-TO-ZERO KONTRASTIMA. Dizajn je neuravnotežen
po konstrukciji (modeli koji zavise samo od mreže dobijaju duplirane
redove preko 12 kombinacija ocenjivač x test_skoka), i postoje tri namerna
interakciona člana -- pri takvom dizajnu se Tip II i Tip III NE slažu oko
glavnih efekata. Tip III bez sum-to-zero kontrasta nije dobro definisan
(glavni efekti zavise od proizvoljnog izbora referentne kategorije), pa se
kontrastno kodiranje EKSPLICITNO postavlja (Sum kontrasti), a ne prepušta
podrazumevanoj biblioteci (patsy podrazumevano koristi Treatment).

Interakcioni članovi mreža x ocenjivač i mreža x test skoka su uključeni
jer su STRUKTURNI, a ne slučajni: šum mikrostrukture pristrasno podiže RV
otprilike linearno sa brojem intradnevnih opservacija (n), a ocenjivač
otporan na skokove (BPV itd.) je tim istim šumom pogođen drugačije -- pa
ponašanje ocenjivača u odnosu na RV zaista zavisi od mreže. model x
test_skoka je uključen jer sposobnost modela osetljivog na dizajn da
iskoristi informaciju o skoku zavisi od toga koji ju je test proizveo.
Izostavljanje ovih članova tiho prebacuje deo efekta u glavne efekte.

design_id se rastavlja isključivo preko parse_design_id() -- inverza
MeasurementDesign.id sa njegovim fiksnim redosledom polja. Nikad ne deliti
design_id ad hoc nigde drugde u kodu.

=====================================================================
ZAVISNOST IZMEĐU DIZAJNA -- OGRANIČENJE KOJE MORA U TEKST RADA
=====================================================================
Različiti dizajni merenja izvedeni su iz ISTIH sirovih 1-minutnih svećica,
pa su RV na 1m i 5m za isti dan visoko korelisani. Pretpostavka
nezavisnosti standardnog F-testa ovde NE VAŽI. Zato:

  * Parcijalni eta-kvadrat se prijavljuje kao DESKRIPTIVAN udeo varijanse
    -- to je primarna prezentacija i ne zahteva nezavisnost.
  * Za svaki prijavljeni F ili p: DVOSMERNO klasterisane standardne greške
    (Cameron-Gelbach-Miller), klasterisane ISTOVREMENO po danu i po
    simbolu -- ne dve odvojene jednosmerne opcije.
  * Opciono: blok butstrap preko dana, sa dužinom bloka JEDNAKOM horizontu
    h -- ista konvencija koju koristi evaluation/mcs.py preko
    diebold_mariano(h=...). Dve inferencijalne putanje na različitim
    konvencijama bloka bi se razilazile po konstrukciji.

=====================================================================
ROBUSNOST NA ASIMETRIJU -- OBAVEZNA, NE OPCIONA
=====================================================================
Dnevni QLIKE je izrazito desno asimetričan, a uzorak obuhvata COVID i
2018, pa šačica dana može odrediti udele varijanse dok izlazna tabela
izgleda savršeno normalno. Zato robustness_table() pokreće istu
dekompoziciju TRI načina i prijavljuje ih JEDNO PORED DRUGOG:
  1. sirov dnevni QLIKE (primarno),
  2. log(QLIKE),
  3. rangovi QLIKE-a UNUTAR DANA preko modela.
Ako su udeli stabilni kroz sva tri, to je jaka tvrdnja o robusnosti za
rad. Ako nisu, to mora biti vidljivo PRE nego što se tekst napiše -- zato
je ovo prvoklasna tabela, a ne zakopana dijagnostika. Dodatno se
prijavljuje primarna dekompozicija sa izbačenim gornjim 1% dana po
gubitku, kao direktan odgovor na pitanje "da li bi ovo važilo da ste
izbacili mart 2020?".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sum-to-zero kontrasti: obavezni za dobro definisan Tip III (vidi docstring).
_SUM_CONTRAST = "C({}, Sum)"


def parse_design_id(design_id: str) -> dict[str, str]:
    """Inverzna operacija od MeasurementDesign.id: mreža__ocenjivač__test_skoka,
    sa opcionim sufiksom __t{prag} koji postoji samo za trbpv dizajne."""
    parts = design_id.split("__")
    grid, estimator, jump_test = parts[0], parts[1], parts[2]
    return {"grid": grid, "estimator": estimator, "jump_test": jump_test}


@dataclass(frozen=True)
class ANOVATable:
    sum_sq: pd.Series
    df: pd.Series
    variance_share: pd.Series          # parcijalni eta^2 po članu (deskriptivno)
    horizon: int | None = None
    n_obs: int = 0
    formula: str = ""
    clustered: pd.DataFrame | None = None   # dvosmerno klasterisane SE/p po članu
    note: str = ""


def _prepare(daily_loss_df: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    df = daily_loss_df.copy()
    parsed = df["design_id"].apply(parse_design_id).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df = df.rename(columns={loss_col: "loss"})
    return df


_DESIGN_SENSITIVE_MAIN = ("model", "grid", "estimator", "jump_test", "regime", "symbol")
_DESIGN_SENSITIVE_INTERACTIONS = (
    ("grid", "estimator"),
    ("grid", "jump_test"),
    ("model", "jump_test"),
)
_GRID_ONLY_MAIN = ("model", "grid", "regime", "symbol")


def _varying(df: pd.DataFrame, factors) -> list[str]:
    """Faktori sa bar dva nivoa u OVOM podskupu. Faktor sa jednim nivoom
    nema nijedan stepen slobode, pa Tip III pada na praznoj matrici
    ograničenja ('must have at least one row in constraint matrix') --
    izbacuje se iz formule umesto da sruši dekompoziciju. Dešava se
    legitimno: jedan režim u uskom prozoru, jedan simbol u testu, jedna
    mreža u grid-only podskupu."""
    return [f for f in factors if f in df.columns and df[f].nunique(dropna=True) > 1]


def _build_formula(df: pd.DataFrame, main_factors, interactions=()) -> str:
    c = _SUM_CONTRAST.format
    present = _varying(df, main_factors)
    terms = [c(f) for f in present]
    for a, b in interactions:
        if a in present and b in present:
            terms.append(f"{c(a)}:{c(b)}")
    if not terms:
        raise ValueError("ANOVA: nijedan faktor nema više od jednog nivoa u ovom podskupu")
    return "loss ~ " + " + ".join(terms)


def _design_sensitive_formula(df: pd.DataFrame) -> str:
    return _build_formula(df, _DESIGN_SENSITIVE_MAIN, _DESIGN_SENSITIVE_INTERACTIONS)


def _grid_only_formula(df: pd.DataFrame) -> str:
    return _build_formula(df, _GRID_ONLY_MAIN)


def _two_way_cluster_cov(fit, df: pd.DataFrame) -> pd.DataFrame:
    """Cameron-Gelbach-Miller dvosmerno klasterisane SE (po danu I po
    simbolu istovremeno): V_day + V_symbol - V_day&symbol."""
    groups_day = df["date"].astype(str).to_numpy()
    groups_sym = df["symbol"].astype(str).to_numpy()
    groups_both = np.char.add(np.char.add(groups_day, "|"), groups_sym)

    def _cov(groups):
        return fit.get_robustcov_results(cov_type="cluster", groups=groups, use_correction=True).cov_params()

    cov = _cov(groups_day) + _cov(groups_sym) - _cov(groups_both)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    from scipy import stats as _stats

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(fit.params.to_numpy(), se, out=np.full_like(se, np.nan), where=se > 0)
    p = 2 * (1 - _stats.norm.cdf(np.abs(t)))
    return pd.DataFrame(
        {"coef": fit.params.to_numpy(), "cluster_se": se, "t": t, "p_value": p},
        index=fit.params.index,
    )


def _fit_one(df: pd.DataFrame, formula_builder, horizon: int | None, *, clustered: bool) -> ANOVATable:
    formula = formula_builder(df)
    fit = smf.ols(formula, data=df).fit()
    # typ=3 zahteva sum-to-zero kontraste, koji su ugrađeni u formulu
    table = sm.stats.anova_lm(fit, typ=3)
    table = table.drop(index=[i for i in table.index if i == "Intercept"], errors="ignore")

    ss_resid = table.loc["Residual", "sum_sq"] if "Residual" in table.index else 0.0
    partial_eta2 = table["sum_sq"] / (table["sum_sq"] + ss_resid)

    cluster_df = None
    if clustered:
        try:
            cluster_df = _two_way_cluster_cov(fit, df)
        except Exception as exc:  # pragma: no cover - zavisi od ranga dizajna
            cluster_df = pd.DataFrame({"error": [str(exc)]})

    return ANOVATable(
        sum_sq=table["sum_sq"],
        df=table["df"],
        variance_share=partial_eta2,
        horizon=horizon,
        n_obs=len(df),
        formula=formula,
        clustered=cluster_df,
        note=(
            "Type III SS with sum-to-zero contrasts; symbol is an explicit blocking "
            "factor; partial eta-squared is DESCRIPTIVE (no independence assumption). "
            "Any F/p should be read against the two-way (day, symbol) clustered SEs."
        ),
    )


def _decompose_per_horizon(
    daily_loss_df: pd.DataFrame, formula_builder, loss_col: str, *, clustered: bool
) -> dict[int, ANOVATable]:
    df = _prepare(daily_loss_df, loss_col)
    out: dict[int, ANOVATable] = {}
    for horizon, cell in df.groupby("horizon", observed=True):
        # formula se gradi PO HORIZONTU: faktor može biti konstantan unutar
        # jednog horizonta a varirati u drugom
        out[int(horizon)] = _fit_one(cell, formula_builder, int(horizon), clustered=clustered)
    return out


def decompose_variance_design_sensitive(
    daily_loss_df: pd.DataFrame, *, loss_col: str = "qlike", clustered: bool = True
) -> dict[int, ANOVATable]:
    """Za modele osetljive na dizajn (CHAR, HAR-J, LightGBM, XGBoost,
    PatchTST, ttm_c, ttm_c_plus_j): glavni efekti model, mreža, ocenjivač,
    test_skoka, režim, simbol (blok), PLUS interakcije mreža:ocenjivač,
    mreža:test_skoka, model:test_skoka.

    Vraća {horizont: ANOVATable} -- JEDNA dekompozicija po horizontu,
    nikad spojeno (vidi docstring modula)."""
    return _decompose_per_horizon(daily_loss_df, _design_sensitive_formula, loss_col, clustered=clustered)


def decompose_variance_grid_only(
    daily_loss_df: pd.DataFrame, *, loss_col: str = "qlike", clustered: bool = True
) -> dict[int, ANOVATable]:
    """Za modele koji zavise samo od mreže (naive, HAR (NNLS-levels),
    log_har, SHAR, HARQ, HAR-IV, ARFIMA, MEM, ttm_rv, log_har_ttm):
    sekundarna provera da Faktor A1 ima značaj i bez informacije o
    skokovima. Ovaj skup je tačno {model : model.depends_on == {"grid"}}.

    Vraća {horizont: ANOVATable} -- po horizontu, nikad spojeno."""
    return _decompose_per_horizon(daily_loss_df, _grid_only_formula, loss_col, clustered=clustered)


# ── robusnost na asimetriju (obavezna) ───────────────────────────────────


def _within_day_ranks(df: pd.DataFrame, loss_col: str) -> pd.Series:
    """Rang QLIKE-a UNUTAR (dan, simbol, dizajn, horizont) preko modela."""
    return df.groupby(["date", "symbol", "design_id", "horizon"], observed=True)[loss_col].rank()


def _drop_top_loss_days(df: pd.DataFrame, loss_col: str, *, share: float = 0.01) -> pd.DataFrame:
    """Izbaci gornji `share` DANA po prosečnom gubitku tog dana (ne
    pojedinačne redove) -- direktan odgovor na 'da li bi važilo bez marta 2020'."""
    per_day = df.groupby("date", observed=True)[loss_col].mean()
    cutoff = per_day.quantile(1.0 - share)
    keep = per_day[per_day <= cutoff].index
    return df[df["date"].isin(keep)]


def robustness_table(
    daily_loss_df: pd.DataFrame,
    *,
    decomposition: str = "design_sensitive",
    loss_col: str = "qlike",
) -> pd.DataFrame:
    """PRVOKLASNA tabela robusnosti na asimetriju: ista dekompozicija
    izvedena četiri načina, jedna do druge:

      variant="raw"          sirov dnevni QLIKE (primarno)
      variant="log"          log(QLIKE)
      variant="within_day_rank"  rangovi QLIKE-a unutar dana preko modela
      variant="raw_ex_top1pct"   sirov QLIKE bez gornjeg 1% dana

    Jedan red po (variant, horizont, član), sa parcijalnim eta-kvadratom.
    Ako su udeli stabilni kroz varijante -- jaka tvrdnja o robusnosti. Ako
    nisu, mora se videti PRE pisanja teksta."""
    decompose = (
        decompose_variance_design_sensitive
        if decomposition == "design_sensitive"
        else decompose_variance_grid_only
    )

    base = daily_loss_df.copy()
    variants: dict[str, pd.DataFrame] = {"raw": base}

    log_df = base.copy()
    log_df[loss_col] = np.log(log_df[loss_col].clip(lower=1e-12))
    variants["log"] = log_df

    rank_df = base.copy()
    rank_df[loss_col] = _within_day_ranks(rank_df, loss_col)
    variants["within_day_rank"] = rank_df

    variants["raw_ex_top1pct"] = _drop_top_loss_days(base, loss_col, share=0.01)

    rows = []
    for variant, frame in variants.items():
        try:
            per_horizon = decompose(frame, loss_col=loss_col, clustered=False)
        except Exception as exc:  # pragma: no cover
            rows.append({"variant": variant, "horizon": None, "term": "ERROR", "partial_eta_sq": np.nan, "note": str(exc)})
            continue
        for horizon, table in per_horizon.items():
            for term, share in table.variance_share.items():
                if term == "Residual":
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "horizon": horizon,
                        "term": term,
                        "partial_eta_sq": float(share),
                        "n_obs": table.n_obs,
                        "note": "",
                    }
                )
    return pd.DataFrame(rows)


# ── alternativna inferencijalna putanja: blok butstrap ───────────────────


def block_bootstrap_variance_shares(
    daily_loss_df: pd.DataFrame,
    *,
    horizon: int,
    decomposition: str = "design_sensitive",
    loss_col: str = "qlike",
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Opciona alternativna inferencijalna putanja: blok butstrap PREKO
    DANA sa dužinom bloka = `horizon`, ista konvencija koju koristi
    evaluation/mcs.py (preko diebold_mariano(h=...)). Dve putanje na
    različitim konvencijama bloka razilazile bi se po konstrukciji.

    Vraća po članu: butstrap sredinu i 2.5/97.5 percentile parcijalnog
    eta-kvadrata."""
    decompose = (
        decompose_variance_design_sensitive
        if decomposition == "design_sensitive"
        else decompose_variance_grid_only
    )
    cell = daily_loss_df[daily_loss_df["horizon"] == horizon]
    days = np.sort(cell["date"].unique())
    block_len = max(int(horizon), 1)
    n_blocks = max(len(days) // block_len, 1)
    rng = np.random.default_rng(seed)

    draws: list[pd.Series] = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(len(days) - block_len, 1), size=n_blocks)
        picked = np.concatenate([days[s : s + block_len] for s in starts])
        sample = cell[cell["date"].isin(picked)]
        try:
            table = decompose(sample, loss_col=loss_col, clustered=False)[horizon]
        except Exception:  # pragma: no cover - degenerisan izvlačenje
            continue
        draws.append(table.variance_share.drop(index=["Residual"], errors="ignore"))

    if not draws:
        return pd.DataFrame(columns=["term", "boot_mean", "ci_lo", "ci_hi", "n_draws"])

    stacked = pd.concat(draws, axis=1)
    return pd.DataFrame(
        {
            "term": stacked.index,
            "boot_mean": stacked.mean(axis=1).to_numpy(),
            "ci_lo": stacked.quantile(0.025, axis=1).to_numpy(),
            "ci_hi": stacked.quantile(0.975, axis=1).to_numpy(),
            "n_draws": len(draws),
        }
    ).reset_index(drop=True)


# ── HOOK: agregacija dve dekompozicije u jedan broj ──────────────────────
# NAMERNO NEIMPLEMENTIRANO. Da li se decompose_variance_design_sensitive i
# decompose_variance_grid_only smeju spojiti u jedan "headline" broj je
# OTVORENO PITANJE sa mentorom. Do te odluke, dve dekompozicije se vraćaju
# ODVOJENO i ne agregiraju se. Ako odluka padne, implementacija ide ovde --
# i mora eksplicitno navesti kako se tretira to što dva skupa modela imaju
# različit broj redova po dizajnu (neuravnoteženost koja je i razlog za
# Tip III iznad).


def aggregate_headline_share(*_args, **_kwargs):  # pragma: no cover - namerni hook
    raise NotImplementedError(
        "Aggregating the design-sensitive and grid-only decompositions into a single "
        "headline number is an open question with the advisor. Until that is decided, "
        "report the two decompositions separately (see the module docstring). This hook "
        "exists so the decision has one obvious implementation site."
    )
