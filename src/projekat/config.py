"""Centralna konfiguracija za sloj merenja: univerzum simbola, period uzorka,
trgovinski kalendar i fiksni parametri dizajna (K, W, TrBPV prag)
koje plan zahteva da budu eksplicitno navedeni, a ne implicitni.
"""

from __future__ import annotations

from pathlib import Path

# ── period uzorka ────────────────────────────────────────────────────────

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# ── univerzum simbola ────────────────────────────────────────────────────
# 25 američkih akcija sa >=16 godina 1-minutne istorije u vault-u, odabranih
# prema broju tikova (vidi plan, korak 2), plus 4 akcije koje nisu iz SAD po
# sedištu (izvorno 5; MT je isključen, vidi EXCLUDED_SYMBOLS ispod) (i dalje se trguju na američkim berzama -- navedeno ograničenje,
# a ne zamena za ne-američki indeks).

US_SYMBOLS = [
    "AAPL", "MSFT", "INTC", "AMD", "BAC", "AMZN", "MU", "C", "JPM",
    "WFC", "CSCO", "XOM", "ORCL", "QCOM", "PFE", "T", "NFLX", "MS", "AMAT",
    "WMT", "CVX", "DIS", "VZ", "F",
]
NON_US_SYMBOLS = ["ACN", "CB", "SU", "AEM"]
UNIVERSE = US_SYMBOLS + NON_US_SYMBOLS

# Izabrani, pa ISKLJUČENI tokom kontrole kvaliteta -- pre bilo kakvog
# modelovanja, sa razlogom zapisanim ovde da odluka bude proverljiva.
# MT (ArcelorMittal): u vault-u su cene 2014-03 .. 2016-03 unutar istog dana
# pomešane između prilagođenog i neprilagođenog nivoa (emisija prava 500:629
# od 2016-03-14), što daje lažne 5-minutne prinose od ~0.23 = ln(629/500) na
# 496 dana. data/corporate.py::verify_split_adjustment ga ispravno odbacuje.
EXCLUDED_SYMBOLS = {
    "MT": "intradnevno pomešane prilagođene/neprilagođene cene 2014-03..2016-03 (496 dana)",
    # NVDA: cene su unazad prilagođene za podele 4:1 (2021) i 10:1 (2024), pa
    # su 2014-2016 oko 0.46 USD i zaokružene na cent -- efektivni korak cene
    # je ~2.2%. Posledice izmerene na izgrađenom panelu: 65 dana sa RV_d = 0
    # (jedini u celom panelu), medijana odnosa RV(1m)/RV(5m) 2.2-2.5 za
    # 2014-2016 naspram 1.14 na celom panelu, i 99.77% ukupne mase QLIKE-a
    # u evaluaciji (naivni model: prosek 14.161 sa NVDA, 2.32 bez nje).
    # Merenje volatilnosti pri tom koraku cene je šum zaokruživanja, ne signal.
    "NVDA": "cene unazad prilagođene na ~0.46 USD 2014-2016, korak ~2.2%: 65 dana sa RV = 0, 99.8% mase QLIKE-a",
}

# ── trgovinski kalendar ──────────────────────────────────────────────────

EXCHANGE_CALENDAR = "XNYS"

# ── Faktor A1: mreže uzorkovanja (sampling grids) ────────────────────────
# Mreža je 1m/5m/15m (NE 10m -- vault nema nativan 10-minutni vremenski okvir:
# 1s,5s,15s,30s,1m,3m,5m,15m,30m,1h,4h,1d,1w,1mo; 15m JESTE nativno dostupan,
# mada se svaka mreža ipak gradi preuzorkovanjem istih učitanih 1-minutnih
# izvornih svećica, tako da A1 menja samo agregaciju).

GRIDS = ("1m", "5m", "15m")
GRID_MINUTES = {"1m": 1, "5m": 5, "15m": 15}

# Očekivan broj svećica, ključan po (mreži, da li je skraćen dan). Pun NYSE
# trgovinski dan traje 09:30-16:00 po istočnom vremenu (390 minuta); dan sa
# ranijim zatvaranjem traje 09:30-13:00 (210 minuta). Empirijski proveren
# vault za oba slučaja.
EXPECTED_BARS_FULL = {"1m": 390, "5m": 78, "15m": 26}
EXPECTED_BARS_HALF = {"1m": 210, "5m": 42, "15m": 14}

# ── Faktor A2: TrBPV prag ────────────────────────────────────────────────
# Fiksiran umesto da bude uveden kao 4. dimenzija dizajna (vidi plan, Odluka
# o pragu za Threshold BPV): zadržan kao navedena dizajnerska odluka,
# dokumentovana u radu, a ne podešavana (nije "tjunovana").

TRBPV_THRESHOLD = 3.0

# ── Faktor A3: Lee-Mykland lokalni prozor K ──────────────────────────────
# K = 4 trgovinske sesije na SVAKOJ mreži -- namerna odluka, a ne preuzimanje
# iz literature. Lee i Mykland (2008) sami tabeliraju K=156 na 15m
# (6 sesija), ali ovaj plan umesto toga koristi K=104 (4 sesije) na 15m: da se
# dužina K-a u *sesijama* menjala između mreža, klasifikacija skoka za dati
# dan zavisila bi od dve stvari koje se obe menjaju sa Faktorom A1
# (frekvencija uzorkovanja I dužina lokalnog prozora), što bi iskrivilo udeo
# varijanse u ANOVA analizi pripisan Faktoru A1. Držanjem K-a fiksnim na 4
# sesije svuda, jedino frekvencija uzorkovanja se menja kada se menja A1.
# K=1560/312/104 na 1m/5m/15m. NIJE "jedan trgovinski mesec" -- ranija
# računska greška (1560/390 = 4.0, a ne 22) je uočena i ispravljena.

LM_LOCAL_WINDOW_K = {"1m": 1560, "5m": 312, "15m": 104}

# ── Faktor A3: lm_fdr klizni (trailing) prozor ───────────────────────────
# Benjamini-Hochberg korekcija se primenjuje preko prethodnih W dana plus
# tekućeg dana (nikad preko celog uzorka -- to bi unelo unapredno gledanje,
# tzv. lookahead bias). W=250 ~= jedna trgovinska godina. Dani pre nego što
# simbol ima svojih prvih W dana koriste običnu `lm` značajnost umesto FDR
# korekcije (obeleženo kolonom `fdr_warmup`).

LM_FDR_TRAILING_WINDOW = 250

# ── Faktor A3: nivoi značajnosti ─────────────────────────────────────────
# BNS: jednostrani test, p = 1 - Phi(Z), značajno ako p < BNS_ALPHA. 0.001
# (a ne 0.05) jer se test primenjuje jednom po danu na ~4.000 dana po
# simbolu: pri 0.05 bi i bez ijednog skoka ~200 dana po simbolu bilo
# proglašeno danom sa skokom, pa bi J na tim danima bio čist šum
# ocenjivača. 0.001 je nivo koji koriste Andersen, Bollerslev i Diebold
# (2007) za uslovnu dekompoziciju.
BNS_ALPHA = 0.001

# LM: Gumbel kritična vrednost za max|L| dana na nivou poverenja 0.999
# (isti nivo lažne uzbune po danu kao BNS iznad, pa se A3 nivoi razlikuju
# po konstrukciji testa, a ne po izabranom nivou značajnosti).
LM_CONFIDENCE = 0.999

# LM-FDR: Benjamini-Hochberg nivo q preko kliznog prozora od W dana;
# dnevna p-vrednost je Gumbelova p = 1 - exp(-exp(-(max|L| - C_n)/S_n)).
LM_FDR_Q = 0.05

# ── QA pragovi (kontrola kvaliteta) ──────────────────────────────────────

MIN_BAR_COVERAGE = 0.80        # izbaci dan ako je ispod 80% sopstvenog očekivanog broja svećica
JUMP_DAY_SHARE_BOUNDS = (0.05, 0.15)   # referentni opseg; assert_bns_jump_share_in_range ga NE poziva nijedan pipeline

# Skraćeni dani (rano zatvaranje) se izbacuju sa SVIH mreža: 15m skraćeni
# dan daje svega 13 prinosa (< MIN_INTRADAY_OBS), pa bi ga 15m izgubio a
# 1m/5m zadržali -- A1 nivoi bi imali različit skup dana (i to baš mirnih
# dana), što je neuravnotežen faktorijalni plan.
DROP_HALF_DAYS = True
MIN_A2_A3_DIVERGENCE = 0.05    # bns i lm moraju da se razlikuju na >5% dana, inače upozorenje
RQ_HALF_DAY_FLAG_PERCENTILE = 0.99

# Apsolutni donji prag broja intradnevnih opservacija za RQ/MedRV/TrBPV,
# nezavisno od (i pored) MIN_BAR_COVERAGE. Potreban jer relativni filter od
# 80% ne može da uhvati 15-minutni skraćeni dan (n=14): 80% od punog
# dnevnog broja za 15m (26) je samo 20.8, pa 15m dan može proći relativni
# filter sa svega 21 svećicom, i ništa ne sprečava ispravno zadržan
# skraćeni dan (n=14, prema ispravci prozora sesije) da bude korišćen za
# ocenjivač četvrtog momenta koji je nepouzdan pri toj veličini uzorka.
# Provereno da ovo isključuje tačno onu jednu ćeliju kojoj je to potrebno:
# 1m/5m skraćeni dani (210/42) oba prolaze iznad 20; 15m puni dani (26)
# takođe prolaze; samo 15m skraćeni dani (14) padaju ispod praga.
MIN_INTRADAY_OBS = 20

# ── Validaciona šema: prošireni prozor sa dnevnim prognozama ─────────────
# Prozor za obuku uvek počinje od START_DATE i samo se proširuje (nikad
# klizni). Prognoza se pravi SVAKOG dana t od FIRST_FORECAST_DATE; svaki
# trenirani model se ponovo ocenjuje svakih REFIT_EVERY_SESSIONS dana
# trgovanja na SVIM podacima strogo pre datuma ponovne ocene (i red za
# obuku ulazi samo ako mu je cilj realizovan pre tog datuma). Između dve
# ponovne ocene parametri su fiksni, a ulazi/stanje se ažuriraju dnevno.
#
# Nema posebnog validacionog preklopa: korekcija povratne transformacije i
# Mincer-Zarnowitz koeficijenti se za dan t ocenjuju iz SOPSTVENIH
# van-uzoračkih grešaka modela čiji je cilj realizovan u poslednjih
# CORRECTION_WINDOW_SESSIONS dana trgovanja STROGO pre t.
#
# 2014. je godina zagrevanja: prognoze se prave (daju prvih 252 van-uzoračke
# greške), ali se ne ocenjuju. Period evaluacije je EVAL_START..EVAL_END.
HORIZONS = (1, 5, 22)
FIRST_FORECAST_DATE = "2014-01-01"
EVAL_START = "2015-01-01"
EVAL_END = "2025-12-31"
REFIT_EVERY_SESSIONS = 21          # ~jedan mesec trgovanja; isto pravilo za svaki model osim PatchTST
CORRECTION_WINDOW_SESSIONS = 252   # ~jedna godina trgovanja van-uzoračkih grešaka
MZ_MIN_PAIRS = 30                  # manje parova -> MZ koeficijenti su NaN (nema rekalibracije)

# PatchTST se trenira od nule jednom godišnje (prvi dan trgovanja u godini),
# ne svakih 21 dan: ~3.900 treniranja umesto ~46.000. 3 semena (ne 5).
PATCHTST_SEEDS = (0, 1, 2)
PATCHTST_EARLY_STOP_SHARE = 0.10   # poslednjih 10% perioda obuke za rano zaustavljanje
PATCHTST_PATIENCE = 5              # epoha bez poboljšanja pre zaustavljanja (max epoha ostaje 30)

# Režim: jedna oznaka po (simbol, dan) iz RV_m na 5m mreži, naspram medijane
# RV_m 5m iz PRVOG perioda obuke (pre FIRST_FORECAST_DATE); ista oznaka za
# sve dizajne, pa režim ne zavisi od Faktora A1.
REGIME_GRID = "5m"
REGIME_QUANTILE = 0.5

# ── Faktor C: TTM foundation model (WI-7) ────────────────────────────────
# Tačno PRIKOVAN checkpoint, nikad generičko "TTM": različita izdanja imaju
# različite skupove podataka za predtreniranje (r1 ~250M uzoraka, r2 ~700M,
# r2.1 ~1B), pa bi provera kontaminacije naspram neprikovanog modela bila
# besmislena -- ne bi se znalo NAD ČIM je model zapravo predtreniran.
# Ime grane prati shemu <kontekst>-<predikcija>-<ft>-<metrika>-<izdanje>.
TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
# Namerno NE "-ft-" grana: varijante sa frekvencijskim prefiks-tuningom
# zahtevaju `freq_token` na ulazu (inače: "Expecting freq_token in
# forward"), što bi uvelo dodatni, neobrazloženi hiperparametar u
# zero-shot putanju. Obična grana ne traži ništa osim konteksta.
TTM_REVISION = "512-192-r2"       # grana = konkretna varijanta (kontekst 512, predikcija 192)
TTM_CONTEXT_LENGTH = 512          # mora odgovarati prikovanoj grani
TTM_PREDICTION_LENGTH = 192       # mora odgovarati prikovanoj grani; >= max(HORIZONS)=22

# Objavljeni datum odsecanja (cutoff) podataka za predtreniranje prikovanog
# checkpoint-a, korišćen ISKLJUČIVO za proveru kontaminacije
# (model/foundation/contamination.py). TTM r2/r2.1 su predtrenirani na
# javnim, NE-finansijskim serijama (potrošnja struje, vremenske prilike,
# saobraćaj, sunčeve pege, Bitcoin cena, web saobraćaj, epidemiološki
# podaci -- vidi karticu modela) objavljenim pre NeurIPS 2024 publikacije.
# Nijedan skup ne sadrži realizovanu volatilnost akcija iz ovog univerzuma,
# ali se preklapanje PERIODA ipak eksplicitno proverava i zapisuje, jer je
# odsustvo preklapanja tvrdnja koja mora biti proverljiva, a ne
# pretpostavljena.
TTM_PRETRAIN_CUTOFF = "2024-01-01"

# ── putanje ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Evaluation output artefacts (results/report.py) and the contamination
# check (model/foundation/contamination.py). NOT inside data/ -- data/ is
# gitignored because it is entirely regenerable from the vault and FRED,
# whereas results are what gets reported in the thesis.
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
