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
# prema broju tikova (vidi plan, korak 2), plus 5 akcija koje nisu iz SAD po
# sedištu (i dalje se trguju na američkim berzama -- navedeno ograničenje,
# a ne zamena za ne-američki indeks).

US_SYMBOLS = [
    "NVDA", "AAPL", "MSFT", "INTC", "AMD", "BAC", "AMZN", "MU", "C", "JPM",
    "WFC", "CSCO", "XOM", "ORCL", "QCOM", "PFE", "T", "NFLX", "MS", "AMAT",
    "WMT", "CVX", "DIS", "VZ", "F",
]
NON_US_SYMBOLS = ["ACN", "MT", "CB", "SU", "AEM"]
UNIVERSE = US_SYMBOLS + NON_US_SYMBOLS

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

# ── QA pragovi (kontrola kvaliteta) ──────────────────────────────────────

MIN_BAR_COVERAGE = 0.80        # izbaci dan ako je ispod 80% sopstvenog očekivanog broja svećica
JUMP_DAY_SHARE_BOUNDS = (0.05, 0.15)   # striktno proveravano (assert) samo za `bns`
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
