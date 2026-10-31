# Dizajn merenja naspram izbora modela u predviđanju realizovane volatilnosti

Kod koji prati diplomski rad. Rad odgovara na pitanje: **da li tačnost prognoze
realizovane volatilnosti (RV) više zavisi od načina na koji se volatilnost meri
ili od izbora modela?**

Projekat je postavljen kao faktorijalni eksperiment. Svaka kombinacija dizajna
merenja i modela prognozira se na istim podacima, a razlike u gubitku prognoze se
zatim razlažu analizom varijanse (ANOVA). Tako se vidi koliki deo varijacije
dolazi od modela, a koliki od merenja.

## Podaci

- Jednominutni barovi za **28 likvidnih američkih akcija**, 2010–2025.
- VIX sa FRED-a, za opis tržišnih režima.
- Dve akcije su isključene posle provere kvaliteta podataka:
  - **MT** — unutar istog dana mešaju se cene prilagođene i neprilagođene za split.
  - **NVDA** — posle prilagođavanja za split cena je u 2014–2016. bila oko 0,46 USD,
    pa je RV merio zaokruživanje na cent, a ne volatilnost.
- Dani sa skraćenim trgovanjem i dani sa premalo podataka izbacuju se jednom, za sve
  mreže uzorkovanja istovremeno. Svi dizajni zato imaju isti skup dana.

## Eksperimentalni faktori

**Dizajn merenja** ima tri faktora, što daje 3 × 3 × 4 = 36 dizajna:

| Faktor | Nivoi |
|---|---|
| A1 — mreža uzorkovanja | 1 min, 5 min, 15 min |
| A2 — estimator otporan na skokove | BPV (bipower variation), MedRV, TrBPV (odsečeni BPV) |
| A3 — test skokova | naivni prag, BNS (Barndorff-Nielsen–Shephard), LM (Lee–Mykland), LM sa FDR kontrolom |

Iz svakog dizajna dobija se dnevni RV, njegova razlaganja na kontinualni deo (C) i
skok (J), polu-varijanse i HAR agregati (dnevni, nedeljni, mesečni).

**Modeli** (16):

| Grupa | Modeli |
|---|---|
| Osnovni | naivni (RV sutra = RV danas) |
| HAR familija | HAR (NNLS na nivoima), log-HAR, HAR-J, CHAR, SHAR, HARQ, HAR-IV |
| Ekonometrijski | ARFIMA, MEM |
| Stabla odlučivanja | LightGBM, XGBoost |
| Osnovni modeli (foundation, zero-shot) | TTM-RV, TTM-C, TTM-C+J, hibrid log-HAR + TTM |

## Šema validacije

- **Proširujući prozor, dnevne prognoze.** Prognoza se pravi svakog trgovačkog dana
  za horizonte **h = 1, 5 i 22 dana**. Cilj je prosečan RV narednih h dana.
- Modeli se ponovo procenjuju na svakih **21 trgovački dan**, na svim podacima pre
  tog datuma. Između dve procene ulazi se ažuriraju svakog dana.
- 2014. je godina zagrevanja. Ocenjuje se period **2015–2025**.
- Modeli u logaritmima vraćaju se u nivoe neparametarskom korekcijom (Duan smearing),
  izračunatom samo iz grešaka poslednjih 252 dana. Nema gledanja u budućnost, što
  proveravaju testovi u `tests/test_walkforward.py`.

## Metrike i analiza

Ocena ima četiri nivoa.

1. **Tačnost.** Glavna metrika je **QLIKE**, jer ostaje pouzdana i kada je RV kao
   aproksimacija stvarne volatilnosti šumovit (Patton, 2011). Uz nju se računaju
   RMSE i MAE, kao i relativni QLIKE u odnosu na **log-HAR**. Relativni QLIKE je
   količnik proseka, a ne prosek količnika.
2. **Značajnost i razlaganje.**
   - Diebold–Mariano test na dnevnoj seriji gubitaka (prosek preko akcija za svaki
     dan), sa Benjamini–Hochberg kontrolom stope lažnih otkrića.
   - Skup najboljih modela (u duhu Model Confidence Set-a) i Mincer–Zarnowitz
     regresija.
   - **ANOVA** (Type III, posebno za svaki horizont) dnevnog QLIKE gubitka po
     faktorima model, mreža, estimator, test skokova, režim i akcija. Rezultat se
     izražava kao parcijalni η².
   - Robusnost se proverava u četiri varijante: sirovi QLIKE, log(QLIKE), rangovi
     modela unutar dana i bez 1% najgorih dana.
3. **Ekonomska ocena.** VaR za h = 1 proverava se na prinosu sledećeg dana
   (Kupiec, Christoffersen, Dynamic Quantile test, Lopez gubitak).
4. **Karakterizacija skokova.** Proverava se koliko se testovi skokova slažu
   (Jaccard, kapa) i koliki je udeo dana sa skokom. **Placebo test** sa 100 permutacija
   proverava da li informacija o skokovima zaista pomaže prognozi.

## Glavni rezultati

Rezultati su iz pokretanja `run_v3`: 246 kombinacija, bez ijednog neuspeha.

- **QLIKE, h = 1, dizajn 5m/BPV/BNS:**
  - Najbolji su LightGBM (0,155) i XGBoost (0,156).
  - HAR familija je zbijena između 0,161 i 0,163 (log-HAR 0,162).
  - Slede HAR na nivoima (0,171), naivni model (0,192) i MEM (0,204).
- **Udeo varijanse (ANOVA, h = 1):** sirovi QLIKE je previše asimetričan da bi išta
  razlučio. Zato čitanje daju log i rang varijante:

  | Varijanta | Model | Mreža |
  |---|---|---|
  | log(QLIKE) | 0,058 | 0,020 |
  | Rangovi unutar dana | 0,110 | 0,061 |

  Merenje, sa interakcijama, nosi oko 36–58% udela koji nosi model. Gotovo ceo
  doprinos merenja dolazi od **mreže uzorkovanja**. Izbor estimatora i testa
  skokova malo utiče.
- **Skokovi:**
  - Prema Diebold–Mariano testu, HAR-J i CHAR se ne razlikuju značajno od log-HAR-a.
  - Placebo test pokazuje da njihova mala prednost ipak potiče od stvarnih dana sa
    skokom (p ≈ 0,01–0,02).
  - Zaključak: signal postoji, ali je mali.
- **TTM modeli:** 81,9% dana u periodu ocene prethodi kraju njihovog pretreniranja.
  Zato su u analizi zadržani, ali označeni kao kontaminirani.

## Struktura koda

```
src/projekat/
  data/        preuzimanje 1m barova, čišćenje, berzanski kalendar, VIX
  measure/     mreže, estimatori, testovi skokova, panel (šema u measure/schema.py)
  model/       modeli, walk-forward, ciljevi, vraćanje iz logaritma
  evaluation/  gubici, DM, MCS, ANOVA, VaR, stratifikacija, dijagnostike
  qa/          provere kvaliteta panela
  results/     čuvanje prognoza i izrada izveštaja
scripts/       ulazne tačke celog lanca
tests/         testovi na sintetičkim podacima (bez mreže)
```

Estimatori, testovi skokova i modeli registruju se preko dekoratora. Za novi
estimator ili model dovoljan je jedan novi fajl.

## Pokretanje

Potreban je [uv](https://docs.astral.sh/uv/) i ključ za LSE vault u `.env`
(pogledati `.env.example`).

```bash
uv sync
uv run pytest                      # testovi

uv run python scripts/fetch_data.py                     # 1. preuzimanje 1m barova
uv run python -m projekat.build_panel ...               # 2. panel (36 dizajna)
uv run python scripts/run_parallel.py --models fast ... # 3. prognoze
uv run python scripts/run_report.py ...                 # 4. evaluacija
uv run python scripts/run_diagnostics.py ...            # 5. placebo i dijagnostike
```

Ceo lanac (koraci 2–5) pokreće se jednom komandom, sa nastavkom posle prekida:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

Podaci (`data/`) i rezultati (`results/`) nisu u repozitorijumu. Mogu se ponovo
izgraditi iz vaulta i sa FRED-a. Celo pokretanje traje oko 9,5 sati.
