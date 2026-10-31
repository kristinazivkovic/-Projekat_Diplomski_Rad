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
- VIX sa FRED-a, sa pomakom od jednog dana, kao ulaz modela HAR-IV, LightGBM i XGBoost.
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

Rezultati su iz pokretanja `run_v3`: 246 kombinacija, bez ijednog neuspeha. Ocena
obuhvata 2015–2025, oko 76.700 parova akcija i dana po kombinaciji.

| Hipoteza | Ishod | Ukratko |
|---|---|---|
| H1 — dizajn merenja je ravnopravan izboru modela | delimično potvrđena | Mreža ima veliki udeo, ali je to gornja granica. Estimator i test skokova malo utiču. |
| H2 — informacija o skokovima poboljšava prognozu | nije potvrđena | Signal postoji, ali je mali i njegov znak zavisi od dizajna. |
| H3 — poredak modela se menja pri ekonomskoj oceni | potvrđena | Poredak po QLIKE i po VaR gubitku se slabo slaže. |

### Tačnost

Prosečan QLIKE na referentnom dizajnu 5m/BPV/BNS (manje je bolje):

| Model | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| LightGBM | **0,155** | 0,132 | 0,203 |
| XGBoost | 0,156 | **0,131** | 0,203 |
| ARFIMA | 0,161 | 0,136 | 0,220 |
| log-HAR (osnova za poređenje) | 0,162 | 0,133 | 0,197 |
| HAR-J | 0,162 | 0,133 | 0,197 |
| CHAR | 0,162 | 0,133 | 0,197 |
| HAR na nivoima | 0,171 | 0,142 | **0,179** |
| TTM-RV | 0,175 | 0,150 | 0,218 |
| naivni | 0,192 | 0,182 | 0,260 |
| MEM | 0,204 | 0,170 | 0,238 |

- **Najbolji model zavisi od horizonta.** Stabla odlučivanja značajno pobeđuju log-HAR
  samo na h = 1 (LightGBM je 4,5% bolji, DM p < 0,001). Na h = 5 i h = 22 ta razlika
  nije značajna.
- **HAR na nivoima ide suprotnim smerom.** Na h = 1 i h = 5 značajno je lošiji od
  log-HAR-a. Na h = 22 je najbolji (9% niži QLIKE), ali ta razlika ne prolazi test
  (p = 0,10).

### Razlaganje varijanse

Sirovi QLIKE je previše asimetričan i na h = 1 svi udeli su oko 1e-5. Zato čitanje
daju log i rang varijante. Tabela prikazuje parcijalni η² za modele osetljive na dizajn:

| Faktor | log, h = 1 | log, h = 5 | log, h = 22 | rangovi, h = 1 | rangovi, h = 5 | rangovi, h = 22 |
|---|---|---|---|---|---|---|
| Model | 0,058 | 0,085 | 0,066 | 0,110 | 0,137 | 0,105 |
| Mreža | 0,020 | 0,006 | 0,002 | 0,061 | 0,029 | 0,013 |
| Estimator | 0,001 | 0,001 | 0,001 | 0,002 | 0,002 | 0,001 |
| Test skokova | < 0,001 | < 0,001 | 0,001 | < 0,001 | 0,001 | 0,001 |

- **Gotovo ceo efekat merenja dolazi od mreže uzorkovanja** i slabi sa horizontom. Među
  modelima koji zavise samo od mreže ona prevazilazi i izbor modela (h = 1: 0,025 prema
  0,001 u log varijanti).
- **Udeo mreže je gornja granica.** Mreža menja i RV prema kome se gubitak ocenjuje, pa
  deo njenog udela odražava težinu samog cilja. Kada se poređenje ograniči na isti cilj,
  čist efekat dizajna na h = 1 iznosi samo **0,001** (log) i **0,003** (rangovi).
- **Udeo modela zavisi od skupa modela.** Veliki deo nosi jedan loš model, TTM-C+J. Čist
  efekat dizajna je 0,019–0,024 udela modela sa njim, a 0,31–0,43 bez njega.

### Skokovi

- **Diebold–Mariano:** na referentnom dizajnu HAR-J i CHAR se ne razlikuju značajno od
  log-HAR-a ni na jednom horizontu. Preko svih dizajna značajni nalazi idu u oba smera,
  a na h = 1 ih je čak više u korist log-HAR-a (25 prema 19).
- **Placebo test:** prednost na 5-minutnoj mreži potiče od stvarnih dana sa skokom
  (p ≈ 0,01–0,02 za BNS, LM i LM-FDR). Izuzetak je naivni test, gde je HAR-J lošiji od
  log-HAR-a (p = 1,0).
- **Udeo dana sa skokom jako zavisi od dizajna.** LM označava 9% dana na 15m, a 51% na
  1m. LM-FDR označava više dana od LM-a (39% prema 20% na 5m), pa **nije**
  konzervativnija varijanta.

### Ekonomska ocena

- VaR 1% **odbacuje se Kupiec testom za svih 246 kombinacija**. Stopa probijanja je
  1,2–3,7% (medijana 1,5%), pa prognoza uz pretpostavku normalnosti potcenjuje rizik.
- Kendallov τ između poretka po QLIKE i po Lopez gubitku iznosi 0,30 na proseku preko
  dizajna (p = 0,12). Slaganje je slabo i ne može se statistički razlikovati od nule.
- HAR na nivoima je 11. od 16 po QLIKE na h = 1, a prvi po Lopez gubitku.

### Ograničenja

- **TTM modeli:** 81,9% dana u periodu ocene (2.264 od 2.766) prethodi kraju njihovog
  pretreniranja, pa su zadržani u analizi, ali označeni kao kontaminirani. Čisto je samo
  razdoblje 2024–2025.
- **Skup najboljih modela** je sekvencijalna eliminacija u duhu Model Confidence Set-a,
  a ne bootstrap procedura Hansena, Lundea i Nasona.
- **VaR** obuhvata samo unutardnevni prinos, bez noćnog gepa.

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

Rezultati izveštenog pokretanja nalaze se u [`results/run_v3/report/`](results/run_v3/report/):
tabele za sva četiri nivoa ocene, a pored njih `results/run_v3/manifest.json` sa verzijom
koda, heš vrednošću panela i verzijama paketa.

Podaci (`data/`), pojedinačne prognoze (`results/run_v3/tables/`, 4,2 GB) i dnevni gubici
(`daily_losses_h*.parquet`, po 1,3 GB) nisu u repozitorijumu zbog veličine. Mogu se ponovo
izgraditi iz vaulta i sa FRED-a. Celo pokretanje traje oko 9,5 sati.
