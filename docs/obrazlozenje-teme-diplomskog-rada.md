# Predlog teme diplomskog rada

*Verzija 2 — revidirana*

## Radni naslov

**Dizajn merenja naspram izbora modela u predviđanju realizovane volatilnosti: koliko razlika u performansama potiče od arhitekture, a koliko od načina merenja?**

Alternativna, uža formulacija: *Značaj kvaliteta merenja skokova u predviđanju realizovane volatilnosti: poređenje ekonometrijskih i mašinskih modela.*

---

## 1. Kako smo došli do ove teme

Tema nije izabrana odmah. Do nje se stiglo kroz niz otkrića koja su redom eliminisala lošije alternative. Taj put je sam po sebi argument za odbranu.

### Polazna tačka: predviđanje cena akcija

Prvobitna ideja bila je poređenje velikih jezičkih modela i transformerskih arhitektura u predviđanju cena akcija — tema koja je u 2026. vizuelno najatraktivnija.

### Otkriće 1 — objavljeni rezultati su sistematski naduvani

Radovi koji prijavljuju tačnost od 90% i više gotovo uvek pate od metodoloških problema. Najozbiljniji je **lookahead bias / kontaminacija podataka**: veliki jezički modeli su tokom treniranja „videli" istorijske finansijske podatke, pa deluje kao da predviđaju, a zapravo se prisećaju.

Empirijski dokaz: isti modeli ostvaruju prinose preko 40% u periodu unutar uzorka, a doživljavaju drastičan pad čim se pređe granica datuma prekida treniranja.

### Otkriće 2 — postoji tvrda gornja granica predvidljivosti prinosa

Referentni rad oblasti (Gu, Kelly & Xiu, *Review of Financial Studies*, 2020) poredi ceo repertoar metoda mašinskog učenja na preko 900 prediktora:

| Pristup | Mesečni out-of-sample R² |
|---|---|
| Linearni model sa 900+ prediktora | negativan (katastrofalno preprilagođavanje) |
| Penalizacija / redukcija dimenzija | ~0,26% |
| Stabla i neuronske mreže | ~0,33–0,40% |

Najbolji modeli na svetu objašnjavaju **manje od pola procenta** varijanse prinosa. Razlog nije algoritamski nego strukturni — većina kretanja cena vođena je nepredvidivim vestima, a količina podataka je fundamentalno ograničena.

**Posledica:** rad koji pokušava da predvidi cenu akcije unapred je osuđen na skroman ili nepostojeći rezultat. To nije neuspeh istraživača, nego svojstvo problema.

### Otkriće 3 — postoji asimetrija koja menja sve

Isti podaci, dve potpuno različite slike:

- **Autokorelacija prinosa ≈ 0.** Današnji prinos ne govori praktično ništa o sutrašnjem.
- **Autokorelacija kvadriranih prinosa je snažno pozitivna i sporo opada.** Ostaje značajna i posle stotinu dana.

Prevedeno: **ne zna se da li će cena ići gore ili dole, ali se prilično dobro zna koliko će se divlje kretati.** Tipičan R² za volatilnost je 50–70%, dakle više od sto puta veći nego kod prinosa.

### Otkriće 4 — zašto ta asimetrija postoji

- **Predvidljivost pravca se sama uništava.** Ako postoji pravilo „posle X cena raste", trgovci ga iskoriste odmah, cena skoči danas, i pravilo prestaje da važi. Arbitraža briše svaki predvidiv smer.
- **Predvidljivost intenziteta se ne može uništiti na isti način.** Znanje da će sutra biti turbulentno ne govori u kom pravcu, pa ne postoji direktna opklada kojom bi se to izbrisalo.

Grupisanje volatilnosti potiče od neravnomernog pristizanja informacija — vesti stižu u talasima, a učesnici reaguju različitim brzinama.

### Otkriće 5 — literatura o transformerima protivreči sama sebi

**Strana A (transformeri pobeđuju):** studija na S&P 500, NASDAQ 100 i Dow Jonesu za period 2000–2025 nalazi da Transformer i PatchTST-lite dosledno postižu najniže greške, sa prednošću izraženom na dužem horizontu. Christensen i saradnici (2023) takođe nalaze da mašinsko učenje pobeđuje HAR porodicu, uz najveće dobitke na dužim horizontima.

**Strana B (transformeri gube):** rad *FinStressTS* (KDD '26) evaluira 15 modela i zaključuje da su performanse vođene arhitekturnom induktivnom pristrasnošću, a ne kapacitetom — jednostavni linearni modeli dosledno nadmašuju transformere u okruženjima vođenim volatilnošću, repovima i **skokovima**.

**Posledica:** postoji konkretno, neodgovoreno pitanje.

### Otkriće 6 — postoji zanemarena dimenzija

Cela rasprava se vodi oko **izbora modela**. Niko sistematski ne ispituje **dizajn merenja ulaznih veličina**, iako je realizovana volatilnost izvedena, a ne posmatrana veličina — svaka njena vrednost rezultat je niza metodoloških odluka.

Rad prihvaćen u februaru 2026. pokazuje da problem višestrukog testiranja u detekciji skokova generiše lažne skokove koji kvare prediktivnu tačnost, i da je razlikovanje pravih od lažnih skokova suštinsko za predviđanje semivarijanse.

**To je praznina koju ovaj rad popunjava.**

---

## 2. Šta rad zapravo istražuje

### Centralno pitanje

> Koliki udeo razlika u tačnosti predviđanja volatilnosti potiče od izbora modela, a koliki od odluka donetih pri merenju ulaznih veličina?

Rad tretira **dizajn merenja kao eksperimentalnu dimenziju ravnopravnu izboru modela** — što, koliko je pregledom literature utvrđeno, do sada nije sistematski rađeno.

### Teorijsko uporište

Premisa nije nova ni ekscentrična. **HARQ model** (Bollerslev, Patton, Quaedvlieg, 2016) interaguje dnevni RV regresor sa realizovanom kvartičnošću upravo zato da bi uzeo u obzir **vremenski promenljivu grešku merenja**. Postoji, dakle, ugledan model izgrađen na ideji da greška merenja sistematski utiče na prognozu.

Ovaj rad tu ideju proširuje: umesto da se greška merenja modeluje unutar jednog modela, ona se **varira eksperimentalno** i meri se njen doprinos u odnosu na izbor arhitekture.

### Hipoteze

**H1.** Dodavanje komponenti skokova kao ulaznih promenljivih poboljšava performanse u odnosu na modele bez njih.

**H2.** Poboljšanje je izraženije na kratkom horizontu (h = 1), gde skokovi dominiraju, nego na dugom (h = 22), gde dominira perzistentnost.

**H3.** Varijacija u dizajnu merenja objašnjava udeo varijanse gubitka uporediv sa udelom koji objašnjava izbor modela.

**H4.** Rangiranje modela se menja pri prelasku sa statističke (QLIKE) na ekonomsku evaluaciju (VaR backtest).

**H5.** Poboljšanja pripisana skokovima nestaju kada se indikatori skokova nasumično permutuju (placebo provera).

Nijedan ishod nije neuspeh: potvrda daje pozitivan nalaz, opovrgavanje potvrđuje ili osporava postojeću literaturu na nezavisnim podacima.

---

## 3. Zašto je ovo bolje od prošlogodišnje teme

Prošlogodišnja tema: *Predviđanje akcija S&P 500 pomoću tehničkih parametara* (LSTM).

| Kriterijum | Prošlogodišnja tema | Predložena tema |
|---|---|---|
| **Predvidljivost problema** | R² < 1%, blizu granice šuma | R² 50–70%, stvaran signal |
| **Zasićenost** | Stotine praktično identičnih radova | Konkretna otvorena kontradikcija u literaturi |
| **Vrsta doprinosa** | Primena poznatih metoda | Metodološki doprinos (nova eksperimentalna dimenzija) |
| **Rizik od praznog rezultata** | Visok — verovatno nema signala | Nizak — signal postoji |
| **Odbranjivost** | Teško objasniti zašto model ne radi | Svaki ishod je interpretabilan nalaz |
| **Industrijska primena** | Trgovanje (sporno da li radi) | Upravljanje rizikom, VaR, cene opcija — regulatorno obavezno |
| **Metodološka zrelost** | Retko uključuje testove značajnosti | DM, MCS, Kupiec, Christoffersen kao standard |

Suštinska razlika: prošlogodišnja tema pokušava da **pobedi tržište**, a ova pokušava da **izmeri rizik**. Prvo je gotovo nemoguće, drugo je posao koji finansijske institucije po regulativi moraju da rade.

**Kontinuitet:** LSTM se zadržava u skupu modela, čime se uspostavlja direktna veza sa prethodnim radom — ista arhitektura primenjena na problem koji ima signal.

---

## 4. Dizajn istraživanja

Eksperiment je **faktorski**: svaka kombinacija dizajna merenja i modela se evaluira, a rezultati se analiziraju po faktorima.

### Faktor A — dizajn merenja (glavni doprinos)

Tri odvojene odluke koje se obično donose prećutno:

**A1. Frekvencija uzorkovanja**

| Varijanta | Postupak | Napomena |
|---|---|---|
| 1 minut | Cene se uzorkuju na svakih 60 sekundi, računaju se log-prinosi, RV je suma njihovih kvadrata po danu. | Najviše opservacija, dakle najmanja greška uzorkovanja, ali bid-ask odbijanja sistematski naduvavaju RV. |
| 5 minuta | Isti postupak na 5-minutnoj mreži. | Standard u literaturi; empirijski kompromis između preciznosti i mikrostrukturnog šuma. |
| 10 minuta | Isti postupak na 10-minutnoj mreži. | Najmanje šuma, ali oko 39 opservacija dnevno — procena postaje nestabilna. |

**A2. Estimator otporan na skokove**

Cilj svakog estimatora je da izmeri **kontinuirani deo** volatilnosti, odnosno ono što bi RV bio da skokova nema. Razlika `RV − estimator` je onda procena skokovite komponente.

| Varijanta | Postupak | Napomena |
|---|---|---|
| Bipower variation (BPV) | Umesto kvadrata jednog prinosa, množe se **apsolutne vrednosti dva susedna** prinosa i sumiraju. Skok utiče samo na dva sabirka umesto da se kvadrira, pa mu je uticaj potisnut. | Barndorff-Nielsen & Shephard (2004); standard u literaturi. |
| Median RV (MedRV) | Uzimaju se **medijane trojki susednih apsolutnih prinosa**. Medijana ignoriše ekstrem u trojci. | Otporniji kad se skokovi jave uzastopno — situacija u kojoj BPV zakaže jer su oba činioca kontaminirana. |
| Threshold BPV | Pre sumiranja se **isključuju svi prinosi iznad praga** izvedenog iz lokalne volatilnosti. | Direktnije, ali uvodi novi slobodan parametar (visina praga) i time novu odluku merenja. |

**A3. Test za detekciju skokova**

| Varijanta | Postupak | Napomena |
|---|---|---|
| Naivni prag | Dan se proglašava danom sa skokom ako dnevni prinos pređe fiksni broj standardnih devijacija. | Bez statističkog testa; koristi se kao donja granica kvaliteta, ne kao ozbiljan metod. |
| BNS test | Poredi se RV sa estimatorom otpornim na skokove; ako je razlika prevelika u odnosu na svoju asimptotsku raspodelu, dan se proglašava danom sa skokom. Skalira se tripower kvartičnošću. | Barndorff-Nielsen & Shephard (2006); daje jednu odluku po danu. |
| Lee-Mykland | Svaki **intradnevni** prinos se deli lokalnom procenom volatilnosti (pomični prozor) i poredi sa raspodelom maksimuma. | Lokalizuje **trenutak** skoka unutar dana, ne samo dan. Omogućava merenje broja i veličine skokova. |
| Korigovan za višestruko testiranje | Isti test kao prethodni, ali sa pragom podešenim za broj izvršenih testova (kontrola udela lažnih otkrića). | Bez korekcije, pri hiljadama dnevnih testova, deo „skokova" su čisti statistički artefakti. |

### Faktor B — izvedene promenljive

- HAR komponente: `RV_d`, `RV_w`, `RV_m`
- Skokovi: `J`, `C` (kontinuirana komponenta), binarni indikator
- Semivarijanse: `RS_pos`, `RS_neg`, predznakom označena varijacija `SJ`
- Realizovana kvartičnost `RQ` (za HARQ)
- Leverage: dnevni prinos `r_d`
- Egzogeno: `VIX` sa docnjom od jednog dana

Osnov za razlaganje: Patton & Sheppard (2015) pokazuju da je negativna semivarijansa znatno važnija za buduću volatilnost od pozitivne, i da negativni skokovi vode višoj, a pozitivni **nižoj** budućoj volatilnosti.

### Faktor C — modeli

| Model | Postupak | Uloga |
|---|---|---|
| Naivni | Sutrašnja vrednost se izjednačava sa današnjom: `RV(t) = RV(t−1)`. | Apsolutni pod. Model koji ga ne nadmaši ne uči ništa. |
| **HAR** | Linearna regresija RV na tri regresora: juče, prosek prethodnih 5 dana, prosek prethodnih 22 dana. Tri horizonta predstavljaju učesnike koji reaguju različitim brzinama. | Standardni benchmark cele oblasti. Tri regresora plus konstanta. |
| **HAR-J** | HAR kod kog se RV razlaže na kontinuirani deo `C` i skokoviti deo `J`, koji ulaze kao **odvojeni** regresori. | Testira da li skokovi imaju drugačiju perzistentnost od kontinuiranog dela. Direktno koristi Faktor A. |
| **CHAR** | HAR u kom se umesto RV koristi **isključivo kontinuirani deo** kao regresor. | Provera da li se odbacivanjem skokova dobija čistiji signal perzistentnosti. |
| **SHAR** | RV se razlaže na pozitivnu i negativnu semivarijansu (`RS_pos`, `RS_neg`), koje ulaze kao odvojeni regresori. | Hvata asimetriju: negativni pomeraji podižu, a pozitivni spuštaju buduću volatilnost. |
| **HARQ** | Koeficijent uz dnevni RV se ne fiksira, nego se **množi realizovanom kvartičnošću** — kada je merenje tog dana bilo nepouzdano, model automatski smanjuje težinu tog regresora. | Teorijski najbliži tezi rada: postojeći, priznat model izgrađen na ideji da greška merenja utiče na prognozu. |
| **HAR-IV** | HAR proširen indeksom implicitne volatilnosti (VIX) sa docnjom od jednog dana. | Testira da li tržište opcija sadrži informaciju koje nema u istoriji cena. |
| **LightGBM / XGBoost** | Sekvencijalno se grade plitka stabla odluke; svako naredno uči na greškama prethodnih. Ulazi su isti regresori kao kod HAR-a plus komponente skokova. | Hvata nelinearne interakcije koje linearni HAR ne može. Prema literaturi najjači ML kandidat na ovom zadatku. |
| **LSTM** | Rekurentna mreža koja kroz kapije (gates) održava unutrašnje stanje i propušta informaciju kroz vreme; uči direktno iz sekvence prošlih vrednosti. | Veza sa prethodnim radom. Literatura pokazuje da daje glatke prognoze koje kasne za skokovima. |
| **PatchTST** | Serija se deli na segmente („patch"-eve) koji postaju tokeni; attention mehanizam uči zavisnosti među njima bez unapred zadate strukture docnji. Svaki kanal se obrađuje nezavisno. | Savremena transformerska arhitektura; centralna u aktuelnoj raspravi. |
| **Chronos zero-shot** (opciono) | Predtrenirani model za vremenske serije kome se serija prosto preda i vrati prognozu — **bez ikakvog treniranja** na finansijskim podacima. | Primena temeljnih modela na volatilnost je u literaturi eksplicitno označena kao neistražena. |

### Faktor D — stratifikacija

Sve metrike se računaju odvojeno po:

- **režimu:** mirni naspram turbulentnih perioda (kvantil `RV_m`)
- **prisustvu skoka:** dani sa skokom naspram dana bez skoka
- **horizontu:** h = 1, 5, 22 dana

Prosek kroz ceo uzorak može sakriti da model pobeđuje u mirnim, a gubi u turbulentnim periodima — što je za upravljanje rizikom upravo obrnuto od korisnog.

### Validacioni protokol

**Anchored walk-forward validacija** — trening uzorak se širi kroz vreme, testiranje uvek na narednom neviđenom segmentu. Eliminiše svaki oblik curenja informacija iz budućnosti.

Za modele sa stohastičkim treningom (LSTM, PatchTST): **najmanje pet random seed-ova**, uz prijavljivanje srednje vrednosti i standardne devijacije. Bez toga se ne može razlikovati efekat arhitekture od šuma inicijalizacije.

---

## 5. Metrike evaluacije

Metrike su organizovane u četiri nivoa jer mere suštinski različite stvari.

### Nivo 1 — tačnost prognoze

**Fundamentalni problem:** prava volatilnost je **nemerljiva**. Prognoze se porede u odnosu na šumovit proxy.

Patton (2011) je pokazao da većina uobičajenih funkcija gubitka daje **pogrešno rangiranje modela** kada je proxy šumovit. Samo dve su robusne u tom smislu: **MSE** i **QLIKE**.

| Metrika | Postupak | Mogućnosti i ograničenja |
|---|---|---|
| **QLIKE** | Za svaki dan se računa odnos stvarne i prognozirane volatilnosti, oduzme se njegov logaritam i jedinica; rezultati se usrednje. Gubitak je nula samo kad se prognoza poklopi sa stvarnom vrednošću. | **Primarna.** Robusna na šum u proxyju. Asimetrična — potcenjivanje volatilnosti kažnjava jače od precenjivanja, što je ekonomski ispravno jer je potceniti rizik skuplje. Standard u savremenoj literaturi. |
| **RMSE** | Kvadrira se razlika stvarne i prognozirane vrednosti, usrednji i korenuje. | Sekundarna. Robusna na šum u proxyju, ali **simetrična** — jednako kažnjava obe vrste greške. Uključena radi uporedivosti sa starijom literaturom. |
| **MAE** | Usrednjava se apsolutna razlika stvarne i prognozirane vrednosti. | Dodatna. **Nije** robusna na šum u proxyju — može dati pogrešno rangiranje. Ako odstupa od QLIKE-a, prednost ima QLIKE i to se eksplicitno navodi. |
| **Relativni QLIKE** | QLIKE modela se deli QLIKE-om HAR baseline-a. | Prikazna mera. Vrednost 0,95 znači 5% bolje od HAR-a. Čini rezultate odmah čitljivim i uporedivim sa objavljenim radovima. |

**Tehnička napomena:** modeluje se `log(RV)` jer je volatilnost izrazito desno asimetrična. Metrike se računaju **nakon vraćanja u originalnu skalu**, uz korekciju za Jensenovu nejednakost — u suprotnom se sistematski potcenjuje nivo.

### Nivo 2 — statistička značajnost i dekompozicija

| Postupak | Kako funkcioniše | Šta omogućava |
|---|---|---|
| **Diebold-Mariano** | Za svaki dan se računa **razlika gubitaka** dva modela; testira se da li je prosek te razlike statistički različit od nule. | Paran test značajnosti razlike između dva modela. Koristi se Harvey-Leybourne-Newbold korekcija za male uzorke. |
| **Model Confidence Set** | Polazi se od svih modela; iterativno se odbacuje najgori sve dok se preostali ne mogu statistički razlikovati međusobno. | Daje **skup modela koji se ne mogu razlikovati od najboljeg** na datom nivou poverenja. Rešava problem višestrukog testiranja pri poređenju mnogo modela. |
| **Dekompozicija varijanse (ANOVA)** | Svaki rezultat se označi vrednostima faktora (model, frekvencija, estimator, test za skokove, režim, horizont), pa se ukupna varijansa gubitka razloži na doprinose pojedinačnih faktora. | **Ključno za H3.** Daje udeo objašnjene varijanse po faktoru — brojčan odgovor na pitanje da li više doprinosi izbor modela ili dizajn merenja. |
| **Mincer-Zarnowitz regresija** | Stvarna RV se regresuje na prognozu; testira se da li je intercept 0 i nagib 1. | Otkriva **sistematsku pristrasnost** — model može imati nizak gubitak, a dosledno potcenjivati nivo volatilnosti. QLIKE to ne razdvaja. |

**Zašto je MCS posebno važan:** ako HAR-J i PatchTST završe u istom skupu poverenja, formalno je dokazano da razlika **nije statistički značajna**. To je mnogo jače od konstatacije da su brojevi slični.

**Zašto je ANOVA neophodna:** bez nje je tvrdnja „merenje utiče više od arhitekture" neproverljiva. Sa njom postaje brojka koju je moguće citirati.

### Nivo 3 — ekonomska evaluacija (Value at Risk)

Prognoza volatilnosti koristi se za izračunavanje VaR-a na nivoima 1% i 5%, koji se zatim testira.

Postupak izračunavanja VaR-a: prognozirana volatilnost se pomnoži kvantilom pretpostavljene raspodele (npr. 2,33 za nivo 1% kod normalne), čime se dobija prag gubitka koji ne bi trebalo da bude premašen. Dan u kom stvarni gubitak pređe taj prag naziva se **prekoračenjem**.

| Test | Kako funkcioniše | Šta omogućava |
|---|---|---|
| **Kupiec (POF)** | Broji se koliko je prekoračenja bilo i testom odnosa verodostojnosti proverava da li se taj broj slaže sa deklarisanim nivoom. | Bezuslovna pokrivenost — da li je **broj** prekoračenja odgovarajući. |
| **Christoffersen** | Prekoračenja se posmatraju kao niz nula i jedinica; testira se da li verovatnoća prekoračenja zavisi od toga da li je prethodni dan bio prekoračenje. | Nezavisnost. Konceptualno ključan: model može imati tačno 5% prekoračenja, ali ako se sva dese u istoj nedelji, to je katastrofalan model rizika. |
| **Dynamic Quantile** | Niz prekoračenja se regresuje na sopstvene docnje i na sam VaR; testira se da li su svi koeficijenti nula. | Stroža verzija testa nezavisnosti — hvata i zavisnost od nivoa VaR-a, ne samo od prethodnog dana. |
| **Lopezova funkcija gubitka** | Prekoračenja se ne samo broje, nego se **kažnjavaju srazmerno veličini** premašaja. | Razlikuje model koji greši malo od modela koji greši katastrofalno, što broj prekoračenja ne vidi. |

**Istraživačka vrednost:** model sa boljim QLIKE-om **ne mora** imati bolji VaR. Promena rangiranja pri prelasku na ekonomsku meru je samostalan nalaz (H4).

### Nivo 4 — karakterizacija detekcije skokova

**Metodološka napomena:** klasifikacione metrike (precision, recall, F1) ovde **nisu primenljive**, jer ne postoji ground truth — skok nije observabilan događaj, nego ono što estimator proglasi skokom. Umesto tačnosti, meri se **slaganje i posledice**.

| Mera | Kako se računa | Šta govori |
|---|---|---|
| Broj i udeo detektovanih skokova | Broj dana proglašenih danima sa skokom, podeljen ukupnim brojem dana. | Koliko je metod konzervativan. Literatura: tipično 5–15% dana. Vrednost izvan tog raspona signalizira loše podešen prag. |
| Jaccard indeks preklapanja | Broj dana koje **obe** varijante proglašavaju skokom, podeljen brojem dana koje proglašava **bar jedna**. | Koliko se metodi međusobno slažu. Nizak indeks znači da izbor testa suštinski menja ulazne podatke — što je sama premisa rada. |
| Prosečna veličina skoka | Srednja vrednost `J` na danima sa detektovanim skokom. | Da li stroži test hvata samo velike skokove ili ravnomerno filtrira. |
| Udeo `J` u ukupnom `RV` | Suma skokovite komponente podeljena sumom RV kroz uzorak. | Ekonomska značajnost skokovite komponente. Ako je udeo zanemarljiv, i efekat na prognozu će biti mali. |
| **Placebo test (H5)** | Indikatori skokova se **nasumično permutuju** kroz vreme, čime se zadržava njihova učestalost ali uništava veza sa stvarnim danima; model se ponovo procenjuje. | Ako poboljšanje opstane i sa permutovanim indikatorima, ono **nije** posledica informacije o skokovima nego artefakt strukture modela. |

Stvarni test kvaliteta detekcije je **nizvodni** — koja varijanta daje bolju prognozu. To je centralno pitanje rada, pa se ništa ne gubi.

### Prikaz rezultata

Glavna tabela je trodimenzionalna, za svaku kombinaciju modela i dizajna merenja:

| | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| Mirni periodi | QLIKE | QLIKE | QLIKE |
| Turbulentni periodi | QLIKE | QLIKE | QLIKE |
| Dani sa skokom | QLIKE | QLIKE | QLIKE |

Uz to: MCS pripadnost po ćeliji, tabela ANOVA dekompozicije, i odvojena tabela VaR testova.

---

## 6. Podaci

### Kritična napomena

**Oxford-Man Realized Library, standardni izvor u ovoj literaturi, više nije dostupna i institut je saopštio da ne planira zamenu.** Skoro svi radovi koji se citiraju koriste nju, pa se njihov setup ne može direktno ponoviti.

### Šta se traži od izvora (sirovi podaci)

| Kolona | Napomena |
|---|---|
| `timestamp` | sa eksplicitnom vremenskom zonom |
| `price` | poslednja transakciona cena ili sredina bid-ask raspona |
| `volume` | za filtriranje i kontrolu kvaliteta |

Traži se: **intradnevni podaci na 1-minutnoj rezoluciji**, iz kojih se agregacijom dobijaju 5- i 10-minutne serije (Faktor A1).

### Mogući izvori

| Izvor | Napomena |
|---|---|
| **Univerzitetska pretplata** (Refinitiv, Bloomberg, WRDS) | najbolja opcija ako postoji — proveriti prvo |
| **Interactive Brokers API** | efektivno besplatan uz finansiran račun; ograničenje ~10 godina intradnevnih barova |
| **Databento**, **Polygon.io / Massive** | vode u tick i Level 2 podacima; plaćaju se po potrošnji |
| **Alpha Vantage** | pristupačan REST pristup OHLCV podacima; **nema tick podatke** |
| Arhivirane Oxford-Man serije (GitHub) | gotovo upotrebljivo do 2022, ali onemogućava variranje Faktora A |

### Izbor uzorka

**Panel od 20–30 pojedinačnih akcija plus indeks**, umesto jedne serije.

Obrazloženje je i statističko i metodološko:
- Jedna serija od 15 godina daje oko 3.700 dnevnih opservacija — nedovoljno za stabilan trening transformerskih arhitektura, koje su testirane na datasetovima sa stotinama hiljada tačaka. Panel daje red veličine više.
- Nalaz potvrđen na 30 akcija robusniji je od nalaza na jednom indeksu.
- Omogućava proveru da li zaključci zavise od likvidnosti i sektora.

Dodatno, **jedan neamerički indeks** (DAX ili FTSE) kao provera robustnosti — većina objavljenih studija koristi isključivo američka tržišta.

### Zamke pri konstrukciji

| Zamka | Posledica ako se previdi |
|---|---|
| Prekonoćni prinosi | RV se računa samo iz trgovačkih sati; uključivanje prekonoćnog skoka kvari meru. Izostavljanje se eksplicitno navodi. |
| Skraćeni trgovački dani | Dani pred praznike izgledaju kao dani niske volatilnosti. Filtrirati po broju opservacija (< ~80% očekivanog). |
| Vremenska zona i DST | Kod više tržišta obavezno UTC interno, konverzija samo pri filtriranju trgovačkih sati. |
| VIX bez docnje | Korišćenje VIX(t) za predviđanje RV(t) je lookahead bias. |
| `RS_pos + RS_neg ≠ RV` | Signal greške u računu semivarijansi. Ugraditi kao automatsku proveru. |

### Rezervni plan

Ako intradnevni podaci ne budu dostupni: **range-based estimatori** iz dnevnih OHLC podataka (Parkinson, Garman-Klass, Rogers-Satchell). Zahtevaju samo `open`, `high`, `low`, `close`.

Mana: nemoguće je računati semivarijanse i bipower variation, pa Faktori A2 i A3 otpadaju. Rad ostaje odbraniv, ali osiromašen. Držati kao rezervu, ne kao prvi izbor.

---

## 7. Odnos prema postojećoj literaturi

### Prema pregledu iz Financial Innovation (januar 2026)

Nije konkurencija — to je survey koji analizira sve modele u literaturi od 2000. do prve polovine 2024. Služi kao **temelj literaturnog pregleda**. Pokriva do sredine 2024, pa se aktuelna rasprava o transformerima dodaje samostalno.

### Prema radu o transformerima (JRFM, decembar 2025)

Najbliži konkurent; razgraničenje mora biti eksplicitno.

| | JRFM rad (dec 2025) | Ovaj rad |
|---|---|---|
| **Šta varira** | isključivo arhitektura modela | arhitektura **i dizajn merenja** |
| **Merenje** | fiksno, prećutno | tri odvojene odluke, sistematski variran |
| **Pitanje** | koji model je najbolji | koliki je udeo svakog faktora |
| **Značajnost** | Diebold-Mariano | DM, **MCS**, **ANOVA dekompozicija** |
| **Evaluacija** | statistička | statistička **i ekonomska** (VaR) |
| **Uzorak** | tri američka indeksa | panel akcija plus neamerički indeks |

**Rečenica za razgraničenje u uvodu:**

> Postojeće studije variraju arhitekturu modela dok tretiraju realizovane mere kao fiksne ulaze. Ovaj rad obrće perspektivu i pita koliki udeo posmatranih razlika u performansama potiče od odluka donetih pri konstrukciji tih mera.

---

## 8. Rizici i njihovo ublažavanje

| Rizik | Ublažavanje |
|---|---|
| Nema pristupa intradnevnim podacima | Range-based estimatori iz dnevnih OHLC podataka (osiromašena verzija) |
| Transformer ne konvergira na malom uzorku | Panel umesto jedne serije; više seed-ova sa prijavljenom varijansom; mala arhitektura |
| Detekcija skokova sadrži grešku koja se prenosi nizvodno | Placebo test (H5); poređenje više varijanti umesto oslanjanja na jednu |
| Rezultati zavise od jednog tržišta | Neamerički indeks kao provera robustnosti |
| Neko objavi sličan rad u međuvremenu | Kombinacija (dizajn merenja + MCS + ANOVA + VaR) je uska; replikacija na nezavisnim podacima ima samostalnu vrednost |
| Obim eksperimenta naraste | Faktori su modularni: A1, Nivo 3 i temeljni modeli mogu se izostaviti bez rušenja jezgra |

---

## 9. Ključna literatura

**Polazna tačka:**
- *Advances in forecasting realized volatility: a review of methodologies* — Financial Innovation, januar 2026

**Temelji:**
- Corsi (2009) — HAR model
- Andersen, Bollerslev & Diebold (2007) — HAR-J, dekompozicija na kontinuiranu i skokovitu komponentu
- Barndorff-Nielsen & Shephard (2004, 2006) — bipower variation i test za skokove
- Patton & Sheppard (2015) — *Good volatility, bad volatility: signed jumps and the persistence of volatility*
- Bollerslev, Patton & Quaedvlieg (2016) — HARQ, greška merenja
- Lee & Mykland (2008) — intradnevna detekcija skokova
- Clements & Preve (2021) — praktični vodič za implementaciju HAR proširenja

**Metodologija evaluacije:**
- Patton (2011) — robusne funkcije gubitka pri šumovitom proxyju
- Hansen, Lunde & Nason — Model Confidence Set
- Diebold & Mariano (1995); Christoffersen; Kupiec; Engle & Manganelli

**Aktuelna rasprava:**
- *Deep Learning and Transformer Architectures for Volatility Forecasting* — JRFM, decembar 2025
- *FinStressTS* — KDD '26
- Christensen i saradnici (2023) — komparativna procena ML metoda
- Rad o detekciji pravih naspram lažnih skokova, prihvaćen februar 2026

**Kontekst:**
- Gu, Kelly & Xiu (2020) — *Empirical Asset Pricing via Machine Learning* (zašto se rad ne bavi prinosima)

---

## 10. Rečenica za odbranu

> Dok se aktuelna literatura spori oko toga koja arhitektura najbolje predviđa volatilnost, ovaj rad pokazuje da odluke donete pri konstrukciji realizovanih mera mogu uticati na rezultat u meri uporedivoj sa izborom samog modela — što znači da deo objavljenih razlika među modelima može poticati od razlika u pripremi podataka, a ne od arhitekture.
