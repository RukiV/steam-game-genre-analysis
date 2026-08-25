# Steam Game Genre Analysis

**Doelwit:** Speletjie-ontwikkelaars en beleggers help om 'n ingelige besluit te neem oor watter genre om in te belê, gebaseer op werklike Steam-data oor gewildheid, sentiment en suksesfaktore per genre.

**Vraag:** Watter genre werk op Steam? — watter genre is die gewildste, watter kenmerke dryf sukses, en hoe lyk die kommentaar (resensies) binne elke genre?

Analiseer 33 speletjies oor **15 genres** (RPG, Shooter, Hero Shooter, Battle Royale, Action, Strategy, Adventure, Survival, Free to Play, Third Person, First Person, Top Down, Single Player, Multiplayer, Indie) met ~160 000 skoon Engelse resensies (Okt 2021 – Aug 2026). Genre-vlak fokus met **netwerkgrafieke** wat wys hoe genres verwant is en watter tipe kommentaar hulle gemeenskappe lewer.

> **🌍 Engels-only:** Resensies word slegs in Engels geskrap (`language='english'`). VADER en TF-IDF werk slegs op Engels; nie-Engelse resensies sou in Fase 4 ge-filter moes word.

> **📦 Data-beleid (GitHub):** Gegenereerde CSVs word **nie** gecommit nie (`data/raw/reviews.csv` en `data/processed/reviews_clean.csv` is te groot vir GitHub). Die notebook is self-contained — dit skraap en herskep alles van nuut af. Sien **Gebruik** hieronder. Klein brondata (`app_details.csv`, `player_counts.csv`) word wel gecommit.

## Projekstruktuur

```
├── data/
│   ├── raw/              # Rou API data (reviews, app_details, player_counts)
│   └── processed/        # Skoongemaakte data met genre + VADER kolomme
├── src/
│   ├── scrape.py         # Steam API-skraper (reviews, app details, huidige spelers)
│   ├── clean.py          # Data skoonmaak + genre-kenmerke
│   ├── eda.py            # Genre-rangorde, gewildheid, suksesfaktore, statistiese toetse
│   ├── nlp_analysis.py   # VADER, TF-IDF (per game en per genre), WordCloud
│   ├── network.py        # Genre-netwerkgrafieke (verwantskap, kommentaar, woorde)
│   ├── regression.py     # Lineêre & logistiese regressie met genre-kenmerke
│   └── utils.py          # Konfigurasie: 33 speletjies, 15 genres, GAME_GENRES
├── notebooks/
│   ├── project.ipynb     # Selfstandige notebook (genre-fokus, geen tydreeks)
│   └── Fase1_Projek.ipynb # Fase 1 projekdokument (inleiding, probleemstelling, metodologie)
├── dashboard/
│   └── app.py            # Streamlit dashboard (Oorsig, NLP & Netwerke, Vergelyk, Voorspellings)
├── rescrape_all.py       # Progressiewe herskraper (50 bladsye per speletjie, crash-veilig)
├── vader_runner.py       # VADER-herlaai en persisteer na CSV
├── run_vader.sh          # Shell-skrip vir VADER-herlaai
├── analysis-findings.md  # Gedetailleerde bevindings
├── requirements.txt
├── .streamlit/
│   └── config.toml       # Streamlit-konfigurasie (headless, geen hot-reload)
└── README.md
```

## Installasie

```bash
# Virtuele omgewing (Python 3.14+)
python3 -m venv venv && source venv/bin/activate

# Installeer afhanklikhede
pip install -r requirements.txt
```

> **⚠️ WSL-gebruikers:** As Streamlit crash met 'n segfault ('AllocateResizableBuffer'), is PyArrow 25+ die oorsaak. Die `requirements.txt` sluit `pyarrow<25` in om dit te voorkom. Sien ook die dashboard se `config.toml` vir `runOnSave = false`.

## Gebruik

### 1. Data insameling en analise (Jupyter)
Die notebook is **self-contained** — skraap outomaties van nuut as data ontbreek:
```bash
source venv/bin/activate && jupyter notebook notebooks/project.ipynb
```
Om van nuut af te skraap, verwyder eers die data:
```bash
rm -f data/raw/*.csv data/processed/*.csv
```

### 2. Dashboard (Streamlit)
```bash
source venv/bin/activate && streamlit run dashboard/app.py
```
Vasgevangde crashes? Hardloop met stderr-herleiding:
```bash
source venv/bin/activate && streamlit run dashboard/app.py 2>crash.log
```
Faulthandler vang dan C-vlak segfaults en skryf dit na `crash.log`.

### 3. Progressiewe herskrapping (crash-veilig)
```bash
source venv/bin/activate && python3 rescrape_all.py
```
Skraap 50 bladsye per speletjie, stoor progressief, en hardloop outomaties `process_reviews()` + VADER na.

### 4. Skraping (onafhanklik van notebook)
```bash
source venv/bin/activate && python3 -c "from src.scrape import scrape_all; scrape_all()"
source venv/bin/activate && python3 -c "from src.clean import process_reviews; process_reviews()"
```

### 5. Slegs VADER-herlaai
```bash
source venv/bin/activate && python3 vader_runner.py
```

## Data Skoonmaak

Die pyplyn pas verskeie skoonmaak- en voorverwerkingstegnieke toe op die resensiedata:

### Resensies (`src/clean.py` — `clean_reviews()`)

| Tegniek | Beskrywing |
|---|---|
| Deduplikasie | Duplikaat-resensies op `review_id` verwyder |
| Ontbrekende teks | Leë/NaN `review_text` weggegooi |
| Minimum lengte | Resensies korter as 10 karakters verwyder (spam) |
| Karakterverhouding-filter | Slegs resensies waar >50% van karakters alfabeties/spasies is — verwyder nonsens-teks en emoji-spam |
| Tydstempel-validasie | `timestamp_created` → numeries; ongeldige waardes verwyder; `review_date`, `review_year`, `review_month`, `review_day_of_week` afgelei |
| Playtime-winsorisering | Uitkenners bo die 99ste persentiel word afgekap (nie verwyder nie) |
| Ontbrekende speeltyd | NaN playtime → 0 |
| Tipe-afdwinging | `voted_up`, `steam_purchase`, `written_during_early_access` → bool |

**Teks-skoonmaak (`clean_text()`):** URL's verwyder → spesiale karakters (behalwe `. ' ! ? , ; -`) vervang met spasies → spasies ineenstort.

**Kenmerk-ingenieurswese:** `review_length`, `word_count`, plus 15 one-hot `genre_{GENRE}`-kolomme uit `GAME_GENRES`.

### NLP (`src/nlp_analysis.py`)

- **VADER-val:** leë/non-str teks kry neutrale verstek (compound=0, neu=1.0)
- **Sentiment-binning:** `compound` → `negative`/`neutral`/`positive` (drempels ±0.05)
- **TF-IDF:** Engelse stopwoorde verwyder, **unigrams slegs** (`ngram_range=(1,1)` — WSL-geheuebeperking), `max_features=1000`

### Regressie (`src/regression.py` — `prepare_regression_features()`)

- **Ontbrekende data:** rye met NaN op kern-kenmerke gedrop
- **Log-transformasie** (`log1p`) op skeefgetrekte kenmerke: `playtime_forever`, `review_length`, `word_count`, `num_games_owned`, `num_reviews`, `votes_up`
- **StandardScaler** toegepas binne die modelle self

## Kenmerke (kolomme)

### Kern-kenmerke — `data/processed/reviews_clean.csv` (43 kolomme)

**Genre & identifikasie** — die kern van die "watter genre werk"-vraag:

| Kolom | Beskrywing |
|---|---|
| `app_id`, `game_name` | Watter speletjie |
| `genre_{GENRE}` (×15) | Een-hot genre-etikette: RPG, Shooter, Hero_Shooter, Battle_Royale, Action, Strategy, Adventure, Survival, Free_to_Play, Third_Person, First_Person, Top_Down, Single_Player, Multiplayer, Indie |

**Speler- & resensiegedrag** — die kenmerke wat sukses dryf:

| Kolom | Beskrywing |
|---|---|
| `playtime_forever` | Totale speeltyd (gekap by 99ste persentiel) |
| `num_games_owned`, `num_reviews` | Speler-ervaring op Steam |
| `votes_up`, `votes_funny`, `weighted_vote_score` | Resensie-sigbaarheid/waarde |
| `steam_purchase`, `received_for_free` | Aankoopkonteks |
| `written_during_early_access` | Early-access-konteks |
| `voted_up` | **Die teiken**: positief (1) / negatief (0) |

**Teks & sentiment** — die kommentaar:

| Kolom | Beskrywing |
|---|---|
| `review_text`, `review_text_clean` | Rou vs. skoongemaakte teks |
| `review_length`, `word_count` | Resensielengte (log-getransformeer in regressie) |
| `vader_compound`, `vader_positive`, `vader_neutral`, `vader_negative`, `vader_sentiment_label` | VADER-sentiment (die teiken vir lineêre regressie) |

### Konteks-kolomme (bewaar, maar nie tydreeks-geanaliseer nie)

`timestamp_created`, `review_date`, `review_year`, `review_month`, `review_day_of_week` — datums word bewaar vir konteks, datumbereik-oorwegings en potensiële seisoenale analise, maar daar is **geen tydreeks-analises** in hierdie projek nie.

### Ondersteunende data

| Lêer | Kolomme | Rol |
|---|---|---|
| `data/raw/app_details.csv` | `app_id`, `name`, `release_date`, `developers`, `publishers`, `genres`, `categories`, `price`, `metacritic_score`, `recommendations` | Speletjie-besonderhede vir genre-rangorde (prys, kritici-telling) |
| `data/raw/player_counts.csv` | `app_id`, `game_name`, `player_count` | **Huidige** spelertellings (snapshot) vir "wat is die gewildste" |

## Speletjies en Genres

### 9 Genres

| Genre | # Speletjies | Gem. Positief | Karakter |
|-------|:-----------:|:-------------:|----------|
| **Action** | 2 | 88.9% | Hoogste sentiment, klein steekproef |
| **Adventure** | 3 | 87.6% | Storiegedrewe, sterk resensies |
| **RPG** | 6 | 78.9% | Hoë basislynverwagting |
| **Survival** | 4 | 71.4% | Hoë speeltyd, lojale gemeenskap |
| **Strategy** | 4 | 61.3% | DLC-moeheid, gemengde gevoelens |
| **Free to Play** | 6 | 52.7% | Mikrobetalings wreek hulself |
| **Shooter** | 6 | 47.0% | Cheater-probleme, kompetisie |
| **Battle Royale** | 2 | 39.8% | Versadigde mark, lae toleransie |
| **Hero Shooter** | 3 | 34.1% | Laagste — monitisering + balans |

### 33 Speletjies

| Speletjie | Genre(s) | % Positief |
|-----------|----------|:----------:|
| Elden Ring | RPG, Action, Third Person, Single Player | 98.6% |
| Cyberpunk 2077 | RPG, Action, Single Player | 98.5% |
| Baldur's Gate 3 | RPG, Strategy, Adventure, Third Person, Top Down, Single Player | 97.8% |
| Palworld | Survival, Third Person, Indie | 94.7% |
| Devil May Cry 5 | Action, Third Person, Single Player | 94.4% |
| Team Fortress 2 | Shooter, Hero Shooter, Action, Free to Play, First Person, Multiplayer | 93.5% |
| God of War | Action, Adventure, Third Person, Single Player | 93.2% |
| The Witcher 3 | RPG, Action, Adventure, Third Person, Single Player | 92.6% |
| The Forest | Action, Adventure, Survival, First Person, Single Player, Indie | 91.6% |
| Skyrim SE | RPG, Action, Adventure, Third Person, First Person, Single Player | 91.2% |
| No Man's Sky | Action, Adventure, Survival, Third Person, First Person, Single Player | 90.9% |
| Valheim | RPG, Action, Adventure, Survival, Third Person, First Person, Single Player, Multiplayer, Indie | 89.6% |
| Ghost of Tsushima | Action, Adventure, Third Person, Single Player | 88.8% |
| Red Dead Redemption 2 | Action, Adventure, Third Person, Single Player | 88.4% |
| Sid Meier's Civilization VI | Strategy, Top Down, Single Player, Multiplayer | 85.9% |
| Age of Empires IV | Strategy, Top Down, Single Player, Multiplayer | 79.9% |
| Destiny 2 | Shooter, First Person, Multiplayer | 68.2% |
| Rust | Action, Survival, Multiplayer, Indie | 67.4% |
| ARK: Survival Evolved | Action, Adventure, Survival, Third Person, First Person, Multiplayer, Indie | 66.9% |
| Stellaris | Strategy, Top Down | 65.9% |
| Dragon's Dogma 2 | RPG, Action, Third Person, Single Player | 64.8% |
| PUBG: BATTLEGROUNDS | Shooter, Battle Royale, Free to Play, Third Person, First Person, Multiplayer | 58.2% |
| Rainbow Six Siege | Shooter, Strategy, First Person, Multiplayer | 55.8% |
| Apex Legends | Shooter, Battle Royale, Free to Play, First Person, Multiplayer | 55.6% |
| Counter-Strike 2 | Shooter, Free to Play, First Person, Multiplayer | 50.7% |
| Overwatch 2 | Shooter, Hero Shooter, Free to Play, First Person, Multiplayer | 45.6% |
| Total War: WARHAMMER III | Strategy, Third Person, Top Down, Multiplayer | 44.4% |
| Fallout 4 | RPG, Action, Adventure, Third Person, First Person, Single Player | 43.8% |
| Battlefield 2042 | Shooter, First Person, Multiplayer | 42.0% |
| Marvel Rivals | Hero Shooter, Free to Play, Third Person, Multiplayer | 36.0% |
| Battlefield 6 | Shooter, First Person, Single Player, Multiplayer | 22.1% |
| Call of Duty HQ | Shooter, Battle Royale, First Person, Single Player, Multiplayer | 21.2% |
| Helldivers 2 | Shooter, Action, Third Person, Multiplayer | 10.3% |

> **Nota:** Persentasies gebaseer op ~161k skoon Engelse resensies (Okt 2021 – Aug 2026).

## Tegnologieë

- **Python** — pandas 3.0+, numpy 2.5+, matplotlib 3.11+, seaborn 0.13+
- **NLP** — VADER, TF-IDF (scikit-learn), WordCloud
- **Netwerkgrafieke** — networkx (genre-verwantskap, kommentaar-ooreenkoms, woordnetwerke)
- **Regressie** — Scikit-learn (Lineêr, Logisties)
- **Dashboard** — Streamlit 1.59+ (4 tabs, genre + game filters, Afrikaans UI)
- **Data** — Steam API (appreviews, appdetails, huidige spelertellings)

### WSL-kompatibiliteit

Die dashboard is geoptimaliseer vir WSL2:
| Probleem | Oplossing |
|----------|-----------|
| PyArrow 25+ segfault | `pyarrow<24`, `string_storage='python'` |
| Hot-reload valse herlaaie | `runOnSave = false` |
| Matplotlib GUI-backend | `matplotlib.use('Agg')` |
| OOM tydens TF-IDF | unigrams slegs, sampel tot 30k |
| OOM tydens WordCloud | sampel tot 500 resensies |
| Crash-logging | `faulthandler.enable()` |

## Sleutelbevindings

1. **Genre bepaal basislyn.** RPG begin teen ~85% positief; Battle Royale teen ~45%.
2. **Speeltyd = kritiek.** Mees belêde spelers gee die hardste kritiek (p ≈ 5 × 10⁻¹⁹³).
3. **Woordkeuse verskil per genre.** Elke genre se gemeenskap het unieke bekommernisse — sigbaar in die genre-netwerke.

Sien `analysis-findings.md` vir volledige uiteensetting.

### Sleuteluitdagings wat aangespreek is

1. **PyArrow-segfault op WSL2** — `AllocateResizableBuffer` crash in PyArrow 25.0.0 tydens boolean indexing op string-kolomme. Opgelos: pin `pyarrow<25`, stel `pd.options.mode.string_storage = 'python'`.
2. **TF-IDF OOM** — Bigramme veroorsaak geheue-oploop op WSL. Opgelos: slegs unigrams, sampel tot 30 000 resensies.
3. **WordCloud OOM** — Alle resensies saamvoeg is te groot. Opgelos: sampel tot 500 resensies.

## Lêers

- `notebooks/project.ipynb` — Self-contained notebook (genre-fokus)
- `notebooks/Fase1_Projek.ipynb` — Fase 1 projekdokument (inleiding, probleemstelling, metodologie, bronnelys)
- `notebooks/Fase2_Projek.ipynb` — Fase 2 data-insameling en -verwerking (skraap, skoonmaak, einddatastel; Fase 3-plan)
- `reports/Fase2_Aanbieding.pptx` — Fase 2-klasaanbieding (12 skyfies, sprekersnotas, grafieke)
- `dashboard/app.py` — Streamlit dashboard met 4 tabs + genre/game filters (Afrikaans)
- `.streamlit/config.toml` — Streamlit-konfigurasie (headless, geen hot-reload)
- `rescrape_all.py` — Progressiewe herskraper, 50 bladsye per speletjie, crash-veilig
- `analysis-findings.md` — Volledige bevindings
- `data/processed/reviews_clean.csv` — 160k skoon resensies met genre + VADER
