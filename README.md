# Steam Game Reviews & Popularity Analysis

Data-analise projek vanuit 'n speletjie-ontwikkelaar se perspektief: **Hoe maak 'n mens 'n suksesvolle speletjie op Steam, gebaseer op genre?**

Analiseer 33 speletjies oor 9 genres (RPG, Shooter, Hero Shooter, Battle Royale, Action, Strategy, Adventure, Survival, Free to Play) met ~65 000 resensies, 836 inhoudsgebeure, 705 historiese spelerdata-rye, en 2831 SteamDB-maandrye.

> **📦 Data-beleid (GitHub):** Gegenereerde CSVs word **nie** gecommit nie (`data/raw/reviews.csv` ~104 MB en `data/processed/reviews_clean.csv` ~198 MB is te groot vir GitHub). Die notebook is self-contained — dit skraap en herskep alles van nuut af. Sien **Gebruik** hieronder. Klein brondata (`steamdb_chart_*.csv`, `content_events.csv`, `player_history.csv`, `app_details.csv`) word wel gecommit.

## Projekstruktuur

```
├── data/
│   ├── raw/              # Rou API data (reviews, app_details, player_counts, content_events, player_history, steamdb_chart_*.csv)
│   └── processed/        # Skoongemaakte data met genre + VADER kolomme + steamdb_monthly.csv
├── src/
│   ├── scrape.py         # Steam API-skraper (reviews, app details, players, content events)
│   ├── steamcharts.py    # SteamCharts historiese spelerdata-skraper
│   ├── steamdb.py        # SteamDB daaglikse resensietellings-laaier + skoonmaak
│   ├── clean.py          # Data skoonmaak + genre-kenmerke
│   ├── eda.py            # Verkennende data analise + statistiese toetse
│   ├── nlp_analysis.py   # VADER, TF-IDF (per game en per genre), WordCloud
│   ├── regression.py     # Lineêre & logistiese regressie met genre-kenmerke
│   ├── timeseries.py     # Tydreeksanalise, seisoenale patrone, verlossingsboë
│   └── utils.py          # Konfigurasie: 33 speletjies, 9 genres, GAME_GENRES
├── notebooks/
│   ├── project.ipynb     # 50-sel selfstandige notebook (alle 4 fases)
│   └── Fase1_Projek.ipynb # Fase 1 projekdokument (inleiding, probleemstelling, metodologie)
├── dashboard/
│   └── app.py            # Streamlit dashboard (5 tabs, genre + game filters, Afrikaans)
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
source venv/bin/activate && python3 -c "from src.scrape import scrape_all_content_events; scrape_all_content_events()"
source venv/bin/activate && python3 src/steamcharts.py
source venv/bin/activate && python3 -c "from src.clean import process_reviews; process_reviews()"
source venv/bin/activate && python3 -c "from src.steamdb import load_steamdb_history; load_steamdb_history(force_reprocess=True)"
```

### 5. Slegs VADER-herlaai
```bash
source venv/bin/activate && python3 vader_runner.py
```

## Data Skoonmaak

Die pyplyn pas verskeie skoonmaak- en voorverwerkingstegnieke toe oor die verskillende databronne:

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

**Kenmerk-ingenieurswese:** `review_length`, `word_count`, `has_early_access`, `is_steam_purchase`, plus 9 one-hot `genre_{GENRE}`-kolomme uit `GAME_GENRES`.

### NLP (`src/nlp_analysis.py`)

- **VADER-val:** leë/non-str teks kry neutrale verstek (compound=0, neu=1.0)
- **Sentiment-binning:** `compound` → `negative`/`neutral`/`positive` (drempels ±0.05)
- **TF-IDF:** Engelse stopwoorde verwyder, **unigrams slegs** (`ngram_range=(1,1)` — WSL-geheuebeperking), `max_features=1000`

### SteamDB (`src/steamdb.py`)

| Tegniek | Beskrywing |
|---|---|
| Kumulatiewe-ry-verwydering | Eerste ry >20× die mediaan van die volgende 5 dae = opgehoopte "voor-bytrekking"-data (Overwatch 2: 731×) → verwyder |
| NaN-dae gefiltreer | `dropna(subset=['positive','negative'])` — slegs dae met beide positiewe én negatiewe data tel |
| Absolute negatiewe | SteamDB stoor negatiewe as negatief → `abs()` |
| Maand-aggregasie | Positiewe/negatiewe per maand gesommeer → `positive_pct` |
| Pre-data-bewaring | Data voor SteamDB se bytrekking bewaar as `pre_total_*` vir korrekte kumulatiewe persentasies |

### SteamCharts (`src/steamcharts.py`)

- Kommas uit getalle verwyder, "Last 30 Days"-ry oorgeslaan, ongeldige waardes gedrop.

### Regressie (`src/regression.py` — `prepare_regression_features()`)

- **Ontbrekende data:** rye met NaN op kern-kenmerke gedrop
- **Log-transformasie** (`log1p`) op skeefgetrekte kenmerke: `playtime_forever`, `review_length`, `word_count`, `num_games_owned`, `num_reviews`, `votes_up`
- **StandardScaler** toegepas binne die modelle self

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
| Elden Ring | Action, RPG | 94.7% |
| Baldur's Gate 3 | RPG | 93.6% |
| The Witcher 3 | RPG, Action | 94.7% |
| Cyberpunk 2077 | RPG, Action | 95.7% |
| Red Dead Redemption 2 | Adventure, Action | 89.0% |
| God of War | Action, Adventure | 92.9% |
| Ghost of Tsushima | Action, Adventure | 92.7% |
| The Forest | Survival, Adventure | 92.9% |
| Skyrim SE | RPG | 94.9% |
| No Man's Sky | Action, Adventure, Survival | 89.2% |
| Team Fortress 2 | Shooter, Free to Play | 90.4% |
| Devil May Cry 5 | Action | 84.3% |
| Sid Meier's Civilization VI | Strategy | 83.7% |
| Paladins | Hero Shooter, Free to Play | 80.1% |
| Valheim | Survival | 70.5% |
| Rust | Survival, Action | 76.7% |
| Age of Empires IV | Strategy | 69.8% |
| Dragon's Dogma 2 | RPG, Action | 62.5% |
| Counter-Strike 2 | Shooter, Free to Play | 56.4% |
| PUBG: BATTLEGROUNDS | Battle Royale, Free to Play | 49.1% |
| Rainbow Six Siege | Shooter | 47.1% |
| ARK: Survival Evolved | Survival, Action | 43.5% |
| Apex Legends | Battle Royale, Free to Play | 39.1% |
| Destiny 2 | Shooter, Free to Play | 39.5% |
| Battlefield 2042 | Shooter | 38.6% |
| Stellaris | Strategy | 36.9% |
| Overwatch 2 | Hero Shooter, Free to Play | 35.5% |
| Total War: WARHAMMER III | Strategy | 30.8% |
| Marvel Rivals | Hero Shooter, Free to Play | 30.4% |
| Fallout 4 | RPG | 27.7% |
| Helldivers 2 | Shooter | 25.3% |
| Call of Duty HQ | Shooter, Free to Play | 17.0% |
| Fall Guys | Battle Royale, Free to Play | 74.2% |

## Tegnologieë

- **Python** — pandas 3.0+, numpy 2.5+, matplotlib 3.11+, seaborn 0.13+
- **NLP** — VADER, TF-IDF (scikit-learn), WordCloud
- **Regressie** — Scikit-learn (Lineêr, Logisties)
- **Tydreeks** — Moving averages, seasonal decomposition, redemption arcs
- **Dashboard** — Streamlit 1.59+ (5 tabs, genre + game filters, Afrikaans UI)
- **Data** — Steam API (appdetails, appreviews, ISteamNews, player counts), SteamCharts, SteamDB

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

1. **Genre bepaal basislyn.** RPG begin teen ~79% positief; Hero Shooter teen ~34%.
2. **Speeltyd = kritiek.** Mees belêde spelers gee die hardste kritiek (p ≈ 10⁻⁶⁰).
3. **Verlossingsboë is werklik.** Cyberpunk 2077 en No Man's Sky herstel van ~50% na 90%+.
4. **Woordkeuse verskil per genre.** Elke genre se gemeenskap het unieke bekommernisse.
5. **Spelers en sentiment is swak gekorreleer.** Mense speel ten spyte van wat hulle sê.

Sien `analysis-findings.md` vir volledige uiteensetting.

### Sleuteluitdagings wat aangespreek is

1. **PyArrow-segfault op WSL2** — `AllocateResizableBuffer` crash in PyArrow 25.0.0 tydens boolean indexing op string-kolomme. Opgelos: pin `pyarrow<25`, stel `pd.options.mode.string_storage = 'python'`.
2. **SteamDB-kumulatiewe eerste ry** — Overwatch 2 se eerste datapunt was 731× groter as die daaglikse gemiddelde (kumulatief oor 8 maande). Opgelos: `_remove_cumulative_first_row()` in `src/steamdb.py`.
3. **Resensie-bomaanvalle** — Helldivers 2 se PSN-koppeling-kontroversie het kunsmatige data-skatting veroorsaak. Opgelos: `ratio_std_threshold=5.0` in `_estimate_ratio()`.
4. **TF-IDF OOM** — Bigramme veroorsaak geheue-oploop op WSL. Opgelos: slegs unigrams, sampel tot 30 000 resensies.
5. **WordCloud OOM** — Alle resensies saamvoeg is te groot. Opgelos: sampel tot 500 resensies.

## Lêers

- `notebooks/project.ipynb` — Self-contained notebook (50 selle, al 4 fases)
- `notebooks/Fase1_Projek.ipynb` — Fase 1 projekdokument (inleiding, probleemstelling, metodologie, bronnelys)
- `dashboard/app.py` — Streamlit dashboard met 5 tabs + genre/game filters (Afrikaans)
- `.streamlit/config.toml` — Streamlit-konfigurasie (headless, geen hot-reload)
- `rescrape_all.py` — Progressiewe herskraper, 50 bladsye per speletjie, crash-veilig
- `analysis-findings.md` — Volledige bevindings
- `data/processed/reviews_clean.csv` — 65k skoon resensies met genre + VADER
- `data/processed/steamdb_monthly.csv` — 2831 maandrye van SteamDB (33 speletjies)
