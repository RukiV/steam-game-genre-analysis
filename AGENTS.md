# Steam Data Analysis — Agent Guide

## Run commands

```bash
# Notebook (full analysis — self-contained, can scrape from scratch)
source venv/bin/activate && cd notebooks && jupyter notebook project.ipynb

# Dashboard
source venv/bin/activate && streamlit run dashboard/app.py

# Progressive rescrape (50 pages per game, crash-safe, saves per game)
source venv/bin/activate && python3 rescrape_all.py

# Re-clean after scraping
source venv/bin/activate && python3 -c "from src.clean import process_reviews; process_reviews()"

# Re-run VADER and persist to CSV (run after cleaning if dashboard is slow)
source venv/bin/activate && python3 -c "
import sys, pandas as pd; sys.path.insert(0,'.');
from src.nlp_analysis import apply_vader;
from src.utils import PROCESSED_DIR;
df = pd.read_csv(f'{PROCESSED_DIR}/reviews_clean.csv');
df = apply_vader(df);
df.to_csv(f'{PROCESSED_DIR}/reviews_clean.csv', index=False);
print('VADER saved')
"

# Force full re-scrape from scratch (delete cached data)
rm -f data/raw/reviews.csv data/raw/app_details.csv data/raw/player_counts.csv data/processed/reviews_clean.csv
```

## Project structure

- `src/` — independent modules (scrape → clean → eda → nlp → regression → network), each exposes its entry functions
- `dashboard/app.py` — Streamlit with 4 tabs, genre + game filters, Afrikaans UI
- `notebooks/project.ipynb` — self-contained (scrapes from scratch if data missing)
- `notebooks/Fase2_Projek.ipynb` — Fase 2 deliverable (aktiewe skraap, skoonmaak, einddatastel-struktuur; Fase 3 slegs beplan)
- `data/raw/` — `reviews.csv` (~163k rows, English-only), `app_details.csv`, `player_counts.csv` (33 games)
- `data/processed/reviews_clean.csv` — ~161k cleaned English reviews + 15 genre one-hot columns + 5 VADER columns

## Key facts

- **33 games** in `src/utils.py` GAMES dict, **15 genres** in GENRES dict
- **GENRES** maps genre → [app_ids]; **GAME_GENRES** maps app_id → [genre tags]; `games_by_genre(genre)` and `games_in_genres(list)` helpers
- **VADER columns**: vader_compound, vader_positive, vader_neutral, vader_negative, vader_sentiment_label. Pre-persisted in CSV.
- **Scraper** uses `language='english'`, `purchase_type=all`, `day_range=9999`, 0.3s delay, dedup via seen_ids, stops after 3 empty pages. `max_pages=50` for rescrape (was 100).
- **Network graphs** (`src/network.py`): genre_relationship_network (gedeelde speletjies), genre_commentary_network (TF-IDF-sentroïede, cosinus-ooreenkoms), genre_word_network (ko-voorkoms van woorde). Albei teks-netwerke filter na Engelse resensies.
- **All paths** use constants from `src.utils`: PROJECT_DIR, RAW_DIR, PROCESSED_DIR, GAMES, GAME_IDS, GENRES, GENRE_IDS
- **Data span**: ~161k clean English reviews across all 33 games, Oct 2021 – Aug 2026. 50 pages (5000 reviews) covers median 365 days per game.
- **Skills** installed at `~/.config/opencode/skills/` (global, not project-local)
- **Notebook self-contained**: Phase 2 cells check for existing data and scrape if missing. Delete `data/raw/*.csv` and `data/processed/*.csv` to force fresh scrape.

## Module gotchas

- `prepare_regression_features()` must NOT include `vader_compound` — it's the target for linear regression. Logistic regression adds it back manually.
- Feature columns for per-game dummies use `GAMES[app_id]` names (e.g., `Counter-Strike_2`), not `game_{id}`.
- `prepare_regression_features()` also adds genre one-hot columns (`genre_{GENRE_KEY}`) from `GAME_GENRES` — only works if `process_reviews()` was run post-utils change.
- `rescrape_all.py` — progressive save (append per game), crash-safe, 50 pages per game. Calls `process_reviews()` + VADER after scraping.
- `genre_commentary_network()` — alle genre-vokabulêre oorvleuel >0.85; gebruik `threshold='auto'` (gem + 0.5·std) om net die sterkste bande te hou.
- `genre_word_network()` en `genre_commentary_network()` — filter na Engels via `language='english'`; werk net as `language`-kolom bestaan. WSL-veilig: sample 500 (woorde) / 3000 per genre (TF-IDF).
- `draw_network()` — nodusgrootte kom uit `size`-attribuut, kantlyndikte uit `weight`; beide grafieke stel dit self.

## Dashboard layout

- **Sidebar**: genre multiselect → game multiselect → date range → min reviews slider
- **Tab1 (Oorsig)**: overview metrics → per-game table/charts → per-genre table/charts → current player counts
- **Tab2 (NLP & Netwerke)**: VADER distribution, word clouds (sampled ≤500), game TF-IDF, genre TF-IDF, 3 genre netwerkgrafieke (verwantskap, kommentaar, woorde)
- **Tab3 (Vergelyk)**: radio toggle between Speletjies/Genres mode, scatter plots
- **Tab4 (Voorspellings)**: linear & logistic regression (includes genre features)

## WSL memory gotchas

- `TfidfVectorizer` — must use `ngram_range=(1,1)` (unigrams only); bigrams cause OOM segfault
- `WordCloud.generate()` — sample text to ≤500 reviews before joining; concatenating all is OOM
- Dashboard scatter plots wrapped in try/except; extreme `playtime_forever` pre-filtered

## Module return conventions

- `basic_statistics()` → dict: total_reviews, positive_pct, avg_playtime, avg_word_count, num_games, date_range
- `per_game_statistics()` → DataFrame: game, total_reviews, positive, negative, positive_pct, avg_playtime, steam_purchase_pct
- `per_genre_statistics()` → DataFrame: genre, total_reviews, positive, negative, positive_pct, avg_playtime, avg_word_count, num_games
- `linear_regression_model()` → dict: r2_score, rmse, feature_importance, intercept, n_train, n_test
- `logistic_regression_voted_up()` → dict: accuracy, auc_roc, feature_importance, classification_report, n_train, n_test
- `genre_relationship_network()` → nx.Graph: nodus = genre (size = aantal speletjies), kantlyn = gedeelde speletjies
- `genre_commentary_network(df)` → nx.Graph: nodus = genre (size = aantal Engelse resensies), kantlyn = cosinus-ooreenkoms (TF-IDF-sentroïede), net bo auto-drempel
- `genre_word_network(df, genre)` → nx.Graph: nodus = top-woord (size = frekwensie), kantlyn = ko-voorkoms in resensies
- `draw_network(G, ax=None, ...)` → matplotlib Axes (spring-uitleg, nodusgrootte = `size`, kantlyndikte = `weight`)

## Dashboard crash fixes

- **Memory optimization**: `load_data()` drops `review_text` and `review_text_clean` columns (156 MB saved). Main DataFrame is ~49 MB instead of 206 MB.
- **Lazy text loading**: `review_text_clean` loaded on-demand via `load_review_texts()` only when Tab2 (NLP & Netwerke) is active. Merged into `filtered_en` via `review_id`.
- **Interrupted-run cleanup**: `plt.close('all')` + `gc.collect()` at the top of the script (before any widgets) cleans up orphaned matplotlib figures from the previous run when Streamlit interrupts it midway. Prevents OOM during rapid genre/game filter switching.
- **Cache resource**: `load_data()` and `load_review_texts()` use `@st.cache_resource` instead of `@st.cache_data` to avoid pickle serialization overhead and memory doubling.
- **PyArrow segfault**: `pd.options.mode.string_storage = 'python'` prevents PyArrow-backed string columns that crash (`AllocateResizableBuffer`) during DataFrame filter and Arrow serialization. Pin `pyarrow<25` to avoid the WSL2 segfault.
- **Disable hot-reload**: `.streamlit/config.toml` sets `runOnSave = false` to prevent Streamlit's file watcher (running on NTFS via WSL 9p) from triggering false re-runs.
- **Crash logging**: `faulthandler.enable()` at the top of `app.py` captures C-level segfault traces to stderr for debugging.
- **Matplotlib backend**: Explicitly set to `'Agg'` before `pyplot` import to prevent it from trying a GUI backend when `DISPLAY=:0` (WSLg) is set.
- **Every tab** wrapped in outer try/except — if a tab fails, it shows a warning and other tabs still work
- **Sidebar** handles empty `filtered` (NaT dates → warning instead of crash)
- **Tab2 WordCloud** uses `len(unique_games) > 0` guard before `st.selectbox`, so no crash when genre filter excludes all English reviews
- **Tab2 TF-IDF** individual try/except per section, already had sample cap 30k
- **Tab2 network graphs** each wrapped in try/except; netwerke gebruik `filtered_en` (moet `review_text_clean` gemerged hê)
- **Tab4 each regression** wrapped separately (linear & logistic don't kill each other)
- **Bar chart colors** fixed: `per_game.sort_values('positive_pct')` used for both color list and sort, instead of mismatched sorts
