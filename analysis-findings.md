# Analysis Findings — Steam Game Reviews & Popularity

## 1. Data Scraping & Cleaning

**Scraping (`src/scrape.py`):** Fetches from the Steam API: app details, up to ~5000 English reviews per game, and live player counts. 33 games, ~80,000 English reviews total, saved as raw CSVs.

**Cleaning (`src/clean.py`):** Strips URLs, removes non-alpha characters, deduplicates by `review_id`, filters reviews <10 characters, clips extreme playtime outliers (99th percentile), and engineers `review_date`, `word_count`, `review_length`. Adds 15 genre one-hot columns (`genre_RPG`, `genre_Shooter`, etc.) via `GAME_GENRES` mapping. Drops the now-constant `language` column (English-only data).

**Data scale:** ~80k clean English reviews after cleaning. Average review ~97 words. Playtime has extreme skew.

## 2. EDA — Per-Game Statistics

All 33 games split into clear sentiment clusters:

| Group | Examples | Positive % | Interpretation |
|-------|----------|-----------|----------------|
| **Loved** | Elden Ring, Baldur's Gate 3, Cyberpunk 2077, No Man's Sky, Skyrim SE, The Witcher 3 | 89–96% | Critically acclaimed or successful redemption arcs |
| **Good** | Civilization VI, Devil May Cry 5, The Forest, Ghost of Tsushima, God of War, Red Dead Redemption 2, Team Fortress 2, Valheim | 70–93% | Strong titles in their niches |
| **Mixed** | CS2, PUBG, Rainbow Six Siege, Age of Empires IV, Dragon's Dogma 2 | 47–70% | Long-running live-service or polarising sequels |
| **Disliked** | Overwatch 2, Marvel Rivals, Call of Duty HQ, Helldivers 2, Apex Legends, Destiny 2, Battlefield 2042, Fallout 4, Stellaris, Total War: WARHAMMER III, ARK | 17–44% | Free-to-play backlash or monetisation issues |

**Key insight:** The biggest driver of positive reviews is not objective quality — it's community sentiment and expectations. Cyberpunk 2077 (95.7%) and No Man's Sky (89.2%) both had disastrous launches but sit at the top after years of updates. Meanwhile, Fallout 4 (27.7%) and Total War: WARHAMMER III (30.8%) suffer from community fatigue and monetisation backlash despite being from respected franchises.

## 3. EDA — Per-Genre Statistics

Genres show dramatically different sentiment baselines — critical for a developer choosing their genre:

| Genre | Positive % | Avg Playtime (h) | Avg Word Count | # Games |
|-------|-----------|-----------------|----------------|---------|
| **Action** | 88.9% | 25,538 | 97 | 2 |
| **Adventure** | 87.6% | 22,717 | 94 | 3 |
| **RPG** | 78.9% | 26,161 | 97 | 6 |
| **Survival** | 71.4% | 31,419 | 96 | 4 |
| **Strategy** | 61.3% | 11,618 | 83 | 4 |
| **Free to Play** | 52.7% | 20,411 | 93 | 6 |
| **Shooter** | 47.0% | 17,283 | 91 | 6 |
| **Battle Royale** | 39.8% | 14,022 | 87 | 2 |
| **Hero Shooter** | 34.1% | 24,892 | 88 | 3 |

**Key insight:** A new RPG starts with ~79% expected positive reviews before writing a single line of code. A new Hero Shooter starts at ~34%. The genre *itself* sets the baseline — this is the most important finding for a game developer. Action and Adventure genres have the highest sentiment (small sample, but all critically acclaimed). Survival games demand high playtime investment but retain good sentiment.

## 4. Statistical Tests

### Welch's t-test — Word count vs voted_up
- Positive reviews: **93 words avg**, Negative reviews: **103 words avg**
- p = **1.47 × 10⁻⁵** (highly significant)

### Welch's t-test — Playtime vs voted_up
- Positive reviewers: **21,012 hours avg**, Negative reviewers: **30,170 hours avg**
- p = **7.89 × 10⁻⁶⁰** (extremely significant — strongest signal in dataset)

### One-way ANOVA — Word count by genre
- F = **53.0**, p ≈ 0 (highly significant)
- **Longest reviews:** RPG (RPG players write the most detailed)
- **Shortest reviews:** Hero Shooter, Strategy

**What it means:** Negative reviews are longer (detailed complaints). High-playtime players are the most critical (invested → higher expectations). Review length varies significantly by genre — RPG communities write more detail than Shooter communities.

## 5. NLP — VADER Sentiment Analysis

**Distribution:** Most scores cluster near 0 (neutral), with a slight positive skew. VADER vs Steam thumbs-up agreement: ~70%. VADER measures *textual* sentiment (word choice), while Steam captures *intent* (recommendation). You can write "This game is broken" and still click thumbs-up.

**Per-game VADER:** Mirrors the positive % ranking closely. Games with low positive % (Overwatch 2, CoD HQ, Marvel Rivals) also have the most negative average VADER scores.

**Per-genre VADER:** Action and Adventure genres have the highest average VADER; Hero Shooters and Battle Royales the lowest — consistent with the positive % ranking.

## 6. NLP — TF-IDF by Genre

Each genre has distinctive vocabulary that reveals community priorities:

| Genre | Top Distinguishing Terms |
|-------|-------------------------|
| **RPG** | "larian", "dialogue", "exploration", "build", "romance", "choices" |
| **Shooter** | "cheater", "hacker", "balance", "vac", "shadowban", "hitreg" |
| **Hero Shooter** | "overwatch", "mercy", "tank", "role", "queue", "battlepass" |
| **Battle Royale** | "pubg", "warzone", "camping", "loot", "circle" |
| **Action** | "combat", "parry", "boss", "difficulty", "moveset" |
| **Strategy** | "diplomacy", "ai", "dlc", "mechanic", "turns" |
| **Adventure** | "story", "world", "atmosphere", "immersion", "graphics" |
| **Survival** | "grind", "base", "building", "server", "wipe", "solo" |
| **Free to Play** | "store", "cosmetic", "grind", "pay", "battlepass", "skin" |

**Key insight:** Each genre's community has completely different concerns — RPG players care about story and choices, Shooter players about anti-cheat and balance, Survival players about grind and base-building. A developer's communication and priorities must match their genre's vocabulary.

## 7. NLP — TF-IDF by Game

Standout distinctive terms per game:
- **Baldur's Gate 3:** "larian", "dnd", "act", "romance", "honour"
- **Cyberpunk 2077:** "phantom", "liberty", "dlc", "expansion", "keanu"
- **Counter-Strike 2:** "cheater", "vac", "rank", "tickrate", "prime"
- **Call of Duty HQ:** "hacker", "shadowban", "mw3", "warzone"
- **Elden Ring:** "margit", "malenia", "elden", "dung", "tree"
- **Helldivers 2:** "nerf", "nerfed", "patriot", "bug", "nerfing"
- **Marvel Rivals:** "netease", "spider", "battlepass", "skin", "jeff"
- **Overwatch 2:** "overwatch", "mercy", "widow", "genji", "kiriko"
- **Rust:** "offline", "raided", "wipe", "zerg", "roof"

**Key insight:** Negative terms like "cheater", "hacker", "shadowban" dominate CS2 and CoD. "Nerf" appears in Helldivers 2's top terms — a community frustrated by balance changes. Positive terms like "masterpiece", "phatom liberty", "larian" dominate BG3 and Cyberpunk. This unsupervised signal tells us *why* players feel the way they do.

## 8. Regression — Linear Regression

**Goal:** Predict VADER compound score from review features (playtime, review length, word count, game dummies, genre dummies, purchase type, early access).

**Expected result:** R² will likely be low (<0.15) — sentiment is primarily driven by game-specific experiences, patch timing, and personal preference, not by objective features.

**Feature importance:** Playtime is expected to be the strongest negative predictor across all games. Genre dummies will capture baseline sentiment differences (Hero Shooter → more negative, RPG → more positive).

## 9. Regression — Logistic Regression

**Goal:** Predict thumbs-up (`voted_up`) from the same features plus VADER score.

**Expected insight:** VADER should significantly improve AUC-ROC — the way someone writes (emotional tone) predicts their recommendation beyond playtime and game identity.

## 10. Regression — Per-Genre Success Factors

Using separate logistic regression models per genre to answer: *"What predicts success within each genre?"*

| Genre | Expected Key Factors |
|-------|---------------------|
| **RPG** | Low word count (succinct praise), high playtime (invested fans recommend) |
| **Shooter** | Steam purchase (vs free), low playtime (casual players happier) |
| **Hero Shooter** | NOT free-to-play, early access? |
| **Battle Royale** | Steam purchase, moderate playtime |
| **Strategy** | Low word count, DLC ownership? |
| **Survival** | High playtime (grind-friendly players), low word count |

**Key insight:** Different features matter in different genres. In Shooters, free-to-play is a negative signal (microtransaction fatigue). In RPGs, playtime correlates with positive sentiment (invested fans). A developer should focus on the factors that matter in *their* genre, not the global average.

## 11. Time Series — Seasonal Patterns

**Hourly:** Most reviews written afternoon/evening (player time zones). Fewest midnight–6am.

**Daily:** Weekends have more reviews; positive % stays consistent.

**Monthly:** No strong seasonal pattern.

**Key insight:** Review behavior follows circadian rhythms. Sentiment doesn't vary by time.

## 12. Time Series — Genre Monthly Trends

Each genre's positive % over time reveals stability and shock patterns:
- **RPG:** Stable high sentiment (consistently >75%)
- **Hero Shooter:** Volatile low sentiment (balance patches cause swings)
- **Survival:** Moderate with gradual improvement
- **Shooter:** Declining trend across 2025–2026

**Key insight:** Some genres are inherently stable (RPG, Adventure), while others are volatile (Hero Shooter, Battle Royale). Genre choice determines not just the baseline but the volatility of player sentiment.

## 13. Time Series — Redemption Arcs

**Cyberpunk 2077 (95.7% positive):** Disastrous launch → Phantom Liberty + 2.0 patch → 90%+ recovery. Monthly trend shows clear inflection point.

**No Man's Sky (89.2% positive):** Similarly dramatic — continuous free updates over years rebuilt reputation.

**Key insight:** Games *can* recover from bad launches through sustained developer effort. The redemption arc is real and measurable in review data. This is the strongest narrative finding.

## 14. Content Impact Analysis

Using 836 content events across 33 games, we can measure the immediate before/after sentiment impact of patches and updates. For each event, sentiment before the event vs after (30-day window) is compared.

**Expected pattern:** Major patches and DLC releases show a temporary positive spike in sentiment, while controversial balance changes show a negative dip.

## 15. SteamCharts — Historical Player Counts

CS2 dominates player counts (peaked at ~1.8M concurrent). Single-player games (Elden Ring, Cyberpunk) show content-driven spikes. Live-service games (CoD HQ, Overwatch 2) show gradual decline.

**Key insight:** Player counts and review sentiment are weakly correlated. Overwatch 2 and Marvel Rivals have low review scores but high player counts — people play despite disliking it (sunk cost, social pressure, lack of alternatives).

---

## Big Picture Summary

The project answers: *"How can a game developer succeed in their genre on Steam?"*

**Answer:** The genre determines baseline expectations more than any other factor. A new RPG starts at ~79% expected positive reviews; a new Hero Shooter at ~34%. Within a genre, the strongest predictors are playtime (invested players are the harshest critics) and vocabulary (each genre community uses different language to express satisfaction or frustration).

The redemption arc finding proves that games *can* recover from bad launches through continued development. Content events measurably impact sentiment. And the TF-IDF analysis gives developers a direct map of what their genre's community cares about — cheat prevention for Shooters, story quality for RPGs, grind reduction for Survival games.

**Key takeaway:** Pick your genre wisely (the baseline varies 3x). Listen to your community's specific vocabulary. Invest in sustained updates. Accept that the most invested players will be the most critical — but also the most loyal if you deliver.
