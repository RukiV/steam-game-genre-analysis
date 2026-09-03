# Fase 3 Plan — Gedetailleerde Selstruktuur

> **Doel:** Een selfstandige `notebooks/Fase3_Projek.ipynb` wat die volledige Fase 1+2+3-pipeline dokumenteer en self uitvoer, sonder `from src import`. Slegs suiwer EDA in Fase 3 (geen VADER/TF-IDF/netwerke — dit is Fase 4).

**Selstruktuur:** ~50 selle, ~15–25 minute hardlooptyd weens Fase 1 se skraap.

---

## Inleiding (3 md-selle)

- **[0] md:** Titel + doelwit + omvang (Fase 1, 2, 3)
- **[1] md:** Navorsingsvraag + hipotese + sub-vrae (uit Fase 1)
- **[2] md:** Etiese oorwegings + bronnelys (kort)

## Opstelling (1 kode-sel, [3])

- Imports (slegs stdlib + pandas + numpy + matplotlib + requests + scipy) — **geen `from src`**
- Inline definieer GAMES (33), GENRES (15), GENRE_IDS, GAME_GENRES, RAW_DIR, PROCESSED_DIR
- `safe_request()` en `ensure_dirs()` funksies inline gekopieër uit `src/utils.py`

## Fase 1: Skraap (3 md + 3 kode-selle)

- **[4] md:** Fase 1 oorsig + parameter-tabel (50 bladsye, Engels, 0.3s, dedup)
- **[5] kode:** `fetch_app_details()` + `scrape_reviews()` + `fetch_player_count()` + `scrape_all()` — gekopieër uit `src/scrape.py`
- **[6] kode:** `scrape_all()` uitvoer (produseer 3 rou CSVs) — ~15–25 min
- **[7] kode:** Druk rye-tellings en vertoon `app_details.csv`-voorskou

## Fase 2: Skoonmaak (1 md + 4 kode-selle)

- **[8] md:** Fase 2 oorsig + skoonmaakstappe-tabel
- **[9] kode:** `clean_text()` + `clean_reviews()` + `process_reviews()` — gekopieër uit `src/clean.py`
- **[10] kode:** Filter na Engels-only + `clean_reviews()` + skryf `reviews_clean.csv` uit
- **[11] kode:** Vertoon kolom-tabel (36 kolomme, datatipes) — `df.info()`
- **[12] kode:** Datakwaliteit-opsomming: rye, positiewe %, gem. speeltyd, gem. woorde, datumreeks, 15 genre one-hot kolomme
- **[13] md:** Samevatting Fase 2 + brug na Fase 3

## Fase 3: EDA (3 afdelings, ~30 selle)

### § 3.1 Basiese Statistieke (5 selle: 2 md + 3 kode)
- **[14] md:** Doel van basiese statistieke
- **[15] kode:** Totale oorsig (`basic_statistics()` inline gekopieër uit `src/eda.py`)
- **[16] kode:** Per-speletjie tabel (33 rye, gesorteer op positiewe %)
- **[17] kode:** Per-genre tabel (15 rye, gesorteer op positiewe %)
- **[18] md:** Voorlopige bevinding — 4 clusters (Loved/Good/Mixed/Disliked)

### § 3.2 Statistiese Toetse (8 selle: 1 md + 6 kode + 1 md)
- **[19] md:** 5 toetse + tabel met voorlopige resultate
- **[20] kode:** Welch t-toets woord-telling: pos 61 vs neg 96, p≈0
- **[21] kode:** Welch t-toets speeltyd: pos 303h vs neg 406h, p=5×10⁻¹⁹³
- **[22] kode:** Eenrigting-ANOVA woord-telling per genre: F=166.3
- **[23] kode:** Eenrigting-ANOVA woord-telling per speletjie (33 groepe)
- **[24] kode:** Chi-kwadraat: genre × voted_up (kontingensie-tabel + Cramer's V)
- **[25] kode:** Spearman-korrelasie-matriks (speeltyd, woorde, lengte, votes_up)
- **[26] md:** Interpretasie + effekgroottes

### § 3.3 Patrone, Tendense & Uitdagings (8 selle: 1 md + 6 kode + 1 md)
- **[27] md:** Doel
- **[28] kode:** Verdelingsgrafieke: speeltyd (log-skaal), woorde, resensielengte
- **[29] kode:** Maandelikse resensie-volume per genre (laaste 365 dae)
- **[30] kode:** Maandelikse positiewe-% per genre (Helldivers-bom sigbaar)
- **[31] kode:** Hittekaart: genre × sentiment-basislyn (RPG 85% vs BR 45%)
- **[32] kode:** Uitdagings: lys resensiebomme, uitskieters, nie-onafhanklike waarnemings
- **[33] kode:** Speeltyd-uitskieter-analise (99ste persentiel afkap)
- **[34] md:** Voorlopige Fase 4-roete

## Fase 4 Vooruitsig + Bronnelys (3 md-selle)

- **[35] md:** Fase 4 word beplan (NLP, regressie, netwerke, dashboard) — **nie uitgevoer**
- **[36] md:** Bronnelys
- **[37] md:** Slotopmerking — Fase 3 status: GEREED

---

## Konvensies

- Alle hulpselfunksies word **heel-bo in die opstelling-selle** gedefinieër (geen `from src`)
- Sinstyl: kort en reguit, geen onnodige uitleg
- Visuele selle: matplotlib met `%matplotlib inline`, 1 figuur per sel
- Geen selle met `print(inspect.getsource(...))` — die kode self word in die notebook geskryf (self-standig)
- Geen VADER-, TF-IDF-, regressie- of netwerkselle — dit is Fase 4
- Genres se getalle sal 15 wees (Single_Player, Multiplayer, Indie ingesluit) — moet GENRES/GAME_GENRES in sel [4] reg kopieer

---

## Verifiëring (na implementering)

1. Hardloop `jupyter nbconvert --to notebook --execute Fase3_Projek.ipynb --inplace` (deur gebruiker)
2. Bevestig: 161 000+ skoon rye, 15 genres, alle 5 toetse het p-waardes
3. Finale oudit: geen verouderde syfers (80k, 9 genres, Sep 2016, ens.)

---

*Plan goedgekeur — gebruiker hardloop self*
