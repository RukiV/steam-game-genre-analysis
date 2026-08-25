# Fase 3 Plan: Verkennende Data-analise (EDA)

> **Doel:** Leer die data ken voordat gevorderde modelle gebou word. Beskrywende + inferensiële statistiek om patrone en afwykings te identifiseer.
>
> **Data:** ~161 000 skoon Engelse resensies · 33 speletjies · 15 genres · Okt 2021 – Aug 2026

---

## 1. Basiese Statistieke (Beskrywend)

Gebruik bestaande `src/eda.py` (`basic_statistics()`, `per_game_statistics()`, `per_genre_statistics()`):

- **Oorsig:** totale resensies, positiewe %, gem. speeltyd, gem. woorde, datumreeks
- **Per speletjie:** resensie-volume, positiewe %, gem. speeltyd, Steam-aankoop-%
- **Per genre:** dieselfde maatstawwe, gesorteer op sentiment
- **Verdelings:** histogramme van speeltyd, woord-telling, resensielengte, VADER-compound

**Aflewering:** opsommingstabelle + verdelingsgrafieke in `Fase3_Projek.ipynb`

## 2. Formele Statistiese Toetse (Inferensieel)

| Toets | Vraag | Voorlopige resultaat |
|---|---|---|
| Welch se t-toets | Woord-telling: positief vs negatief | Pos = 61 woorde, Neg = 96 woorde, p ≈ 0 ✅ |
| Welch se t-toets | Speeltyd: positief vs negatief | Pos = 303 h, Neg = 406 h, p ≈ 5×10⁻¹⁹³ ✅ |
| Eenrigting-ANOVA | Woord-telling verskil per genre? | F = 166.3, p ≈ 2×10⁻²⁸¹ ✅ |
| Chi-kwadraat | Hang `voted_up` saam met genre? | *om doen* |
| Korrelasie (Spearman) | Speeltyd ↔ VADER-compound | *om doen* |

**Aflewering:** toetstablelle met effekgroottes in die notebook

## 3. Patrone, Tendense & Uitdagings

**Patrone (voorlopig bevestig):**
- Genre bepaal die sentiment-basislyn: RPG ~85% vs Battle Royale ~45%
- Strukturele splitsing: enkel-speler/story-genres (75–85%) teenoor mededingende multiplayer-genres (45–62%)
- Negatiewe resensies is ~58% langer as positiewe (gedetailleerde klagtes)

**Tendense:**
- Maandelikse positiewe-% per genre oor tyd (stabiliteit vs wisselvalligheid)
- Redemption arcs: Cyberpunk 2077 (98.5%), No Man's Sky (90.9%)

**Uitdagings om aan te dui:**
- **Resensiebomme:** Helldivers 2 Mar–Mei 2026 het tot 1.6% positief gedaal — skep kunsmatige pieke
- **Uitskieters:** speeltyd by 99ste persentiel afgekap; 2 speletjies met lae Engelse volume strek verder terug (AoE IV: 1755 dae)
- **Nie-afhanklike waarnemings:** veelvuldige resensies per speler kan korreleer

## 4. Aanvanklike Plan vir Diepgaande Analise (Fase 4)

Gebaseer op bostaande voorlopige bevindinge:

1. **NLP-sentiment:** VADER per speletjie/genre (bevestig of tekstuele sentiment met `voted_up` ooreenstem — verwag ~75%)
2. **TF-IDF + woordwolke:** onderskeidende woordeskat per genre (bv. RPG: "story"/"mods"; Shooter: "maps"/"cheaters")
3. **Netwerkgrafieke:** genre-verwantskap (gedeelde speletjies), kommentaar-ooreenkoms (TF-IDF-sentroïede), woordnetwerke
4. **Regressie:** lineêr (VADer-compound voorspel uit kenmerke; verwag lae R²) + logistiek (`voted_up` voorspel; toets of VADER AUC verbeter)
5. **Dashboard:** al die bogenoemde interaktief in Streamlit met genre/speletjie-filters

---

**Volgende stap:** bou `notebooks/Fase3_Projek.ipynb` met die struktuur hierbo, hergebruik `src/eda.py`, voeg chi-kwadraat + korrelasietoetse by.
