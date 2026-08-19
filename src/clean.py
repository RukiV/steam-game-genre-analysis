import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import RAW_DIR, PROCESSED_DIR, GAMES, GAME_GENRES, GENRE_IDS, load_raw


def clean_reviews(df):
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.drop_duplicates(subset='review_id')

    df = df[df['review_text'].notna() & (df['review_text'].str.strip() != '')]

    df = df[df['review_text'].str.len() >= 10]

    char_threshold = 0.5
    df = df[df['review_text'].apply(
        lambda x: sum(c.isalpha() or c.isspace() for c in str(x)) / max(len(str(x)), 1) > char_threshold
    )]

    df['timestamp_created'] = pd.to_numeric(df['timestamp_created'], errors='coerce')
    df = df[df['timestamp_created'].notna()]
    df['review_date'] = pd.to_datetime(df['timestamp_created'], unit='s')
    df['review_year'] = df['review_date'].dt.year
    df['review_month'] = df['review_date'].dt.month
    df['review_day_of_week'] = df['review_date'].dt.dayofweek

    df['playtime_forever'] = pd.to_numeric(df['playtime_forever'], errors='coerce').fillna(0).astype(float)

    outliers = df['playtime_forever'] > df['playtime_forever'].quantile(0.99)
    df.loc[outliers, 'playtime_forever'] = df['playtime_forever'].quantile(0.99)

    df['voted_up'] = df['voted_up'].astype(bool)
    df['steam_purchase'] = df['steam_purchase'].astype(bool)
    df['written_during_early_access'] = df['written_during_early_access'].astype(bool)

    df['review_text_clean'] = df['review_text'].apply(clean_text)

    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()

    # Alles is nou Engels-only — die language-kolom is nie meer nodig nie
    if 'language' in df.columns:
        df = df.drop(columns=['language'])

    genres_for_app = df['app_id'].map(GAME_GENRES)
    for genre in GENRE_IDS:
        df[f'genre_{genre}'] = genres_for_app.apply(
            lambda tags: 1 if isinstance(tags, list) and genre in tags else 0
        )

    return df


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s\'.!?,;-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_reviews(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = load_raw('reviews.csv')

    if df is None or df.empty:
        raise ValueError("No review data found. Run scrape.py first.")

    print(f"Loaded {len(df)} raw reviews")

    # Na professor se terugvoer: filter na Engels-only
    # (VADER en TF-IDF werk slegs op Engels)
    df = df[df['language'] == 'english'].copy()
    print(f"Filtered to English: {len(df)} reviews")

    df = clean_reviews(df)
    print(f"After cleaning: {len(df)} reviews")

    out_path = f'{PROCESSED_DIR}/reviews_clean.csv'

    # Bewaar VADER-kolomme van die vorige skoon CSV indien beskikbaar —
    # VADER is duur om te bereken (~50 min vir 160k), so moenie dit verloor nie.
    if os.path.exists(out_path):
        prev_cols = pd.read_csv(out_path, nrows=0).columns
        vader_cols = [c for c in prev_cols if c.startswith('vader_')]
        if vader_cols:
            prev = pd.read_csv(out_path, usecols=['review_id'] + vader_cols)
            df = df.merge(prev, on='review_id', how='left')
            print(f"VADER-kolomme bewaar van vorige skoonmaak: {vader_cols}")

    df.to_csv(out_path, index=False)
    print(f"Saved clean reviews to {out_path}")

    stats = {
        'total_raw': len(pd.read_csv(f'{RAW_DIR}/reviews.csv')),
        'total_clean': len(df),
        'positive': int(df['voted_up'].sum()),
        'negative': int((~df['voted_up']).sum()),
        'avg_review_length': float(df['review_length'].mean()),
        'games': df['app_id'].nunique(),
    }

    for app_id, name in GAMES.items():
        game_df = df[df['app_id'] == app_id]
        if not game_df.empty:
            pos_ratio = game_df['voted_up'].mean() * 100
            print(f"  {name:25s}: {len(game_df):>6} reviews, {pos_ratio:5.1f}% positive")
            stats[f'{name}_count'] = int(len(game_df))
            stats[f'{name}_pos_pct'] = round(pos_ratio, 1)

    return df, stats
