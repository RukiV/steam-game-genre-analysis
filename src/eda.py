import pandas as pd
import numpy as np
from scipy import stats
from src.utils import GAMES, GENRES, GENRE_IDS, GAME_GENRES


def basic_statistics(df):
    stats_summary = {
        'total_reviews': len(df),
        'positive_reviews': int(df['voted_up'].sum()),
        'negative_reviews': int((~df['voted_up']).sum()),
        'positive_pct': round(df['voted_up'].mean() * 100, 2),
        'avg_playtime': round(df['playtime_forever'].mean(), 1),
        'median_playtime': round(df['playtime_forever'].median(), 1),
        'avg_review_length': round(df['review_length'].mean(), 1),
        'median_review_length': round(df['review_length'].median(), 1),
        'avg_word_count': round(df['word_count'].mean(), 1),
        'date_range': f"{df['review_date'].min().date()} to {df['review_date'].max().date()}",
        'num_games': df['app_id'].nunique(),
        'num_languages': df['language'].nunique(),
    }
    return stats_summary


def per_game_statistics(df):
    per_game = []
    for app_id, name in GAMES.items():
        g = df[df['app_id'] == app_id]
        if g.empty:
            continue
        per_game.append({
            'game': name,
            'total_reviews': len(g),
            'positive': int(g['voted_up'].sum()),
            'negative': int((~g['voted_up']).sum()),
            'positive_pct': round(g['voted_up'].mean() * 100, 1),
            'avg_playtime': round(g['playtime_forever'].mean(), 0),
            'avg_review_length': round(g['review_length'].mean(), 0),
            'avg_word_count': round(g['word_count'].mean(), 1),
            'steam_purchase_pct': round(g['steam_purchase'].mean() * 100, 1),
        })
    return pd.DataFrame(per_game)


def review_length_sentiment_test(df):
    pos = df[df['voted_up']]['word_count']
    neg = df[~df['voted_up']]['word_count']
    if len(pos) < 2 or len(neg) < 2:
        return None
    t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False)
    return {
        'test': "Welch's t-test",
        'variable': 'word_count by voted_up',
        'positive_mean': round(pos.mean(), 2),
        'negative_mean': round(neg.mean(), 2),
        't_statistic': round(t_stat, 4),
        'p_value': f'{p_val:.2e}',
        'significant': p_val < 0.05,
    }


def playtime_sentiment_test(df):
    pos = df[df['voted_up']]['playtime_forever']
    neg = df[~df['voted_up']]['playtime_forever']
    if len(pos) < 2 or len(neg) < 2:
        return None
    t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False)
    return {
        'test': "Welch's t-test",
        'variable': 'playtime_forever by voted_up',
        'positive_mean': round(pos.mean(), 2),
        'negative_mean': round(neg.mean(), 2),
        't_statistic': round(t_stat, 4),
        'p_value': f'{p_val:.2e}',
        'significant': p_val < 0.05,
    }


def anova_by_game(df):
    groups = [g['word_count'].values for _, g in df.groupby('app_id') if len(g) > 1]
    if len(groups) < 2:
        return None
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        'test': 'One-way ANOVA',
        'variable': 'word_count by game',
        'f_statistic': round(f_stat, 4),
        'p_value': f'{p_val:.2e}',
        'significant': p_val < 0.05,
        'num_groups': len(groups),
    }


def correlation_analysis(df):
    corr_cols = ['playtime_forever', 'review_length', 'word_count',
                 'votes_up', 'num_games_owned', 'num_reviews']
    corr_df = df[corr_cols].select_dtypes(include=[np.number]).corr()
    return corr_df


def playtime_by_rating_trend(df):
    result = df.groupby('voted_up')['playtime_forever'].describe()
    return result


def per_genre_statistics(df):
    rows = []
    for genre in GENRE_IDS:
        g = df[df[f'genre_{genre}'] == 1]
        if g.empty:
            continue
        rows.append({
            'genre': genre.replace('_', ' '),
            'total_reviews': len(g),
            'positive': int(g['voted_up'].sum()),
            'negative': int((~g['voted_up']).sum()),
            'positive_pct': round(g['voted_up'].mean() * 100, 1),
            'avg_playtime': round(g['playtime_forever'].mean(), 0),
            'avg_review_length': round(g['review_length'].mean(), 0),
            'avg_word_count': round(g['word_count'].mean(), 1),
            'num_games': g['app_id'].nunique(),
        })
    return pd.DataFrame(rows)


def genre_comparison(df, genre_list):
    cols = [f'genre_{g}' for g in genre_list]
    mask = df[cols].sum(axis=1) > 0
    return df[mask]


def genre_success_factors(df, genre):
    g = df[df[f'genre_{genre}'] == 1].copy()
    if g.empty or g['app_id'].nunique() < 2:
        return None
    from sklearn.linear_model import LinearRegression
    features = ['playtime_forever', 'review_length', 'word_count']
    g = g.dropna(subset=features + ['voted_up'])
    X = g[features]
    y = g['voted_up'].astype(int)
    model = LinearRegression()
    model.fit(X, y)
    return dict(zip(features, [round(c, 4) for c in model.coef_]))
