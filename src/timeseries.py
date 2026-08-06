import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils import GAMES, GENRE_IDS, RAW_DIR
import os


def daily_review_volume(df):
    daily = df.copy()
    daily['date'] = daily['review_date'].dt.date
    agg_dict = dict(
        total_reviews=('review_id', 'count'),
        positive_reviews=('voted_up', 'sum'),
        negative_reviews=('voted_up', lambda x: (~x).sum()),
        avg_playtime=('playtime_forever', 'mean'),
    )
    if 'vader_compound' in df.columns:
        agg_dict['avg_sentiment'] = ('vader_compound', 'mean')
    volume = daily.groupby(['app_id', 'date']).agg(**agg_dict).reset_index()
    volume['date'] = pd.to_datetime(volume['date'])
    volume['positive_pct'] = (volume['positive_reviews'] / volume['total_reviews'] * 100).round(1)
    volume['game_name'] = volume['app_id'].map(GAMES)
    return volume


def weekly_rolling_average(df, window=7):
    daily = daily_review_volume(df)
    daily = daily.sort_values('date')
    daily['rolling_positive_pct'] = daily.groupby('app_id')['positive_pct'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    if 'avg_sentiment' in daily.columns:
        daily['rolling_avg_sentiment'] = daily.groupby('app_id')['avg_sentiment'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    daily['rolling_review_volume'] = daily.groupby('app_id')['total_reviews'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return daily


def monthly_aggregation(df):
    monthly = df.copy()
    monthly['year_month'] = monthly['review_date'].dt.to_period('M')
    agg_dict = dict(
        total_reviews=('review_id', 'count'),
        positive_reviews=('voted_up', 'sum'),
        negative_reviews=('voted_up', lambda x: (~x).sum()),
        avg_playtime=('playtime_forever', 'mean'),
        avg_review_length=('review_length', 'mean'),
    )
    if 'vader_compound' in df.columns:
        agg_dict['avg_sentiment'] = ('vader_compound', 'mean')
    agg = monthly.groupby(['app_id', 'year_month']).agg(**agg_dict).reset_index()
    agg['positive_pct'] = (agg['positive_reviews'] / agg['total_reviews'] * 100).round(1)
    agg['year_month'] = agg['year_month'].astype(str)
    agg['date'] = pd.to_datetime(agg['year_month'])
    agg['game_name'] = agg['app_id'].map(GAMES)
    return agg


def seasonal_patterns(df):
    hourly = df.copy()
    hourly['hour'] = pd.to_datetime(hourly['timestamp_created'], unit='s').dt.hour
    hourly['day_of_week'] = hourly['review_date'].dt.day_name()
    hourly['month'] = hourly['review_date'].dt.month_name()

    by_hour = hourly.groupby('hour').agg(
        total=('review_id', 'count'),
        positive_pct=('voted_up', 'mean'),
    ).reset_index()
    by_hour['positive_pct'] = (by_hour['positive_pct'] * 100).round(1)

    by_day = hourly.groupby('day_of_week').agg(
        total=('review_id', 'count'),
        positive_pct=('voted_up', 'mean'),
    ).reset_index()
    by_day['positive_pct'] = (by_day['positive_pct'] * 100).round(1)

    by_month = hourly.groupby('month').agg(
        total=('review_id', 'count'),
        positive_pct=('voted_up', 'mean'),
    ).reset_index()
    by_month['positive_pct'] = (by_month['positive_pct'] * 100).round(1)

    return {'by_hour': by_hour, 'by_day': by_day, 'by_month': by_month}


def genre_monthly_trend(df, genre):
    col = f'genre_{genre}'
    g = df[df[col] == 1].copy()
    if g.empty:
        return pd.DataFrame()
    g['year_month'] = g['review_date'].dt.to_period('M').astype(str)
    agg = g.groupby('year_month').agg(
        total_reviews=('review_id', 'count'),
        positive_pct=('voted_up', 'mean'),
    ).reset_index()
    agg['positive_pct'] = (agg['positive_pct'] * 100).round(1)
    agg['date'] = pd.to_datetime(agg['year_month'])
    return agg


def load_content_events():
    path = f'{RAW_DIR}/content_events.csv'
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()


def content_impact_analysis(df, game_name, event_dates, window_days=30):
    g = df[df['game_name'] == game_name].copy()
    if g.empty:
        return None
    g['date'] = g['review_date'].dt.date
    results = []
    for ed in event_dates:
        ed = pd.Timestamp(ed).date()
        before = g[(g['date'] >= ed - timedelta(days=window_days)) & (g['date'] < ed)]
        after = g[(g['date'] >= ed) & (g['date'] <= ed + timedelta(days=window_days))]
        results.append({
            'event_date': ed,
            'before_pos_pct': round(before['voted_up'].mean() * 100, 1) if len(before) > 0 else None,
            'after_pos_pct': round(after['voted_up'].mean() * 100, 1) if len(after) > 0 else None,
            'before_count': len(before),
            'after_count': len(after),
        })
    return pd.DataFrame(results)


def redemption_arc_analysis(df, app_ids=None):
    if app_ids is None:
        app_ids = [1091500, 275850]  # Cyberpunk 2077, No Man's Sky

    results = {}
    for app_id in app_ids:
        game_df = df[df['app_id'] == app_id].copy()
        if game_df.empty:
            continue
        game_df['year_month'] = game_df['review_date'].dt.to_period('M')
        agg_dict = dict(
            total=('review_id', 'count'),
            positive_pct=('voted_up', 'mean'),
        )
        if 'vader_compound' in df.columns:
            agg_dict['avg_sentiment'] = ('vader_compound', 'mean')
        monthly = game_df.groupby('year_month').agg(**agg_dict).reset_index()
        monthly['positive_pct'] = monthly['positive_pct'] * 100
        monthly['year_month'] = monthly['year_month'].astype(str)
        monthly = monthly.sort_values('year_month')

        early = monthly.head(3)['positive_pct'].mean()
        late = monthly.tail(3)['positive_pct'].mean()
        change = late - early

        results[GAMES.get(app_id, str(app_id))] = {
            'monthly_data': monthly,
            'early_avg_pct': round(early, 1),
            'late_avg_pct': round(late, 1),
            'change_pct': round(change, 1),
            'improved': change > 0,
        }
    return results
