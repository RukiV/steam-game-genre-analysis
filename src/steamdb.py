import os
import glob
import numpy as np
import pandas as pd
from src.utils import RAW_DIR, PROCESSED_DIR, GAMES, GAME_IDS, GENRES, ensure_dirs

STEAMDB_MONTHLY_PATH = os.path.join(PROCESSED_DIR, 'steamdb_monthly.csv')


def _remove_cumulative_first_row(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove cumulative first rows (data that accumulated before SteamDB started tracking)."""
    n = len(df)
    if n < 6:
        return df

    first_pos = df.loc[0, 'positive']
    first_neg = abs(df.loc[0, 'negative'])
    if pd.isna(first_pos) or pd.isna(first_neg):
        return df

    first_total = first_pos + first_neg
    if first_total < 5000:
        return df

    next5 = df.iloc[1:6]
    nxt_totals = []
    for _, r in next5.iterrows():
        p = r['positive']
        n = r['negative']
        if not pd.isna(p) and not pd.isna(n):
            nxt_totals.append(p + abs(n))

    if not nxt_totals:
        return df

    med = np.median(nxt_totals)
    if med > 0 and first_total / med > 20:
        df = df.iloc[1:].reset_index(drop=True)

    return df


def _estimate_ratio(daily: pd.DataFrame, window: int = 10, ratio_std_threshold: float = 5.0) -> pd.DataFrame:
    """Fill NaN in one column using the pos:neg ratio from nearby complete rows.

    Skips estimation when the ratio values in the window are too volatile (std > threshold)
    to avoid creating artifacts during extreme events like review bombings.
    """
    df = daily.copy()
    n = len(df)

    for i in range(n):
        pos = df.loc[i, 'positive']
        neg = df.loc[i, 'negative']
        pos_missing = pd.isna(pos)
        neg_missing = pd.isna(neg)

        if not pos_missing and not neg_missing:
            continue
        if pos_missing and neg_missing:
            continue

        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        valid = df.iloc[lo:hi]
        mask = valid['positive'].notna() & valid['negative'].notna()
        valid = valid[mask]

        if len(valid) == 0:
            continue

        if pos_missing:
            valid = valid[valid['negative'] != 0]
            if len(valid) == 0:
                continue
            ratios = valid['positive'] / valid['negative'].abs()
            if ratios.std() > ratio_std_threshold:
                continue
            df.loc[i, 'positive'] = round(abs(neg) * ratios.median())
        else:
            valid = valid[valid['positive'] != 0]
            if len(valid) == 0:
                continue
            inv = valid['negative'].abs() / valid['positive']
            if inv.std() > ratio_std_threshold:
                continue
            df.loc[i, 'negative'] = -round(pos * inv.median())

    df[['positive', 'negative']] = df[['positive', 'negative']].fillna(0).astype(int)
    return df


def load_steamdb_history(force_reprocess=False):
    monthly_path = STEAMDB_MONTHLY_PATH
    if os.path.exists(monthly_path) and not force_reprocess:
        return pd.read_csv(monthly_path, parse_dates=['date'])

    pattern = os.path.join(RAW_DIR, 'steamdb_chart_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No steamdb_chart_*.csv files found in {RAW_DIR}")

    rows = []
    pre_data = {}
    for fpath in files:
        base = os.path.basename(fpath)
        app_id = int(base.replace('steamdb_chart_', '').replace('.csv', ''))
        name = GAMES.get(app_id, str(app_id))
        df = pd.read_csv(fpath, parse_dates=['DateTime'])
        df.columns = ['date', 'positive', 'negative']
        df['app_id'] = app_id
        df['game_name'] = name
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['positive'] = pd.to_numeric(df['positive'], errors='coerce')
        df['negative'] = pd.to_numeric(df['negative'], errors='coerce')

        pre_pos, pre_neg = 0, 0
        if len(df) > 0:
            first_pos = df['positive'].iloc[0]
            first_neg = abs(df['negative'].iloc[0])
            if not pd.isna(first_pos) and not pd.isna(first_neg) and first_pos + first_neg > 5000:
                next5 = df.iloc[1:6]
                nxt_totals = []
                for _, r in next5.iterrows():
                    p = r['positive']
                    n = r['negative']
                    if not pd.isna(p) and not pd.isna(n):
                        nxt_totals.append(p + abs(n))
                if nxt_totals and np.median(nxt_totals) > 0 and (first_pos + first_neg) / np.median(nxt_totals) > 20:
                    pre_pos, pre_neg = int(first_pos), int(first_neg)
        pre_data[app_id] = (pre_pos, pre_neg)

        df = _remove_cumulative_first_row(df)
        rows.append(df)

    daily = pd.concat(rows, ignore_index=True)
    daily = daily.dropna(subset=['positive', 'negative'])
    daily['positive'] = daily['positive'].astype(int)
    daily['negative'] = daily['negative'].astype(int)
    daily['year_month'] = pd.to_datetime(daily['date']).dt.to_period('M').astype(str)
    monthly = daily.groupby(['app_id', 'game_name', 'year_month']).agg(
        total_positive=('positive', 'sum'),
        total_negative=('negative', 'sum'),
        total_reviews=('positive', 'count'),
        days=('date', 'count'),
    ).reset_index()
    monthly['total_negative'] = monthly['total_negative'].abs()
    monthly['total_reviews'] = monthly['total_positive'] + monthly['total_negative']
    monthly['positive_pct'] = (
        monthly['total_positive'] / monthly['total_reviews'] * 100
    ).round(1)
    monthly['date'] = pd.to_datetime(monthly['year_month'].astype(str) + '-01')
    monthly = monthly.sort_values(['app_id', 'date']).reset_index(drop=True)

    pre_pos_map = monthly['app_id'].map({k: v[0] for k, v in pre_data.items()})
    pre_neg_map = monthly['app_id'].map({k: v[1] for k, v in pre_data.items()})
    monthly['pre_total_positive'] = pre_pos_map.fillna(0).astype(int)
    monthly['pre_total_negative'] = pre_neg_map.fillna(0).astype(int)

    monthly['cumulative_pct'] = 0.0
    for app_id in monthly['app_id'].unique():
        mask = monthly['app_id'] == app_id
        idxs = monthly[mask].index
        pre_pos = monthly.loc[idxs[0], 'pre_total_positive']
        pre_neg = monthly.loc[idxs[0], 'pre_total_negative']
        cum_pos, cum_neg = pre_pos, pre_neg
        for idx in idxs:
            cum_pos += monthly.loc[idx, 'total_positive']
            cum_neg += monthly.loc[idx, 'total_negative']
            total = cum_pos + cum_neg
            monthly.loc[idx, 'cumulative_pct'] = round(cum_pos / total * 100, 1) if total > 0 else 0.0

    ensure_dirs()
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved SteamDB monthly data: {len(monthly)} rows, {monthly['app_id'].nunique()} games")
    return monthly


def load_steamdb_daily():
    """Load and return daily SteamDB data (positive/negative/total reviews per day) for all games."""
    pattern = os.path.join(RAW_DIR, 'steamdb_chart_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No steamdb_chart_*.csv files found in {RAW_DIR}")

    rows = []
    for fpath in files:
        base = os.path.basename(fpath)
        app_id = int(base.replace('steamdb_chart_', '').replace('.csv', ''))
        name = GAMES.get(app_id, str(app_id))
        df = pd.read_csv(fpath, parse_dates=['DateTime'])
        df.columns = ['date', 'positive', 'negative']
        df['app_id'] = app_id
        df['game_name'] = name
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['positive'] = pd.to_numeric(df['positive'], errors='coerce')
        df['negative'] = pd.to_numeric(df['negative'], errors='coerce')
        df = _remove_cumulative_first_row(df)
        rows.append(df)

    daily = pd.concat(rows, ignore_index=True)
    daily = daily.dropna(subset=['positive', 'negative'])
    daily['positive'] = daily['positive'].astype(int)
    daily['negative'] = daily['negative'].astype(int)
    daily['total_reviews'] = daily['positive'] + daily['negative'].abs()
    daily['positive_pct'] = (daily['positive'] / daily['total_reviews'].replace(0, np.nan) * 100).round(1)
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


def steamdb_redemption_arcs(app_ids=None, monthly=None):
    if app_ids is None:
        app_ids = [1091500, 275850]

    if monthly is None:
        monthly = load_steamdb_history()

    results = {}
    for app_id in app_ids:
        game_df = monthly[monthly['app_id'] == app_id].copy()
        if game_df.empty:
            continue
        game_df = game_df.sort_values('date')
        early = game_df.head(3)['positive_pct'].mean()
        late = game_df.tail(3)['positive_pct'].mean()
        change = late - early

        results[GAMES.get(app_id, str(app_id))] = {
            'monthly_data': game_df[['year_month', 'total_reviews', 'positive_pct', 'date']].copy(),
            'early_avg_pct': round(early, 1),
            'late_avg_pct': round(late, 1),
            'change_pct': round(change, 1),
            'improved': change > 0,
        }
    return results


def steamdb_seasonal_patterns(daily, game_names):
    sd = daily[daily['game_name'].isin(game_names)].copy()
    if sd.empty:
        return {'by_day': pd.DataFrame(), 'by_month': pd.DataFrame()}
    sd['day_of_week'] = sd['date'].dt.day_name()
    sd['month'] = sd['date'].dt.month_name()
    by_day = sd.groupby('day_of_week').agg(
        total=('total_reviews', 'sum'),
        positive_pct=('positive_pct', 'mean'),
    ).reset_index()
    by_day['positive_pct'] = by_day['positive_pct'].round(1)
    by_month = sd.groupby('month').agg(
        total=('total_reviews', 'sum'),
        positive_pct=('positive_pct', 'mean'),
    ).reset_index()
    by_month['positive_pct'] = by_month['positive_pct'].round(1)
    return {'by_day': by_day, 'by_month': by_month}


def steamdb_genre_trend(genre_key, monthly):
    app_ids = GENRES.get(genre_key, [])
    g = monthly[monthly['app_id'].isin(app_ids)]
    if g.empty:
        return pd.DataFrame()
    agg = g.groupby('year_month').agg(
        total_positive=('total_positive', 'sum'),
        total_negative=('total_negative', 'sum'),
    ).reset_index()
    agg['total_reviews'] = agg['total_positive'] + agg['total_negative']
    agg['positive_pct'] = (agg['total_positive'] / agg['total_reviews'] * 100).round(1)
    agg['date'] = pd.to_datetime(agg['year_month'].astype(str) + '-01')
    return agg.sort_values('date')
